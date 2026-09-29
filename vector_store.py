"""
ChromaDB vector store for resume chunks and job-description retrieval.
"""
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config


class VectorStore:
    """Embeds resume chunks, stores them in ChromaDB, and retrieves by job description."""

    def __init__(self, persist_path: Optional[str] = None, collection_name: Optional[str] = None):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        db_path = persist_path or config.VECTOR_DB_PATH
        try:
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False),
            )
        except TypeError:
            self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.embedding_model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def delete_resume(self, resume_id: str) -> None:
        existing = self.collection.get(where={"resume_id": resume_id})
        ids = existing.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)

    def upsert_resume_chunks(
        self,
        resume_id: str,
        chunks: List[str],
        extra_metadata: Optional[Dict] = None,
    ) -> int:
        """
        Replace any previous chunks for this resume_id, then insert the new ones.
        Returns the number of chunks stored.
        """
        if not resume_id:
            raise ValueError("resume_id is required")
        if not chunks:
            raise ValueError("Cannot index an empty resume")

        self.delete_resume(resume_id)
        embeddings = self.embed_texts(chunks)
        ids = [f"{resume_id}::chunk::{index}" for index in range(len(chunks))]
        metadatas = []
        for index, _chunk in enumerate(chunks):
            metadata = {
                "resume_id": resume_id,
                "chunk_index": index,
                "doc_type": "resume_chunk",
            }
            if extra_metadata:
                for key, value in extra_metadata.items():
                    if value is not None and isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def count_resume_chunks(self, resume_id: str) -> int:
        result = self.collection.get(where={"resume_id": resume_id})
        return len(result.get("ids") or [])

    def retrieve_resume_chunks(
        self,
        job_description: str,
        resume_id: str,
        top_k: int = None,
    ) -> List[Dict]:
        """
        Retrieve the resume chunks that are most similar to the job description.
        Similarity is cosine similarity derived from Chroma cosine distance.
        """
        chunk_count = self.count_resume_chunks(resume_id)
        if chunk_count == 0:
            return []

        k = min(top_k or config.TOP_K_CHUNKS, chunk_count)
        query_embedding = self.embed_texts([job_description])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"resume_id": resume_id},
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        for index, chunk_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            similarity = max(0.0, min(1.0, 1.0 - distance))
            matches.append(
                {
                    "id": chunk_id,
                    "text": documents[index] if index < len(documents) else "",
                    "similarity": round(similarity, 4),
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                }
            )
        return matches

    def retrieval_score(self, matches: List[Dict]) -> float:
        """Average cosine similarity of retrieved chunks (0-1)."""
        if not matches:
            return 0.0
        return float(sum(match["similarity"] for match in matches) / len(matches))

    def clear_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
