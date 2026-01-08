"""
Vector Store Management for Resume Embeddings
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import config


class VectorStore:
    """Manages vector embeddings and semantic search"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_resume(self, resume_id: str, resume_text: str, metadata: Dict = None):
        """Add a resume to the vector store"""
        embedding = self.embedding_model.encode(resume_text).tolist()
        
        self.collection.add(
            ids=[resume_id],
            embeddings=[embedding],
            documents=[resume_text],
            metadatas=[metadata or {}]
        )
    
    def add_job_description(self, job_id: str, job_description: str, metadata: Dict = None):
        """Add a job description to the vector store"""
        embedding = self.embedding_model.encode(job_description).tolist()
        
        self.collection.add(
            ids=[f"job_{job_id}"],
            embeddings=[embedding],
            documents=[job_description],
            metadatas=[metadata or {}]
        )
    
    def semantic_search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search on resumes/job descriptions
        Returns list of matches with scores
        """
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        matches = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # Convert distance to similarity score (1 - distance for cosine similarity)
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Cosine distance to similarity
                
                matches.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'similarity': similarity,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })
        
        return matches
    
    def calculate_semantic_score(self, resume_text: str, job_description: str) -> float:
        """
        Calculate semantic similarity score between resume and job description
        Returns a score between 0 and 1
        """
        matches = self.semantic_search(job_description, top_k=1)
        
        if matches:
            # Use the best match score
            return matches[0]['similarity']
        else:
            # If no matches, calculate direct similarity
            resume_embedding = self.embedding_model.encode(resume_text)
            job_embedding = self.embedding_model.encode(job_description)
            
            # Cosine similarity
            import numpy as np
            similarity = np.dot(resume_embedding, job_embedding) / (
                np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
            )
            return float(similarity)
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        self.client.delete_collection(name=config.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
