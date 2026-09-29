"""
RAG resume screening pipeline.

Flow:
1. Clean resume and job-description text
2. Split the resume into chunks
3. Embed chunks and store them in ChromaDB
4. Retrieve the chunks most similar to the job description
5. Send retrieved context + job description to Gemini
6. Combine retrieval similarity and the structured LLM evaluation
"""
import uuid
from typing import Dict, List, Optional

import config
from llm_evaluator import LLMEvaluationError, LLMEvaluator
from utils import clean_text, split_resume_into_chunks
from vector_store import VectorStore


class RAGResumeScreener:
    """Indexes a resume, retrieves relevant sections, then evaluates with Gemini."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_evaluator = LLMEvaluator()

    def screen_resume(
        self,
        resume_text: str,
        job_description: str,
        resume_id: Optional[str] = None,
    ) -> Dict:
        resume_id = resume_id or f"resume_{uuid.uuid4().hex[:12]}"
        job_description = clean_text(job_description)
        resume_text = clean_text(resume_text)

        if not job_description:
            raise ValueError("Job description is empty after cleaning")
        if not resume_text:
            raise ValueError("Resume text is empty after cleaning")

        chunks = split_resume_into_chunks(resume_text)
        if not chunks:
            raise ValueError("Resume could not be split into retrievable chunks")

        self.vector_store.upsert_resume_chunks(resume_id, chunks)
        return self._evaluate_indexed_resume(resume_id, job_description, chunk_count=len(chunks))

    def batch_screen_resumes(self, resumes: List[Dict], job_description: str) -> List[Dict]:
        """
        Index every resume first, then retrieve and evaluate each one against the same JD.
        Each item in `resumes` must include `text` and should include `id`.
        """
        job_description = clean_text(job_description)
        if not job_description:
            raise ValueError("Job description is empty after cleaning")
        if not resumes:
            return []

        indexed = []
        results = []

        for index, resume in enumerate(resumes, start=1):
            resume_id = str(resume.get("id") or f"resume_{index}")
            resume_text = clean_text(resume.get("text") or "")
            if not resume_text:
                results.append(
                    self._error_result(
                        resume_id,
                        "Resume text is empty after cleaning",
                        error_type="invalid_input",
                    )
                )
                continue
            try:
                chunks = split_resume_into_chunks(resume_text)
                if not chunks:
                    raise ValueError("Resume could not be split into retrievable chunks")
                self.vector_store.upsert_resume_chunks(resume_id, chunks)
                indexed.append({"resume_id": resume_id, "chunk_count": len(chunks)})
            except Exception as exc:
                results.append(
                    self._error_result(
                        resume_id,
                        str(exc),
                        error_type="indexing_error",
                    )
                )

        for item in indexed:
            result = self._evaluate_indexed_resume(
                item["resume_id"],
                job_description,
                chunk_count=item["chunk_count"],
            )
            result["resume_id"] = item["resume_id"]
            results.append(result)

        results.sort(
            key=lambda item: (
                0 if item.get("status") == "success" else 1,
                -(item.get("final_score") if item.get("final_score") is not None else -1.0),
            )
        )
        return results

    def _evaluate_indexed_resume(
        self,
        resume_id: str,
        job_description: str,
        chunk_count: Optional[int] = None,
    ) -> Dict:
        retrieved = self.vector_store.retrieve_resume_chunks(
            job_description,
            resume_id,
            top_k=config.TOP_K_CHUNKS,
        )
        if not retrieved:
            return self._error_result(
                resume_id,
                "No resume chunks were retrieved from ChromaDB for this candidate.",
                error_type="retrieval_error",
            )

        retrieval_score = round(self.vector_store.retrieval_score(retrieved), 3)
        retrieved_context = self._format_retrieved_context(retrieved)

        try:
            llm_result = self.llm_evaluator.evaluate_match(retrieved_context, job_description)
        except LLMEvaluationError as exc:
            return self._error_result(
                resume_id,
                str(exc),
                error_type="llm_error",
                extra={
                    "retrieval_score": retrieval_score,
                    "retrieved_chunks": retrieved,
                    "chunk_count": chunk_count or len(retrieved),
                },
            )

        llm_score = llm_result["score"]
        final_score = round(
            retrieval_score * config.RETRIEVAL_WEIGHT + llm_score * config.LLM_WEIGHT,
            3,
        )
        recommendation = llm_result.get("recommendation") or self._recommendation_from_score(
            final_score
        )

        return {
            "status": "success",
            "resume_id": resume_id,
            "final_score": final_score,
            "retrieval_score": retrieval_score,
            "llm_score": llm_score,
            "llm_details": llm_result,
            "recommendation": recommendation,
            "retrieved_chunks": retrieved,
            "chunk_count": chunk_count or len(retrieved),
            "weights": {
                "retrieval": config.RETRIEVAL_WEIGHT,
                "llm": config.LLM_WEIGHT,
            },
            "error": None,
        }

    @staticmethod
    def _format_retrieved_context(retrieved: List[Dict]) -> str:
        blocks = []
        for index, chunk in enumerate(retrieved, start=1):
            similarity = chunk.get("similarity", 0.0)
            text = chunk.get("text", "").strip()
            blocks.append(f"[Excerpt {index} | similarity={similarity:.2f}]\n{text}")
        return "\n\n".join(blocks)

    @staticmethod
    def _recommendation_from_score(score: float) -> str:
        if score >= 0.8:
            return "Strongly Recommended"
        if score >= 0.65:
            return "Recommended"
        if score >= 0.5:
            return "Consider"
        if score >= 0.35:
            return "Weak Match"
        return "Not Recommended"

    @staticmethod
    def _error_result(
        resume_id: str,
        message: str,
        error_type: str,
        extra: Optional[Dict] = None,
    ) -> Dict:
        result = {
            "status": "error",
            "resume_id": resume_id,
            "final_score": None,
            "retrieval_score": None,
            "llm_score": None,
            "llm_details": {
                "score": None,
                "reasoning": message,
                "matched_skills": [],
                "missing_skills": [],
                "recommendation": "",
            },
            "recommendation": None,
            "retrieved_chunks": [],
            "chunk_count": 0,
            "weights": {
                "retrieval": config.RETRIEVAL_WEIGHT,
                "llm": config.LLM_WEIGHT,
            },
            "error": message,
            "error_type": error_type,
        }
        if extra:
            result.update(extra)
        return result
