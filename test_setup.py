"""
Lightweight checks for imports, chunking, and ChromaDB retrieval.
Does not call Gemini, so no API key is required.
"""
import os
import shutil
import tempfile

os.environ.setdefault("VECTOR_DB_PATH", tempfile.mkdtemp(prefix="resume_screener_test_"))

from rag_pipeline import RAGResumeScreener  # noqa: E402
from utils import clean_text, split_resume_into_chunks  # noqa: E402
from vector_store import VectorStore  # noqa: E402


def test_error_is_not_a_zero_score() -> None:
    result = RAGResumeScreener._error_result(
        "candidate_1",
        "Gemini evaluation failed (TimeoutError): timed out",
        "llm_error",
        extra={"retrieval_score": 0.71},
    )
    assert result["status"] == "error"
    assert result["final_score"] is None
    assert result["llm_score"] is None
    assert result["retrieval_score"] == 0.71
    assert "Gemini" in result["error"]


def test_imports() -> None:
    import config  # noqa: F401
    import llm_evaluator  # noqa: F401
    import rag_pipeline  # noqa: F401
    import utils  # noqa: F401
    import vector_store  # noqa: F401


def test_chunking_and_retrieval() -> None:
    resume = clean_text(
        """
        Jane Doe
        Data Scientist

        Skills
        Python, SQL, scikit-learn, LangChain, ChromaDB, AWS.

        Experience
        Built a retrieval-augmented search service that indexed support articles
        with embeddings and returned the most relevant passages to an LLM.

        Education
        M.S. in Computer Science.
        """
    )
    job = clean_text(
        """
        We need a data scientist who can build RAG systems with embeddings,
        vector databases, and Python.
        """
    )
    chunks = split_resume_into_chunks(resume, chunk_size=180, chunk_overlap=40)
    assert chunks, "Expected resume chunks"

    db_path = os.environ["VECTOR_DB_PATH"]
    store = VectorStore(persist_path=db_path, collection_name="test_resume_chunks")
    stored = store.upsert_resume_chunks("candidate_jane", chunks)
    assert stored == len(chunks)

    matches = store.retrieve_resume_chunks(job, "candidate_jane", top_k=3)
    assert matches, "Expected retrieved chunks for the supplied resume"
    assert all(match["metadata"].get("resume_id") == "candidate_jane" for match in matches)
    score = store.retrieval_score(matches)
    assert 0.0 <= score <= 1.0

    store.delete_resume("candidate_jane")
    assert store.count_resume_chunks("candidate_jane") == 0

    python_resume = [
        "Backend engineer using Django, PostgreSQL, and Redis for APIs.",
        "Deployed services on AWS ECS and wrote pytest coverage.",
    ]
    ml_resume = [
        "Machine learning engineer using PyTorch, transformers, and vector databases.",
        "Built RAG pipelines with ChromaDB and sentence embeddings.",
    ]
    store.upsert_resume_chunks("python_dev", python_resume)
    store.upsert_resume_chunks("ml_dev", ml_resume)
    ml_job = "Looking for RAG, embeddings, and vector database experience."
    ml_matches = store.retrieve_resume_chunks(ml_job, "ml_dev", top_k=2)
    python_matches = store.retrieve_resume_chunks(ml_job, "python_dev", top_k=2)
    assert ml_matches, "Expected matches from the ML resume"
    assert all(match["metadata"]["resume_id"] == "ml_dev" for match in ml_matches)
    assert all(match["metadata"]["resume_id"] == "python_dev" for match in python_matches)
    assert store.retrieval_score(ml_matches) >= store.retrieval_score(python_matches)

    store.clear_collection()
    shutil.rmtree(db_path, ignore_errors=True)


if __name__ == "__main__":
    test_imports()
    test_error_is_not_a_zero_score()
    test_chunking_and_retrieval()
    print("Setup checks passed: imports, chunking, and ChromaDB retrieval.")
