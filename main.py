"""
Command-line entry point for the AI Resume Screener.
"""
import json
import sys

from rag_pipeline import RAGResumeScreener
from utils import clean_text


def _read_multiline(prompt: str) -> str:
    print(f"\n{prompt} (press Enter on an empty line to finish):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return clean_text("\n".join(lines))


def _print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print("SCREENING RESULT")
    print("=" * 60)

    if result.get("status") != "success":
        print("\nStatus: ERROR (not a candidate score)")
        print(f"Error type: {result.get('error_type')}")
        print(f"Details: {result.get('error')}")
        if result.get("retrieval_score") is not None:
            print(f"Retrieval score (partial): {result['retrieval_score']:.1%}")
        return

    print(f"\nFinal score: {result['final_score']:.1%}")
    print(f"Recommendation: {result['recommendation']}")
    print("\nScore breakdown")
    print(
        f"  Retrieval score: {result['retrieval_score']:.1%} "
        f"(weight {result['weights']['retrieval']})"
    )
    print(
        f"  LLM score: {result['llm_score']:.1%} "
        f"(weight {result['weights']['llm']})"
    )

    details = result.get("llm_details") or {}
    if details.get("reasoning"):
        print("\nLLM reasoning")
        print(f"  {details['reasoning']}")
    if details.get("matched_skills"):
        print("\nMatched skills")
        for skill in details["matched_skills"]:
            print(f"  - {skill}")
    if details.get("missing_skills"):
        print("\nMissing skills")
        for skill in details["missing_skills"]:
            print(f"  - {skill}")

    retrieved = result.get("retrieved_chunks") or []
    if retrieved:
        print(f"\nRetrieved excerpts ({len(retrieved)} of {result.get('chunk_count')} chunks)")
        for index, chunk in enumerate(retrieved, start=1):
            preview = (chunk.get("text") or "").replace("\n", " ")[:180]
            print(f"  {index}. similarity={chunk.get('similarity', 0):.2f} | {preview}")


def main() -> None:
    print("Initializing AI Resume Screener...")
    try:
        screener = RAGResumeScreener()
    except Exception as exc:
        print(f"Failed to initialize screener: {exc}")
        sys.exit(1)

    job_description = _read_multiline("Enter job description")
    if not job_description:
        print("Error: Job description cannot be empty")
        sys.exit(1)

    resume_text = _read_multiline("Enter resume text")
    if not resume_text:
        print("Error: Resume text cannot be empty")
        sys.exit(1)

    print("\nIndexing resume, retrieving relevant sections, and evaluating with Gemini...")
    try:
        result = screener.screen_resume(resume_text, job_description)
    except Exception as exc:
        print(f"Screening failed: {exc}")
        sys.exit(1)

    _print_result(result)

    output_file = "screening_result.json"
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    print(f"\nResults saved to '{output_file}'")

    if result.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
