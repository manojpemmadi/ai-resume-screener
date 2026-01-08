"""
Main script for AI Resume Screener
"""
from rag_pipeline import RAGResumeScreener
import json
import sys


def main():
    """Main function for AI Resume Screener"""
    
    # Initialize the RAG pipeline
    print("Initializing AI Resume Screener...")
    screener = RAGResumeScreener()
    
    # Get job description
    print("\nEnter job description (press Enter twice to finish):")
    job_description_lines = []
    while True:
        try:
            line = input()
            if line == "" and job_description_lines and job_description_lines[-1] == "":
                break
            job_description_lines.append(line)
        except EOFError:
            break
    
    job_description = "\n".join(job_description_lines).strip()
    
    if not job_description:
        print("Error: Job description cannot be empty")
        sys.exit(1)
    
    # Get resume text
    print("\nEnter resume text (press Enter twice to finish):")
    resume_lines = []
    while True:
        try:
            line = input()
            if line == "" and resume_lines and resume_lines[-1] == "":
                break
            resume_lines.append(line)
        except EOFError:
            break
    
    resume_text = "\n".join(resume_lines).strip()
    
    if not resume_text:
        print("Error: Resume text cannot be empty")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Screening Resume against Job Description")
    print("="*60)
    
    # Screen the resume
    result = screener.screen_resume(resume_text, job_description)
    
    # Display results
    print("\n[SCREENING RESULTS]")
    print(f"\nFinal Score: {result['final_score']:.1%}")
    print(f"Recommendation: {result['recommendation']}")
    
    print(f"\n[SCORE BREAKDOWN]")
    print(f"  Semantic Search Score: {result['semantic_score']:.1%} (Weight: {result['weights']['semantic_search']})")
    print(f"  LLM Evaluation Score: {result['llm_score']:.1%} (Weight: {result['weights']['llm']})")
    
    if result['llm_details'].get('reasoning'):
        print(f"\n[LLM REASONING]")
        print(f"  {result['llm_details']['reasoning']}")
    
    if result['llm_details'].get('matched_skills'):
        print(f"\n[MATCHED SKILLS]")
        for skill in result['llm_details']['matched_skills']:
            print(f"  - {skill}")
    
    if result['llm_details'].get('missing_skills'):
        print(f"\n[MISSING SKILLS]")
        for skill in result['llm_details']['missing_skills']:
            print(f"  - {skill}")
    
    print("\n" + "="*60)
    
    # Save results to JSON
    output_file = 'screening_result.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[SUCCESS] Results saved to '{output_file}'")


if __name__ == "__main__":
    main()
