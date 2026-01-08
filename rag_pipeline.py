"""
RAG Pipeline combining Semantic Search and LLM Evaluation
"""
from vector_store import VectorStore
from llm_evaluator import LLMEvaluator
import config
from typing import Dict, List


class RAGResumeScreener:
    """
    Main RAG Pipeline for Resume Screening
    Combines semantic search (0.6 weight) and LLM evaluation (0.4 weight)
    """
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_evaluator = LLMEvaluator()
    
    def screen_resume(self, resume_text: str, job_description: str) -> Dict:
        """
        Screen a resume against a job description using RAG pipeline
        
        Args:
            resume_text: The candidate's resume text
            job_description: The job description text
            
        Returns:
            Dictionary containing:
            - final_score: Weighted combination of semantic and LLM scores
            - semantic_score: Score from semantic search (0-1)
            - llm_score: Score from LLM evaluation (0-1)
            - llm_details: Detailed LLM evaluation results
            - recommendation: Overall recommendation
        """
        # Step 1: Semantic Search Evaluation (0.6 weight)
        semantic_score = self.vector_store.calculate_semantic_score(
            resume_text, job_description
        )
        
        # Step 2: LLM Evaluation (0.4 weight)
        llm_result = self.llm_evaluator.evaluate_match(resume_text, job_description)
        llm_score = llm_result['score']
        
        # Step 3: Weighted Combination
        final_score = (
            semantic_score * config.SEMANTIC_SEARCH_WEIGHT +
            llm_score * config.LLM_WEIGHT
        )
        
        # Step 4: Generate Recommendation
        recommendation = self._generate_recommendation(final_score)
        
        return {
            'final_score': round(final_score, 3),
            'semantic_score': round(semantic_score, 3),
            'llm_score': round(llm_score, 3),
            'llm_details': llm_result,
            'recommendation': recommendation,
            'weights': {
                'semantic_search': config.SEMANTIC_SEARCH_WEIGHT,
                'llm': config.LLM_WEIGHT
            }
        }
    
    def batch_screen_resumes(self, resumes: List[Dict], job_description: str) -> List[Dict]:
        """
        Screen multiple resumes against a job description
        
        Args:
            resumes: List of dictionaries with 'id' and 'text' keys
            job_description: The job description text
            
        Returns:
            List of screening results, sorted by final_score (descending)
        """
        results = []
        
        for resume in resumes:
            result = self.screen_resume(resume['text'], job_description)
            result['resume_id'] = resume.get('id', 'unknown')
            results.append(result)
        
        # Sort by final score (highest first)
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results
    
    def _generate_recommendation(self, score: float) -> str:
        """Generate recommendation based on final score"""
        if score >= 0.8:
            return "Strongly Recommended - Excellent match"
        elif score >= 0.65:
            return "Recommended - Good match"
        elif score >= 0.5:
            return "Consider - Moderate match"
        elif score >= 0.35:
            return "Weak Match - Review carefully"
        else:
            return "Not Recommended - Poor match"
    
    def add_resume_to_index(self, resume_id: str, resume_text: str, metadata: Dict = None):
        """Add a resume to the vector store for future searches"""
        self.vector_store.add_resume(resume_id, resume_text, metadata)
    
    def add_job_to_index(self, job_id: str, job_description: str, metadata: Dict = None):
        """Add a job description to the vector store"""
        self.vector_store.add_job_description(job_id, job_description, metadata)
