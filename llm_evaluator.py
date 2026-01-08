"""
LLM-based Resume Evaluation using Gemini
"""
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate
import config
from typing import Dict


class LLMEvaluator:
    """Evaluates resume-job description match using Gemini LLM"""
    
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=config.TEMPERATURE
        )
        
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume screener. Your task is to evaluate how well a candidate's resume matches a job description.
            
            Analyze the following aspects:
            1. Required skills match
            2. Experience relevance
            3. Education and qualifications
            4. Overall fit for the role
            
            Provide a score between 0 and 1, where:
            - 0.9-1.0: Excellent match, highly qualified
            - 0.7-0.89: Good match, qualified
            - 0.5-0.69: Moderate match, some qualifications
            - 0.3-0.49: Weak match, few qualifications
            - 0.0-0.29: Poor match, not qualified
            
            Respond with ONLY a JSON object containing:
            {{
                "score": <float between 0 and 1>,
                "reasoning": "<brief explanation>",
                "matched_skills": ["<skill1>", "<skill2>", ...],
                "missing_skills": ["<skill1>", "<skill2>", ...]
            }}"""),
            ("human", """Job Description:
{job_description}

Resume:
{resume_text}

Evaluate the match and provide your assessment.""")
        ])
    
    def evaluate_match(self, resume_text: str, job_description: str) -> Dict:
        """
        Evaluate resume-job description match using LLM
        Returns a dictionary with score and reasoning
        """
        try:
            messages = self.evaluation_prompt.format_messages(
                job_description=job_description,
                resume_text=resume_text
            )
            
            response = self.llm.invoke(messages)
            
            # Parse the response
            response_text = response.content.strip()
            
            # Try to extract JSON from response
            import json
            import re
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    'score': float(result.get('score', 0.0)),
                    'reasoning': result.get('reasoning', ''),
                    'matched_skills': result.get('matched_skills', []),
                    'missing_skills': result.get('missing_skills', [])
                }
            else:
                # Fallback: try to extract score from text
                score_match = re.search(r'"score":\s*([0-9.]+)', response_text)
                if score_match:
                    score = float(score_match.group(1))
                    return {
                        'score': score,
                        'reasoning': response_text,
                        'matched_skills': [],
                        'missing_skills': []
                    }
                else:
                    # Default fallback
                    return {
                        'score': 0.5,
                        'reasoning': response_text,
                        'matched_skills': [],
                        'missing_skills': []
                    }
        
        except Exception as e:
            print(f"Error in LLM evaluation: {str(e)}")
            return {
                'score': 0.0,
                'reasoning': f'Error: {str(e)}',
                'matched_skills': [],
                'missing_skills': []
            }
