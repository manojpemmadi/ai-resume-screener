"""
Gemini evaluation of retrieved resume context against a job description.
"""
import os
from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate

import config


class LLMEvaluationError(Exception):
    """Raised when the LLM call or structured parse fails. This is not a low candidate score."""


class ResumeEvaluation(BaseModel):
    """Structured screening result produced by Gemini."""

    score: float = Field(
        ...,
        description="Match score between 0.0 and 1.0 based only on the retrieved resume excerpts.",
    )
    reasoning: str = Field(
        ...,
        description="Short explanation grounded in the retrieved excerpts and job description.",
    )
    matched_skills: List[str] = Field(
        default_factory=list,
        description="Skills or qualifications evidenced in the retrieved resume excerpts.",
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="Important job requirements not evidenced in the retrieved excerpts.",
    )
    recommendation: str = Field(
        ...,
        description=(
            "One of: Strongly Recommended, Recommended, Consider, Weak Match, Not Recommended"
        ),
    )

    @field_validator("score")
    @classmethod
    def clamp_score(cls, value: float) -> float:
        numeric = float(value)
        if numeric < 0.0:
            return 0.0
        if numeric > 1.0:
            return 1.0
        return numeric

    @field_validator("matched_skills", "missing_skills", mode="before")
    @classmethod
    def ensure_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)


class LLMEvaluator:
    """Evaluates retrieved resume context against a job description using Gemini."""

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found. Create a .env file with GEMINI_API_KEY=your_key."
            )

        os.environ.setdefault("GEMINI_API_KEY", config.GEMINI_API_KEY)
        os.environ.setdefault("GOOGLE_API_KEY", config.GEMINI_API_KEY)
        llm_kwargs = {
            "model": config.LLM_MODEL,
            "temperature": config.TEMPERATURE,
        }
        try:
            self.llm = ChatGoogleGenerativeAI(api_key=config.GEMINI_API_KEY, **llm_kwargs)
        except TypeError:
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=config.GEMINI_API_KEY,
                **llm_kwargs,
            )
        self.structured_llm = self._build_structured_llm()
        self.evaluation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert technical recruiter. Evaluate how well a candidate "
                    "matches a job description using ONLY the retrieved resume excerpts. "
                    "Treat information that is not in the excerpts as not evidenced. "
                    "Do not invent experience, employers, or skills.\n\n"
                    "Score from 0.0 to 1.0:\n"
                    "- 0.90-1.00: Excellent match\n"
                    "- 0.70-0.89: Good match\n"
                    "- 0.50-0.69: Moderate match\n"
                    "- 0.30-0.49: Weak match\n"
                    "- 0.00-0.29: Poor match\n\n"
                    "Recommendation must be one of: Strongly Recommended, Recommended, "
                    "Consider, Weak Match, Not Recommended.",
                ),
                (
                    "human",
                    "Job description:\n{job_description}\n\n"
                    "Retrieved resume excerpts (most relevant to this job):\n{retrieved_context}\n\n"
                    "Evaluate the candidate for this role.",
                ),
            ]
        )

    def _build_structured_llm(self):
        try:
            return self.llm.with_structured_output(
                ResumeEvaluation,
                method="json_schema",
            )
        except TypeError:
            return self.llm.with_structured_output(ResumeEvaluation)

    def evaluate_match(
        self,
        retrieved_context: str,
        job_description: str,
    ) -> Dict:
        """
        Run a structured Gemini evaluation on retrieved resume context.
        Raises LLMEvaluationError on API or parsing failure.
        """
        if not job_description or not job_description.strip():
            raise ValueError("Job description is empty")
        if not retrieved_context or not retrieved_context.strip():
            raise ValueError("Retrieved resume context is empty")

        messages = self.evaluation_prompt.format_messages(
            job_description=job_description.strip(),
            retrieved_context=retrieved_context.strip(),
        )

        try:
            result = self.structured_llm.invoke(messages)
        except Exception as primary_error:
            try:
                fallback_llm = self.llm.with_structured_output(
                    ResumeEvaluation,
                    method="function_calling",
                )
                result = fallback_llm.invoke(messages)
            except Exception as exc:
                raise LLMEvaluationError(
                    f"Gemini evaluation failed ({type(primary_error).__name__}): {primary_error}"
                ) from exc

        if result is None:
            raise LLMEvaluationError("Gemini returned no structured evaluation.")

        if isinstance(result, dict):
            try:
                result = ResumeEvaluation.model_validate(result)
            except Exception as exc:
                raise LLMEvaluationError(
                    f"Gemini returned data that did not match the evaluation schema: {exc}"
                ) from exc

        if not isinstance(result, ResumeEvaluation):
            raise LLMEvaluationError(
                f"Unexpected Gemini response type: {type(result).__name__}"
            )

        return {
            "score": round(float(result.score), 3),
            "reasoning": result.reasoning.strip(),
            "matched_skills": result.matched_skills,
            "missing_skills": result.missing_skills,
            "recommendation": result.recommendation.strip(),
        }
