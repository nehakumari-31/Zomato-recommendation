"""
Phase 3: Integrate LLM (Groq)

This module handles integration with Groq LLM for recommendation generation.
"""

from .groq_client import (
    GroqClient,
    GroqConfig,
    GroqError,
    LLMRecommendation,
    LLMRecommendationResponse,
    build_recommendation_prompt,
    parse_llm_recommendation_json,
)

__all__ = [
    "GroqClient",
    "GroqConfig",
    "GroqError",
    "LLMRecommendation",
    "LLMRecommendationResponse",
    "build_recommendation_prompt",
    "parse_llm_recommendation_json",
]
