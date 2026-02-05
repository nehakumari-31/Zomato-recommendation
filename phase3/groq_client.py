"""
Phase 3: Groq LLM integration

This module provides:
- Prompt builder for restaurant recommendation ranking
- Strict JSON response parser
- Thin HTTP client for Groq Chat Completions (OpenAI-compatible)

Notes:
- Unit tests DO NOT call the real Groq API (they mock HTTP).
- API key is read from env by default in the client config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import json
import os
import re

import httpx
from dotenv import load_dotenv


class GroqError(RuntimeError):
    """Raised for Groq API or parsing errors."""


@dataclass(frozen=True)
class GroqConfig:
    """
    Configuration for Groq API.

    Groq provides an OpenAI-compatible Chat Completions API at:
    https://api.groq.com/openai/v1/chat/completions
    """

    api_key: str
    model: str = "llama-3.1-8b-instant"
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class LLMRecommendation:
    """
    One recommended restaurant from the LLM.
    """

    name: str
    reason: str


@dataclass(frozen=True)
class LLMRecommendationResponse:
    """
    Parsed response returned by the LLM.
    """

    recommendations: List[LLMRecommendation]


def build_recommendation_prompt(
    *,
    city: str,
    cuisine: str,
    price: str,
    candidates: Sequence[Dict[str, Any]],
    top_n: int = 10,
) -> str:
    """
    Build a prompt that forces the model to pick ONLY from provided candidates.

    candidates: list of dicts with (at minimum) `name`, plus helpful fields such as
    rating_numeric, votes, cost_numeric, rest_type, location, cuisines.
    """
    # Keep prompt compact: include only the fields we might use.
    safe_candidates = []
    for c in candidates:
        safe_candidates.append(
            {
                "name": c.get("name"),
                "city": c.get("city_normalized") or c.get("listed_in_city") or c.get("city"),
                "location": c.get("location"),
                "cuisines": c.get("cuisines"),
                "rating": c.get("rating_numeric") or c.get("rate"),
                "votes": c.get("votes"),
                "cost_for_two": c.get("cost_numeric") or c.get("approx_cost_for_two") or c.get("approx_cost(for two people)"),
                "rest_type": c.get("rest_type"),
            }
        )

    schema = {
        "recommendations": [
            {
                "name": "string (MUST match exactly one candidate name)",
                "reason": "string (short, factual, based on provided candidate fields)",
            }
        ]
    }

    prompt = f"""
You are a restaurant recommendation engine.

User preferences:
- City: {city}
- Cuisine: {cuisine}
- Price: {price} (approx cost for two)

You MUST recommend ONLY from the provided candidate restaurants.
Do NOT invent or rename restaurants.

Task:
- Pick the best {top_n} restaurants from the candidates.
- Rank them from best to worst.
- Provide a short reason for each pick, grounded only in the candidate data.

Return STRICT JSON matching this schema:
{json.dumps(schema, ensure_ascii=False)}

Candidates (JSON):
{json.dumps(safe_candidates, ensure_ascii=False)}
""".strip()

    return prompt


def _extract_json(text: str) -> str:
    """
    Extract JSON object from a response that might contain extra text or code fences.
    """
    if text is None:
        raise GroqError("Empty LLM response.")

    # Remove fenced code blocks if present
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    # Try to locate first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise GroqError("LLM response does not contain a JSON object.")

    return text[start : end + 1]


def parse_llm_recommendation_json(text: str) -> LLMRecommendationResponse:
    """
    Parse and validate the LLM JSON output.
    """
    raw_json = _extract_json(text)
    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise GroqError(f"Invalid JSON from LLM: {e}") from e

    recs = obj.get("recommendations")
    if not isinstance(recs, list):
        raise GroqError("JSON must contain a 'recommendations' list.")

    parsed: List[LLMRecommendation] = []
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            raise GroqError(f"Recommendation at index {i} must be an object.")
        name = r.get("name")
        reason = r.get("reason", "")
        if not isinstance(name, str) or not name.strip():
            raise GroqError(f"Recommendation at index {i} has invalid 'name'.")
        if not isinstance(reason, str):
            raise GroqError(f"Recommendation at index {i} has invalid 'reason'.")
        parsed.append(LLMRecommendation(name=name.strip(), reason=reason.strip()))

    return LLMRecommendationResponse(recommendations=parsed)


class GroqClient:
    """
    Thin Groq Chat Completions client (OpenAI-compatible).
    """

    def __init__(self, config: Optional[GroqConfig] = None):
        load_dotenv()  # Load .env file at init
        if config is None:
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if not api_key:
                raise GroqError("Missing GROQ_API_KEY (set env var or pass GroqConfig).")
            config = GroqConfig(api_key=api_key)
        self.config = config

        self._http = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )

    def close(self) -> None:
        self._http.close()

    def chat_completion(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        """
        Call Groq chat completions and return assistant content as string.
        """
        payload = {
            "model": self.config.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        try:
            resp = self._http.post("/chat/completions", json=payload)
        except Exception as e:
            raise GroqError(f"Groq request failed: {e}") from e

        if resp.status_code >= 400:
            raise GroqError(f"Groq API error {resp.status_code}: {resp.text}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise GroqError(f"Unexpected Groq response format: {data}") from e

    def get_recommendations(
        self,
        *,
        city: str,
        cuisine: str,
        price: str,
        candidates: Sequence[Dict[str, Any]],
        top_n: int = 10,
    ) -> LLMRecommendationResponse:
        """
        High-level helper used by Phase 4:
        - builds prompt
        - calls Groq
        - parses strict JSON
        """
        system = "You return strict JSON only. No prose."
        user_prompt = build_recommendation_prompt(
            city=city, cuisine=cuisine, price=price, candidates=candidates, top_n=top_n
        )
        text = self.chat_completion(system=system, user=user_prompt)
        return parse_llm_recommendation_json(text)

