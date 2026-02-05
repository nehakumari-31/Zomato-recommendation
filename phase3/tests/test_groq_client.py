import json

import pytest

from phase3.groq_client import (
    GroqError,
    LLMRecommendationResponse,
    build_recommendation_prompt,
    parse_llm_recommendation_json,
)


def test_build_prompt_contains_user_prefs_and_candidates():
    prompt = build_recommendation_prompt(
        city="Bangalore",
        cuisine="North Indian",
        price="500-1000",
        candidates=[
            {"name": "A", "rating_numeric": 4.2, "votes": 100, "cost_numeric": 800},
            {"name": "B", "rating_numeric": 4.0, "votes": 50, "cost_numeric": 600},
        ],
        top_n=2,
    )
    assert "City: Bangalore" in prompt
    assert "Cuisine: North Indian" in prompt
    assert "Price: 500-1000" in prompt
    assert '"name": "A"' in prompt
    assert "Pick the best 2 restaurants" in prompt


def test_parse_llm_json_happy_path():
    text = json.dumps(
        {
            "recommendations": [
                {"name": "Restaurant A", "reason": "High rating and good value"},
                {"name": "Restaurant B", "reason": "Popular choice"},
            ]
        }
    )
    parsed = parse_llm_recommendation_json(text)
    assert isinstance(parsed, LLMRecommendationResponse)
    assert len(parsed.recommendations) == 2
    assert parsed.recommendations[0].name == "Restaurant A"


def test_parse_llm_json_with_code_fence():
    text = """```json
    {"recommendations":[{"name":"A","reason":"x"}]}
    ```"""
    parsed = parse_llm_recommendation_json(text)
    assert len(parsed.recommendations) == 1
    assert parsed.recommendations[0].name == "A"


def test_parse_llm_json_missing_recommendations():
    with pytest.raises(GroqError):
        parse_llm_recommendation_json('{"foo": 1}')


def test_parse_llm_json_invalid_json():
    with pytest.raises(GroqError):
        parse_llm_recommendation_json("{not json}")


def test_parse_llm_json_invalid_item():
    with pytest.raises(GroqError):
        parse_llm_recommendation_json('{"recommendations":[{"name":"","reason":"x"}]}')


def test_extract_json_from_extra_text():
    text = "Here you go:\n" + '{"recommendations":[{"name":"A","reason":"x"}]}' + "\nThanks!"
    parsed = parse_llm_recommendation_json(text)
    assert parsed.recommendations[0].name == "A"

