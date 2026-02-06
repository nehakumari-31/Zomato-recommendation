"""
Test cases for Groq LLM integration in recommendation engine.

These tests specifically check that the LLM integration is working properly
and that API calls are made as expected.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from httpx import Response

from phase1.data_loader import ZomatoDataLoader
from phase2.input_validation import ValidatedUserInput, PricePreference
from phase3.groq_client import GroqClient, GroqConfig, GroqError, LLMRecommendation, LLMRecommendationResponse
from phase4.recommendation_engine import RecommendationEngine


def test_groq_api_call_made_when_getting_recommendations():
    """Test that Groq API is actually called when getting recommendations."""
    # Create mock data loader
    mock_data_loader = Mock(spec=ZomatoDataLoader)
    dummy_data = pd.DataFrame({
        'name': ['Restaurant A', 'Restaurant B', 'Restaurant C'],
        'city_normalized': ['Bangalore', 'Bangalore', 'Bangalore'],
        'cuisines_list': [['North Indian'], ['Chinese'], ['North Indian']],
        'rating_numeric': [4.5, 3.8, 4.2],
        'cost_numeric': [800.0, 500.0, 700.0],
        'votes': [500, 200, 400],
        'address': ['Address A', 'Address B', 'Address C'],
        'url': ['url_a', 'url_b', 'url_c'],
        'rest_type': ['Casual Dining', 'Quick Bites', 'Casual Dining'],
        'location': ['Location A', 'Location B', 'Location C']
    })
    mock_data_loader.clean_and_validate.return_value = dummy_data
    mock_data_loader.get_processed_data.return_value = dummy_data

    # Create a real GroqClient with mocked HTTP calls
    with patch('httpx.Client') as mock_http_class:
        # Create a mock client instance
        mock_http_instance = Mock()
        mock_http_class.return_value = mock_http_instance
        
        # Mock the successful response from Groq API
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "recommendations": [
                            {"name": "Restaurant A", "reason": "Excellent rating and great value for money"},
                            {"name": "Restaurant C", "reason": "Good quality North Indian cuisine"}
                        ]
                    })
                }
            }]
        }
        mock_http_instance.post.return_value = mock_response

        # Create GroqClient with fake API key
        config = GroqConfig(api_key="fake_api_key")
        groq_client = GroqClient(config=config)

        # Create recommendation engine
        engine = RecommendationEngine(mock_data_loader, groq_client)

        # Make a recommendation request
        user_input = ValidatedUserInput(
            city="Bangalore",
            cuisine="North Indian",
            price=PricePreference(exact=800.0)
        )
        
        recommendations = engine.get_recommendations(user_input)

        # Verify that the HTTP client was called
        assert mock_http_instance.post.called
        call_args = mock_http_instance.post.call_args
        assert call_args is not None
        assert call_args[0][0] == "/chat/completions"  # Check the endpoint
        
        # Verify that we got recommendations
        assert len(recommendations) > 0
        assert recommendations[0].name == "Restaurant A"


def test_groq_api_failure_handled_gracefully():
    """Test that Groq API failures are handled gracefully with fallback."""
    # Create mock data loader
    mock_data_loader = Mock(spec=ZomatoDataLoader)
    dummy_data = pd.DataFrame({
        'name': ['Restaurant A', 'Restaurant B'],
        'city_normalized': ['Bangalore', 'Bangalore'],
        'cuisines_list': [['North Indian'], ['Chinese']],
        'rating_numeric': [4.5, 3.8],
        'cost_numeric': [800.0, 500.0],
        'votes': [500, 200],
        'address': ['Address A', 'Address B'],
        'url': ['url_a', 'url_b'],
        'rest_type': ['Casual Dining', 'Quick Bites'],
        'location': ['Location A', 'Location B']
    })
    mock_data_loader.clean_and_validate.return_value = dummy_data
    mock_data_loader.get_processed_data.return_value = dummy_data

    # Create a real GroqClient with mocked HTTP calls that fail
    with patch('httpx.Client') as mock_http_class:
        # Create a mock client instance
        mock_http_instance = Mock()
        mock_http_class.return_value = mock_http_instance
        
        # Mock a failed response from Groq API
        mock_response = Mock(spec=Response)
        mock_response.status_code = 429  # Rate limit error
        mock_response.text = "Rate limit exceeded"
        mock_http_instance.post.return_value = mock_response

        # Create GroqClient with fake API key
        config = GroqConfig(api_key="fake_api_key")
        groq_client = GroqClient(config=config)

        # Create recommendation engine
        engine = RecommendationEngine(mock_data_loader, groq_client)

        # Make a recommendation request
        user_input = ValidatedUserInput(
            city="Bangalore",
            cuisine="North Indian",
            price=PricePreference(exact=800.0)
        )
        
        recommendations = engine.get_recommendations(user_input)

        # Verify that the HTTP client was called
        assert mock_http_instance.post.called
        call_args = mock_http_instance.post.call_args
        assert call_args is not None
        assert call_args[0][0] == "/chat/completions"
        
        # Verify fallback behavior - should still return recommendations with deterministic reason
        assert len(recommendations) > 0
        assert recommendations[0].name == "Restaurant A"  # Highest rated
        assert recommendations[0].reason == "Deterministically ranked based on rating and votes."


def test_groq_client_get_recommendations_method():
    """Test the get_recommendations method on GroqClient directly."""
    with patch('httpx.Client') as mock_http_class:
        # Create a mock client instance
        mock_http_instance = Mock()
        mock_http_class.return_value = mock_http_instance
        
        # Mock the successful response from Groq API
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "recommendations": [
                            {"name": "Restaurant X", "reason": "Top choice for North Indian cuisine"},
                            {"name": "Restaurant Y", "reason": "Great ambiance and service"}
                        ]
                    })
                }
            }]
        }
        mock_http_instance.post.return_value = mock_response

        # Create GroqClient with fake API key
        config = GroqConfig(api_key="fake_api_key")
        groq_client = GroqClient(config=config)

        # Test the get_recommendations method
        candidates = [
            {"name": "Restaurant X", "rating": 4.5, "cost": 800},
            {"name": "Restaurant Y", "rating": 4.2, "cost": 700}
        ]
        
        result = groq_client.get_recommendations(
            city="Bangalore",
            cuisine="North Indian", 
            price="700-900",
            candidates=candidates
        )

        # Verify the result
        assert isinstance(result, LLMRecommendationResponse)
        assert len(result.recommendations) == 2
        assert result.recommendations[0].name == "Restaurant X"
        assert result.recommendations[0].reason == "Top choice for North Indian cuisine"
        
        # Verify API was called
        assert mock_http_instance.post.called
        call_args = mock_http_instance.post.call_args
        assert call_args[0][0] == "/chat/completions"
        
        # Check that the request body contains our prompt
        request_body = call_args[1]['json']
        assert 'model' in request_body
        assert request_body['model'] == "llama-3.1-8b-instant"
        assert len(request_body['messages']) == 2


if __name__ == "__main__":
    pytest.main([__file__])