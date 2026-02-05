import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import os
from io import StringIO # For capturing console output in tests
from rich.console import Console # For console in RecommendationEngine tests

from phase1.data_loader import ZomatoDataLoader
from phase2.input_validation import ValidatedUserInput, PricePreference, ValidationError
from phase3.groq_client import GroqClient, GroqError, LLMRecommendation, LLMRecommendationResponse
from phase4.recommendation_engine import RecommendationEngine, RecommendedRestaurant


@pytest.fixture
def dummy_dataframe():
    data = {
        'name': ['Rest A', 'Rest B', 'Rest C', 'Rest D', 'Rest E', 'Rest F'],
        'listed_in_city': ['Bangalore', 'Bangalore', 'Mumbai', 'Bangalore', 'Delhi', 'Bangalore'],
        'city_normalized': ['Bangalore', 'Bangalore', 'Mumbai', 'Bangalore', 'Delhi', 'Bangalore'],
        'cuisines': ['North Indian', 'Chinese', 'Italian, Chinese', 'North Indian, South Indian', 'Continental', 'North Indian'],
        'cuisines_list': [['North Indian'], ['Chinese'], ['Italian', 'Chinese'], ['North Indian', 'South Indian'], ['Continental'], ['North Indian']],
        'rate': ['4.5/5', '3.8/5', '4.0/5', '4.2/5', '3.0/5', '3.5/5'],
        'rating_numeric': [4.5, 3.8, 4.0, 4.2, 3.0, 3.5],
        'votes': [500, 200, 300, 400, 100, 150],
        'approx_cost_for_two': ['₹800', '₹500', '₹1200', '₹700', '₹300', '₹600'],
        'cost_numeric': [800.0, 500.0, 1200.0, 700.0, 300.0, 600.0],
        'address': ['Addr A', 'Addr B', 'Addr C', 'Addr D', 'Addr E', 'Addr F'],
        'url': ['url_a', 'url_b', 'url_c', 'url_d', 'url_e', 'url_f'],
        'rest_type': ['Casual Dining', 'Quick Bites', 'Fine Dining', 'Casual Dining', 'Cafe', 'Casual Dining'],
        'location': ['Location A', 'Location B', 'Location C', 'Location D', 'Location E', 'Location F']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_data_loader(dummy_dataframe):
    loader = Mock(spec=ZomatoDataLoader)
    loader.clean_and_validate.return_value = dummy_dataframe
    loader.get_processed_data.return_value = dummy_dataframe
    loader.get_unique_cities.return_value = ["Bangalore", "Mumbai", "Delhi"]
    loader.get_unique_cuisines.return_value = ["North Indian", "Chinese", "Italian", "South Indian", "Continental"]
    return loader


@pytest.fixture
def mock_groq_client():
    client = Mock(spec=GroqClient)
    client.get_recommendations.return_value = LLMRecommendationResponse(
        recommendations=[
            LLMRecommendation(name="Rest A", reason="LLM chose it"),
            LLMRecommendation(name="Rest D", reason="Another LLM pick"),
        ]
    )
    client.close.return_value = None
    return client


@pytest.fixture
def mock_console_for_engine():
    return Console(file=StringIO(), markup=True)


def test_init_engine(mock_data_loader, mock_groq_client, mock_console_for_engine):
    engine = RecommendationEngine(mock_data_loader, mock_groq_client, console=mock_console_for_engine)
    assert engine.data_loader == mock_data_loader
    assert engine.groq_client == mock_groq_client
    mock_data_loader.clean_and_validate.assert_called_once()


def test_filter_by_city(mock_data_loader, mock_groq_client, mock_console_for_engine):
    engine = RecommendationEngine(mock_data_loader, mock_groq_client, console=mock_console_for_engine)
    filtered = engine._filter_by_city(engine.df, "Bangalore")
    assert len(filtered) == 4
    assert "Mumbai" not in filtered["city_normalized"].unique()


def test_filter_by_cuisine(mock_data_loader, mock_groq_client, mock_console_for_engine):
    engine = RecommendationEngine(mock_data_loader, mock_groq_client, console=mock_console_for_engine)
    filtered = engine._filter_by_cuisine(engine.df, "North Indian")
    assert len(filtered) == 3
    assert "Chinese" not in filtered["cuisines_list"].explode().unique()


def test_filter_by_price_exact(mock_data_loader, mock_groq_client, mock_console_for_engine):
    engine = RecommendationEngine(mock_data_loader, mock_groq_client, console=mock_console_for_engine)
    price_pref = PricePreference(exact=700.0)
    filtered = engine._filter_by_price(engine.df, price_pref)
    assert len(filtered) == 3
    assert "Rest A" in filtered["name"].tolist()
    assert "Rest D" in filtered["name"].tolist()
    assert "Rest F" in filtered["name"].tolist()


def test_filter_by_price_range(mock_data_loader, mock_groq_client, mock_console_for_engine):
    engine = RecommendationEngine(mock_data_loader, mock_groq_client, console=mock_console_for_engine)
    price_pref = PricePreference(min_value=500.0, max_value=800.0)
    filtered = engine._filter_by_price(engine.df, price_pref)
    assert len(filtered) == 4
    assert "Rest A" in filtered["name"].tolist()
    assert "Rest B" in filtered["name"].tolist()
    assert "Rest D" in filtered["name"].tolist()
    assert "Rest F" in filtered["name"].tolist()


def test_filter_by_price_category_budget(mock_data_loader, mock_groq_client, mock_console_for_engine):
    engine = RecommendationEngine(mock_data_loader, mock_groq_client, console=mock_console_for_engine)
    price_pref = PricePreference(category="budget")
    filtered = engine._filter_by_price(engine.df, price_pref)
    assert len(filtered) == 2
    assert "Rest B" in filtered["name"].tolist()
    assert "Rest E" in filtered["name"].tolist()


def test_deterministic_rank(mock_data_loader, mock_groq_client, mock_console_for_engine):
    engine = RecommendationEngine(mock_data_loader, mock_groq_client, console=mock_console_for_engine)
    ranked = engine._deterministic_rank(engine.df)
    assert ranked.iloc[0]["name"] == "Rest A" # Rating 4.5, Votes 500
    assert ranked.iloc[1]["name"] == "Rest D" # Rating 4.2, Votes 400


def test_get_recommendations_llm_success(mock_data_loader, mock_groq_client, mock_console_for_engine):
    engine = RecommendationEngine(mock_data_loader, mock_groq_client, console=mock_console_for_engine)
    user_input = ValidatedUserInput(city="Bangalore", cuisine="North Indian", price=PricePreference(exact=800.0))
    
    recommendations = engine.get_recommendations(user_input)

    assert len(recommendations) > 0
    assert mock_groq_client.get_recommendations.called_once()
    assert any(rec.reason == "LLM chose it" for rec in recommendations)
    assert "Calling Groq LLM for final ranking..." in mock_console_for_engine.file.getvalue()


def test_get_recommendations_llm_fallback(mock_data_loader, mock_console_for_engine):
    mock_groq_client_fail = Mock(spec=GroqClient)
    mock_groq_client_fail.get_recommendations.side_effect = GroqError("Mock LLM failure")
    mock_groq_client_fail.close.return_value = None

    engine = RecommendationEngine(mock_data_loader, mock_groq_client_fail, console=mock_console_for_engine)
    user_input = ValidatedUserInput(city="Bangalore", cuisine="North Indian", price=PricePreference(exact=800.0))

    recommendations = engine.get_recommendations(user_input)
    output = mock_console_for_engine.file.getvalue()

    assert len(recommendations) > 0
    assert all(rec.reason == "Deterministically ranked based on rating and votes." for rec in recommendations)
    assert "Groq LLM call failed: Mock LLM failure. Falling back to deterministic ranking." in output
    mock_groq_client_fail.get_recommendations.assert_called_once()


def test_get_recommendations_no_initial_filter_results(mock_data_loader, mock_groq_client, mock_console_for_engine):
    engine = RecommendationEngine(mock_data_loader, mock_groq_client, console=mock_console_for_engine)
    user_input = ValidatedUserInput(city="NonExistentCity", cuisine="North Indian", price=PricePreference(exact=800.0))
    recommendations = engine.get_recommendations(user_input)
    assert len(recommendations) == 0
    assert "No restaurants found after initial filtering." in mock_console_for_engine.file.getvalue()
    mock_groq_client.get_recommendations.assert_not_called()


def test_get_recommendations_no_llm_candidates(mock_data_loader, mock_groq_client, mock_console_for_engine):
    engine = RecommendationEngine(mock_data_loader, mock_groq_client, llm_candidate_limit=0, console=mock_console_for_engine)
    user_input = ValidatedUserInput(city="Bangalore", cuisine="North Indian", price=PricePreference(exact=800.0))
    recommendations = engine.get_recommendations(user_input)
    assert len(recommendations) == 0
    assert "No candidates generated for LLM. Returning empty list." in mock_console_for_engine.file.getvalue()
    mock_groq_client.get_recommendations.assert_not_called()
