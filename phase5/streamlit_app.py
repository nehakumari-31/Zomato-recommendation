import streamlit as st
import pandas as pd
import requests
import json
import os
import sys
from requests import exceptions as req_exceptions
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path so `phase1`, `phase2`, `phase3`, etc. are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set page config
st.set_page_config(
    page_title="Zomato Restaurant Recommendation System",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS for Zomato-like styling
st.markdown("""
<style>
    .main-header {
        background-color: #e23744;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .filter-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .restaurant-card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .restaurant-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .restaurant-name {
        font-size: 18px;
        font-weight: bold;
        color: #222324;
        margin-bottom: 5px;
    }
    
    .restaurant-location {
        font-size: 14px;
        color: #686b70;
        margin-bottom: 5px;
    }
    
    .restaurant-cuisines {
        font-size: 14px;
        color: #686b70;
        margin-bottom: 10px;
    }
    
    .restaurant-details {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #eee;
    }
    
    .cost-info {
        font-size: 14px;
        color: #686b70;
    }
    
    .rating-container {
        background-color: #3d9b6d;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        display: flex;
        align-items: center;
    }
    
    .reason-box {
        background-color: #f8f9fa;
        border-left: 4px solid #e23744;
        padding: 10px;
        margin-top: 10px;
        border-radius: 0 4px 4px 0;
        font-size: 13px;
        color: #686b70;
    }
    
    .order-link {
        color: #e23744;
        text-decoration: none;
        font-weight: 500;
        font-size: 14px;
    }
    
    .stButton>button {
        background-color: #e23744;
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 4px;
        font-size: 16px;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #d32f2f;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🍽️ Zomato Restaurant Recommendation System</h1><p>Find the perfect restaurant for your taste and budget</p></div>', unsafe_allow_html=True)

# Initialize session state
if 'cities' not in st.session_state:
    st.session_state.cities = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []


def _show_backend_run_instructions():
    """Show instructions to the user for starting the backend server."""
    BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8001')
    st.info(f"Backend not reachable at {BACKEND_URL}. To run the backend locally:")
    st.code("python3 phase5/run_server.py", language='bash')
    st.write("Or run python3 phase5/server.py directly (default port 8000), or set the BACKEND_URL environment variable to point to a running backend.")


def _get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get configuration from env var, falling back to Streamlit secrets.

    Checks `os.environ` first, then `st.secrets` (if available), then returns `default`.
    """
    val = os.getenv(name)
    if val:
        return val
    try:
        # st.secrets acts like a dict when present
        return st.secrets.get(name, default) if hasattr(st, 'secrets') else default
    except Exception:
        return default

# Function to fetch cities from backend
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_cities():
    try:
        # If EMBED_BACKEND env var / secret is set, initialize and use the embedded engine
        if str(_get_setting('EMBED_BACKEND', '')).lower() in ('1', 'true', 'yes'):
            try:
                from phase1.data_loader import ZomatoDataLoader
                from phase4.recommendation_engine import RecommendationEngine
                from unittest.mock import Mock
                from phase3.groq_client import GroqError, GroqClient, GroqConfig

                api_key = _get_setting('GROQ_API_KEY')
                if not api_key:
                    groq_client = Mock()
                    groq_client.get_recommendations.side_effect = GroqError("API key not available")
                else:
                    config = GroqConfig(api_key=api_key, model="llama-3.1-8b-instant")
                    groq_client = GroqClient(config=config)

                data_loader = ZomatoDataLoader(dataset_name="ManikaSaini/zomato-restaurant-recommendation")
                data_loader.load_dataset()
                data_loader.clean_and_validate()
                engine = RecommendationEngine(data_loader, groq_client)
                st.session_state.engine = engine
                return engine.data_loader.get_unique_cities()
            except Exception as e:
                st.error(f"Failed to initialize embedded backend for cities: {e}")
                # fallthrough to trying external backend

        BACKEND_URL = _get_setting('BACKEND_URL', 'http://localhost:8001')
        response = requests.get(f"{BACKEND_URL}/api/cities", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to load cities: {response.status_code}")
            return []
    except req_exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        _show_backend_run_instructions()
        return []
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return []


def _preload_embedded_engine_with_progress():
    """Preload the embedded recommendation engine and show progress in the UI.

    This is called on app startup when `EMBED_BACKEND` is enabled. It stores
    the built engine in `st.session_state.engine` and exposes progress messages.
    """
    if 'engine' in st.session_state:
        return st.session_state.engine

    if str(_get_setting('EMBED_BACKEND', '')).lower() not in ('1', 'true', 'yes'):
        return None

    placeholder = st.empty()
    progress = st.progress(0)
    try:
        placeholder.info('Initializing embedded recommendation engine...')
        progress.progress(5)

        from unittest.mock import Mock
        from phase3.groq_client import GroqError, GroqClient, GroqConfig
        from phase1.data_loader import ZomatoDataLoader
        from phase4.recommendation_engine import RecommendationEngine

        api_key = _get_setting('GROQ_API_KEY')
        if not api_key:
            groq_client = Mock()
            groq_client.get_recommendations.side_effect = GroqError('API key not available')
        else:
            placeholder.info('Configuring LLM client...')
            progress.progress(15)
            config = GroqConfig(api_key=api_key, model='llama-3.1-8b-instant')
            groq_client = GroqClient(config=config)

        placeholder.info('Loading dataset (this may take a while)...')
        progress.progress(30)
        data_loader = ZomatoDataLoader(dataset_name='ManikaSaini/zomato-restaurant-recommendation')
        data_loader.load_dataset()
        progress.progress(65)

        placeholder.info('Cleaning and validating dataset...')
        data_loader.clean_and_validate()
        progress.progress(85)

        placeholder.info('Creating recommendation engine...')
        engine = RecommendationEngine(data_loader, groq_client)
        st.session_state.engine = engine
        progress.progress(100)
        placeholder.success('Embedded engine ready')
        return engine
    except Exception as e:
        placeholder.error(f'Failed to initialize embedded engine: {e}')
        progress.empty()
        return None

# Load cities
with st.spinner('Loading cities...'):
    # Preload embedded engine (if enabled) so the app is responsive and cities are available
    if str(_get_setting('EMBED_BACKEND', '')).lower() in ('1', 'true', 'yes'):
        _preload_embedded_engine_with_progress()

    if not st.session_state.cities:
        st.session_state.cities = load_cities()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

# Option to run embedded backend inside Streamlit
use_embedded = st.sidebar.checkbox("Run backend inside this Streamlit app (no external server)", value=False)

city = st.sidebar.selectbox(
    "Area/Locality",
    options=[""] + [city for city in st.session_state.cities],
    format_func=lambda x: "Select area/locality" if x == "" else x
)

cuisine = st.sidebar.selectbox(
    "Cuisine",
    options=["", "north indian", "south indian", "chinese", "italian", "mexican", "thai", "japanese", "continental", "fast food", "street food", "desserts", "beverages"],
    format_func=lambda x: "Select cuisine" if x == "" else x.title()
)

price = st.sidebar.selectbox(
    "Price for Two",
    options=["", "200", "500", "800", "1200", "1500", "2000"],
    format_func=lambda x: "Select price range" if x == "" else f"₹{x}" + (" and below" if x in ["200", "500"] else " and below" if x in ["800", "1200", "1500", "2000"] else "")
)

# Main content
st.markdown('<div class="filter-container">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([3, 3, 3, 2])

with col1:
    st.selectbox(
        "Area/Locality",
        options=[""] + [city for city in st.session_state.cities],
        format_func=lambda x: "Select area/locality" if x == "" else x,
        key="city_main"
    )

with col2:
    st.selectbox(
        "Cuisine",
        options=["", "north indian", "south indian", "chinese", "italian", "mexican", "thai", "japanese", "continental", "fast food", "street food", "desserts", "beverages"],
        format_func=lambda x: "Select cuisine" if x == "" else x.title(),
        key="cuisine_main"
    )

with col3:
    st.selectbox(
        "Price for Two",
        options=["", "200", "500", "800", "1200", "1500", "2000"],
        format_func=lambda x: "Select price range" if x == "" else f"₹{x}" + (" and below" if x in ["200", "500"] else " and below" if x in ["800", "1200", "1500", "2000"] else ""),
        key="price_main"
    )

with col4:
    if st.button("Find Restaurants", use_container_width=True):
        if st.session_state.city_main and st.session_state.cuisine_main and st.session_state.price_main:
            with st.spinner('Asking AI for personalized recommendations...'):
                # If user chose embedded backend, run recommendation pipeline locally
                if use_embedded:
                    try:
                        from phase2.input_validation import validate_user_input

                        # Initialize engine lazily
                        def _init_embedded_engine() -> Optional[object]:
                            if 'engine' in st.session_state:
                                return st.session_state.engine
                            try:
                                from unittest.mock import Mock
                                from phase3.groq_client import GroqConfig, GroqClient, GroqError
                                from phase4.recommendation_engine import RecommendationEngine
                                from phase1.data_loader import ZomatoDataLoader

                                api_key = _get_setting('GROQ_API_KEY')
                                if not api_key:
                                    groq_client = Mock()
                                    groq_client.get_recommendations.side_effect = GroqError("API key not available")
                                else:
                                    config = GroqConfig(api_key=api_key, model="llama-3.1-8b-instant")
                                    groq_client = GroqClient(config=config)

                                data_loader = ZomatoDataLoader(dataset_name="ManikaSaini/zomato-restaurant-recommendation")
                                data_loader.load_dataset()
                                data_loader.clean_and_validate()
                                engine = RecommendationEngine(data_loader, groq_client)
                                st.session_state.engine = engine
                                return engine
                            except Exception as e:
                                st.error(f"Failed to initialize embedded backend: {e}")
                                return None

                        engine = _init_embedded_engine()
                        if engine is None:
                            st.error("Embedded backend initialization failed. Try running the external backend instead.")
                        else:
                            # Validate inputs against dataset lists
                            available_cities = engine.data_loader.get_unique_cities()
                            available_cuisines = engine.data_loader.get_unique_cuisines()
                            try:
                                validated = validate_user_input(
                                    city=st.session_state.city_main,
                                    cuisine=st.session_state.cuisine_main,
                                    price=st.session_state.price_main,
                                    available_cities=available_cities,
                                    available_cuisines=available_cuisines,
                                )

                                recs = engine.get_recommendations(validated)
                                # Convert dataclass objects to serializable dicts
                                st.session_state.recommendations = [
                                    {
                                        'id': abs(hash(r.name)) % 10000,
                                        'name': r.name,
                                        'address': r.address,
                                        'city': r.city,
                                        'cuisines': r.cuisines,
                                        'rating': r.rating,
                                        'cost_for_two': r.cost_for_two,
                                        'url': r.url or '',
                                        'rest_type': '',
                                        'location': r.city,
                                        'reason': r.reason or 'Selected based on your preferences.'
                                    }
                                    for r in recs
                                ]
                                if not st.session_state.recommendations:
                                    st.warning("No restaurants found matching your criteria. Try adjusting your filters.")
                            except Exception as e:
                                st.error(f"Input validation failed: {e}")
                    except Exception as e:
                        st.error(f"Embedded backend error: {e}")
                else:
                    try:
                        BACKEND_URL = _get_setting('BACKEND_URL', 'http://localhost:8001')
                        response = requests.post(
                            f"{BACKEND_URL}/api/recommendations",
                            json={
                                "city": st.session_state.city_main,
                                "cuisine": st.session_state.cuisine_main,
                                "price": float(st.session_state.price_main)
                            },
                            headers={'Content-Type': 'application/json'},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            st.session_state.recommendations = response.json()
                            if not st.session_state.recommendations:
                                st.warning("No restaurants found matching your criteria. Try adjusting your filters.")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except req_exceptions.RequestException as e:
                        st.error(f"Error connecting to backend: {str(e)}")
                        _show_backend_run_instructions()
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
        else:
            st.warning("Please select all filters")

st.markdown('</div>', unsafe_allow_html=True)

# Display recommendations
if st.session_state.recommendations:
    st.header(f"🍽️ Top Restaurants for You ({len(st.session_state.recommendations)} found)")
    
    for restaurant in st.session_state.recommendations:
        with st.container():
            st.markdown(f"""
            <div class="restaurant-card">
                <div class="restaurant-name">{restaurant.get('name', 'Unknown')}</div>
                <div class="restaurant-location">{restaurant.get('location', restaurant.get('city', 'Unknown'))}</div>
                <div class="restaurant-cuisines">{', '.join(restaurant.get('cuisines', []))}</div>
                
                <div class="restaurant-details">
                    <div class="cost-info">₹{restaurant.get('cost_for_two', '0')} for two</div>
                    <div class="rating-container">⭐ {restaurant.get('rating', 'N/A')}</div>
                </div>
                
                <div class="reason-box">
                    <strong>Why this place:</strong> {restaurant.get('reason', 'Selected based on your preferences.')}
                </div>
                
                <a href="{restaurant.get('url', '#')}" target="_blank" class="order-link">
                    Order Now →
                </a>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Select your preferences above to get restaurant recommendations")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; padding: 20px;">
    <p>By continuing past this page, you agree to our Terms of Service, Cookie Policy, Privacy Policy and Content Policies. All trademarks are properties of their respective owners. © 2024 Zomato™ Ltd. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)