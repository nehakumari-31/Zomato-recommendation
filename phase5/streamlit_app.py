import streamlit as st
import pandas as pd
import requests
import json
from typing import List, Dict, Any

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

# Function to fetch cities from backend
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_cities():
    try:
        response = requests.get('http://localhost:8001/api/cities')
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to load cities: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return []

# Load cities
with st.spinner('Loading cities...'):
    if not st.session_state.cities:
        st.session_state.cities = load_cities()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

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
                try:
                    response = requests.post(
                        'http://localhost:8001/api/recommendations',
                        json={
                            "city": st.session_state.city_main,
                            "cuisine": st.session_state.cuisine_main,
                            "price": float(st.session_state.price_main)
                        },
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        st.session_state.recommendations = response.json()
                        if not st.session_state.recommendations:
                            st.warning("No restaurants found matching your criteria. Try adjusting your filters.")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")
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