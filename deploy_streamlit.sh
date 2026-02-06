#!/bin/bash

# Script to deploy the Zomato Restaurant Recommendation System with Streamlit

echo "🚀 Deploying Zomato Restaurant Recommendation System with Streamlit..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Start the backend server in the background
echo "📡 Starting backend server..."
cd phase5
python3 run_server.py &
BACKEND_PID=$!

# Wait a moment for the backend to start
sleep 5

# Start the Streamlit app
echo "🎨 Starting Streamlit frontend..."
streamlit run streamlit_app.py --server.port=8501

# Kill the backend when Streamlit stops
kill $BACKEND_PID 2>/dev/null

echo "✅ Deployment completed!"