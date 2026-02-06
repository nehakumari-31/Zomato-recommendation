#!/usr/bin/env python3
"""
Simple Python server that serves the React frontend and handles API requests
for the Zomato restaurant recommendation system.
"""

import http.server
import socketserver
import json
import os
import sys
import urllib.parse
from http import HTTPStatus

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4.recommendation_engine import RecommendationEngine, RecommendedRestaurant
from phase3.groq_client import GroqClient, GroqConfig
from phase2.input_validation import ValidatedUserInput, PricePreference
from phase1.data_loader import ZomatoDataLoader
from unittest.mock import Mock
import pandas as pd


class RestaurantRecommendationHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Initialize the recommendation engine once
        self._initialize_engine()
        super().__init__(*args, **kwargs)
    
    def _initialize_engine(self):
        """Initialize the recommendation engine with the actual dataset"""
        try:
            # Load API key from environment
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("Warning: GROQ_API_KEY not found, using deterministic ranking from real data only")
                self.has_api_key = False
                # Create a special client that always raises GroqError to trigger fallback
                from unittest.mock import Mock
                from phase3.groq_client import GroqError
                self.groq_client = Mock()
                self.groq_client.get_recommendations.side_effect = GroqError("API key not available")
            else:
                self.has_api_key = True
                self.config = GroqConfig(api_key=api_key, model="llama-3.1-8b-instant")
                self.groq_client = GroqClient(config=self.config)

            # Use the actual data loader from Phase 1 - ALWAYS load real data
            from phase1.data_loader import ZomatoDataLoader
            self.data_loader = ZomatoDataLoader(dataset_name="ManikaSaini/zomato-restaurant-recommendation")
            # Load and clean the dataset
            self.data_loader.load_dataset()  # Load the dataset from HuggingFace
            self.data_loader.clean_and_validate()  # Clean and validate the data
            
            # Create recommendation engine with real data
            self.engine = RecommendationEngine(self.data_loader, self.groq_client)

        except Exception as e:
            print(f"Error initializing engine: {e}")
            # Don't fallback to mock data - we should fail if real data loading fails
            raise
    
    def _create_sample_data(self):
        """Create sample data for the recommendation engine."""
        sample_restaurants = [
            {
                'name': 'Spice Garden',
                'city_normalized': 'mumbai', 
                'cuisines_list': ['North Indian', 'Mughlai', 'Kashmiri'],
                'rating_numeric': 4.2,
                'cost_numeric': 800.0,
                'votes': 450,
                'address': '123 MG Road, Mumbai',
                'url': 'https://zomato.com/spicegarden',
                'rest_type': 'Casual Dining',
                'location': 'Downtown Mumbai'
            },
            {
                'name': 'Golden Dragon',
                'city_normalized': 'mumbai',
                'cuisines_list': ['Chinese', 'Thai', 'Asian'],
                'rating_numeric': 4.0,
                'cost_numeric': 650.0,
                'votes': 320,
                'address': '456 Koramangala, Mumbai',
                'url': 'https://zomato.com/goldendragon',
                'rest_type': 'Casual Dining',
                'location': 'Koramangala'
            },
            {
                'name': 'Ocean\'s Delight',
                'city_normalized': 'mumbai',
                'cuisines_list': ['Seafood', 'Continental', 'Mediterranean'],
                'rating_numeric': 4.5,
                'cost_numeric': 1200.0,
                'votes': 280,
                'address': '789 Coastal Road, Mumbai',
                'url': 'https://zomato.com/oceansdelight',
                'rest_type': 'Fine Dining',
                'location': 'Coastal Area'
            },
            {
                'name': 'Saravana Bhavan',
                'city_normalized': 'chennai',
                'cuisines_list': ['South Indian', 'Vegetarian', 'Sweets'],
                'rating_numeric': 4.3,
                'cost_numeric': 300.0,
                'votes': 650,
                'address': '101 T Nagar, Chennai',
                'url': 'https://zomato.com/saravanabhavan',
                'rest_type': 'Quick Bites',
                'location': 'T Nagar'
            },
            {
                'name': 'Taj Mahal Tea House',
                'city_normalized': 'delhi',
                'cuisines_list': ['Mughlai', 'North Indian', 'Biryani'],
                'rating_numeric': 4.1,
                'cost_numeric': 750.0,
                'votes': 520,
                'address': '202 Connaught Place, Delhi',
                'url': 'https://zomato.com/tajmahaltea',
                'rest_type': 'Casual Dining',
                'location': 'Connaught Place'
            },
            {
                'name': 'Mainland China',
                'city_normalized': 'delhi',
                'cuisines_list': ['Chinese', 'Asian', 'Sushi'],
                'rating_numeric': 4.0,
                'cost_numeric': 900.0,
                'votes': 410,
                'address': '303 Nehru Place, Delhi',
                'url': 'https://zomato.com/mainlandchina',
                'rest_type': 'Casual Dining',
                'location': 'Nehru Place'
            },
            {
                'name': 'Theobroma',
                'city_normalized': 'bangalore',
                'cuisines_list': ['Bakery', 'Desserts', 'Cafe'],
                'rating_numeric': 4.4,
                'cost_numeric': 400.0,
                'votes': 780,
                'address': '404 Indiranagar, Bangalore',
                'url': 'https://zomato.com/theobroma',
                'rest_type': 'Cafe',
                'location': 'Indiranagar'
            },
            {
                'name': 'Punjab Grill',
                'city_normalized': 'bangalore',
                'cuisines_list': ['North Indian', 'Punjabi', 'Mughlai'],
                'rating_numeric': 4.6,
                'cost_numeric': 1500.0,
                'votes': 890,
                'address': '505 Koramangala, Bangalore',
                'url': 'https://zomato.com/punjabgrill',
                'rest_type': 'Fine Dining',
                'location': 'Koramangala'
            }
        ]
        
        return pd.DataFrame(sample_restaurants)
    
    def _get_recommendations(self, city, cuisine, price):
        """Get restaurant recommendations based on user preferences."""
        try:
            # Create user input object
            price_pref = PricePreference(exact=float(price))
            user_input = ValidatedUserInput(
                city=city.lower(),
                cuisine=cuisine.replace('-', ' ').title(),  # Format cuisine properly
                price=price_pref
            )

            # Get recommendations through the full pipeline using real data
            # The RecommendationEngine should handle missing API key internally and use deterministic ranking
            recommendations = self.engine.get_recommendations(user_input)

            # Convert to JSON serializable format
            result = []
            for rec in recommendations:
                result.append({
                    'id': abs(hash(rec.name)) % 10000,  # Simple ID generation
                    'name': rec.name,
                    'address': getattr(rec, 'address', 'Address not available'),
                    'city': getattr(rec, 'city', city).title(),
                    'cuisines': getattr(rec, 'cuisines', [cuisine.replace('-', ' ').title()]),
                    'rating': getattr(rec, 'rating', 0),
                    'cost_for_two': getattr(rec, 'cost_for_two', float(price)),
                    'url': getattr(rec, 'url', ''),
                    'rest_type': getattr(rec, 'rest_type', 'Restaurant'),
                    'location': getattr(rec, 'location', getattr(rec, 'city', city).title()),
                    'reason': getattr(rec, 'reason', f"Selected based on your preferences for {cuisine.replace('-', ' ').title()} cuisine in {city.title()} at your budget")
                })

            return result
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            import traceback
            traceback.print_exc()
            # Return empty list in case of error
            return []
    
    def _get_cities(self):
        """Get list of available cities from the real dataset."""
        try:
            # Use the real data loader
            return self.data_loader.get_unique_cities()
        except Exception as e:
            print(f"Error getting cities from real dataset: {e}")
            # Fallback to empty list
            return []

    def _get_cuisines(self):
        """Get list of available cuisines from the real dataset."""
        try:
            # Use the real data loader
            return self.data_loader.get_unique_cuisines()
        except Exception as e:
            print(f"Error getting cuisines from real dataset: {e}")
            # Fallback to empty list
            return []
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/api/cities':
            self._send_json_response(self._get_cities())
        elif self.path == '/api/cuisines':
            self._send_json_response(self._get_cuisines())
        elif self.path == '/' or self.path == '/index.html':
            # Serve the static HTML file
            self._serve_static_file('dist/index.html')
        elif self.path.startswith('/static/'):
            # Serve static files
            file_path = self.path[1:]  # Remove leading slash
            self._serve_static_file(file_path)
        else:
            # Default to serving index.html for SPA routing
            self._serve_static_file('dist/index.html')
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/recommendations':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                city = data.get('city', '')
                cuisine = data.get('cuisine', '')
                price = data.get('price', 0)
                
                if not city or not cuisine or not price:
                    self._send_error_response('City, cuisine, and price are required', 400)
                    return
                
                recommendations = self._get_recommendations(city, cuisine, price)
                self._send_json_response(recommendations)
                
            except json.JSONDecodeError:
                self._send_error_response('Invalid JSON', 400)
            except Exception as e:
                self._send_error_response(f'Error processing request: {str(e)}', 500)
        else:
            self._send_error_response('Not found', 404)
    
    def _send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _send_error_response(self, message, status_code=400):
        """Send error response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        error_response = {'error': message}
        self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def _serve_static_file(self, file_path):
        """Serve static files"""
        # Resolve file path
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        
        if not os.path.exists(full_path):
            self._send_error_response('File not found', 404)
            return
        
        # Get content type
        content_type = self._get_content_type(full_path)
        
        # Read and serve file
        try:
            with open(full_path, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self._send_error_response(f'Error serving file: {str(e)}', 500)
    
    def _get_content_type(self, file_path):
        """Determine content type based on file extension"""
        if file_path.endswith('.html'):
            return 'text/html'
        elif file_path.endswith('.css'):
            return 'text/css'
        elif file_path.endswith('.js'):
            return 'application/javascript'
        elif file_path.endswith('.json'):
            return 'application/json'
        elif file_path.endswith('.png'):
            return 'image/png'
        elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
            return 'image/jpeg'
        else:
            return 'text/plain'
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        print(f"[{self.address_string()}] {format % args}")


def run_server(port=8000):
    """Run the server"""
    with socketserver.TCPServer(("", port), RestaurantRecommendationHandler) as httpd:
        print(f"🚀 Server running at http://localhost:{port}")
        print("🍽️  Zomato Restaurant Recommendation System")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
            httpd.server_close()


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run_server(port)