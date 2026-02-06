#!/usr/bin/env python3
"""
Run the integrated Zomato Restaurant Recommendation Server.

This script launches both the frontend (static files) and backend (API) servers.
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path

def run_server(port=8000):
    """
    Run the integrated server
    
    Args:
        port (int): Port number to run the server on (default: 8000)
    """
    print("=" * 60)
    print("ZOMATO RESTAURANT RECOMMENDATION SYSTEM")
    print("Integrated Frontend + Backend Server")
    print("=" * 60)
    print(f"🚀 Starting Zomato Restaurant Recommendation Server...")
    print(f"🍽️  This will serve both the frontend and backend")
    
    # Change to the phase5 directory
    os.chdir(Path(__file__).parent)
    
    # Import and run the server
    try:
        from server import run_server as start_server
        start_server(port=port)
    except Exception as e:
        print(f"❌ Error running server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Use port 8001 instead of 8000 to avoid conflicts
    PORT = 8001
    print(f"⚠️  Using port {PORT} instead of 8000 to avoid conflicts")
    run_server(PORT)