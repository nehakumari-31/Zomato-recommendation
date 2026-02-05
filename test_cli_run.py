#!/usr/bin/env python3
"""
Test script to run CLI with predefined inputs for demonstration.
"""
import sys
from unittest.mock import patch
from phase5.cli import run_cli_recommendation, console

# Mock console.input to provide predefined responses
call_count = [0]  # Use list to make it mutable
def mock_input(prompt):
    # Remove Rich markup for matching
    import re
    clean_prompt = re.sub(r'\[[^\]]*\]', '', prompt).strip()
    
    responses = [
        "Indiranagar",    # City (area in Bangalore)
        "North Indian",   # Cuisine  
        "500-1000"        # Price
    ]
    
    if call_count[0] < len(responses):
        result = responses[call_count[0]]
        call_count[0] += 1
        return result
    return ""

if __name__ == "__main__":
    with patch.object(console, 'input', side_effect=mock_input):
        try:
            run_cli_recommendation()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()
