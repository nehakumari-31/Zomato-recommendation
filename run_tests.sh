#!/bin/bash
# Quick test runner for Phase 1

echo "Running Phase 1 tests..."
pytest phase1/tests/ -v

echo ""
echo "To run with coverage:"
echo "  pytest phase1/tests/ --cov=phase1 --cov-report=html"
