#!/bin/bash

# Helper script to run tests with live output
# Usage: ./run_tests_live.sh [quick|performance|accuracy|all]

TEST_TYPE=${1:-quick}
echo "🚀 Running tests with live output: $TEST_TYPE"
echo "Press Ctrl+C to cancel if needed"
echo ""

# Run with unbuffered output for immediate feedback
if [ "$TEST_TYPE" = "quick" ]; then
    PYTHONUNBUFFERED=1 python3 test_runner.py --quick --notes "Live test run: quick"
else
    PYTHONUNBUFFERED=1 python3 test_runner.py --types $TEST_TYPE --notes "Live test run: $TEST_TYPE"
fi 