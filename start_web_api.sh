#!/bin/bash

# ImpactOS AI Web API Startup Script
# This script starts the FastAPI web service for ImpactOS AI system integration

set -e

# Default configuration
HOST="0.0.0.0"
PORT="8000"
MODE="development"
LOG_LEVEL="info"

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Start ImpactOS AI Web API service"
    echo ""
    echo "OPTIONS:"
    echo "  --host HOST         Host to bind to (default: 0.0.0.0)"
    echo "  --port PORT         Port to bind to (default: 8000)"
    echo "  --mode MODE         Run mode: development|production (default: development)"
    echo "  --log-level LEVEL   Log level: debug|info|warning|error (default: info)"
    echo "  --help              Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Start in development mode"
    echo "  $0 --mode production --port 8080     # Start in production mode on port 8080"
    echo "  $0 --host localhost --port 8001      # Start on localhost:8001"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "development" && "$MODE" != "production" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be 'development' or 'production'"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "src/web_api.py" ]]; then
    echo "Error: web_api.py not found in src/ directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment is activated (optional but recommended)
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Warning: No virtual environment detected. Consider activating one:"
    echo "  source impactos-env-new/bin/activate"
    echo ""
fi

# Check for required environment variables
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "Warning: OPENAI_API_KEY environment variable not set"
    echo "GPT-4 orchestration will be disabled"
    echo ""
fi

# Display startup information
echo "🚀 Starting ImpactOS AI Web API Service"
echo "========================================"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Mode: $MODE"
echo "Log Level: $LOG_LEVEL"
echo ""

# Change to src directory for imports to work correctly
cd src

# Start the service based on mode
if [[ "$MODE" == "development" ]]; then
    echo "Starting in development mode with auto-reload..."
    python web_api.py --host "$HOST" --port "$PORT" --reload --log-level "$LOG_LEVEL"
else
    echo "Starting in production mode..."
    # Production mode with optimized settings
    uvicorn web_api:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        --workers 4 \
        --loop uvloop \
        --http httptools
fi 