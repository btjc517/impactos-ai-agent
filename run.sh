#!/bin/bash

# ImpactOS AI Layer MVP - Helper Script
# This script activates the virtual environment and runs commands

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 ImpactOS AI Layer MVP${NC}"
echo -e "${BLUE}Activating virtual environment...${NC}"

# Activate virtual environment and run command
source impactos-env-new/bin/activate && python3 src/main.py "$@" 