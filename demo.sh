#!/bin/bash

# 🚀 ImpactOS AI System - Quick Demo
echo "🚀 ImpactOS AI System Demo"

# Activate environment
source impactos-env-new/bin/activate

# 1. Show system info
echo "📋 System Overview:"
python src/main.py schema

# 2. Ingest sample data (fast demo)
echo "📊 Ingesting sample data..."
python src/main.py ingest data/TakingCare_Benevity_Synthetic_Data.xlsx --verify

# 3. Natural language queries
echo "❓ Sample Queries:"
python src/main.py query "What volunteering data do we have?"
python src/main.py query "Show me carbon emissions metrics"
python src/main.py query "What frameworks are our metrics mapped to?"

# 4. Show framework mappings
echo "🗺️ Framework Mappings:"
python src/main.py frameworks

echo "✅ Demo complete! Try your own queries with:"
echo "   ./run.sh query 'Your question here'" 