#!/bin/bash

# 🚀 ImpactOS AI System - Fresh Demo (Clean Database)
echo "🚀 ImpactOS AI System Demo (Fresh Start)"

# Activate environment
source impactos-env-new/bin/activate

echo "🗑️  Creating fresh database..."
# Remove existing database to start clean
rm -f db/impactos.db
rm -f db/faiss_index.*

# 1. Show system info (this will create new database)
echo "📋 System Overview:"
python src/main.py schema

# 2. Ingest sample data
echo "📊 Ingesting fresh sample data..."
python src/main.py ingest data/TakingCare_Benevity_Synthetic_Data.xlsx --verify

# 3. Natural language queries
echo "❓ Sample Queries:"
python src/main.py query "What volunteering data do we have?"
python src/main.py query "Show me carbon emissions metrics"
python src/main.py query "What frameworks are our metrics mapped to?"

# 4. Show framework mappings
echo "🗺️ Framework Mappings:"
python src/main.py frameworks

echo ""
echo "✅ Fresh demo complete!"
echo "   ✨ Started with clean database"
echo "   🔍 Try your own queries with: ./run.sh query 'Your question here'"
echo "   📋 Check data status with: ./run.sh list" 