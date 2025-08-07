#!/bin/bash

# 🚀 ImpactOS AI System - Clean Demo (No Duplicates)
echo "🚀 ImpactOS AI System Demo (Clean)"

# Activate environment
source impactos-env-new/bin/activate

# 1. Show system info
echo "📋 System Overview:"
python src/main.py schema

# 2. Check if sample data is already ingested
echo "🔍 Checking for existing data..."
SAMPLE_FILE="TakingCare_Benevity_Synthetic_Data.xlsx"

# Quick check if data exists by trying a query
echo "Testing if data is available..."
if python src/main.py query "What data sources do we have?" | grep -q "TakingCare_Benevity"; then
    echo "✅ Sample data already available - skipping ingestion"
    SKIP_INGESTION=true
else
    echo "📊 No sample data found - ingesting fresh data..."
    python src/main.py ingest data/$SAMPLE_FILE --verify
    SKIP_INGESTION=false
fi

# 3. Natural language queries
echo "❓ Sample Queries:"
python src/main.py query "What volunteering data do we have?"
python src/main.py query "Show me carbon emissions metrics"
python src/main.py query "What frameworks are our metrics mapped to?"

# 4. Show framework mappings
echo "🗺️ Framework Mappings:"
python src/main.py frameworks

# 5. Summary
echo ""
echo "✅ Demo complete!"
if [ "$SKIP_INGESTION" = true ]; then
    echo "   ℹ️  Used existing data (no duplicates created)"
else
    echo "   ✨ Ingested fresh data"
fi
echo "   🔍 Try your own queries with: ./run.sh query 'Your question here'"
echo "   📋 Check data status with: ./run.sh list" 