# ImpactOS AI - Quick Commands

## Setup
```bash
# Activate environment
source impactos-env-new/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY=your-key-here
```

## Ingest Data
```bash
# CSV/Excel files
python src/main.py ingest --input data/file.xlsx --type excel

# PDF files
python src/main.py ingest --input data/report.pdf --type pdf
```

## Query Data
```bash
# Interactive mode
python src/main.py query --interactive

# Direct query
python src/main.py query "What are our volunteering hours?"
```

## Start API
```bash
# Development mode
./start_web_api.sh

# Or directly
python src/web_api.py --host localhost --port 8000
```

## Use API
```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are our carbon emissions?"}'

# Ingest endpoint
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample.xlsx" \
  -F "file_type=excel"
```

## Test
```bash
# Run all tests
cd src/testing && python test_runner.py

# Quick validation
python test_runner.py --quick
``` 

source /Users/benjamincheesebrough/Desktop/impactos-ai-agent/impactos-env-new/bin/activate
PYTHONPATH=src python3 src/testing/llm_eval.py --suite all \
  --models gpt-5-mini,gpt-4o-mini \
  --max-output-tokens 256 \
  --reasoning minimal \
  --verbosity low \
  --enforce-json auto \
  --concurrency 3

./run.sh metrics --file TakingCare_Benevity_Synthetic_Data.xlsx --json