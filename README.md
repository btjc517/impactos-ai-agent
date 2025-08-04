# ImpactOS AI Agent

A CLI tool to ingest and query social value data

## Overview

This project provides a Python-based command-line interface for ingesting CSV and PDF files, processing them into a SQLite database with a schema for impact metrics, commitments, sources, and framework mappings, and enabling terminal Q&A with citations using advanced AI technologies.

## Technologies Used

- **LangChain**: For building AI-powered query chains
- **OpenAI**: For natural language processing and generation
- **pandas**: For data manipulation and analysis
- **PyPDF2**: For PDF document processing
- **sentence-transformers**: For text embeddings
- **FAISS**: For efficient similarity search
- **SQLite**: For local database storage

## Setup Instructions

### Prerequisites
- Python 3.12 or higher

### Installation

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv impactos-env
   source impactos-env/bin/activate  # On Windows: impactos-env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install langchain openai pandas pypdf2 sqlite3 faiss-cpu sentence-transformers
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Project Structure

```
impactos-ai-agent/
├── src/                    # Python source files
│   ├── ingest.py          # Data ingestion logic
│   ├── query.py           # Query processing
│   └── schema.py          # Database schema definitions
├── data/                  # Sample CSV and PDF files
├── db/                    # SQLite database files
├── tests/                 # Unit tests
├── .gitignore            # Git ignore patterns
└── README.md             # This file
```

## Usage

### Data Ingestion
```bash
python src/ingest.py --input data/sample.csv --type csv
python src/ingest.py --input data/report.pdf --type pdf
```

### Querying Data
```bash
python src/query.py "What are the key impact metrics for education initiatives?"
```

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
This project follows Python naming conventions:
- **Files**: lowercase_with_underscores.py
- **Functions/Variables**: lowercase_with_underscores
- **Classes**: CamelCase
- **Database Tables**: snake_case

## Contributing

Please follow the git commit message conventions:
- `feat:` for new features
- `fix:` for bug fixes
- `chore:` for maintenance
- `docs:` for documentation
- `test:` for tests
- `refactor:` for code improvements

Keep commit messages concise (50 characters max for subject line) and use imperative mood. 