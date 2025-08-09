# ImpactOS AI Agent

A comprehensive AI-powered platform for ingesting, analyzing, and querying social value data with advanced framework mapping and natural language processing capabilities.

## Overview

ImpactOS AI Agent is a sophisticated Python-based system designed to handle the complete lifecycle of social impact data. From ingesting diverse data sources to providing intelligent Q&A capabilities, the system serves as a foundational platform for impact measurement and reporting that will iteratively expand toward the full ImpactOS vision.

### Key Capabilities

- **Intelligent Data Ingestion**: Process CSV, Excel, and PDF files with AI-powered extraction and validation
- **Framework Mapping**: Map impact metrics to established frameworks (MAC, UN SDGs, TOMs, B Corp)
- **Vector Search**: Advanced similarity search using FAISS for contextual data retrieval
- **Natural Language Q&A**: Query your data using natural language with cited, accurate responses
- **Comprehensive Testing**: Production-ready testing infrastructure with performance tracking
- **Extensible Architecture**: Modular design supporting iterative feature development

## Technologies Used

- **AI & ML**: LangChain, OpenAI GPT-5, sentence-transformers, PyTorch
- **Data Processing**: pandas, PyPDF2, openpyxl, polars
- **Search & Storage**: FAISS, SQLite, SQLAlchemy
- **Testing & Performance**: Custom test runners, metrics collection, performance tracking
- **Development**: Python 3.12, Git integration, configuration management

## Quick Start

### Prerequisites
- Python 3.12 or higher
- OpenAI API key

### Installation

1. **Clone and setup environment:**
   ```bash
   git clone [repository-url]
   cd impactos-ai-agent
   python -m venv impactos-env-new
   source impactos-env-new/bin/activate  # On Windows: impactos-env-new\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp config/system_config.json.example config/system_config.json
   # Edit config file with your OpenAI API key and preferences
   ```

## Usage

### Quick Start with Run Script
```bash
# Process data and start interactive mode
./run.sh --ingest data/sample.xlsx --interactive

# Query specific data
./run.sh --query "What are our total volunteering hours?"
```

### Advanced Usage

#### Data Ingestion
```bash
# Ingest CSV files
python src/main.py ingest --input data/sample.csv --type csv

# Process Excel files with multiple sheets
python src/main.py ingest --input data/impact_report.xlsx --type excel

# Handle PDF reports
python src/main.py ingest --input data/annual_report.pdf --type pdf
```

#### Querying Data
```bash
# Interactive query mode
python src/main.py query --interactive

# Direct queries
python src/main.py query "What are the key impact metrics for education initiatives?"

# Framework-specific queries
python src/main.py query "Map our environmental initiatives to UN SDGs"
```

#### Testing and Validation
```bash
# Run comprehensive test suite
cd src/testing
python test_runner.py --types all

# Quick validation tests
python test_runner.py --quick

# Performance benchmarking
python test_runner.py --types performance
```

## Project Structure

```
impactos-ai-agent/
├── src/                          # Core application modules
│   ├── main.py                   # CLI interface and application entry point
│   ├── ingest.py                 # Data ingestion and processing pipeline
│   ├── query.py                  # Natural language query processing
│   ├── vector_search.py          # FAISS-based vector search functionality
│   ├── frameworks.py             # Framework mapping (MAC, SDGs, TOMs, B Corp)
│   ├── schema.py                 # Database schema and models
│   ├── config.py                 # Configuration management
│   ├── extract_v2.py            # Advanced data extraction with AI
│   ├── enhanced_loader.py        # Sophisticated data loading capabilities
│   ├── verify.py                 # Data validation and quality assurance
│   └── testing/                  # Comprehensive testing infrastructure
│       ├── test_runner.py        # Test orchestration and execution
│       ├── test_cases.py         # Standardized test case definitions
│       ├── test_database.py      # Test data management
│       ├── metrics_collector.py  # Performance and accuracy metrics
│       ├── performance_tracker.py # System performance monitoring
│       └── config/               # Testing-specific configuration
├── data/                         # Sample data and test datasets
├── db/                          # Database files and indices
├── config/                      # System configuration files
├── requirements.txt             # Python dependencies
├── run.sh                       # Convenient execution script
└── docs/                        # Documentation (generated)
```

## Configuration

The system uses JSON-based configuration files for flexible behavior control:

```json
{
  "database": {
    "path": "db/impactos.db",
    "vector_index_path": "db/faiss_index"
  },
  "ai": {
    "model": "gpt-5",
    "embedding_model": "all-MiniLM-L6-v2",
    "confidence_threshold": 0.7
  },
  "frameworks": {
    "enabled": ["MAC", "SDG", "TOMS", "B_CORP"],
    "mapping_confidence_threshold": 0.8
  }
}
```

## Development

### Running Tests
```bash
# Full test suite with performance tracking
cd src/testing
python test_runner.py

# Specific test types
python test_runner.py --types accuracy
python test_runner.py --types performance

# Quick validation
python test_runner.py --quick
```

### Code Quality Standards
- **PEP 8 Compliance**: Automated formatting and linting
- **Comprehensive Testing**: >90% coverage on core functionality
- **Documentation**: Google-style docstrings for all public APIs
- **Type Hints**: Full type annotation for better IDE support
- **Performance**: Benchmarked performance with regression detection

### Adding New Features
1. **Assess Existing Architecture**: Understand current patterns and interfaces
2. **Design for Extension**: Add to existing modules rather than creating parallel systems
3. **Implement with Tests**: Include comprehensive test coverage
4. **Validate Performance**: Ensure no regression in system performance
5. **Update Documentation**: Keep all docs current with changes

## Roadmap

### Current Capabilities (MVP+)
- ✅ Advanced data ingestion with AI extraction
- ✅ Multi-framework mapping and validation
- ✅ Vector search with hybrid retrieval
- ✅ Natural language Q&A with citations
- ✅ Comprehensive testing and performance tracking
- ✅ Configuration-driven behavior

### Near-term Enhancements
- 🔄 Real-time data processing pipeline
- 🔄 Advanced analytics and trend analysis
- 🔄 Enhanced framework mapping with ML validation
- 🔄 API endpoints for programmatic access
- 🔄 Advanced query optimization

### Long-term Vision
- 📋 Web-based user interface
- 📋 Multi-tenant architecture
- 📋 Enterprise security and audit features
- 📋 Real-time collaboration tools
- 📋 Advanced AI-powered insights

## Contributing

This project follows iterative development principles:

1. **Maintain Backward Compatibility**: Ensure existing functionality continues working
2. **Extend Rather Than Replace**: Add to existing systems when possible
3. **Comprehensive Testing**: All changes must include appropriate test coverage
4. **Performance Awareness**: Monitor and maintain system performance
5. **Documentation**: Keep all documentation current with changes

### Commit Guidelines
- `feat:` for new features that extend existing capabilities
- `fix:` for bug fixes and error corrections
- `perf:` for performance improvements
- `refactor:` for code improvements without behavior changes
- `test:` for adding or improving tests
- `docs:` for documentation updates
- `breaking:` for changes that break backward compatibility

## License

[License information to be added]

## Support

For questions, issues, or contributions, please refer to the project documentation or open an issue in the repository. 