# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Environment Setup
```bash
# Activate virtual environment
source impactos-env-new/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY=your-key-here
```

### Development Workflow
```bash
# Process data and start interactive mode
./run.sh --ingest data/sample.xlsx --interactive

# Run comprehensive test suite
cd src/testing
python test_runner.py --types all

# Quick validation tests
python test_runner.py --quick

# Performance benchmarking
python test_runner.py --types performance

# Start web API
./start_web_api.sh

# Direct CLI usage (bypassing run.sh)
python src/main.py ingest --input data/file.xlsx --type excel
python src/main.py query --interactive
python src/main.py query "What are our total volunteering hours?"
```

### Testing Commands
- `python src/testing/test_runner.py`: Full test suite with performance tracking
- `python src/testing/test_runner.py --types accuracy`: Accuracy-focused tests
- `python src/testing/test_runner.py --types performance`: Performance benchmarks
- `python src/testing/test_runner.py --quick`: Quick validation tests

## Architecture Overview

### Core Data Flow
1. **Ingestion Pipeline** (`src/ingest.py`): Processes CSV/Excel/PDF files using AI-powered extraction
2. **Storage Layer**: SQLite database with FAISS vector indexing for hybrid search
3. **Query System** (`src/query.py`): Natural language processing with GPT-4 orchestration
4. **Framework Mapping** (`src/frameworks.py`): Maps metrics to standards (MAC, UN SDGs, TOMs, B Corp)
5. **Vector Search** (`src/vector_search.py`): FAISS-based similarity search with configurable thresholds

### Key Components
- **Enhanced File Loader** (`src/enhanced_loader.py`): Bulletproof data extraction with Polars integration
- **Schema Management** (`src/schema.py`): Database models and initialization
- **Configuration System** (`src/config.py`): Centralized settings management
- **Testing Infrastructure** (`src/testing/`): Comprehensive test runners with performance tracking

### Flexible Data Processing
- **Dynamic Processing Pipeline**: Configurable data transformation stages (avoid hardcoded medallion patterns)
- **Adaptive Schema**: Schema should adapt to different data sources rather than enforcing rigid structures
- **Dynamic Facts System**: Facts and concepts should be dynamically discovered and configured, not static

## Development Guidelines

### Code Standards
- Follow PEP 8 compliance with Google-style docstrings
- Use existing patterns: extend rather than rewrite components
- Configuration-driven behavior via `config/system_config.json`
- Comprehensive error handling with meaningful logging
- All new features require test coverage using existing infrastructure

### Extension Strategy
- **Dynamic Configuration**: Avoid hardcoding data processing patterns, schema assumptions, or framework mappings
- **Flexible Architecture**: Design systems that adapt to different data sources and use cases rather than enforcing rigid structures
- **Runtime Adaptability**: Facts, concepts, and processing pipelines should be configurable at runtime, not compile-time
- **Modular Design**: New features should integrate cleanly without breaking existing functionality
- **Backward Compatibility**: Ensure existing CLI and programmatic interfaces continue working

### Testing Requirements
- Use the comprehensive testing infrastructure in `src/testing/`
- All changes must include appropriate test coverage
- Performance impact must be measured and documented
- Regression tests for critical functionality

## Configuration Management

### Key Configuration Files
- `config/system_config.json`: Main system configuration with vector search, query processing, extraction, and analysis settings
- `src/testing/config/system_config.json`: Testing-specific configuration
- Environment variables: `OPENAI_API_KEY`, `IMPACTOS_DB_PATH`, `IMPACTOS_FAISS_INDEX_PATH`

### Configuration Sections
- **vector_search**: FAISS parameters, similarity thresholds, embedding settings
- **query_processing**: Result limits, model selection, caching configuration
- **extraction**: AI extraction parameters, confidence thresholds, batch processing
- **scalability**: Memory limits, connection pooling, timeout settings
- **analysis**: Intent classification, category keywords, framework mapping thresholds

## Flexibility and Dynamic Configuration

### Avoid Static Patterns
- **Medallion Architecture**: Should be configurable transformation stages, not hardcoded Bronze→Silver→Gold layers
- **Facts System**: Facts and concepts should be dynamically discoverable and configurable, not static JSON files
- **Schema Enforcement**: Schema should adapt to data sources rather than forcing rigid data structures
- **Framework Mappings**: Should support dynamic framework addition without code changes

### Dynamic System Design
- **Runtime Configuration**: Core processing logic should be configurable at runtime through config files
- **Adaptive Processing**: Data pipelines should adapt to different data source characteristics
- **Extensible Frameworks**: Framework support should be plugin-based rather than hardcoded
- **Flexible Fact Discovery**: System should discover and learn facts from data rather than relying on static definitions

## Data Sources and Processing

### Supported File Types
- CSV/Excel files with automatic column detection and type inference
- PDF documents with OCR and structured text extraction
- Enhanced loading with bulletproof accuracy using Polars for type safety

### Storage Architecture
- SQLite database for relational data (metrics, commitments, sources, framework mappings)
- FAISS vector indices for similarity search with configurable embedding dimensions
- Comprehensive metadata tracking with confidence scores and provenance

## Framework Integration

The system supports multiple social value frameworks:
- **MAC (Measurement and Accounting Criteria)**: UK Social Value Model
- **UN SDGs**: Sustainable Development Goals
- **TOMs**: Themes, Outcomes and Measures
- **B Corp Assessment**: B Corporation framework

Framework mappings are dynamically loaded and extensible through the concept graph system.

## Performance and Quality

### Performance Monitoring
- Built-in performance tracking with metrics collection
- Memory usage monitoring and optimization
- Query latency profiling and optimization
- Comprehensive benchmarking infrastructure

### Quality Assurance
- >90% test coverage on core functionality
- Automated accuracy measurement and tracking
- Regression detection across test runs
- Performance benchmarking with historical comparison

## API and Integration

### CLI Interface
- Main entry point: `src/main.py` with subcommands for `ingest`, `query`, and `metrics`
- Interactive query mode for exploratory data analysis
- Batch processing capabilities for large datasets

### Web API
- FastAPI-based service (`src/web_api.py`) with health checks and query endpoints
- Development and production deployment modes
- RESTful API design ready for future web UI integration

## Environment Variables

- `OPENAI_API_KEY`: Required for AI-powered extraction and query processing
- `IMPACTOS_DB_PATH`: Override default database path (default: `db/impactos.db`)
- `IMPACTOS_FAISS_INDEX_PATH`: Override FAISS index path (default: `db/faiss_index`)
- `TOKENIZERS_PARALLELISM`: Set to `false` to avoid threading issues

## Troubleshooting

### Common Issues
- Missing OpenAI API key: Set environment variable or GPT features will be disabled
- Database initialization: System auto-initializes database on first run
- Virtual environment: Recommended to use `impactos-env-new` for dependency isolation
- Memory issues: Configure limits in `config/system_config.json` scalability section

### Performance Optimization
- Adjust batch sizes in configuration for memory-constrained environments
- Tune FAISS parameters for optimal search performance
- Use result caching for frequently repeated queries
- Monitor performance metrics through the testing infrastructure