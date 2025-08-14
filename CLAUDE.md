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

### Commit Conventions
- Use Conventional Commits with types: `feat:` (new feature), `fix:` (bug fix), `docs:` (documentation), `chore:` (maintenance), `refactor:` (code improvement without changing behavior), `test:` (adding tests), `perf:` (performance improvement), `breaking:` (breaking changes)
- Subject line: Imperative mood, concise (≤50 chars), e.g., "feat: Add framework mapping validation"
- Body: Include rationale for changes, impact on existing functionality, and any migration notes
- Commit incremental, logical changes that maintain system stability

### Python Coding Standards
- **PEP 8 Compliance**: Indentation (4 spaces), line length ≤79 chars, imports at top (standard, third-party, local)
- **Naming Conventions**: 
  - Variables/functions: snake_case (e.g., `process_impact_metrics`)
  - Classes: CamelCase (e.g., `FrameworkMapper`)
  - Constants: UPPER_SNAKE_CASE
- **Docstrings**: Use Google style for functions/classes with parameter types and return values
- **Error Handling**: Use try/except for file I/O, API calls; log errors meaningfully with context
- **Dependencies**: Use existing dependencies when possible; justify new dependencies with clear benefits
- **Backward Compatibility**: Ensure existing APIs and interfaces continue working unless marked deprecated
- **Testing**: Add comprehensive tests for all new functionality using existing testing infrastructure

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

### Architecture & Workflow Standards
- **Extension Strategy**: Prefer extending existing modules (e.g., adding new framework support to `frameworks.py`) over creating parallel implementations
- **Interface Stability**: Maintain stable public interfaces for core components (QuerySystem, IngestionPipeline, etc.)
- **Configuration Management**: Use configuration files in `config/` for behavior changes rather than hardcoded values
- **Testing Integration**: Leverage existing test infrastructure in `src/testing/` for all new features
- **Performance Considerations**: Use performance tracking to ensure new features don't degrade existing functionality
- **Modularity**: Design new features as independent modules that integrate cleanly with existing architecture

### Development Workflow
- **Pre-Development Assessment**: Always examine existing code patterns and interfaces before implementing new features
- **Incremental Development**: Break large features into smaller, testable increments that can be safely deployed
- **Feature Flags**: Use configuration-based feature flags for gradual rollout of new capabilities
- **Documentation Updates**: Update README, docstrings, and inline documentation with any changes
- **Testing Requirements**: Every change must include appropriate test coverage and pass existing tests
- **Performance Validation**: Use existing performance tracking to validate that changes meet quality standards

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

## Development Templates & Patterns

### Test Case Template
When creating new test cases for `src/testing/test_cases.py`:

```python
TestCase(
    id="your_test_id",
    query="Natural language question",
    query_type="aggregation|descriptive|analytical",
    complexity="simple|medium|complex",
    expected_answer_keywords=["keyword1", "keyword2"],
    expected_sources=["file1.xlsx", "file2.csv"],
    expected_metrics={"metric_name": "expected_value"},
    expected_frameworks=["MAC", "SDG"],
    description="Clear description of what this test validates",
    min_response_quality=0.7
)
```

### Framework Mapping Template
When adding new framework mappings in `frameworks.py`:

```python
framework_mapping = {
    'framework_name': {
        'category_id': 'Human-readable description',
        'another_id': 'Another description'
    }
}
```

### Configuration Addition Template
When adding new configuration options in `config.py`:

```python
@dataclass
class NewFeatureConfig:
    """Configuration for new feature."""
    enabled: bool = True
    threshold: float = 0.5
    max_items: int = 100
    
    def validate(self) -> bool:
        """Validate configuration values."""
        return 0.0 <= self.threshold <= 1.0 and self.max_items > 0
```

### Error Handling Pattern
Standard error handling for the project:

```python
try:
    result = risky_operation()
    logger.info(f"Operation completed successfully: {len(result)} items")
    return result
except SpecificException as e:
    logger.warning(f"Expected error in operation: {e}")
    return fallback_result()
except Exception as e:
    logger.error(f"Unexpected error in operation: {e}")
    raise ProcessingError(f"Operation failed: {e}") from e
```

### Performance Measurement Pattern
When adding performance tracking:

```python
with self.metrics_collector.time_operation('operation_name'):
    result = expensive_operation()
    self.metrics_collector.record_operation_result(len(result))
```

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

### Security & Production Considerations
- **API Security**: Design internal APIs with future external exposure in mind
- **Data Protection**: Handle sensitive data according to existing patterns in the codebase
- **Environment Configuration**: Use environment variables for sensitive configuration (API keys, database URLs)
- **Audit Trails**: Consider logging and audit requirements for enterprise use cases
- **Error Handling**: Implement comprehensive error handling with appropriate logging levels

### Long-term Vision Alignment
- **Scalability**: Design features to work from single-user CLI to multi-tenant web application
- **Web UI Readiness**: Structure backend logic to be easily consumable by future web interfaces
- **API Potential**: Design internal interfaces that could be exposed as REST APIs
- **Enterprise Features**: Consider requirements for role-based access, audit logging, and advanced security
- **Open Source Sustainability**: Maintain clear separation between core open-source features and potential commercial extensions

### Quality Assurance
- **Test Coverage**: Maintain high test coverage using existing testing infrastructure
- **Performance Benchmarks**: Use performance tracking to ensure system improvements over time
- **Code Quality**: Regular refactoring to maintain code quality while preserving functionality
- **Documentation Currency**: Keep all documentation aligned with current system capabilities

### Performance Optimization
- Adjust batch sizes in configuration for memory-constrained environments
- Tune FAISS parameters for optimal search performance
- Use result caching for frequently repeated queries
- Monitor performance metrics through the testing infrastructure