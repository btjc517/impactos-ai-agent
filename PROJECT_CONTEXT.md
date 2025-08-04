# Project Context for Cursor AI: ImpactOS AI Agent

This file provides essential context for Cursor's AI to generate code, suggestions, and improvements for the ImpactOS AI Agent. Read and follow this fully before responding. If anything is unclear, missing, or requires prior steps, ask for clarification or suggest preemptive actions (e.g., "This task requires completing the ingestion pipeline first—shall I guide you through that?").

## Project Overview
- **Goal**: Build a comprehensive Python-based ImpactOS AI system that ingests social value data (e.g., volunteering hours, carbon emissions, donations), extracts/normalizes metrics, maps to frameworks (e.g., UK SV Model/MAC, UN SDGs, TOMs, B Corp), stores in SQLite (relational) and FAISS (vector embeddings), and enables natural language Q&A with citations using LangChain and GPT-4. **This project will iteratively expand toward the full ImpactOS vision**
- **Current State**: The system includes production-ready CLI interface, comprehensive testing infrastructure, vector search capabilities, framework mapping, and advanced query processing.
- **Key Features**:
  - Ingestion: Load CSV/Excel (pandas), PDF (PyPDF2/Unstructured for OCR), extract via GPT-4 (JSON output), map to schema, embed chunks (sentence-transformers), store with confidence/provenance.
  - Schema: SQLite tables for ImpactMetric, Commitment, Source, Framework Mappings with comprehensive testing database.
  - Q&A: CLI and programmatic query interface, hybrid search (FAISS vector + SQL aggregates), GPT-4 orchestration for cited answers; handles low confidence with fallback strategies.
  - Testing: Comprehensive test suite with performance tracking, accuracy measurement, and progress monitoring across test runs.
  - Frameworks: Support for MAC, SDGs, TOMs, B Corp with extensible mapping system.
- **Iterative Development Approach**: Each development cycle adds capabilities while maintaining backward compatibility and system stability. Focus on extensible architecture that can scale from current CLI to future web UI, API, and enterprise features.
- **Data Flow**: Ingest raw/messy client data → Extract metrics → Normalize/map to frameworks → Store → Query with citations. Production testing with synthetic and real client data.

## Development Approach
- **Environment**: Python 3.12, virtual env `impactos-env-new`, comprehensive dependencies including advanced ML libraries (transformers, torch, sentence-transformers) for production deployment.
- **Architecture Principles**:
  - **Modularity**: Each component (ingest, query, frameworks, testing) should be independently extensible
  - **Backward Compatibility**: New features should not break existing functionality
  - **Extensibility**: Design patterns that support adding new data sources, frameworks, and query types
  - **Testability**: Comprehensive test coverage with performance and accuracy tracking
  - **Configuration-Driven**: Use configuration files to control behavior without code changes
- **Structure**:
  - `src/`: Modular components (schema.py, ingest.py, query.py, frameworks.py, vector_search.py, main.py)
  - `src/testing/`: Comprehensive testing infrastructure with test runners, databases, performance tracking
  - `data/`: Sample files and datasets for testing and development
  - `db/`: SQLite files and FAISS indices for both production and testing
  - `config/`: Configuration files for different environments and use cases
- **Iterative Development Workflow**:
  - Assess current system capabilities before implementing new features
  - Design new features as extensions to existing modules where possible
  - Implement feature flags for gradual rollout of new capabilities
  - Maintain comprehensive test coverage for all changes
  - Use performance benchmarks to ensure system improvements
  - Document breaking changes and migration paths when necessary
- **Quality Standards**:
  - **No Regressions**: All existing functionality must continue working
  - **Performance**: New features should not degrade existing performance without justification
  - **Testing**: Every change must include appropriate test coverage
  - **Documentation**: Keep all project documentation current with changes
- **Long-term Vision Integration**:
  - Design APIs and interfaces that can support future web UI development
  - Plan for scalability from single-user CLI to multi-tenant web application
  - Consider enterprise requirements (security, audit trails, performance)
  - Build toward eventual SaaS offering while maintaining open-source core

## Development Guidelines
- **Feature Development**:
  - **Extend Rather Than Replace**: Prefer extending existing functionality over complete rewrites
  - **Configuration Over Code**: Use configuration files for behavior changes when possible
  - **Interface Stability**: Maintain stable interfaces for core components
  - **Graceful Degradation**: New features should fail gracefully without breaking core functionality
- **Testing Requirements**:
  - All new features must include test cases
  - Performance impact must be measured and documented
  - Regression tests for critical functionality
  - Use the existing testing infrastructure in `src/testing/`
- **Best Practices**:
  - **No Hallucinations**: Base all code/suggestions on provided context, existing architecture, or proven patterns
  - **Architecture Consistency**: Follow established patterns in the codebase
  - **Incremental Changes**: Break large features into smaller, testable increments
  - **Code Quality**: Maintain PEP 8, comprehensive docstrings, proper error handling
  - **Version Compatibility**: Consider impact on data migrations and API compatibility
- **Communication Standards**:
  - Document rationale for architectural decisions
  - Explain how changes fit into long-term vision
  - Provide migration guidance for breaking changes
  - Keep README and documentation current

## Relevant Resources
- Current Implementation: Examine existing modules in `src/` for established patterns and interfaces
- Testing: Use `src/testing/` infrastructure for validation and performance measurement
- Configuration: Reference `config/` files for system behavior customization
- Frameworks: Extend existing framework mappings in `frameworks.py` rather than creating new systems
- Data: Use existing sample data in `data/` and add new samples as needed for testing

## Future Considerations
- **Web UI Integration**: Current CLI and core logic should be reusable in future web interface
- **API Development**: Design internal interfaces that can be exposed as REST APIs
- **Scalability**: Consider multi-user, multi-tenant requirements in architectural decisions
- **Enterprise Features**: Plan for audit logging, role-based access, advanced security
- **Open Source Sustainability**: Maintain clear separation between open-source core and potential commercial features

Use this context to ensure all development aligns with the long-term iterative expansion strategy while maintaining the robustness and quality of the existing system.