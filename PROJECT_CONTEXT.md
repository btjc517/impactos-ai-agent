# Project Context for Cursor AI: ImpactOS AI Layer MVP Phase One

This file provides essential context for Cursor's AI to generate code, suggestions, and improvements for the ImpactOS AI Layer MVP. Read and follow this fully before responding. If anything is unclear, missing, or requires prior steps, ask for clarification or suggest preemptive actions (e.g., "This task requires completing the ingestion pipeline first—shall I guide you through that?").

## Project Overview
- **Goal**: Build a standalone Python CLI tool (Phase One of MVP) to ingest local CSV/PDF files containing social value data (e.g., volunteering hours, carbon emissions, donations), extract/normalize metrics, map to frameworks (e.g., UK SV Model/MAC, UN SDGs, TOMs, B Corp), store in SQLite (relational) and FAISS (vector embeddings), and enable terminal Q&A with citations using LangChain and GPT-4.
- **Key Features**:
  - Ingestion: Load CSV (pandas), PDF (PyPDF2/Unstructured for OCR), extract via GPT-4 (JSON output), map to schema, embed chunks (sentence-transformers 'all-MiniLM-L6-v2'), store with confidence/provenance.
  - Schema: SQLite tables for ImpactMetric (e.g., name="volunteering_hours", value=float, unit="hours"), Commitment, Source, Framework Mappings.
  - Q&A: CLI query, hybrid search (FAISS vector + SQL aggregates), GPT-4 orchestration for cited answers; handle low confidence (<0.7 similarity → "Insufficient data").
  - Testing: Use synthetic data (e.g., from "TakingCare" Excels or "ImpactOS...xlsx"), aim for 90% citation coverage on 10-20 files.
- **Phased Approach**: This is Phase One (CLI-only, 4 weeks). Later phases add web UI, quality gates, more frameworks.
- **Data Flow**: Ingest raw/messy client data → Extract metrics → Normalize/map to frameworks → Store → Query with citations. Use mocks to simulate client inputs.

## Development Approach
- **Environment**: Python 3.12, virtual env `impactos-env`, dependencies: langchain, openai, pandas, pypdf2, sqlite3, faiss-cpu, sentence-transformers (optionally unstructured for OCR).
- **Structure**:
  - `src/`: Code files (e.g., `schema.py` for DB init, `ingest.py` for loading/parsing, `query.py` for retrieval/QA).
  - `data/`: Sample files (CSVs/XLSX/PDFs, e.g., TakingCare mocks).
  - `db/`: SQLite files (e.g., `impactos.db`), FAISS index.
  - `tests/`: Unit tests.
  - Other: `.gitignore`, `README.md`, `CURSOR_GUIDELINES.md` (for commits/coding standards).
- **Workflow**:
  - Follow spec documents (e.g., MVP spec PDF, roadmap).
  - Build incrementally: Setup → Schema → Ingestion → Q&A → Tests → Mockups.
  - Handle edges: Invalids ("!!!INVALID!!!"), anomalies, low confidence, duplicates (Levenshtein/fuzzy matching).
  - API: Use OpenAI API key from env; prompt GPT-4 for extraction (e.g., "Extract as JSON: {name, value, unit, timestamp}").
- **Best Practices**:
  - **No Hallucinations**: Base all code/suggestions on provided context, spec, or tools. If info is missing (e.g., unclear schema field), ask: "Need more details on X—can you share context?"
  - **Preemptive Actions**: If a task depends on priors (e.g., querying requires ingestion), suggest: "This needs the ingestion pipeline first. Shall I implement that?"
  - **Tools Usage**: Use available tools (e.g., code_execution for testing snippets) if needed for verification.
  - **Commits**: Follow `CURSOR_GUIDELINES.md` (Airbnb-inspired: `feat:`, imperative, concise).
  - **Code Quality**: PEP 8, docstrings, error handling, modularity.
  - **Testing**: Add tests for each feature; use mocks to validate (e.g., ingest TakingCare payroll → query total donations).

## Relevant Resources
- Specs: MVP Specification PDF (Phase One CLI details), Roadmap PDF (overall architecture).
- Samples: Use "data/" files (e.g., TakingCare XLSX—convert to CSV if needed; test mapping to MAC/SDG).
- Frameworks: Map metrics to MAC (e.g., 8.1 for community), SDGs (e.g., Goal 11), TOMs (e.g., NT90 for volunteering).
- Questions: Always ask if context is insufficient—e.g., "To implement this, need clarification on framework config. What specifics?"

Use this to ensure accurate, effective assistance. Prioritize clarity and completeness.