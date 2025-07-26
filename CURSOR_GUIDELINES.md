# Cursor AI Guidelines for ImpactOS AI Layer MVP

These guidelines help Cursor's AI generate code, commits, and suggestions consistently. Follow them strictly when assisting.

## Commit Conventions
- Use Conventional Commits inspired by Airbnb's JavaScript style guide[](https://github.com/airbnb/javascript#commit-messages), adapted for this Python project.
- Prefix messages with types: `feat:` (new feature), `fix:` (bug fix), `docs:` (documentation), `chore:` (maintenance), `refactor:` (code improvement without changing behavior), `test:` (adding tests).
- Subject line: Imperative mood, concise (≤50 chars), e.g., "feat: Add ingestion CLI command".
- Body: Optional, detailed explanation if needed. Reference issues if applicable.
- Examples:
  - `feat: Implement PDF parsing in ingest.py`
  - `fix: Handle invalid values in data extraction`
- Commit small, logical changes frequently.

## Coding Standards (Python-Specific)
- Follow PEP 8: Indentation (4 spaces), line length ≤79 chars, imports at top (standard, third-party, local).
- Naming: Variables/functions: snake_case (e.g., `ingest_file`). Classes: CamelCase (e.g., `ImpactMetric`). Constants: UPPER_SNAKE_CASE.
- Docstrings: Use Google style for functions/classes.
- Error Handling: Use try/except for file I/O, API calls; log errors meaningfully.
- Dependencies: Only use installed libs (e.g., langchain, pandas); no new installs.
- Testing: Add unit tests in `tests/`; aim for 90% coverage on key functions.

## Project Structure and Workflow
- Files: Keep code in `src/` (e.g., `ingest.py`), samples in `data/`, DB in `db/`.
- Branching: Use `feature/<task-name>` for new work, `fix/<issue>` for bugs.
- Prompts to Cursor: Be specific, e.g., "Refactor this function to handle low-confidence data, following PEP 8."
- AI Assistance: Generate code snippets with explanations; suggest improvements but don't overwrite without confirmation.

## Other Best Practices
- Security: Avoid hardcoding API keys; use env vars (e.g., `os.getenv('OPENAI_API_KEY')`).
- Performance: Optimize for small datasets in Phase One; profile if needed.
- Documentation: Update `README.md` with new features/usage.
- Reviews: Suggest commits that are review-ready (small, focused).

Follow these to maintain consistency. If unclear, ask for clarification in prompts.