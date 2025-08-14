## Query Logging and SQL Determinism

- All SQL used for deterministic answers is compiled inside the code and logged with timing.
- No free-text SQL is executed; LLMs may only plan high-level intent with structured outputs.

Where logged
- CLI: `src/main.py` uses `QuerySystem.query_structured_instrumented` which returns `timings`.
- Web API: `/query` also captures timings and logs via `telemetry`.

Future work
- Introduce a dedicated compiler module that emits SQL strings and parameters for each query template, and a sandboxed executor which logs `(sql, params, ms, tier)` per request.

