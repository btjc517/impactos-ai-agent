from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
import os
import re

from openai import OpenAI

from database_adapter import create_database_adapter
from ir_models import IntermediateRepresentation
from ir_planner import build_ir_prompt, generate_ir_with_validation
from llm_utils import choose_model, call_chat_completion
from metric_loader import MetricCatalog
import logging

logger = logging.getLogger(__name__)


# -----------------------------
# Public Tool Interfaces (MVP)
# -----------------------------

def find_metrics(query_terms: str | List[str], *, metrics_dir: str = "metrics", k: int = 5) -> List[Dict[str, Any]]:
    """Return top catalog entries (planner snippets) for provided terms.

    If a list is provided, aggregate top results across the list and truncate to k.
    """
    cat = MetricCatalog(metrics_dir=metrics_dir)
    snippets: List[Dict[str, Any]] = []

    def _gather(term: str) -> List[Dict[str, Any]]:
        # Exact id match first
        m = cat.get_metric(term)
        if m:
            sn = cat.to_snippet(m['id'])
            return [sn] if sn else []
        # Synonym search
        hits = cat.search_by_synonym(term)
        out: List[Dict[str, Any]] = []
        for h in hits[:k]:
            sn = cat.to_snippet(h['id'])
            if sn:
                out.append(sn)
        return out

    if isinstance(query_terms, list):
        seen: set[str] = set()
        for t in query_terms:
            for sn in _gather(str(t)):
                mid = sn.get('id')
                if mid and mid not in seen:
                    seen.add(mid)
                    snippets.append(sn)
        return snippets[:k]
    else:
        return _gather(str(query_terms))[:k]


def generate_ir(client: Optional[OpenAI], *, question: str, catalog_snippets: List[Dict[str, Any]], time_policy: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate an IR JSON object strictly matching the schema.

    Uses validated planner with one-shot repair. Asserts IR time matches provided
    time_policy (start/end/label/fiscal_used).
    If client is None, returns a minimal default IR using the time policy.
    """
    # Few-shot retrieval: harvest short example questions from provided snippets
    few_shots: List[Dict[str, Any]] = []
    try:
        for sn in (catalog_snippets or [])[:5]:
            for q in (sn.get('example_questions') or [])[:1]:
                few_shots.append({'question': q})
    except Exception:
        pass

    if client is None:
        # Deterministic minimal IR without LLM
        default = IntermediateRepresentation(
            operation='aggregate',
            metric_id=(catalog_snippets[0]['id'] if catalog_snippets else None),
            measures=[],
            filters=[],
            time={
                'start': time_policy.get('start'),
                'end': time_policy.get('end'),
                'label': time_policy.get('label'),
                'fiscal': time_policy.get('fiscal_used'),
                'policy_id': time_policy.get('policy_id'),
            },
            group_by=[],
            order_by=[],
            limit=None,
        )
        return default.model_dump(), {'prompt_version': os.getenv('PROMPTS_VERSION') or 'ir.v1', 'validation_passed': True, 'repaired': False}

    # Use validated planner with repair and meta for telemetry
    try:
        # Pass snippet IDs to generator to re-gather robust snippets internally
        cat_terms = [sn['id'] for sn in (catalog_snippets or []) if sn.get('id')]
        ir_obj, meta = generate_ir_with_validation(
            client,
            question=question,
            catalog_terms=cat_terms,
            time_policy=time_policy,
            metrics_dir='metrics',
            few_shots=few_shots if few_shots else None,
        )
    except Exception as e:
        raise ValueError({
            'error': 'ir_generation_failed',
            'message': 'IR could not be generated',
            'details': str(e),
        })

    # Assert time block frozen
    try:
        t = (ir_obj or {}).get('time') or {}
        expected = {
            'start': time_policy.get('start'),
            'end': time_policy.get('end'),
            'label': time_policy.get('label'),
            'fiscal': time_policy.get('fiscal_used'),
            'policy_id': time_policy.get('policy_id'),
        }
        if (
            t.get('start') != expected['start'] or
            t.get('end') != expected['end'] or
            t.get('label') != expected['label'] or
            t.get('fiscal') != expected['fiscal'] or
            t.get('policy_id') != expected['policy_id']
        ):
            raise ValueError({'error': 'ir_time_mismatch', 'message': 'IR time does not match injected time policy', 'expected': expected, 'actual': t})
    except Exception as e:
        raise

    return ir_obj, meta


def render_sql(ir: Dict[str, Any], *, metrics_dir: str = "metrics", dialect: str = 'sqlite') -> Tuple[str, Tuple[Any, ...]]:
    """Render SQL + params from an IR using the metric catalog when available.

    MVP rules:
    - If metric has calc_sql in catalog, wrap it and apply time window if columns exist
    - Else, query impact_metrics with optional time filter using sources.processed_timestamp
    Supports operation in {aggregate, trend}. Others raise ValueError.
    """
    cat = MetricCatalog(metrics_dir=metrics_dir)
    ir_obj = IntermediateRepresentation(**ir)  # validate again for safety
    operation = ir_obj.operation
    metric_id = ir_obj.metric_id

    # Time window handling (inclusive bounds)
    start = ir_obj.time.start
    end = ir_obj.time.end
    time_filter_sql = ""
    params: List[Any] = []
    if start and end:
        time_filter_sql = " AND COALESCE(im.extracted_timestamp, s.processed_timestamp) BETWEEN ? AND ?"
        params.extend([start, end])

    # Catalog-based
    sql: str
    if metric_id:
        m = cat.get_metric(metric_id)
        calc_sql = (m or {}).get('calc_sql') if m else None
        if calc_sql:
            # Wrap catalog SQL as a subquery named t
            if operation in ('aggregate', 'trend'):
                # Contract: subquery exposes t.value, t.date_key, optional t.tenant_id and dimension keys.
                # Apply tenant/time filters outside.
                where_outer: List[str] = ["1=1"]
                if start and end:
                    where_outer.append("t.date_key BETWEEN ? AND ?")
                tenant_id = os.getenv('TENANT_ID')
                if tenant_id:
                    where_outer.append("t.tenant_id = ?")
                params_out: List[Any] = []
                if start and end:
                    params_out.extend([start, end])
                if tenant_id:
                    params_out.append(tenant_id)
                # ORDER BY and LIMIT can be added generically when provided in IR
                order_by_sql = ""
                if ir_obj.order_by:
                    order_terms: List[str] = []
                    for ob in ir_obj.order_by:
                        # Only allow simple column names (safe heuristic)
                        col = re.sub(r"[^a-zA-Z0-9_]+", "", ob.field or "")
                        if not col:
                            continue
                        order_terms.append(f"{col} {'ASC' if ob.dir == 'asc' else 'DESC'}")
                    if order_terms:
                        order_by_sql = " ORDER BY " + ", ".join(order_terms)
                limit_sql = ""
                if isinstance(ir_obj.limit, int) and ir_obj.limit > 0:
                    limit_sql = " LIMIT ?"
                sql = f"SELECT * FROM ({calc_sql}) AS t WHERE " + " AND ".join(where_outer) + order_by_sql + limit_sql
                if limit_sql:
                    params_out.append(int(ir_obj.limit))
                return sql, tuple(params_out)
        # Fallback to impact_metrics by metric_name
        metric_name = (m or {}).get('title') or metric_id
    else:
        metric_name = None

    if operation not in ('aggregate', 'trend'):
        raise ValueError(f"Unsupported IR operation: {operation}")

    # Default aggregate over impact_metrics
    where_parts: List[str] = ["im.metric_value IS NOT NULL"]
    # Translate filters with bound params only
    allowed_fields = {
        'metric_name': 'im.metric_name',
        'metric_unit': 'im.metric_unit',
        'metric_category': 'im.metric_category',
        'context_description': 'im.context_description',
        'source_sheet_name': 'im.source_sheet_name',
        'source_column_name': 'im.source_column_name',
        'source_cell_reference': 'im.source_cell_reference',
        'filename': 's.filename',
    }
    # Extended date dims via dim_date
    date_group_fields = {
        'month': 'dd.month',
        'quarter': 'dd.quarter',
        'fy': 'dd.fiscal_year',
    }

    # Validate filters and group_by fields against allowlists
    invalid_fields: List[str] = []
    for f in (ir_obj.filters or []):
        if f.field not in allowed_fields:
            invalid_fields.append(f.field)
    for g in (ir_obj.group_by or []):
        if g not in allowed_fields and g not in date_group_fields:
            invalid_fields.append(g)
    if invalid_fields:
        raise ValueError({'error': 'compile_unknown_field', 'fields': list(sorted(set(invalid_fields)))})
    def _escape_like(val: str) -> str:
        # Escape % and _ with backslash for LIKE safety
        return str(val).replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')

    def _add_filter(f):
        field = allowed_fields.get(f.field)
        if not field:
            return
        op = f.op.upper()
        if op in ('=', '!=', '>', '>=', '<', '<='):
            where_parts.append(f"{field} {op} ?")
            params.append(f.value)
        elif op == 'LIKE':
            where_parts.append(f"{field} LIKE ? ESCAPE '\\'")
            params.append(_escape_like(f.value))
        elif op == 'IN':
            vals = f.value if isinstance(f.value, list) else [f.value]
            if not vals:
                # Compile to false to avoid empty IN ()
                where_parts.append("1=0")
                return
            placeholders = ','.join(['?' for _ in vals])
            where_parts.append(f"{field} IN ({placeholders})")
            params.extend(vals)
        elif op == 'BETWEEN':
            vals = f.value if isinstance(f.value, list) else [f.value]
            if len(vals) >= 2:
                where_parts.append(f"{field} BETWEEN ? AND ?")
                params.extend(vals[:2])
    for f in (ir_obj.filters or []):
        try:
            _add_filter(f)
        except Exception:
            continue
    if metric_name:
        where_parts.append("LOWER(im.metric_name) = LOWER(?)")
        params.insert(0, metric_name)
    # Automatic tenant filter: prefer provided TENANT_ID; fallback to strict env fragment
    tenant_id = os.getenv('TENANT_ID')
    if tenant_id:
        where_parts.append("s.tenant_id = ?")
        params.append(tenant_id)
    else:
        tenant_sql = os.getenv('TENANT_FILTER_SQL')
        tenant_param = os.getenv('TENANT_FILTER_PARAM')
        if tenant_sql:
            frag = tenant_sql.strip()
            if all(tok not in frag.upper() for tok in [';', '--', '/*']) and re.match(r"^[a-zA-Z0-9_\.\s=<>\?%,'-]+$", frag or ''):
                where_parts.append(f"({frag})")
                if tenant_param is not None:
                    params.append(tenant_param)

    where_sql = " AND ".join(where_parts)

    # Build SELECT list based on measures and group_by
    group_cols: List[str] = []
    select_cols: List[str] = []
    join_dim_date = any(g in date_group_fields for g in (ir_obj.group_by or []))
    if join_dim_date:
        # Prepare join clause on date cast
        if dialect == 'postgres':
            date_join = "JOIN dim_date dd ON CAST(COALESCE(im.extracted_timestamp, s.processed_timestamp) AS DATE) = dd.date_key"
        else:
            date_join = "JOIN dim_date dd ON DATE(COALESCE(im.extracted_timestamp, s.processed_timestamp)) = dd.date_key"
    else:
        date_join = ""
    for g in (ir_obj.group_by or []):
        col = allowed_fields.get(g) or date_group_fields.get(g)
        if col:
            group_cols.append(col)
            select_cols.append(f"{col} AS {g}")

    agg_cols: List[str] = []
    def _add_measure(me):
        # Support a small allow-list of aggregate expressions over metric_value
        alias = re.sub(r"[^a-zA-Z0-9_]+", "", me.alias or "value") or "value"
        expr_norm = (me.expr or '').strip().lower()
        allowed_measure_columns = {'im.metric_value'}
        if expr_norm in ('sum(metric_value)', 'sum'):
            agg_cols.append(f"SUM(im.metric_value) AS {alias}")
            return alias
        if expr_norm in ('avg(metric_value)', 'avg'):
            agg_cols.append(f"AVG(im.metric_value) AS {alias}")
            return alias
        if expr_norm in ('min(metric_value)', 'min'):
            agg_cols.append(f"MIN(im.metric_value) AS {alias}")
            return alias
        if expr_norm in ('max(metric_value)', 'max'):
            agg_cols.append(f"MAX(im.metric_value) AS {alias}")
            return alias
        if expr_norm in ('count(*)', 'count'):
            agg_cols.append(f"COUNT(*) AS {alias}")
            return alias
        # If expr specifies a column, reject unless allow-listed
        m = re.match(r"^(sum|avg|min|max)\(([^)]+)\)$", (me.expr or '').strip(), flags=re.IGNORECASE)
        if m:
            col_raw = m.group(2).strip()
            col_map = allowed_fields.get(col_raw) or (col_raw if col_raw.startswith('im.') else None)
            if col_map and col_map in allowed_measure_columns:
                agg_cols.append(f"{m.group(1).upper()}({col_map}) AS {alias}")
                return alias
        raise ValueError({'error': 'compile_unknown_measure_column', 'expr': me.expr})

    order_terms: List[str] = []
    if ir_obj.measures:
        aliases: List[str] = []
        for me in ir_obj.measures:
            aliases.append(_add_measure(me))
        # Dialect-specific filename aggregation
        if dialect == 'postgres':
            filenames_agg = "STRING_AGG(DISTINCT s.filename, ',') AS filenames"
        else:
            filenames_agg = "GROUP_CONCAT(DISTINCT s.filename) AS filenames"
        select_list = select_cols + agg_cols + [filenames_agg]
        group_by_sql = (" GROUP BY " + ", ".join(group_cols)) if group_cols else ""
        # ORDER BY support: restrict to group cols or measure aliases
        if ir_obj.order_by:
            for ob in ir_obj.order_by:
                if ob.field in (ir_obj.group_by or []) or ob.field in aliases:
                    order_terms.append(f"{ob.field} {'ASC' if ob.dir == 'asc' else 'DESC'}")
    else:
        # Default measure: SUM total_value and COUNT
        if dialect == 'postgres':
            filenames_agg = "STRING_AGG(DISTINCT s.filename, ',') AS filenames"
        else:
            filenames_agg = "GROUP_CONCAT(DISTINCT s.filename) AS filenames"
        select_list = [
            "im.metric_name",
            "SUM(im.metric_value) AS total_value",
            "im.metric_unit",
            "COUNT(*) AS count",
            filenames_agg,
        ]
        # default group
        group_cols = ["im.metric_name", "im.metric_unit"] + group_cols
        group_by_sql = (" GROUP BY " + ", ".join(group_cols))
        order_terms.append("total_value DESC")

    order_by_sql = (" ORDER BY " + ", ".join(order_terms)) if order_terms else ""
    sql = (
        "SELECT " + ", ".join(select_list) +
        " FROM impact_metrics im JOIN sources s ON im.source_id = s.id " +
        (" " + date_join if date_join else "") +
        f"WHERE {where_sql}{time_filter_sql}" +
        group_by_sql +
        order_by_sql
    )
    # LIMIT injection based on env caps if not single aggregate
    max_rows = int(os.getenv('MAX_ROWS', '10000'))
    max_groups = int(os.getenv('MAX_GROUPS', '2000'))
    effective_cap = max_rows
    is_grouped = bool(ir_obj.group_by) or (operation == 'trend')
    if is_grouped:
        effective_cap = min(max_rows, max_groups)
    # Clamp existing limit if present; else append
    lim = ir_obj.limit
    if lim is not None:
        try:
            lim = int(lim)
        except Exception:
            lim = effective_cap
        lim = max(1, min(lim, effective_cap))
    else:
        if is_grouped:
            lim = effective_cap
    if lim is not None:
        sql += " LIMIT ?"
        params.append(int(lim))
    return sql, tuple(params)


def _strip_sql_comments(text: str) -> str:
    """Remove SQL comments conservatively: -- to EOL and /* ... */ blocks.
    Repeatedly removes block comments to handle nesting without full parsing.
    """
    if not text:
        return text
    # Remove line comments
    no_line = re.sub(r"--.*?$", "", text, flags=re.MULTILINE)
    # Remove block comments (repeat until none remain)
    prev = None
    cur = no_line
    while prev != cur and '/*' in cur:
        prev = cur
        cur = re.sub(r"/\*.*?\*/", "", cur, flags=re.DOTALL)
    return cur


def _is_sql_safe(sql: str) -> bool:
    """Simple sandbox checks to reduce risk when executing model-influenced SQL strings.

    - Disallow semicolons and multiple statements
    - Disallow DDL/DML keywords
    - Only allow SELECT queries
    """
    s = (sql or '').strip()
    # Immediate reject on raw comment markers or semicolons
    if ';' in s or '--' in s or '/*' in s:
        return False
    # Strip comments before parsing any further
    su = _strip_sql_comments(s).upper()
    # Must start with SELECT or WITH
    if not (su.startswith('SELECT') or su.startswith('WITH')):
        return False
    # Forbid EXPLAIN and other dangerous tokens
    banned = [
        'EXPLAIN', 'PRAGMA', 'ATTACH', 'DETACH', 'VACUUM', 'ANALYZE', 'INSERT', 'UPDATE', 'DELETE',
        'CREATE', 'ALTER', 'DROP', 'REINDEX', 'REPLACE', 'TRUNCATE', 'UNION', 'INTERSECT', 'EXCEPT'
    ]
    if any(tok in su for tok in banned):
        return False
    # WITH RECURSIVE default forbidden
    allow_recursive = (os.getenv('SANDBOX_SQL_ALLOW_RECURSIVE', '').lower() in ('1','true','yes'))
    if re.search(r"\bWITH\s+RECURSIVE\b", su):
        if not allow_recursive:
            return False
    # Positive allow-list of clauses
    allowed = set(['SELECT', 'WITH', 'FROM', 'JOIN', 'ON', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'OFFSET'])
    extra = os.getenv('SANDBOX_SQL_ALLOWLIST', '').strip()
    if extra:
        for tok in extra.split(','):
            t = tok.strip().upper()
            if t:
                allowed.add(t)
    # Check occurrences of reserved words; allow identifiers and functions
    clauses = re.findall(r"\b[A-Z]+\b(?:\s+BY)?", su)
    for c in clauses:
        c_norm = c.strip()
        if c_norm in ('AS', 'AND', 'OR', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'DISTINCT', 'NULL', 'IS', 'NOT', 'IN', 'LIKE', 'BETWEEN'):
            continue
        if c_norm in allowed:
            continue
        # Skip numeric literals and function names by heuristic: if followed by '(' in original
        if re.search(r"\b" + re.escape(c_norm) + r"\s*\(", su):
            continue
        # Any other ALLCAP token not in allowed → reject
        return False
    return True


def run_planner(client: Optional[OpenAI], *, question: str, time_policy: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal orchestration: tools-only local flow -> IR -> SQL -> rows.

    Returns dict: { ir, sql, params, rows, citations } or structured error.
    """
    # Step 1: find metrics
    snippets = find_metrics(question)
    # Step 2: IR
    ir, meta = generate_ir(client, question=question, catalog_snippets=snippets, time_policy=time_policy)
    # Step 2b: assert time already done in generate_ir
    # Step 3: SQL
    try:
        sql, params = render_sql(ir)
    except Exception as e:
        return {'error': 'render_sql_failed', 'message': str(e)}
    # Phase 8: sandbox checks
    if not _is_sql_safe(sql):
        return {'error': 'sql_sandbox_rejected', 'message': 'SQL failed sandbox checks'}
    # Step 4: execute via internal function only (not exposed to model)
    try:
        rows = run_sql(sql, params)
    except Exception as e:
        return {'error': 'sql_execution_failed', 'message': str(e)}
    # Step 5: citations
    try:
        citations = fetch_citations(ir=ir, rows=rows)
    except Exception as e:
        return {'error': 'citation_failed', 'message': str(e)}
    # K-anonymity stub: suppress small groups
    try:
        min_group_size = int(os.getenv('MIN_GROUP_SIZE', '0'))
        redacted = False
        if min_group_size > 0 and rows and isinstance(rows, list):
            # If grouped results (presence of any group_by in IR), drop groups with count < min_group_size
            if ir.get('group_by'):
                kept: List[Dict[str, Any]] = []
                for r in rows:
                    c = r.get('count') or r.get('cnt') or r.get('records')
                    try:
                        cval = int(c)
                    except Exception:
                        cval = None
                    if cval is None or cval >= min_group_size:
                        kept.append(r)
                    else:
                        redacted = True
                rows = kept
    except Exception:
        pass
    return {
        'ir': ir,
        'sql': sql,
        'params': list(params),
        'rows': rows,
        'citations': citations,
        'meta': meta,
        'privacy': {'redacted': bool(locals().get('redacted', False))},
    }


def run_sql(sql: str, params: Tuple[Any, ...] = tuple()) -> List[Dict[str, Any]]:
    """Execute SQL with parameters via the database adapter. This is the ONLY execution path.
    """
    adapter = create_database_adapter()
    try:
        # Apply execution caps
        timeout_ms = int(os.getenv('QUERY_TIMEOUT_MS', '8000'))
        try:
            if adapter.db_type == 'sqlite':
                # busy_timeout is not a true query timeout but helps reduce lock waits
                adapter.execute_update(f"PRAGMA busy_timeout={timeout_ms}")
            else:
                adapter.execute_update("SET LOCAL statement_timeout = %s", (timeout_ms,))
        except Exception:
            pass
        rows = adapter.execute_query(sql, params)
        # Attach lightweight schema metadata for downstream caching/UI
        columns: List[Dict[str, Any]] = []
        row_count = len(rows) if isinstance(rows, list) else 0
        if rows:
            sample = rows[0]
            for k, v in sample.items():
                t = type(v).__name__
                columns.append({'name': k, 'type': t})
        # Store schema info in a side-channel attribute on list (not ideal but non-breaking)
        try:
            setattr(rows, '_schema_meta', {'columns': columns, 'row_count': row_count})
        except Exception:
            pass
        return rows
    finally:
        adapter.disconnect()


def fetch_citations(*, ir: Optional[Dict[str, Any]] = None, rows: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """Return source reference filenames from result rows or IR context.

    MVP: extract unique filenames from rows if present; else empty list.
    """
    refs: List[str] = []
    if rows:
        seen: set[str] = set()
        for r in rows:
            fn = r.get('filename') or r.get('filenames')
            if isinstance(fn, str):
                # filenames may be comma-separated
                for part in fn.split(','):
                    val = part.strip()
                    if val and val not in seen:
                        seen.add(val)
                        refs.append(val)
        return refs
    return refs


# ----------------------------------------
# Minimal Orchestrator using Function Calls
# ----------------------------------------

def get_openai_tools_spec() -> List[Dict[str, Any]]:
    """Return OpenAI tool definitions for function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "find_metrics",
                "description": "Find top metric catalog entries/snippets matching query terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_terms": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "k": {"type": "integer", "minimum": 1, "maximum": 7},
                },
                "required": ["query_terms"],
                "additionalProperties": False,
            },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_ir",
                "description": "Generate validated IR JSON strictly matching schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "catalog_snippets": {"type": "array", "items": {"type": "object"}},
                    "time_policy": {"type": "object"},
                },
                "required": ["question", "catalog_snippets", "time_policy"],
                "additionalProperties": False,
            },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "render_sql",
                "description": "Render SQL and params from IR using metric catalog. SQL is not exposed to the model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ir": {"type": "object"},
                },
                "required": ["ir"],
                "additionalProperties": False,
            },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_citations",
                "description": "Fetch source reference filenames from IR context or SQL rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ir": {"type": "object"},
                    "rows": {"type": "array", "items": {"type": "object"}},
                },
                "additionalProperties": False,
            },
            },
        },
    ]


## Note: run_planner is defined above with Optional[OpenAI] client and returns
## a payload including meta and privacy. The legacy duplicate definition below
## has been removed to avoid overriding the correct implementation.


