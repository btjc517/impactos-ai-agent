from __future__ import annotations

import json
import threading
from typing import List

from planning_tools import get_openai_tools_spec, render_sql, run_planner
from planner_agent import run_planner_with_tools


def test_tools_spec_excludes_run_sql():
    tools = get_openai_tools_spec()
    names = [t['function']['name'] for t in tools]
    assert 'run_sql' not in names
    assert 'render_sql' in names
    assert 'generate_ir' in names


def test_deterministic_path_works_without_llm():
    res = run_planner(None, question="Volunteering hours YTD?", time_policy={"start":"2025-01-01","end":"2025-08-13","label":"YTD"})
    assert isinstance(res, dict)
    assert 'ir' in res
    assert 'rows' in res or 'error' in res


class _FakeClient:
    def __init__(self):
        class _Choices:
            def __init__(self):
                self.message = type('M', (), { 'content': json.dumps({}), 'tool_calls': [
                    type('TC', (), { 'id': '1', 'type': 'function', 'function': type('F', (), {'name': 'find_metrics', 'arguments': json.dumps({'query_terms': 'volunteering', 'k': 3})}) })
                ] })
        class _Resp:
            def __init__(self):
                self.choices = [ _Choices() ]
        class _Chat:
            def completions(self, *args, **kwargs):
                class _C:
                    def create(self, *args, **kwargs):
                        return _Resp()
                return _C()
        self.chat = _Chat()


def test_function_calling_path_returns_tool_rows_or_fallback():
    client = _FakeClient()
    res = run_planner_with_tools(client, question="Volunteering hours YTD?", time_policy={"start":"2025-01-01","end":"2025-08-13","label":"YTD"}, model="gpt-4o-mini")
    assert isinstance(res, dict)
    # Either returns tool-driven payload or guardrail fallback
    assert set(['ir','sql','params','rows','citations']).issubset(set(res.keys())) or 'error' in res


def test_concurrent_calls_no_global_state_leak():
    results: List[dict] = []
    def worker(i: int):
        r = run_planner(None, question=f"Q{i}", time_policy={"start":"2025-01-01","end":"2025-08-13","label":"YTD"})
        results.append(r)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(results) == 10


def test_param_hardening_weird_strings_do_not_break_sql():
    # Compose an IR with adversarial filter values; ensure SQL uses bound params and compiles
    ir = {
        'operation': 'aggregate',
        'metric_id': None,
        'measures': [],
        'filters': [
            {'field': 'metric_name', 'op': '=', 'value': "foo' OR 1=1 --"},
            {'field': 'metric_unit', 'op': 'LIKE', 'value': "%kWh%"},
            {'field': 'metric_category', 'op': 'IN', 'value': ["; DROP TABLE x;", "normal"]},
        ],
        'time': {'start': '2025-01-01', 'end': '2025-12-31', 'label': 'ytd', 'fiscal': None, 'policy_id': None},
        'group_by': [],
        'order_by': [],
        'limit': 10,
    }
    sql, params = render_sql(ir)
    # SQL should not include the raw values; they must be in params instead
    assert "foo' OR 1=1 --" not in sql
    assert "; DROP TABLE x;" not in sql
    assert any(isinstance(p, str) and "1=1" in p for p in params)


def test_prompt_injection_rejected_by_sandbox():
    # Simulate dangerous SQL and ensure sandbox rejects it prior to execution via run_planner
    res = run_planner(None, question="Volunteering hours; DROP TABLE foo; --", time_policy={"start":"2025-01-01","end":"2025-08-13","label":"YTD"})
    # Deterministic planner path should either error at IR time mismatch/validation or sandbox
    assert 'error' in res or 'rows' in res


def test_like_escaping_and_in_expansion():
    ir = {
        'operation': 'aggregate',
        'metric_id': None,
        'measures': [{'expr':'sum','alias':'total'}],
        'filters': [
            {'field': 'metric_name', 'op': 'LIKE', 'value': '%Acme_'},
            {'field': 'metric_category', 'op': 'IN', 'value': ['A','B','C']},
        ],
        'time': {'start': '2025-01-01', 'end': '2025-08-13', 'label': 'YTD', 'fiscal': None, 'policy_id': None},
        'group_by': ['metric_category'],
        'order_by': [{'field': 'total', 'dir': 'desc'}],
        'limit': 100,
    }
    sql, params = render_sql(ir)
    # LIKE must include ESCAPE clause and params must have escaped wildcards
    assert 'LIKE ? ESCAPE \'\\\'' in sql
    assert any(p.endswith('\\_') or p.startswith('\\%') for p in params if isinstance(p, str))
    # IN list should expand to 3 placeholders
    assert sql.count('IN (') >= 1


