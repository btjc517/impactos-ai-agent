from __future__ import annotations

import json

from planning_tools import render_sql, run_planner


def test_compile_unknown_field_error():
    ir = {
        'operation': 'aggregate',
        'metric_id': None,
        'measures': [{'expr':'sum','alias':'total'}],
        'filters': [{'field':'nonexistent', 'op':'=', 'value':'x'}],
        'time': {'start':'2025-01-01','end':'2025-01-10','label':'w1','fiscal':None,'policy_id':None},
        'group_by': [],
        'order_by': [],
    }
    try:
        render_sql(ir)
        assert False, 'Expected compile_unknown_field'
    except ValueError as ve:
        e = getattr(ve, 'args', [{}])[0]
        assert isinstance(e, dict) and e.get('error') == 'compile_unknown_field'


def test_sandbox_rejects_non_select():
    # Non-select SQL arises only if compiler is compromised; directly hit sandbox via planner path
    res = run_planner(None, question='; DROP TABLE x; --', time_policy={'start':'2025-01-01','end':'2025-01-10','label':'w1'})
    assert 'error' in res


