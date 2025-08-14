from __future__ import annotations

from planning_tools import render_sql, run_planner


def test_dim_date_grouping_month_sqlite():
    ir = {
        'operation': 'aggregate',
        'metric_id': None,
        'measures': [{'expr':'sum','alias':'total'}],
        'filters': [],
        'time': {'start':'2025-01-01','end':'2025-01-31','label':'Jan 2025','fiscal':None,'policy_id':None},
        'group_by': ['month'],
        'order_by': [{'field':'total','dir':'desc'}],
        'limit': 10,
    }
    sql, params = render_sql(ir, dialect='sqlite')
    assert 'JOIN dim_date dd' in sql
    assert 'dd.month AS month' in sql


def test_privacy_k_anonymity_stub():
    import os
    os.environ['MIN_GROUP_SIZE'] = '3'
    res = run_planner(None, question='Q', time_policy={'start':'2025-01-01','end':'2025-01-31','label':'Jan'})
    assert isinstance(res, dict)
    assert 'privacy' in res and 'redacted' in res['privacy']

