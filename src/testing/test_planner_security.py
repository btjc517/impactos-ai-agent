from __future__ import annotations

import json

from planner_agent import get_openai_tools_spec


def test_tools_spec_does_not_expose_sql():
    names = [t['function']['name'] for t in get_openai_tools_spec()]
    assert 'run_sql' not in names
    assert 'render_sql' in names


