import json
import pytest

from pydantic import ValidationError

from ir_models import IntermediateRepresentation


def test_valid_ir_minimal():
    data = {
        "operation": "aggregate",
        "metric_id": "volunteering_hours",
        "measures": [{"expr": "volunteering_hours", "alias": "hours"}],
        "filters": [],
        "time": {"mode": "ytd", "start": "2025-01-01", "end": "2025-08-13", "label": "YTD", "fiscal": False},
        "group_by": ["dim_site"],
        "order_by": [{"field": "hours", "dir": "desc"}],
        "limit": 10,
        "need_chart": True,
        "explain": False,
        "multi_step": []
    }
    ir = IntermediateRepresentation(**data)
    assert ir.operation == 'aggregate'


def test_invalid_ir_then_repair_like():
    # invalid: limit negative, missing measures, bad dir
    bad = {
        "operation": "aggregate",
        "metric_id": "donations",
        "measures": [],
        "filters": [{"field": "dim_site", "op": "=", "value": "HQ"}],
        "time": {"mode": "last_12_months", "start": None, "end": None, "label": None, "fiscal": None},
        "group_by": [],
        "order_by": [{"field": "donations", "dir": "descending"}],
        "limit": -5,
        "need_chart": True,
        "explain": False,
        "multi_step": []
    }
    # Pydantic should raise; caller repair loop will fix alias dir and limit and insert a default measure
    with pytest.raises(ValidationError):
        IntermediateRepresentation(**bad)


