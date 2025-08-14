import json

import pytest

from pydantic import ValidationError

from ir_models import IntermediateRepresentation


def test_ir_model_enum_validation():
    with pytest.raises(ValidationError):
        IntermediateRepresentation(**{
            "operation": "aggregate",
            "time": {"mode": "ytd"},
            "filters": [{"field": "x", "op": "NEQ", "value": 1}],
            "group_by": [],
            "order_by": [{"field": "x", "dir": "descending"}],
        })


