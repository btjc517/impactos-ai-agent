from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from datetime import date
from pydantic import BaseModel, Field, field_validator, model_validator


class IRTime(BaseModel):
    mode: Optional[str] = Field(default=None, description="e.g., last_12_months, ytd, fiscal_ytd, custom")
    start: Optional[str] = None  # YYYY-MM-DD or null
    end: Optional[str] = None
    label: Optional[str] = None
    fiscal: Optional[bool] = None
    policy_id: Optional[str] = None


AllowedScalar = Union[str, float, int, bool, date]
AllowedValue = Union[AllowedScalar, List[AllowedScalar]]


class IRFilter(BaseModel):
    field: str
    op: Literal['=', '!=', '>', '>=', '<', '<=', 'IN', 'BETWEEN', 'LIKE'] = '='
    value: AllowedValue


class IROrderBy(BaseModel):
    field: str
    dir: Literal['asc', 'desc'] = 'asc'


class IRMeasure(BaseModel):
    expr: str
    alias: str


class IRStep(BaseModel):
    id: str
    depends_on: List[str] = Field(default_factory=list)
    ir: Dict[str, Any]


class IntermediateRepresentation(BaseModel):
    operation: Literal['aggregate', 'trend', 'compare', 'describe']
    metric_id: Optional[str] = None
    measures: List[IRMeasure] = Field(default_factory=list)
    filters: List[IRFilter] = Field(default_factory=list)
    time: IRTime
    group_by: List[str] = Field(default_factory=list)
    order_by: List[IROrderBy] = Field(default_factory=list)
    limit: Optional[int] = None
    need_chart: bool = False
    explain: bool = False
    multi_step: List[IRStep] = Field(default_factory=list)
    time_grain: Optional[Literal['day', 'week', 'month', 'quarter', 'year', 'fy']] = None
    insufficient_data: Optional[bool] = None
    errors: Optional[List[str]] = None

    @model_validator(mode='after')
    def _normalize(self) -> 'IntermediateRepresentation':
        # If operation=aggregate or trend, measures should not be empty; add default when metric_id present
        if self.operation in ('aggregate', 'trend') and not self.measures:
            if self.metric_id:
                self.measures = [IRMeasure(expr=self.metric_id, alias=self.metric_id)]
        # limit must be positive when present
        if self.limit is not None and self.limit <= 0:
            raise ValueError('limit must be > 0')
        return self


def ir_json_schema() -> Dict[str, Any]:
    return IntermediateRepresentation.model_json_schema()


