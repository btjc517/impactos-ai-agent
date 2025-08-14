"""
Pydantic models for Metric Registry YAML with JSON Schema export.

These models define the canonical shape of a metric definition used by
ImpactOS. They are used for:
- YAML validation when loading registry files
- JSON Schema generation for external validation
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import re
import string
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class PrivacyThresholds(BaseModel):
    min_group_size: int = Field(ge=0, description="Minimum group size for k-anonymity style protections (>= 0)")


class ChartDefaults(BaseModel):
    # v2 payload structure
    type: Optional[str] = Field(default=None)
    x: Optional[str] = Field(default=None)
    y: Optional[str] = Field(default=None)
    series_key: Optional[str] = Field(default=None)
    unit: Optional[str] = Field(default=None)
    legend: Optional[bool] = Field(default=None)
    caption: Optional[str] = Field(default=None)


class FrameworkMapping(BaseModel):
    # Categories or codes within each framework. Keep lists flexible; can be empty.
    svm: List[str] = Field(default_factory=list, description="UK Social Value Model categories/codes")
    mac: List[str] = Field(default_factory=list, description="MAC (alias of UK SVM) categories/codes")
    tom: List[str] = Field(default_factory=list, description="TOMs categories/codes")
    sdg: List[str] = Field(default_factory=list, description="UN SDG goal IDs or target codes")


class MetricDefinition(BaseModel):
    # Identity and core definition
    id: str = Field(..., description="Stable metric identifier (slug)")
    title: str = Field(..., description="Human-friendly title")
    unit: Optional[str] = Field(None, description="Unit label, e.g., hours, GBP, kgCO2e")
    measure_type: Optional[str] = Field(
        default=None,
        description="Type of measure, e.g., count, amount, ratio, percentage, score",
    )
    default_agg: Optional[str] = Field(
        default=None, description="Default aggregation, e.g., sum, avg, count, max, min"
    )
    directionality: Optional[Literal["increase", "decrease", "neutral"]] = Field(
        default="neutral",
        description="Whether higher values are better (increase), worse (decrease), or neutral",
    )
    denominator_metric_id: Optional[str] = Field(
        default=None, description="Metric id used as denominator when measure_type is ratio/percentage"
    )

    # Dimensionality and temporal behavior
    allowed_dimensions: List[str] = Field(
        default_factory=list, description="Supported dimension names, e.g., site, department, region"
    )
    time_grain_supported: List[str] = Field(
        default_factory=list, description="Supported time grains, e.g., day, week, month, quarter, year"
    )
    default_time_window: Optional[str] = Field(
        default=None, description="Default time window, e.g., 12M, YTD, rolling_12m"
    )
    fiscal_required: bool = Field(
        default=False, description="Whether a fiscal calendar is required for correct evaluation"
    )

    # Calculation definition (one of)
    calc_sql: Optional[str] = Field(
        default=None,
        description="SQL expression or query used to compute the metric (when applicable)",
    )
    dsl: Optional[str] = Field(
        default=None,
        description="Domain-specific expression describing the metric logic (alternative to calc_sql)",
    )

    # Quality, privacy, and visualization hints
    validations: List[Any] = Field(
        default_factory=list, description="Validation rules (free-form objects or strings)"
    )
    privacy_thresholds: Optional[PrivacyThresholds] = Field(
        default=None, description="Privacy thresholds such as minimum group size"
    )
    chart_defaults: ChartDefaults = Field(
        default_factory=ChartDefaults, description="Default charting hints (v2 payload)"
    )

    # Discovery aids
    synonyms: List[str] = Field(default_factory=list, description="Synonyms or aliases for discovery")
    example_questions: List[str] = Field(
        default_factory=list, description="Example NL questions demonstrating usage"
    )

    # Optional dimension compatibility constraints for planners/compilers
    dimension_compat: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional matrix to block incompatible dimension combinations",
    )

    # Framework mappings and provenance
    framework_mapping: FrameworkMapping = Field(
        default_factory=FrameworkMapping, description="Mappings into common reporting frameworks"
    )
    mapping_version: Optional[str] = Field(
        default=None, description="Version string for mapping provenance"
    )
    evidence: List[str] = Field(
        default_factory=list, description="Evidence or citation links supporting the metric definition"
    )

    # Internal bookkeeping (not required in YAML, but allowed)
    created_at: Optional[str] = Field(
        default=None, description="Creation timestamp (ISO 8601); set by tooling if omitted"
    )

    @model_validator(mode="after")
    def _ensure_calc_defined(self) -> "MetricDefinition":
        # Enforce XOR: exactly one of calc_sql or dsl must be provided
        has_sql = bool(self.calc_sql and self.calc_sql.strip())
        has_dsl = bool(self.dsl and self.dsl.strip())
        if has_sql == has_dsl:
            # both True or both False -> invalid
            raise ValueError("Exactly one of calc_sql or dsl must be provided (XOR)")
        # If ratio, denominator_metric_id required
        if (self.measure_type or '').strip().lower() == 'ratio' and not (self.denominator_metric_id and self.denominator_metric_id.strip()):
            raise ValueError("denominator_metric_id is required when measure_type == 'ratio'")
        # default_agg must be from allowed set when provided
        if self.default_agg:
            if self.default_agg not in {"sum", "avg", "count", "ratio", "min", "max"}:
                raise ValueError("default_agg must be one of {sum,avg,count,ratio,min,max}")
        # time_grain_supported values restricted
        allowed_grains = {"day", "month", "quarter", "year", "fy"}
        bad = [g for g in (self.time_grain_supported or []) if g not in allowed_grains]
        if bad:
            raise ValueError(f"Invalid time_grain_supported values: {bad}")
        # Auto-add percentage sanity validation
        if (self.unit or '') == '%':
            rng = {"type": "range", "min": 0, "max": 100, "when_unit": "%"}
            if rng not in self.validations:
                self.validations.append(rng)
        return self

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if not v:
            raise ValueError("id cannot be empty")
        if not re.fullmatch(r"[a-z0-9_]+", v):
            raise ValueError("id must match [a-z0-9_]+ (slug with underscores)")
        return v

    @field_validator("title")
    @classmethod
    def _normalize_title(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("title cannot be empty")
        return v

    @field_validator("synonyms")
    @classmethod
    def _normalize_synonyms(cls, v: List[str]) -> List[str]:
        # Case-insensitive de-dup, strip punctuation and whitespace
        normed: List[str] = []
        seen = set()
        table = str.maketrans('', '', string.punctuation)
        for s in v or []:
            s0 = (s or '').strip().lower()
            s0 = s0.translate(table)
            if not s0:
                continue
            if s0 in seen:
                continue
            seen.add(s0)
            normed.append(s0)
        return normed

    @field_validator("allowed_dimensions")
    @classmethod
    def _validate_dimensions(cls, dims: List[str]) -> List[str]:
        # Try concept graph-driven validation first
        allowed: Optional[set] = None
        try:
            from semantic_resolver import ConceptGraph  # type: ignore
            g = ConceptGraph()
            allowed = {c['key'] for c in g.list_concepts('dimension')}
        except Exception:
            # Fallback to a conservative catalog
            allowed = {
                'dim_site', 'dim_person', 'dim_department', 'dim_supplier', 'dim_project',
                'dim_region', 'dim_cost_center', 'dim_customer', 'dim_product'
            }
        bad = [d for d in (dims or []) if d not in allowed]
        if bad:
            raise ValueError(f"Unknown dimensions: {bad}. Allowed: {sorted(allowed)}")
        return dims


def metric_json_schema() -> Dict[str, Any]:
    """Return JSON Schema for MetricDefinition suitable for saving to file."""
    schema = MetricDefinition.model_json_schema()
    # Add a helpful title and description
    schema["title"] = "ImpactOS Metric Definition"
    schema["description"] = "Schema for metric YAML files used by the ImpactOS metric registry"
    return schema


