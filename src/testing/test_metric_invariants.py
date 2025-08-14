import pytest
from pydantic import ValidationError

from metric_models import MetricDefinition


def base_metric(**overrides):
    base = dict(
        id="test_metric",
        title="Test Metric",
        unit="hours",
        measure_type="count",
        default_agg="sum",
        directionality="neutral",
        allowed_dimensions=["dim_site"],
        time_grain_supported=["month"],
        fiscal_required=False,
        dsl="value",
    )
    base.update(overrides)
    return base


def test_xor_calc_enforced():
    # both missing
    with pytest.raises(ValidationError):
        MetricDefinition(**base_metric(dsl=None))
    # both present
    with pytest.raises(ValidationError):
        MetricDefinition(**base_metric(calc_sql="x", dsl="y"))


def test_ratio_requires_denominator():
    with pytest.raises(ValidationError):
        MetricDefinition(**base_metric(measure_type="ratio", dsl="x", denominator_metric_id=None))
    # valid when provided
    m = MetricDefinition(**base_metric(measure_type="ratio", dsl="x", denominator_metric_id="total_hours"))
    assert m.denominator_metric_id == "total_hours"


def test_default_agg_whitelist():
    with pytest.raises(ValidationError):
        MetricDefinition(**base_metric(default_agg="median"))
    MetricDefinition(**base_metric(default_agg="avg"))


def test_time_grain_allowed():
    with pytest.raises(ValidationError):
        MetricDefinition(**base_metric(time_grain_supported=["week"]))
    MetricDefinition(**base_metric(time_grain_supported=["quarter", "fy"]))


def test_privacy_min_group_size_non_negative():
    with pytest.raises(ValidationError):
        MetricDefinition(**base_metric(privacy_thresholds={"min_group_size": -1}))
    MetricDefinition(**base_metric(privacy_thresholds={"min_group_size": 0}))


def test_id_slug_enforced():
    with pytest.raises(ValidationError):
        MetricDefinition(**base_metric(id="Bad ID!"))
    m = MetricDefinition(**base_metric(id="good_id_1"))
    assert m.id == "good_id_1"


def test_unit_percentage_adds_validation():
    m = MetricDefinition(**base_metric(unit="%"))
    assert any(v.get("type") == "range" and v.get("min") == 0 and v.get("max") == 100 for v in m.validations)


