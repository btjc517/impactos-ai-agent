import os
import json
import tempfile

from metric_csv_converter import MetricCSVConverter
from metric_loader import MetricCatalog


def test_csv_yaml_loader_roundtrip(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    # Minimal CSV with XOR calc (dsl only) and valid fields
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        """id,title,unit,measure_type,default_agg,directionality,allowed_dimensions,time_grain_supported,fiscal_required,dsl,framework_sdg\n"
        "engagement_rate,Engagement Rate,%,ratio,avg,increase,dim_site|dim_person,month|year,true,rate(engaged, total),3\n".replace('|', ',')
    )

    conv = MetricCSVConverter(metrics_dir=str(metrics_dir))
    result = conv.convert(str(csv_path))
    assert result["created"] == 1

    # Load catalog and check access
    catalog = MetricCatalog(metrics_dir=str(metrics_dir))
    m = catalog.get_metric("engagement_rate")
    assert m is not None
    assert m["id"] == "engagement_rate"
    assert m["unit"] == "%"
    assert m["chart_defaults"] is not None
    # JSON schema should exist
    assert (metrics_dir / "metric.schema.json").exists()


