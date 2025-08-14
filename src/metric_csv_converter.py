from __future__ import annotations

import os
import csv
import json
import hashlib
from typing import Any, Dict, List, Optional

import yaml

from metric_models import MetricDefinition
from metric_loader import write_metric_schema
from semantic_resolver import SemanticResolver


def _slugify(value: str) -> str:
    v = (value or '').strip().lower()
    out = []
    for ch in v:
        if ch.isalnum():
            out.append(ch)
        elif ch in (' ', '-', '/', '|', ':'):
            out.append('-')
    slug = ''.join(out).strip('-')
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug or 'metric'


def _safe_split(csv_field: str) -> List[str]:
    if csv_field is None:
        return []
    parts = [p.strip() for p in str(csv_field).split(',')]
    return [p for p in parts if p]


def _fingerprint_record(row: Dict[str, Any]) -> str:
    # Stable fingerprint for a CSV row used to avoid unnecessary rewrites
    material = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode('utf-8')).hexdigest()


class MetricCSVConverter:
    """Convert a metric catalog CSV into per-metric YAML files.

    Expected CSV columns (flexible, semantic resolver used where possible):
      id,title,unit,measure_type,default_agg,directionality,denominator_metric_id,
      allowed_dimensions,time_grain_supported,default_time_window,fiscal_required,
      calc_sql,dsl,validations,privacy_min_group_size,chart_defaults_json,
      synonyms,example_questions,framework_svm,framework_mac,framework_tom,framework_sdg,
      mapping_version,evidence
    """

    def __init__(self, metrics_dir: str = "metrics", db_path: str = "db/impactos.db"):
        self.metrics_dir = metrics_dir
        self.db_path = db_path
        os.makedirs(self.metrics_dir, exist_ok=True)
        # Ensure schema file exists for consumers
        write_metric_schema(os.path.join(self.metrics_dir, "metric.schema.json"))
        try:
            self.resolver = SemanticResolver(db_path)
        except Exception:
            self.resolver = None

    def convert(self, csv_path: str) -> Dict[str, Any]:
        created, updated, skipped = 0, 0, 0
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Build metric definition using row values; apply reasonable coercions
                # Enforce slug hygiene: [a-z0-9_]+ (translate hyphens to underscores)
                base_id = (row.get('id') or _slugify(row.get('title') or 'metric'))
                base_id = base_id.replace('-', '_')
                metric_id = ''.join(ch for ch in base_id if (ch.isalnum() or ch == '_'))
                title = row.get('title') or metric_id.replace('-', ' ').title()
                unit = row.get('unit') or None
                measure_type = row.get('measure_type') or None
                default_agg = row.get('default_agg') or None
                directionality = (row.get('directionality') or 'neutral').lower()
                denominator_metric_id = row.get('denominator_metric_id') or None
                allowed_dimensions = _safe_split(row.get('allowed_dimensions'))
                time_grain_supported = _safe_split(row.get('time_grain_supported'))
                default_time_window = row.get('default_time_window') or None
                fiscal_required_str = (row.get('fiscal_required') or '').strip().lower()
                fiscal_required = fiscal_required_str in ('1', 'true', 'yes', 'y')

                calc_sql = row.get('calc_sql') or None
                dsl = row.get('dsl') or None

                # validations: allow JSON array or comma-separated strings
                validations_raw = row.get('validations')
                validations: List[Any] = []
                if validations_raw:
                    try:
                        obj = json.loads(validations_raw)
                        if isinstance(obj, list):
                            validations = obj
                        else:
                            validations = [obj]
                    except Exception:
                        validations = _safe_split(validations_raw)

                # privacy thresholds
                privacy_min_group_size = row.get('privacy_min_group_size')
                privacy = None
                try:
                    if privacy_min_group_size is not None and str(privacy_min_group_size).strip() != "":
                        privacy = {"min_group_size": int(privacy_min_group_size)}
                except Exception:
                    privacy = None

                # chart defaults: accept JSON dict
                chart_defaults_raw = row.get('chart_defaults_json')
                chart_defaults: Dict[str, Any] = {}
                if chart_defaults_raw:
                    try:
                        obj = json.loads(chart_defaults_raw)
                        if isinstance(obj, dict):
                            chart_defaults = obj
                    except Exception:
                        chart_defaults = {}

                synonyms = _safe_split(row.get('synonyms'))
                example_questions = _safe_split(row.get('example_questions'))

                # framework mappings; avoid hardcoding by using resolver to clean codes when possible
                framework_svm = _safe_split(row.get('framework_svm'))
                framework_mac = _safe_split(row.get('framework_mac'))
                framework_tom = _safe_split(row.get('framework_tom'))
                framework_sdg = _safe_split(row.get('framework_sdg'))

                if self.resolver:
                    # Try to resolve each token to canonical keys using the concept graph
                    def _resolve_many(framework_key: str, tokens: List[str]) -> List[str]:
                        out: List[str] = []
                        for t in tokens:
                            res = self.resolver.resolve('framework_category', f"{framework_key} | {t}")
                            if res.get('outcome') == 'accepted':
                                out.append(res['key'])
                            else:
                                out.append(t)
                        return list(dict.fromkeys(out))  # de-dupe, preserve order

                    framework_svm = _resolve_many('uk_sv_model', framework_svm)
                    framework_mac = _resolve_many('mac', framework_mac)
                    framework_tom = _resolve_many('toms', framework_tom)
                    framework_sdg = _resolve_many('un_sdgs', framework_sdg)

                mapping_version = row.get('mapping_version') or None
                evidence = _safe_split(row.get('evidence'))

                metric_obj = MetricDefinition(
                    id=metric_id,
                    title=title,
                    unit=unit,
                    measure_type=measure_type,
                    default_agg=default_agg,
                    directionality=directionality,  # validated by Pydantic Literal
                    denominator_metric_id=denominator_metric_id,
                    allowed_dimensions=allowed_dimensions,
                    time_grain_supported=time_grain_supported,
                    default_time_window=default_time_window,
                    fiscal_required=fiscal_required,
                    calc_sql=calc_sql,
                    dsl=dsl,
                    validations=validations,
                    privacy_thresholds=privacy,
                    chart_defaults=chart_defaults,
                    synonyms=synonyms,
                    example_questions=example_questions,
                    framework_mapping={
                        'svm': framework_svm,
                        'mac': framework_mac,
                        'tom': framework_tom,
                        'sdg': framework_sdg,
                    },
                    mapping_version=mapping_version,
                    evidence=evidence,
                )

                # Write YAML file named metrics/{id}.yaml
                yaml_path = os.path.join(self.metrics_dir, f"{metric_obj.id}.yaml")

                # If existing file has same fingerprint (by serialized dict), skip write
                serialized = metric_obj.model_dump(mode='python', exclude_none=True)
                new_fp = hashlib.sha256(json.dumps(serialized, sort_keys=True).encode('utf-8')).hexdigest()
                old_fp = None
                if os.path.exists(yaml_path):
                    try:
                        with open(yaml_path, 'r') as yf:
                            current = yaml.safe_load(yf) or {}
                        old_fp = hashlib.sha256(json.dumps(current, sort_keys=True).encode('utf-8')).hexdigest()
                    except Exception:
                        old_fp = None

                if new_fp == old_fp:
                    skipped += 1
                    continue

                with open(yaml_path, 'w') as yf:
                    yaml.safe_dump(serialized, yf, sort_keys=False)
                if old_fp is None:
                    created += 1
                else:
                    updated += 1

        # Save or update schema alongside
        write_metric_schema(os.path.join(self.metrics_dir, "metric.schema.json"))
        return {"created": created, "updated": updated, "skipped": skipped}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert metric catalog CSV to YAML registry")
    parser.add_argument("csv_path", help="Path to metric catalog CSV")
    parser.add_argument("--metrics-dir", default="metrics", help="Output directory for YAML files")
    parser.add_argument("--db-path", default="db/impactos.db", help="Path to SQLite DB for semantic resolution")
    args = parser.parse_args()

    converter = MetricCSVConverter(metrics_dir=args.metrics_dir, db_path=args.db_path)
    result = converter.convert(args.csv_path)
    print(json.dumps(result, indent=2))


