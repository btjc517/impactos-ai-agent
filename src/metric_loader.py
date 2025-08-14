from __future__ import annotations

import os
import json
import hashlib
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from metric_models import MetricDefinition, metric_json_schema
from semantic_resolver import SemanticResolver


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


class MetricCatalog:
    """Loads and validates metric YAML files, offering a simple catalog API.

    - Validates each YAML against `MetricDefinition` (Pydantic)
    - Exposes get_metric(id) and search_by_synonym(term)
    - Persists fingerprints for cache invalidation in metrics/.fingerprints.json
    - Uses semantic + fuzzy resolver for synonym search fallback (avoids hardcoding)
    """

    def __init__(self, metrics_dir: str = "metrics", db_path: str = "db/impactos.db"):
        env_dir = os.getenv('METRICS_DIR')
        self.metrics_dir = env_dir if env_dir else metrics_dir
        self.db_path = db_path
        self._metrics: Dict[str, MetricDefinition] = {}
        self._synonym_to_ids: Dict[str, List[str]] = {}
        self._fingerprints_path = os.path.join(self.metrics_dir, ".fingerprints.json")
        self._resolver = None
        try:
            self._resolver = SemanticResolver(db_path)
        except Exception:
            self._resolver = None
        self._load_all()

    def _load_all(self) -> None:
        os.makedirs(self.metrics_dir, exist_ok=True)
        for name in os.listdir(self.metrics_dir):
            if not name.endswith('.yaml') and not name.endswith('.yml'):
                continue
            full = os.path.join(self.metrics_dir, name)
            try:
                with open(full, 'r') as f:
                    data = yaml.safe_load(f) or {}
                metric = MetricDefinition(**data)
                # Strict dimension_compat key check (fast failure per-metric)
                if metric.dimension_compat:
                    compat_keys = set(metric.dimension_compat.keys())
                    allowed = set(metric.allowed_dimensions or [])
                    extra = sorted(list(compat_keys - allowed))
                    if extra:
                        print(f"[metric-loader] dimension_compat keys not in allowed_dimensions for {metric.id}: {extra}. Skipping.")
                        continue
            except ValidationError as ve:
                # Soft-fail: log and skip
                print(f"[metric-loader] Validation error in {name}: {ve}")
                continue
            except Exception as e:
                print(f"[metric-loader] Failed to load {name}: {e}")
                continue

            self._metrics[metric.id] = metric
            # Index synonyms
            for syn in metric.synonyms:
                s = (syn or '').strip().lower()
                if not s:
                    continue
                self._synonym_to_ids.setdefault(s, []).append(metric.id)

        # Persist fingerprints after successful load
        self._persist_fingerprints()

    def _persist_fingerprints(self) -> None:
        fingerprints: Dict[str, str] = {}
        for name in os.listdir(self.metrics_dir):
            if name.endswith('.yaml') or name.endswith('.yml'):
                full = os.path.join(self.metrics_dir, name)
                fingerprints[name] = _file_sha256(full)
        tmp_path = self._fingerprints_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(fingerprints, f, indent=2)
        os.replace(tmp_path, self._fingerprints_path)

    def get_metric(self, metric_id: str) -> Optional[Dict[str, Any]]:
        m = self._metrics.get(metric_id)
        return m.model_dump() if m else None

    def search_by_synonym(self, term: str) -> List[Dict[str, Any]]:
        q = (term or '').strip().lower()
        if not q:
            return []
        # Direct synonym exact match first
        ids = self._synonym_to_ids.get(q, [])
        found: List[Dict[str, Any]] = [self._metrics[i].model_dump() for i in ids]
        if found:
            return found

        # Fuzzy + semantic over metric titles and synonyms using SemanticResolver if available
        # We model metrics in resolver as a concept type 'metric' with aliases (synonyms)
        if self._resolver:
            # Build text as title + synonyms
            ranked: List[tuple[str, float]] = []
            try:
                # Use resolver.resolve over a pseudo concept collection:
                # Since resolver pulls from DB, we simulate by local scoring using its fuzzy and embed functions
                # Fallback: simple rapidfuzz-like behavior via resolver._normalize_text and process.extract
                from rapidfuzz import process, fuzz
                choices = []
                id_map = []
                for mid, m in self._metrics.items():
                    text = (m.title or '') + " | " + " | ".join(m.synonyms or [])
                    choices.append(text)
                    id_map.append(mid)
                matches = process.extract(q, choices, scorer=fuzz.WRatio, limit=5)
                for _, score, idx in matches:
                    if score >= 70:
                        ranked.append((id_map[idx], float(score) / 100.0))
            except Exception:
                ranked = []
            # Return ranked unique ids
            seen = set()
            out: List[Dict[str, Any]] = []
            for mid, _ in sorted(ranked, key=lambda x: x[1], reverse=True):
                if mid in seen:
                    continue
                seen.add(mid)
                out.append(self._metrics[mid].model_dump())
            if out:
                return out

        # Last resort: substring search over titles and synonyms
        for m in self._metrics.values():
            hay = (m.title or '').lower() + " " + " ".join([s.lower() for s in (m.synonyms or [])])
            if q in hay:
                found.append(m.model_dump())
        return found

    def to_snippet(self, metric_id: str) -> Optional[Dict[str, Any]]:
        m = self._metrics.get(metric_id)
        if not m:
            return None
        # Trim calc previews for planner
        PREVIEW_LEN = 512
        calc_sql_preview = (m.calc_sql or '').strip()
        dsl_preview = (m.dsl or '').strip()
        if calc_sql_preview:
            calc_sql_preview = calc_sql_preview[:PREVIEW_LEN]
        if dsl_preview:
            dsl_preview = dsl_preview[:PREVIEW_LEN]
        # Up to two ultra-short example questions
        examples: List[str] = []
        for q in (m.example_questions or [])[:2]:
            q0 = (q or '').strip()
            if len(q0) > 120:
                q0 = q0[:117] + '...'
            if q0:
                examples.append(q0)
        body = {
            'id': m.id,
            'title': m.title,
            'unit': m.unit,
            'default_agg': m.default_agg,
            'allowed_dimensions': m.allowed_dimensions,
            'time_grain_supported': m.time_grain_supported,
            'measure_type': m.measure_type,
            'denominator_metric_id': m.denominator_metric_id,
            'example_questions': examples,
        }
        if calc_sql_preview:
            body['calc_sql'] = calc_sql_preview
        if dsl_preview:
            body['dsl'] = dsl_preview
        return body

    def validate(self, strict: bool = False) -> int:
        """Validate loaded metrics and return number of issues found.

        Issues include: missing evidence, unknown framework codes, dimension_compat key violations (already skipped).
        In strict mode, callers should fail CI when issues > 0.
        """
        issues = 0
        resolver = self._resolver
        framework_key_map = {
            'svm': 'uk_sv_model',
            'mac': 'mac',
            'tom': 'toms',
            'sdg': 'un_sdgs',
        }
        for m in self._metrics.values():
            # Missing evidence warning
            if not (m.evidence or []):
                print(f"[metric-validate] Missing evidence: {m.id}")
                issues += 1
            # Unknown framework codes
            if resolver:
                for short_key, fw_name in framework_key_map.items():
                    codes = getattr(m.framework_mapping, short_key)
                    for code in (codes or []):
                        res = resolver.resolve('framework_category', f"{fw_name} | {code}")
                        if res.get('outcome') != 'accepted':
                            print(f"[metric-validate] Unknown framework code for {m.id}: {short_key} -> {code}")
                            issues += 1
            else:
                # Cannot validate framework codes without resolver
                print("[metric-validate] Warning: semantic resolver unavailable; framework codes not validated")
        return issues


def write_metric_schema(schema_path: str) -> None:
    """Write JSON Schema for MetricDefinition to the given path."""
    os.makedirs(os.path.dirname(schema_path) or '.', exist_ok=True)
    schema = metric_json_schema()
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)


