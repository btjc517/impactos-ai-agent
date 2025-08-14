from __future__ import annotations

import json
from typing import Any, Dict, Tuple, List

from pydantic import ValidationError

from ir_models import IntermediateRepresentation, ir_json_schema
from metric_loader import MetricCatalog
from llm_utils import choose_model, call_chat_completion


def build_ir_prompt(question: str, catalog_snippets: Any, time_policy: Dict[str, Any], few_shots: Optional[List[Dict[str, Any]]] = None) -> str:
    schema = ir_json_schema()
    instruction = (
        "You are a planner that produces a strict Intermediate Representation (IR) as JSON.\n"
        "Use ONLY the provided time window; do not invent or modify dates.\n"
        "Return a single JSON object; no commentary.\n"
    )
    content = {
        'instruction': instruction,
        'question': question,
        'catalog_snippets': catalog_snippets,
        'time': {
            'start': time_policy.get('start'),
            'end': time_policy.get('end'),
            'label': time_policy.get('label'),
            'fiscal': time_policy.get('fiscal_used'),
            'policy_id': time_policy.get('policy_id'),
        },
        'schema': schema,
        'few_shots': few_shots or [],
    }
    return json.dumps(content)


def _gather_catalog_snippets(metrics_dir: str, terms: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return up to k planner snippets by synonym search on the catalog."""
    try:
        cat = MetricCatalog(metrics_dir=metrics_dir)
        # Prefer direct id match first
        m = cat.get_metric(terms)
        if m:
            return [cat.to_snippet(m['id'])]
        hits = cat.search_by_synonym(terms)
        out = []
        for h in hits[:k]:
            sn = cat.to_snippet(h['id'])
            if sn:
                out.append(sn)
        return out[:k]
    except Exception:
        return []


def generate_ir_with_validation(client, question: str, catalog_terms: str | List[str], time_policy: Dict[str, Any], *, metrics_dir: str = 'metrics', few_shots: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate IR with deterministic decoding, validate against schema, repair once if needed.

    Returns (ir_dict, meta) or raises ValueError with details.
    """
    params = choose_model('intent')  # small deterministic model
    params['temperature'] = 0.0
    params['max_tokens'] = min(2000, params.get('max_tokens', 2000))
    # Gather snippets from catalog
    if isinstance(catalog_terms, list):
        snippets: List[Dict[str, Any]] = []
        for t in catalog_terms:
            snippets.extend(_gather_catalog_snippets(metrics_dir, str(t)))
        catalog_snippets = snippets[:5]
    else:
        catalog_snippets = _gather_catalog_snippets(metrics_dir, str(catalog_terms))
    prompt = build_ir_prompt(question, catalog_snippets, time_policy, few_shots=few_shots)
    content, meta = call_chat_completion(
        client,
        messages=[{"role": "user", "content": prompt}],
        model=params['model'],
        max_tokens=params['max_tokens'],
        reasoning={'effort': 'low'},
        text={'verbosity': 'low'},
        enforce_json=True,
    )
    try:
        ir_data = json.loads(content) if content else {}
    except Exception:
        ir_data = {}
    # Validate
    validation_passed = False
    repair_invoked = False
    try:
        ir = IntermediateRepresentation(**ir_data)
        validation_passed = True
        meta.update({'prompt_version': 'ir.v1', 'repaired': False, 'validation_passed': True})
        return ir.model_dump(), meta
    except ValidationError as ve:
        # One-shot repair loop
        repair_invoked = True
        repair_instruction = (
            "The previous IR was invalid. Fix ONLY the errors listed and return a valid JSON IR matching the schema.\n"
            f"Errors: {str(ve)}\n"
        )
        repair_payload = {
            'repair_instruction': repair_instruction,
            'previous_ir': ir_data,
            'schema': ir_json_schema(),
        }
        repair_prompt = json.dumps(repair_payload)
        content2, meta2 = call_chat_completion(
            client,
            messages=[{"role": "user", "content": repair_prompt}],
            model=params['model'],
            max_tokens=params['max_tokens'],
            reasoning={'effort': 'low'},
            text={'verbosity': 'low'},
            enforce_json=True,
        )
        try:
            repaired = json.loads(content2) if content2 else {}
        except Exception:
            repaired = {}
        try:
            ir2 = IntermediateRepresentation(**repaired)
            validation_passed = True
            meta.update({'prompt_version': 'ir.v1', 'repaired': True, 'validation_passed': True})
            return ir2.model_dump(), meta
        except ValidationError as ve2:
            meta.update({'prompt_version': 'ir.v1', 'repaired': True, 'validation_passed': False})
            raise ValueError({
                'error': 'invalid_ir',
                'message': 'IR could not be validated after one repair attempt',
                'details': str(ve2),
            })


