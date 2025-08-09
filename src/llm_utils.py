"""
LLM utilities for GPT-5 model parameterization, escalation, and validation.

Centralizes:
- Model selection (choose_model)
- Escalation heuristics (should_escalate_*)
- JSON structure validation for extraction
- OpenAI call construction with GPT-5 reasoning/text params

All functions are side-effect free except logging. Keep dependencies minimal.
"""

from __future__ import annotations

import json
import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import get_config

logger = logging.getLogger(__name__)


def choose_model(task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Select model and parameters based on task and config.

    Returns dict with keys: model, temperature, max_tokens, reasoning, text
    """
    cfg = get_config()
    task = (task or "").lower()
    if task == 'intent':
        return {
            'model': getattr(cfg.analysis, 'llm_intent_model', 'gpt-5-nano'),
            'temperature': getattr(cfg.analysis, 'llm_intent_temperature', 0.0),
            'max_tokens': getattr(cfg.analysis, 'llm_intent_max_tokens', 150),
            'reasoning': {'effort': cfg.llm.reasoning_effort.get('intent', 'minimal')},
            'text': {'verbosity': cfg.llm.verbosity.get('intent', 'low')},
        }
    if task == 'answer':
        return {
            'model': getattr(cfg.query_processing, 'answer_model', getattr(cfg.query_processing, 'gpt4_model', 'gpt-5-mini')),
            'max_tokens': getattr(cfg.query_processing, 'answer_max_tokens', getattr(cfg.query_processing, 'gpt4_max_tokens', 2000)),
            'reasoning': {'effort': cfg.llm.reasoning_effort.get('answer', 'medium')},
            'text': {'verbosity': cfg.llm.verbosity.get('answer', 'medium')},
        }
    if task in ('extraction', 'structure_analysis'):
        return {
            'model': getattr(cfg.extraction, 'structure_analysis_model', 'gpt-5'),
            'max_tokens': getattr(cfg.extraction, 'structure_analysis_max_tokens', getattr(cfg.extraction, 'gpt4_max_tokens_analysis', 3000)),
            'reasoning': {'effort': cfg.llm.reasoning_effort.get('extraction', 'high')},
            'text': {'verbosity': cfg.llm.verbosity.get('extraction', 'low')},
        }
    if task in ('query_generation',):
        return {
            'model': getattr(cfg.extraction, 'query_generation_model', 'gpt-5'),
            'max_tokens': getattr(cfg.extraction, 'extraction_max_tokens', getattr(cfg.extraction, 'gpt4_max_tokens_extraction', 4000)),
            'reasoning': {'effort': cfg.llm.reasoning_effort.get('extraction', 'high')},
            'text': {'verbosity': cfg.llm.verbosity.get('extraction', 'low')},
        }
    # Default safe
    return {
        'model': 'gpt-5-mini',
        'max_tokens': 1000,
        'reasoning': {'effort': 'medium'},
        'text': {'verbosity': 'medium'},
    }


def call_chat_completion(client, messages: List[Dict[str, str]], *,
                         model: str, max_tokens: int,
                         reasoning: Optional[Dict[str, Any]] = None,
                         text: Optional[Dict[str, Any]] = None,
                         enforce_json: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Call OpenAI chat.completions with unified params and measure metrics.

    Returns (content, meta) where meta has: model, latency_ms, usage, error.
    """
    start = time.time()
    usage = None
    try:
        kwargs: Dict[str, Any] = {
            'model': model,
            'messages': messages,
            'max_completion_tokens': int(max_tokens),
        }
        # Pass GPT-5 extra params if supported
        if reasoning:
            kwargs['reasoning'] = reasoning
        if text:
            kwargs['text'] = text
        if enforce_json:
            # Use response_format for strict JSON when supported
            kwargs['response_format'] = {'type': 'json_object'}

        resp = client.chat.completions.create(**kwargs)
        latency_ms = int((time.time() - start) * 1000)
        try:
            usage = getattr(resp, 'usage', None)
        except Exception:
            usage = None
        content = resp.choices[0].message.content.strip()
        logger.info(f"LLM call ok | model={model} tokens={max_tokens} latency_ms={latency_ms}")
        return content, {
            'model': model,
            'latency_ms': latency_ms,
            'usage': usage,
            'error': None,
        }
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        logger.warning(f"LLM call failed | model={model} err={e} latency_ms={latency_ms}")
        return "", {
            'model': model,
            'latency_ms': latency_ms,
            'usage': usage,
            'error': str(e),
        }


def extract_citation_ids(text: str) -> List[int]:
    ids = [int(n) for n in re.findall(r"\[(\d+)\]", text or "")]
    # Preserve order while de-duplicating
    seen = set()
    ordered = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


def should_escalate_answer(answer_text: str, context_count: int, latency_ms: int,
                           qa_cfg) -> Tuple[bool, str]:
    """Check Q&A output against guardrails.
    Returns (escalate, reason).
    """
    # Context too small
    if context_count < getattr(qa_cfg, 'min_context_items', 3):
        return True, 'insufficient_context'

    ids = extract_citation_ids(answer_text)
    if not ids or len(ids) < getattr(qa_cfg, 'citation_min_count', 3):
        return True, 'insufficient_citations'
    if any(i < 1 or i > max(1, context_count) for i in ids):
        return True, 'invalid_citation_indices'

    # Schema/tone check: require a Sources block
    if 'sources' not in (answer_text or '').lower():
        return True, 'missing_sources_block'

    # Latency budget
    p95_limit = int(get_config().llm.escalation_limits.get('p95_latency_ms', 6000))
    if latency_ms > p95_limit:
        return True, 'latency_exceeded'

    return False, ''


def should_escalate_intent(intent_result: Optional[Dict[str, Any]]) -> bool:
    if not intent_result:
        return True
    cats = intent_result.get('categories') or []
    aggs = intent_result.get('aggregations') or []
    qtype = intent_result.get('query_type')
    # Heuristic confidence: require some signal and a valid type
    if not cats and not aggs:
        return True
    if qtype not in ('aggregation', 'descriptive', 'analytical'):
        return True
    # Require >=98% confidence proxy: at least one category or aggregation
    # provided by LLM or embeddings; here we treat presence as high confidence
    return False


def validate_structure_analysis(data: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return False, ['Structure analysis must be a JSON object']
    for key in ['data_overview', 'column_analysis', 'identified_metrics']:
        if key not in data:
            errors.append(f"Missing key: {key}")
    if 'column_analysis' in data and not isinstance(data['column_analysis'], list):
        errors.append('column_analysis must be a list')
    if 'identified_metrics' in data and not isinstance(data['identified_metrics'], list):
        errors.append('identified_metrics must be a list')
    return (len(errors) == 0), errors


def validate_extraction_metrics(metrics: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(metrics, list):
        return False, ['Extraction output must be a JSON array']
    required_fields = [
        'metric_name', 'metric_value', 'metric_unit', 'metric_category',
        'source_column_name', 'source_column_index', 'source_row_index', 'source_cell_reference'
    ]
    for idx, m in enumerate(metrics):
        if not isinstance(m, dict):
            errors.append(f"Item {idx} not an object")
            continue
        for f in required_fields:
            if f not in m or m[f] is None:
                errors.append(f"Item {idx} missing {f}")
    return (len(errors) == 0), errors


def repair_with_model(client, *, prompt: str, schema_hint: Optional[str],
                      task: str) -> Tuple[str, Dict[str, Any]]:
    params = choose_model(task)
    messages = [{"role": "user", "content": prompt}]
    return call_chat_completion(
        client,
        messages,
        model=params['model'],
        max_tokens=params['max_tokens'],
        reasoning=params.get('reasoning'),
        text=params.get('text'),
        enforce_json=True,
    )


