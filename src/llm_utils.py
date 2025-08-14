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


def get_llm_client():
    """Get configured OpenAI client."""
    try:
        import openai
        return openai.OpenAI()
    except ImportError:
        logger.error("OpenAI package not available")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise


# Safe ceilings for output/completion tokens per model.
# These values include headroom below hard limits to avoid 400s.
MODEL_OUTPUT_TOKEN_CEILINGS: Dict[str, int] = {
    # GPT-4o family
    'gpt-4o-mini': 12000,           # hard ~16384; clamp with headroom
    'gpt-4o-mini-2024-07-18': 12000,
    'gpt-4o': 8000,
    'gpt-4o-2024-08-06': 8000,
    # GPT-5 family (conservative defaults; can be tuned via config in future)
    'gpt-5': 6000,
    'gpt-5-mini': 6000,
}


def _get_model_ceiling(model: str, requested: int) -> int:
    """Return a safe ceiling for the given model, defaulting to requested.

    Uses prefix matching so aliases like 'gpt-4o-mini-...'
    resolve to the appropriate ceiling.
    """
    if not model:
        return requested
    # Exact match first
    if model in MODEL_OUTPUT_TOKEN_CEILINGS:
        return MODEL_OUTPUT_TOKEN_CEILINGS[model]
    # Prefix match by known keys
    for key, ceiling in MODEL_OUTPUT_TOKEN_CEILINGS.items():
        if model.startswith(key):
            return ceiling
    return requested


def _clamp_requested_tokens(model: str, requested: int) -> int:
    """Clamp requested tokens to a safe per-model ceiling.

    Always returns at least 1 token.
    """
    try:
        requested_int = int(requested)
    except Exception:
        requested_int = 1
    ceiling = _get_model_ceiling(model, requested_int)
    clamped = max(1, min(requested_int, int(ceiling)))
    return clamped


def choose_model(task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Select model and parameters based on task and config.

    Returns dict with keys: model, temperature, max_tokens, reasoning, text
    """
    cfg = get_config()
    task = (task or "").lower()
    if task == 'intent':
        return {
            'model': getattr(cfg.analysis, 'llm_intent_model', 'gpt-4o-mini'),
            'temperature': getattr(cfg.analysis, 'llm_intent_temperature', 0.0),
            'max_tokens': getattr(cfg.analysis, 'llm_intent_max_tokens', 150),
            'reasoning': {'effort': cfg.llm.reasoning_effort.get('intent', 'minimal')},
            'text': {'verbosity': cfg.llm.verbosity.get('intent', 'low')},
        }
    if task == 'answer':
        return {
            'model': getattr(cfg.query_processing, 'answer_model', getattr(cfg.query_processing, 'gpt4_model', 'gpt-4o-mini')),
            'max_tokens': getattr(cfg.query_processing, 'answer_max_tokens', getattr(cfg.query_processing, 'gpt4_max_tokens', 2000)),
            'temperature': float(getattr(cfg.query_processing, 'answer_temperature', 0.0) or 0.0),
            'reasoning': {'effort': cfg.llm.reasoning_effort.get('answer', 'medium')},
            'text': {'verbosity': cfg.llm.verbosity.get('answer', 'medium')},
        }
    if task in ('extraction', 'structure_analysis'):
        return {
            'model': getattr(cfg.extraction, 'structure_analysis_model', 'gpt-4o-mini'),
            'max_tokens': getattr(cfg.extraction, 'structure_analysis_max_tokens', getattr(cfg.extraction, 'gpt4_max_tokens_analysis', 3000)),
            'reasoning': {'effort': cfg.llm.reasoning_effort.get('extraction', 'high')},
            'text': {'verbosity': cfg.llm.verbosity.get('extraction', 'low')},
        }
    if task in ('query_generation',):
        return {
            'model': getattr(cfg.extraction, 'query_generation_model', 'gpt-4o-mini'),
            'max_tokens': getattr(cfg.extraction, 'extraction_max_tokens', getattr(cfg.extraction, 'gpt4_max_tokens_extraction', 4000)),
            'reasoning': {'effort': cfg.llm.reasoning_effort.get('extraction', 'high')},
            'text': {'verbosity': cfg.llm.verbosity.get('extraction', 'low')},
        }
    # Default safe
    return {
        'model': 'gpt-4o-mini',
        'max_tokens': 1000,
        'reasoning': {'effort': 'medium'},
        'text': {'verbosity': 'medium'},
    }


def call_chat_completion(client, messages: List[Dict[str, str]], *,
                         model: str, max_tokens: int,
                         reasoning: Optional[Dict[str, Any]] = None,
                         text: Optional[Dict[str, Any]] = None,
                         enforce_json: bool = False,
                         temperature: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
    """Call OpenAI chat.completions with unified params and measure metrics.

    Returns (content, meta) where meta has: model, latency_ms, usage, error.
    """
    start = time.time()
    # Log request parameters (model and key flags) before making the call
    try:
        param_summary = {
            'model': model,
            'max_tokens': int(max_tokens),
            'enforce_json': bool(enforce_json),
            'reasoning_effort': (reasoning or {}).get('effort'),
            'text_verbosity': (text or {}).get('verbosity'),
            'temperature': float(temperature) if temperature is not None else None,
            'messages_count': len(messages) if isinstance(messages, list) else 1,
        }
        logger.info(f"LLM call start | params={param_summary}")
    except Exception:
        # Never let logging break the call path
        pass
    usage = None
    # Prefer Responses API for GPT-5 features (verbosity/reasoning/CFG) per cookbook
    # https://cookbook.openai.com/examples/gpt-5/gpt-5_new_params_and_tools
    if model.startswith('gpt-5'):
        try:
            # Prefer string input for simple prompts; list-of-messages when multi-turn
            if isinstance(messages, list) and len(messages) == 1 and isinstance(messages[0], dict) and messages[0].get('role'):
                input_payload: Any = messages[0].get('content', '')
            else:
                input_payload = messages

            # Compose text param and enforce JSON if requested
            text_param: Optional[Dict[str, Any]] = dict(text) if isinstance(text, dict) else {}
            if enforce_json:
                # Per cookbook, use text.format to hint JSON object output
                fmt = text_param.get('format', {}) if text_param else {}
                fmt['type'] = 'json_object'
                text_param = text_param or {}
                text_param['format'] = fmt

            # Clamp tokens for Responses API
            clamped_tokens = _clamp_requested_tokens(model, max_tokens)
            kwargs_resp: Dict[str, Any] = {
                'model': model,
                'input': input_payload,
                'max_output_tokens': int(clamped_tokens),
            }
            if reasoning:
                kwargs_resp['reasoning'] = reasoning
            if text_param:
                kwargs_resp['text'] = text_param
            try:
                resp = client.responses.create(**kwargs_resp)
            except Exception as e:
                # Adaptive single retry for token-limit errors
                err_msg_local = str(e).lower()
                if 'max_tokens is too large' in err_msg_local or 'max_output_tokens' in err_msg_local:
                    reduced = max(1, int(min(_get_model_ceiling(model, clamped_tokens), clamped_tokens * 0.75)))
                    kwargs_resp['max_output_tokens'] = reduced
                    logger.info(f"Adaptive retry (responses) due to token limit | model={model} tokens={clamped_tokens}->{reduced}")
                    resp = client.responses.create(**kwargs_resp)
                else:
                    raise
            latency_ms = int((time.time() - start) * 1000)
            # Extract concatenated text from responses output
            output_text = ""
            try:
                items = getattr(resp, 'output', []) or []
                for item in items:
                    contents = getattr(item, 'content', None) or []
                    for content in contents:
                        # content.text may be an object with .value or a dict
                        text_obj = getattr(content, 'text', None)
                        if text_obj is None and isinstance(content, dict):
                            text_obj = content.get('text')
                        if text_obj is not None:
                            val = getattr(text_obj, 'value', None)
                            if val is None and isinstance(text_obj, dict):
                                val = text_obj.get('value')
                            if val is None and isinstance(text_obj, str):
                                val = text_obj
                            if val:
                                output_text += val
            except Exception:
                pass
            if not output_text:
                # Fallback to standard fields if present
                output_text = getattr(resp, 'output_text', '') or ''
                if not output_text:
                    # Extreme fallback: try choices API-shape if any
                    try:
                        output_text = resp.choices[0].message.content.strip()
                    except Exception:
                        output_text = ''
            usage = getattr(resp, 'usage', None)
            if not output_text:
                # Fallback to Chat if Responses returned no text
                try:
                    chat_kwargs = {
                        'model': model,
                        'messages': [{"role": "user", "content": input_payload if isinstance(input_payload, str) else messages}],
                        'max_completion_tokens': int(_clamp_requested_tokens(model, max_tokens)),
                    }
                    if enforce_json:
                        chat_kwargs['response_format'] = {'type': 'json_object'}
                    if reasoning:
                        chat_kwargs['reasoning'] = reasoning
                    if text:
                        chat_kwargs['text'] = text
                    chat_resp = client.chat.completions.create(**chat_kwargs)
                    content = chat_resp.choices[0].message.content.strip()
                    if content:
                        logger.info(f"LLM call ok (chat fallback) | model={model} tokens={max_tokens} latency_ms={latency_ms}")
                        return content, {
                            'model': model,
                            'latency_ms': latency_ms,
                            'usage': usage,
                            'error': None,
                        }
                except Exception as _:
                    pass
            logger.info(f"LLM call ok (responses) | model={model} tokens={max_tokens} latency_ms={latency_ms}")
            return (output_text or '').strip(), {
                'model': model,
                'latency_ms': latency_ms,
                'usage': usage,
                'error': None,
            }
        except Exception as e:
            logger.warning(f"Responses API failed; falling back to chat | model={model} err={e}")

    # Build base kwargs for Chat Completions
    clamped_chat_tokens = _clamp_requested_tokens(model, max_tokens)
    kwargs: Dict[str, Any] = {
        'model': model,
        'messages': messages,
        'max_completion_tokens': int(clamped_chat_tokens),
    }
    # GPT-4 family: do not include reasoning/text params; prefer temperature
    if model.startswith('gpt-5'):
        if reasoning:
            kwargs['reasoning'] = reasoning
        if text:
            kwargs['text'] = text
    # Temperature: default to 0.0 if not provided
    if temperature is None:
        try:
            # Prefer deterministic output for non-GPT-5 models
            temperature = 0.0
        except Exception:
            temperature = 0.0
    kwargs['temperature'] = float(temperature)
    if enforce_json:
        kwargs['response_format'] = {'type': 'json_object'}

    def _try_call(call_kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[Exception]]:
        try:
            resp = client.chat.completions.create(**call_kwargs)
            latency_ms_local = int((time.time() - start) * 1000)
            try:
                usage_local = getattr(resp, 'usage', None)
            except Exception:
                usage_local = None
            content_local = resp.choices[0].message.content.strip()
            return content_local, {
                'model': model,
                'latency_ms': latency_ms_local,
                'usage': usage_local,
                'error': None,
            }, None
        except Exception as ex:
            return "", {}, ex

    # If JSON is required, enforce JSON mode for Chat as well
    if enforce_json:
        kwargs['response_format'] = {'type': 'json_object'}

    # First try with full kwargs
    content, meta, err = _try_call(kwargs)
    if err is None:
        logger.info(f"LLM call ok | model={model} tokens={clamped_chat_tokens} latency_ms={meta['latency_ms']}")
        return content, meta

    # Retry logic: strip unsupported kwargs progressively
    err_msg = str(err)
    logger.warning(f"LLM call failed | model={model} err={err_msg}")

    # Adaptive single retry for token-limit errors: clamp to 75% and model ceiling
    err_lc = err_msg.lower()
    if 'max_tokens is too large' in err_lc or 'max_completion_tokens' in err_lc:
        try:
            current = int(kwargs.get('max_completion_tokens') or clamped_chat_tokens)
        except Exception:
            current = clamped_chat_tokens
        reduced_tokens = max(1, int(min(_get_model_ceiling(model, current), current * 0.75)))
        if reduced_tokens < current:
            kwargs['max_completion_tokens'] = reduced_tokens
            logger.info(f"Adaptive retry (chat) due to token limit | model={model} tokens={current}->{reduced_tokens}")
            content, meta, err_adapt = _try_call(kwargs)
            if err_adapt is None:
                logger.info(f"LLM call ok (adaptive) | model={model} tokens={reduced_tokens} latency_ms={meta['latency_ms']}")
                return content, meta
            err_msg = str(err_adapt)
            logger.warning(f"LLM adaptive retry failed | model={model} err={err_msg}")

    # 1) Remove reasoning/text if unsupported
    if 'reasoning' in kwargs or 'text' in kwargs:
        kwargs.pop('reasoning', None)
        kwargs.pop('text', None)
        content, meta, err2 = _try_call(kwargs)
        if err2 is None:
            logger.info(f"LLM call ok (no reasoning/text) | model={model} tokens={kwargs.get('max_completion_tokens')} latency_ms={meta['latency_ms']}")
            return content, meta
        err_msg = str(err2)
        logger.warning(f"LLM retry failed (no reasoning/text) | model={model} err={err_msg}")

    # 2) Drop response_format if unsupported (keep parameter naming normalized)

    if 'response_format' in kwargs:
        kwargs.pop('response_format', None)
        content, meta, err4 = _try_call(kwargs)
        if err4 is None:
            logger.info(f"LLM call ok (no response_format) | model={model} tokens={kwargs.get('max_completion_tokens')} latency_ms={meta['latency_ms']}")
            return content, meta
        err_msg = str(err4)
        logger.warning(f"LLM retry failed (no response_format) | model={model} err={err_msg}")

    latency_ms = int((time.time() - start) * 1000)
    return "", {
        'model': model,
        'latency_ms': latency_ms,
        'usage': usage,
        'error': err_msg,
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
        temperature=params.get('temperature'),
    )


