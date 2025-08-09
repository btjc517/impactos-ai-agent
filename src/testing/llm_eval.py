"""
LLM evaluation CLI to compare models and parameter settings.

Runs one or more prompts across a grid of models and params using the
project's unified call helper, recording:
- content (truncated), latency_ms, usage, error, json_parse_ok

Usage examples:
  python3 src/testing/llm_eval.py --prompts "Say hello in one short sentence." \
    --models gpt-5-mini,gpt-4o-mini --max-output-tokens 64 --enforce-json false

  python3 src/testing/llm_eval.py --prompts "Return ONLY a JSON object with keys a and b." \
    --models gpt-5,gpt-5-mini,gpt-4o-mini --max-output-tokens 256 --enforce-json true

Requires OPENAI_API_KEY in environment.
Outputs a JSONL report under src/testing/results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Use the project's LLM helper for robust Responses+Chat fallback
from llm_utils import call_chat_completion


@dataclass
class EvalCase:
    prompt: str
    model: str
    max_output_tokens: int
    reasoning_effort: str
    text_verbosity: str
    enforce_json: bool
    suite_name: str = "custom"


@dataclass
class EvalResult:
    prompt: str
    model: str
    max_output_tokens: int
    reasoning_effort: str
    text_verbosity: str
    enforce_json: bool
    latency_ms: Optional[int]
    usage: Optional[Dict[str, Any]]
    error: Optional[str]
    content_preview: str
    json_parse_ok: Optional[bool]
    suite_name: str


def ensure_results_dir() -> Path:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def materialize_usage(usage_obj: Any) -> Optional[Dict[str, Any]]:
    if not usage_obj:
        return None
    try:
        if hasattr(usage_obj, "model_dump"):
            return usage_obj.model_dump()
        if isinstance(usage_obj, dict):
            return usage_obj
        return json.loads(json.dumps(usage_obj, default=str))
    except Exception:
        return None


def run_eval_case(client: Optional[OpenAI], case: EvalCase) -> EvalResult:
    # Create a local client if not provided (safer for concurrency)
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": case.prompt}]
    text_param: Dict[str, Any] = {"verbosity": case.text_verbosity}
    if case.enforce_json:
        text_param["format"] = {"type": "json_object"}

    content = ""
    meta: Dict[str, Any] = {}
    try:
        content, meta = call_chat_completion(
            client,
            messages,
            model=case.model,
            max_tokens=case.max_output_tokens,
            reasoning={"effort": case.reasoning_effort},
            text=text_param,
            enforce_json=case.enforce_json,
        )
    except Exception as e:
        return EvalResult(
            prompt=case.prompt,
            model=case.model,
            max_output_tokens=case.max_output_tokens,
            reasoning_effort=case.reasoning_effort,
            text_verbosity=case.text_verbosity,
            enforce_json=case.enforce_json,
            latency_ms=None,
            usage=None,
            error=str(e),
            content_preview="",
            json_parse_ok=False if case.enforce_json else None,
            suite_name=case.suite_name,
        )

    # Prepare result fields
    latency_ms = meta.get("latency_ms") if isinstance(meta, dict) else None
    usage = materialize_usage(meta.get("usage")) if isinstance(meta, dict) else None
    error = meta.get("error") if isinstance(meta, dict) else None

    # Evaluate JSON parse if requested
    json_ok: Optional[bool] = None
    if case.enforce_json:
        try:
            _ = json.loads(content) if content else None
            json_ok = bool(content)
        except Exception:
            json_ok = False

    preview = (content or "").strip()
    if len(preview) > 240:
        preview = preview[:240] + "…"

    return EvalResult(
        prompt=case.prompt,
        model=case.model,
        max_output_tokens=case.max_output_tokens,
        reasoning_effort=case.reasoning_effort,
        text_verbosity=case.text_verbosity,
        enforce_json=case.enforce_json,
        latency_ms=latency_ms,
        usage=usage,
        error=error,
        content_preview=preview,
        json_parse_ok=json_ok,
        suite_name=case.suite_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM model/params evaluation")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "Say hello in one short sentence.",
            "Return ONLY a valid JSON object with keys a and b; no markdown.",
        ],
        help="One or more prompts to evaluate",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="",
        choices=["", "structure_analysis", "query_generation", "extraction_metrics", "intent", "answer_with_citations", "all"],
        help="Predefined prompt suite matching project use cases",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gpt-5,gpt-5-mini,gpt-5-nano,gpt-4o-mini",
        help="Comma-separated list of models",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Token cap per generation",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default="minimal,medium",
        help="Comma-separated reasoning efforts (minimal, medium, high)",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="low,medium",
        help="Comma-separated text verbosity levels",
    )
    parser.add_argument(
        "--enforce-json",
        type=str,
        default="auto",
        choices=["true", "false", "auto"],
        help="If true, require JSON. If auto, only for prompts that request JSON.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="",
        help="Optional path for JSONL results. Default under src/testing/results/",
    )
    parser.add_argument(
        "--pretty-outfile",
        type=str,
        default="",
        help="Optional path to write pretty JSON (array) results with indent=2",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--console",
        type=str,
        default="pretty",
        choices=["pretty", "lines", "off"],
        help="Console output style: pretty (multiline), lines (one-liners), off",
    )
    return parser.parse_args()


def str_to_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "y", "on")


# --- Prompt suites ---
def build_suite_prompts() -> List[tuple[str, str]]:
    """Return list of (suite_name, prompt) pairs for project use cases."""
    # 1) Structure analysis (expects JSON object with required keys)
    structure_analysis_prompt = (
        "You are analyzing a small social value dataset. Columns: 'Volunteer Hours' (numeric), "
        "'Donations' (GBP numeric), 'Department' (categorical). Rows: 15. "
        "Return ONLY a valid JSON object with keys: data_overview, column_analysis (list), identified_metrics (list). "
        "No prose, no markdown."
    )
    # 2) Query generation (expects JSON array of queries based on a provided structure)
    query_generation_prompt = (
        "Based on the following structure analysis, generate precise pandas queries as a JSON array. "
        "Return ONLY a JSON array.\n\nSTRUCTURE ANALYSIS:\n"
        + json.dumps({
            "data_overview": {"total_rows": 15, "total_columns": 3, "data_quality_score": 0.9,
                               "primary_data_type": "social_value_metrics"},
            "column_analysis": [
                {"column_name": "Volunteer Hours", "column_index": 0, "data_type": "numeric",
                 "social_value_category": "community_engagement", "contains_metrics": True},
                {"column_name": "Donations", "column_index": 1, "data_type": "numeric",
                 "social_value_category": "charitable_giving", "contains_metrics": True},
                {"column_name": "Department", "column_index": 2, "data_type": "string",
                 "contains_metrics": False}
            ],
            "identified_metrics": [
                {"metric_name": "total_volunteer_hours", "metric_category": "community_engagement",
                 "extraction_method": "column_sum", "target_column": "Volunteer Hours", "confidence": 0.9},
                {"metric_name": "average_donation_amount", "metric_category": "charitable_giving",
                 "extraction_method": "column_average", "target_column": "Donations", "confidence": 0.85}
            ]
        }, indent=2)
    )
    # 3) Extraction metrics JSON (expects array of normalized metrics with citations fields)
    extraction_metrics_prompt = (
        "From the following simplified table snippet, extract metrics and return ONLY a JSON array. "
        "Each item must include: metric_name, metric_value, metric_unit, metric_category, "
        "source_column_name, source_column_index, source_row_index, source_cell_reference. "
        "No prose.\n\nTABLE:\nRow1: Volunteer Hours=12.5; Donations=30.00; Department=HR\n"
        "Row2: Volunteer Hours=8.0; Donations=45.00; Department=Ops\nRow3: Volunteer Hours=15.0; Donations=25.00; Department=HR"
    )
    # 4) Intent classification (expects a small JSON object)
    intent_prompt = (
        "Question: 'What is the total number of volunteer hours?' "
        "Return ONLY a JSON object with keys: categories (list), aggregations (list), query_type."
    )
    # 5) Answer with citations (natural language with [1], [2] and a Sources block)
    answer_with_citations_prompt = (
        "Context [1]: 'Volunteer hours totaled 35.5 across 3 entries.'\n"
        "Context [2]: 'Donations averaged £33.33 across 3 entries.'\n"
        "Answer the question 'Summarize the key impacts.' in 2-3 sentences, citing sources like [1], [2], "
        "and include a Sources block listing [1] and [2]."
    )

    return [
        ("structure_analysis", structure_analysis_prompt),
        ("query_generation", query_generation_prompt),
        ("extraction_metrics", extraction_metrics_prompt),
        ("intent", intent_prompt),
        ("answer_with_citations", answer_with_citations_prompt),
    ]


def main() -> int:
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    reasoning_efforts = [r.strip() for r in args.reasoning.split(",") if r.strip()]
    text_verbosity_levels = [v.strip() for v in args.verbosity.split(",") if v.strip()]

    # Build prompts from suite if provided
    prompts: List[tuple[str, str]] = []
    if args.suite:
        pairs = build_suite_prompts()
        if args.suite != "all":
            prompts = [pair for pair in pairs if pair[0] == args.suite]
        else:
            prompts = pairs
    else:
        prompts = [("custom", p) for p in args.prompts]

    out_dir = ensure_results_dir()
    timestamp = int(time.time())
    out_path = (
        Path(args.outfile)
        if args.outfile
        else out_dir / f"llm_eval_{timestamp}.jsonl"
    )

    total_runs = 0
    successes = 0
    cases: List[EvalCase] = []
    for suite_name, prompt in prompts:
        wants_json = ("json" in prompt.lower() and "object" in prompt.lower())
        enforce_json = (
            str_to_bool(args.enforce_json)
            if args.enforce_json in ("true", "false")
            else wants_json
        )
        for model in models:
            for effort in reasoning_efforts:
                for verbosity in text_verbosity_levels:
                    cases.append(EvalCase(
                        prompt=prompt,
                        model=model,
                        max_output_tokens=args.max_output_tokens,
                        reasoning_effort=effort,
                        text_verbosity=verbosity,
                        enforce_json=enforce_json,
                        suite_name=suite_name,
                    ))

    from concurrent.futures import ThreadPoolExecutor
    results: List[EvalResult] = []
    with open(out_path, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(run_eval_case, None, case) for case in cases]
            for fut, case in zip(futures, cases):
                res = fut.result()
                results.append(res)
                total_runs += 1
                enforce_json = case.enforce_json
                if (not enforce_json and (res.content_preview != "")) or (
                    enforce_json and bool(res.json_parse_ok)
                ):
                    successes += 1
                f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
                # Pretty console output (multiline)
                print_console_result(res)
    # Write pretty JSON array for easier reading
    pretty_path = out_dir / f"llm_eval_{timestamp}.pretty.json"
    with open(pretty_path, "w", encoding="utf-8") as pf:
        json.dump([asdict(r) for r in results], pf, indent=2, ensure_ascii=False)
    print(f"Wrote pretty results to: {pretty_path}")
    print(f"Wrote results to: {out_path}")
    print(f"Successes: {successes}/{total_runs}")
    return 0


def print_console_result(res: EvalResult) -> None:
    ok = res.json_parse_ok if res.enforce_json else (res.content_preview != "")
    status = "OK" if ok else "!!"
    print(f"[{status}] suite={res.suite_name} | model={res.model} | tokens={res.max_output_tokens} | latency_ms={res.latency_ms}")
    print(f"  reasoning={res.reasoning_effort} | verbosity={res.text_verbosity} | json_ok={res.json_parse_ok}")
    if res.error:
        print(f"  error: {res.error}")
    print("  preview:")
    preview = (res.content_preview or "").replace("\n", " ")
    if len(preview) > 160:
        preview = preview[:160] + "…"
    print(f"    {preview}")

if __name__ == "__main__":
    raise SystemExit(main())
