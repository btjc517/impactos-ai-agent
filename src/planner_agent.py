from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from planning_tools import (
    find_metrics as tool_find_metrics,
    generate_ir as tool_generate_ir,
    render_sql as tool_render_sql,
    run_sql as tool_run_sql,
    fetch_citations as tool_fetch_citations,
    get_openai_tools_spec,
    run_planner,
)
from telemetry import telemetry
import time


def _dispatch_tool(name: str, args: Dict[str, Any], client: Optional[OpenAI]) -> Dict[str, Any]:
    """Execute a tool locally and return a JSON-serializable result."""
    if name == "find_metrics":
        query_terms = args.get("query_terms")
        k = int(args.get("k") or 5)
        res = tool_find_metrics(query_terms, k=k)
        return {"snippets": res}
    if name == "generate_ir":
        q = args.get("question", "")
        snippets = args.get("catalog_snippets") or []
        tp = args.get("time_policy") or {}
        res = tool_generate_ir(client, question=q, catalog_snippets=snippets, time_policy=tp)
        return {"ir": res}
    if name == "render_sql":
        ir = args.get("ir") or {}
        sql, params = tool_render_sql(ir)
        return {"sql": sql, "params": list(params)}
    # run_sql is intentionally NOT exposed as a callable tool
    if name == "fetch_citations":
        ir = args.get("ir")
        rows = args.get("rows")
        refs = tool_fetch_citations(ir=ir, rows=rows)
        return {"citations": refs}
    raise ValueError(f"Unknown tool: {name}")


def _chat_tools_call(client: OpenAI, *, question: str, time_policy: Dict[str, Any], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Run a tools-enabled chat session where the model must call tools to complete the task."""
    tools = get_openai_tools_spec()
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a planner. You MUST use tools to: find_metrics -> generate_ir -> render_sql -> run_sql -> fetch_citations. "
                "Never output or invent SQL yourself. Return a final JSON object with keys ir, sql, params, rows, citations."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({"question": question, "time_policy": time_policy}),
        },
    ]

    # Track tool-derived state to assemble final result without trusting free-text
    state: Dict[str, Any] = {"ir": None, "sql": None, "params": None, "rows": None, "citations": None}
    tool_sequence: List[Dict[str, Any]] = []
    t_start = time.monotonic()

    # Loop to satisfy tool calls
    for _ in range(8):
        t_loop = time.monotonic()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_completion_tokens=800,
            temperature=0.0,
        )
        loop_ms = int((time.monotonic() - t_loop) * 1000)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            # Record the assistant turn
            assistant_payload: Dict[str, Any] = {"role": "assistant"}
            if msg.content:
                assistant_payload["content"] = msg.content
            if tool_calls:
                # Use model_dump if available, else best-effort dicts
                try:
                    assistant_payload["tool_calls"] = [tc.model_dump() for tc in tool_calls]
                except Exception:
                    assistant_payload["tool_calls"] = [
                        {
                            "id": getattr(tc, "id", None),
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": getattr(getattr(tc, "function", None), "name", None),
                                "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                            },
                        }
                        for tc in tool_calls
                    ]
            messages.append(assistant_payload)
            # Execute tool calls and append results
            for tc in tool_calls:
                name = tc.function.name
                args_str = tc.function.arguments or "{}"
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}
                t_tool = time.monotonic()
                result = _dispatch_tool(name, args, client)
                tool_ms = int((time.monotonic() - t_tool) * 1000)
                tool_sequence.append({"name": name, "duration_ms": tool_ms})
                # Update local state strictly from tool outputs
                try:
                    if name == "find_metrics":
                        pass  # not part of final payload
                    elif name == "generate_ir":
                        state["ir"] = result.get("ir")
                        # Capture IR generation meta flags when available from local planner
                        if isinstance(result.get("ir"), dict):
                            # Not available here; meta is provided by planning_tools.run_planner path
                            pass
                    elif name == "render_sql":
                        state["sql"] = result.get("sql")
                        state["params"] = result.get("params")
                    elif name == "fetch_citations":
                        state["citations"] = result.get("citations")
                except Exception:
                    pass
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
            continue
        # No tool calls; expect final JSON
        content = (msg.content or "").strip()
        try:
            data = json.loads(content) if content else {}
            # Minimal schema check
            if isinstance(data, dict) and {"ir", "sql", "params", "rows", "citations"}.issubset(set(data.keys())):
                # Reject free-text SQL path unless tools were actually called producing rows
                if state["rows"] is not None and state["sql"] is not None:
                    # Prefer tool-derived state to ensure enforcement
                    return {
                        "ir": state["ir"] or data.get("ir"),
                        "sql": state["sql"],
                        "params": state["params"] or [],
                        "rows": state["rows"],
                        "citations": state["citations"] or data.get("citations") or [],
                    }
                # If model skipped tools, fall back to local orchestrator
                return run_planner(client, question=question, time_policy=time_policy)
        except Exception:
            pass
        # If model didn't comply, but we have IR and SQL from tools, execute SQL internally now
        if state["sql"] is not None:
            try:
                rows = tool_run_sql(state["sql"], tuple(state.get("params") or ()))
            except Exception as e:
                return {"error": "render_or_exec_failed", "message": str(e)}
            refs = tool_fetch_citations(ir=state.get("ir"), rows=rows)
            payload = {
                "ir": state.get("ir"),
                "sql": state.get("sql"),
                "params": state.get("params") or [],
                "rows": rows,
                "citations": refs,
                "meta": state.get("meta"),
            }
            # Telemetry (tools path)
            try:
                if telemetry.is_enabled():
                    total_ms = int((time.monotonic() - t_start) * 1000)
                    event = telemetry.build_event(
                        question=question,
                        answer=None,
                        status='ok',
                        source='planner-tools',
                        model=model,
                        total_ms=total_ms,
                        timings={'loop_ms': loop_ms, 'tools': tool_sequence},
                        chart=None,
                        logs_text=None,
                        error=None,
                        metadata={'tool_seq': [t['name'] for t in tool_sequence]},
                    )
                    telemetry.send_query_event(event)
            except Exception:
                pass
            return payload
        # Else fall back to local orchestrator as guardrail
        return run_planner(client, question=question, time_policy=time_policy)

    # Fallback after too many loops
    return run_planner(client, question=question, time_policy=time_policy)


def run_planner_with_tools(client: OpenAI, *, question: str, time_policy: Dict[str, Any], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Public entry: tools-based run. Falls back to deterministic orchestrator if needed."""
    return _chat_tools_call(client, question=question, time_policy=time_policy, model=model)


