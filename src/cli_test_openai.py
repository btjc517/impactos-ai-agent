"""
Small CLI to test OpenAI Responses/Chat API and print response structure.

Usage:
  python3 src/cli_test_openai.py --prompt "Hello" --model gpt-4o-mini

Requires environment variable OPENAI_API_KEY to be set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from openai import OpenAI


def extract_output_text(resp: Any) -> str:
    """Try to extract concatenated text from Responses API output."""
    # Try standard convenience field first
    text = getattr(resp, "output_text", "") or ""
    if text:
        return text.strip()

    # Walk output -> content -> text.value
    try:
        items = getattr(resp, "output", []) or []
        chunks: List[str] = []
        for item in items:
            contents = getattr(item, "content", None) or []
            for content in contents:
                # content.text may be object with .value, dict, or str
                text_obj = getattr(content, "text", None)
                if text_obj is None and isinstance(content, dict):
                    text_obj = content.get("text")
                val = None
                if text_obj is not None:
                    val = getattr(text_obj, "value", None)
                    if val is None and isinstance(text_obj, dict):
                        val = text_obj.get("value")
                    if val is None and isinstance(text_obj, str):
                        val = text_obj
                if val:
                    chunks.append(str(val))
        if chunks:
            return "".join(chunks).strip()
    except Exception:
        pass

    # As a last resort, try Chat Completions shape if present
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Test OpenAI Responses/Chat API")
    parser.add_argument(
        "--prompt",
        required=False,
        default="Say a short hello from ImpactOS.",
        help="User prompt to send",
    )
    parser.add_argument(
        "--model",
        required=False,
        default="gpt-4o-mini",
        help="Model name (e.g., gpt-4o-mini, gpt-4o)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Max output tokens",
    )
    parser.add_argument(
        "--use-chat",
        action="store_true",
        help="Force Chat Completions API instead of Responses",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Also print full raw JSON response",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in environment.", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key)

    def _call_chat() -> int:
        try:
            chat = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                max_completion_tokens=args.max_output_tokens,
            )
            raw = chat.model_dump()
            text_out = chat.choices[0].message.content.strip()
            print("output_text:\n" + text_out)
            if args.print_raw:
                print("\nraw_response:")
                print(json.dumps(raw, indent=2))
            return 0
        except Exception as e2:
            print(f"Chat API failed: {e2}", file=sys.stderr)
            return 1

    if args.use_chat:
        return _call_chat()

    try:
        resp = client.responses.create(
            model=args.model,
            input=args.prompt,
            modalities=["text"],
            max_output_tokens=args.max_output_tokens,
            reasoning={"effort": "medium"},
            text={"verbosity": "medium", "format": {"type": "text"}},
        )
    except Exception as e:
        print(f"Responses API failed: {e}", file=sys.stderr)
        return _call_chat()

    # Success path for Responses API
    text_out = extract_output_text(resp)
    if not text_out:
        # If no text extracted, try Chat as a fallback
        return _call_chat()
    print("output_text:\n" + text_out)

    # Print usage/metadata if available
    usage = getattr(resp, "usage", None)
    if usage:
        try:
            usage_dict: Dict[str, Any] = (
                usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
            )
            print("\nusage:")
            print(json.dumps(usage_dict, indent=2))
        except Exception:
            pass

    if args.print_raw:
        try:
            print("\nraw_response:")
            if hasattr(resp, "model_dump_json"):
                print(resp.model_dump_json(indent=2))
            else:
                print(json.dumps(resp, indent=2, default=str))
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


