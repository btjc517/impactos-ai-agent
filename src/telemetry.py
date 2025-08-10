"""
Telemetry utilities for recording query events to Supabase.

This module provides:
- SupabaseTelemetry: minimal HTTP client to write events via PostgREST
- LogCapture: context manager to capture in-memory logs for a code block

Environment variables used:
- SUPABASE_URL: Base URL of the Supabase project (e.g., https://xyz.supabase.co)
- SUPABASE_SERVICE_ROLE_KEY: Service role key for privileged inserts

Notes:
- Uses httpx (already in requirements) to avoid adding new dependencies
- Safe to import in environments without Supabase configuration; calls will no-op
"""

from __future__ import annotations

import os
import json
import uuid
import time
import logging
from typing import Any, Dict, Optional

import httpx


class LogCapture(logging.Handler):
    """In-memory log capture handler for collecting logs during a request.

    Usage:
        with capture_logs():
            ...
        logs_text = handler.get_value()
    """

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level)
        self._records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._records.append(msg)
        except Exception:
            # Avoid raising from logging paths
            pass

    def get_value(self, max_chars: int = 100_000) -> str:
        text = "\n".join(self._records)
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text


class capture_logs:
    """Context manager to attach a LogCapture handler temporarily.

    By default, attaches to the root logger to capture logs from all modules.
    """

    def __init__(self, level: int = logging.INFO, logger: Optional[logging.Logger] = None) -> None:
        self.level = level
        self.logger = logger or logging.getLogger()
        self.handler = LogCapture(level=self.level)
        # Consistent, readable format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)

    def __enter__(self) -> LogCapture:
        self.logger.addHandler(self.handler)
        return self.handler

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.logger.removeHandler(self.handler)
        except Exception:
            pass


class SupabaseTelemetry:
    """Minimal Supabase telemetry client using PostgREST endpoints.

    Inserts rows into public.ai_query_events with a single HTTP POST.
    """

    def __init__(self, supabase_url: Optional[str], service_key: Optional[str]) -> None:
        self.supabase_url = (supabase_url or '').rstrip('/')
        self.service_key = service_key or ''
        self._enabled = bool(self.supabase_url and self.service_key)
        self._client: Optional[httpx.Client] = None

    @classmethod
    def from_env(cls) -> "SupabaseTelemetry":
        return cls(
            supabase_url=os.getenv('SUPABASE_URL'),
            service_key=os.getenv('SUPABASE_SERVICE_ROLE_KEY'),
        )

    def is_enabled(self) -> bool:
        return self._enabled

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers = {
                'apikey': self.service_key,
                'Authorization': f'Bearer {self.service_key}',
                'Content-Type': 'application/json',
                'Prefer': 'return=minimal',
            }
            self._client = httpx.Client(headers=headers, timeout=httpx.Timeout(5.0, connect=3.0))
        return self._client

    def send_query_event(self, event: Dict[str, Any]) -> None:
        """Synchronously send a single query event. No-ops if not configured.

        The event should match the ai_query_events schema. Unknown keys are ignored by PostgREST
        if the table has a JSONB metadata field to hold extra data.
        """
        if not self._enabled:
            return
        try:
            url = f"{self.supabase_url}/rest/v1/ai_query_events"
            client = self._get_client()
            client.post(url, content=json.dumps([event]))  # send as array for bulk insert semantics
        except Exception:
            # Avoid impacting the request path on telemetry failure
            pass

    def build_event(
        self,
        *,
        question: str,
        answer: Optional[str],
        status: str,
        source: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        total_ms: Optional[int] = None,
        timings: Optional[Dict[str, Any]] = None,
        chart: Optional[Dict[str, Any]] = None,
        logs_text: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a sanitized event payload for ai_query_events."""
        payload: Dict[str, Any] = {
            'id': request_id or str(uuid.uuid4()),
            'question': question,
            'answer': (answer or '')[:100_000],
            'status': status,
            'source': source,
            'user_id': user_id,
            'session_id': session_id,
            'model': model,
            'total_ms': int(total_ms) if total_ms is not None else None,
            'timings': timings or {},
            'chart': chart or None,
            'logs': (logs_text or '')[:200_000],
            'error': (error or None),
            'metadata': metadata or {},
        }
        return payload


# Module-level telemetry instance for convenient reuse
telemetry = SupabaseTelemetry.from_env()


