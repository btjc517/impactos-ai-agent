"""
Shared embedding model registry to avoid repeated HF Hub downloads and 429s.

- Provides a singleton SentenceTransformer per process
- Uses a fixed cache folder to persist model files
- Tries local-only load first; on miss, retries online with exponential backoff

Environment tips:
- Set HF_HOME or TRANSFORMERS_CACHE to a persistent directory
- For CI/offline runs: export HF_HUB_OFFLINE=1 to force local-only
"""

from __future__ import annotations

import os
import time
import logging
import threading
from typing import Optional, Dict, Any

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


def _is_rate_limited_error(err: Exception) -> bool:
    msg = str(err).lower()
    # Common signals for HF rate limiting / 429s
    return (
        '429' in msg or
        'too many requests' in msg or
        'rate limit' in msg or
        'rate-limited' in msg
    )


def get_embedding_model(
    model_name: str = 'all-MiniLM-L6-v2',
    *,
    cache_folder: str = '.cache/hf',
    max_retries: int = 5,
    initial_backoff_s: float = 0.5,
) -> Optional[SentenceTransformer]:
    """Return a shared SentenceTransformer instance.

    Attempts a local-only load first to avoid network. If unavailable, retries
    online with exponential backoff. On repeated rate limits, falls back to
    local-only once again (if partially cached).
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    with _LOCK:
        if _EMBEDDING_MODEL is not None:
            return _EMBEDDING_MODEL

        os.makedirs(cache_folder, exist_ok=True)

        # 1) Try local-only: avoids any HF requests if cache exists
        try:
            logger.info("Loading shared SentenceTransformer (local-only if cached)")
            _EMBEDDING_MODEL = SentenceTransformer(
                model_name,
                cache_folder=cache_folder,
                model_kwargs={
                    'local_files_only': True,
                },
            )
            return _EMBEDDING_MODEL
        except Exception as e:
            logger.debug(f"Local-only embedding load miss: {e}")

        # 2) Online load with exponential backoff
        backoff = initial_backoff_s
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    logger.info("Loading shared SentenceTransformer (online, may download)")
                else:
                    logger.info(f"Retrying embedding load (attempt {attempt+1}/{max_retries})")
                _EMBEDDING_MODEL = SentenceTransformer(
                    model_name,
                    cache_folder=cache_folder,
                )
                return _EMBEDDING_MODEL
            except Exception as e:
                if _is_rate_limited_error(e) and attempt < max_retries - 1:
                    logger.warning(f"HF rate limit encountered; backing off {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                logger.warning(f"Embedding load failed: {e}")

        # 3) Final fallback: try local-only again in case partial cache exists
        try:
            logger.info("Final fallback: local-only embedding load")
            _EMBEDDING_MODEL = SentenceTransformer(
                model_name,
                cache_folder=cache_folder,
                model_kwargs={
                    'local_files_only': True,
                },
            )
            return _EMBEDDING_MODEL
        except Exception as e:
            logger.error(f"Failed to initialize shared embedding model: {e}")
            _EMBEDDING_MODEL = None
            return None


def reset_embedding_model() -> None:
    """Reset the cached embedding model (for tests)."""
    global _EMBEDDING_MODEL
    with _LOCK:
        _EMBEDDING_MODEL = None


