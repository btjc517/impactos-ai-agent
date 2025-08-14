import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from ingest import DataIngestion
from vector_search import FAISSVectorSearch


def _init_db(db_path: str):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS sources (id INTEGER PRIMARY KEY, filename TEXT, file_type TEXT, file_size_bytes INTEGER, processing_status TEXT, processed_timestamp TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS impact_metrics (id INTEGER PRIMARY KEY, source_id INTEGER, metric_name TEXT, metric_value REAL, metric_unit TEXT, metric_category TEXT, extraction_confidence REAL, context_description TEXT, source_sheet_name TEXT, source_column_name TEXT, source_column_index INTEGER, source_row_index INTEGER, source_cell_reference TEXT, source_formula TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, metric_id INTEGER, embedding_vector_id TEXT, text_chunk TEXT, chunk_type TEXT)")
        conn.commit()


class _NoMetricsIngestion(DataIngestion):
    """Subclass to force zero metrics on second ingest."""

    def _extract_metrics(self, data, source_id: int):
        return []


def test_reingest_zero_metrics_preserves_prior_sources_and_faiss(tmp_path: Path, monkeypatch):
    db_path = str(tmp_path / "impactos_test.db")
    index_path = str(tmp_path / "faiss_index_test")
    _init_db(db_path)

    # Seed a prior source and one metric + embedding
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO sources (filename, file_type, file_size_bytes, processing_status) VALUES (?, ?, ?, 'completed')", ("file.xlsx", "xlsx", 100,))
        prior_source_id = cur.lastrowid
        cur.execute("INSERT INTO impact_metrics (source_id, metric_name, metric_value, metric_unit, metric_category, extraction_confidence, context_description) VALUES (?, 'test_metric', 1.0, 'count', 'test', 0.9, 'ctx')", (prior_source_id,))
        prior_metric_id = cur.lastrowid
        conn.commit()

    # Build FAISS with one vector corresponding to the prior metric
    vs = FAISSVectorSearch(db_path=db_path, index_path=index_path)
    vs.add_embeddings([
        {
            'vector': [0.1] * vs.embedding_dimension,
            'text_chunk': 'seed',
            'metric_id': prior_metric_id,
            'chunk_type': 'metric',
            'metric_name': 'test_metric',
            'metric_category': 'test',
            'filename': 'file.xlsx',
            'source_info': {}
        }
    ])
    before_total = vs.get_stats()['total_vectors']
    assert before_total == 1

    # Ingest same filename but force zero metrics; ensure prior is preserved and FAISS unchanged
    ingestion = _NoMetricsIngestion(db_path=db_path)

    # Provide a tiny CSV to pass validation/load
    csv_path = tmp_path / "file.xlsx"  # use same base name to trigger dedup path
    csv_path.write_text("dummy")

    # Monkeypatch loader to bypass actual file reading and return minimal DataFrame-like object
    def _load_file_mock(self, file_path: str):
        class _DF:
            empty = False
            columns = []
        return _DF()

    ingestion._load_file = _load_file_mock.__get__(ingestion, _NoMetricsIngestion)

    ok = ingestion.ingest_file(str(csv_path), use_query_based=False)
    assert ok is True

    # Ensure prior source still present
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sources WHERE filename = ?", ("file.xlsx",))
        count_sources = cur.fetchone()[0]
        assert count_sources >= 1

    # Ensure FAISS vector count unchanged
    vs2 = FAISSVectorSearch(db_path=db_path, index_path=index_path)
    after_total = vs2.get_stats()['total_vectors']
    assert after_total == before_total


