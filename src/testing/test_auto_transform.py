import os
import sqlite3
from pathlib import Path
import pandas as pd


def setup_db(tmp_path):
    db_path = tmp_path / 'impactos.db'
    os.environ['IMPACTOS_DB_PATH'] = str(db_path)
    conn = sqlite3.connect(db_path)
    root = Path(__file__).resolve().parents[2]
    for mig in [
        '20250813T000000Z__medallion_v1.sql',
        '20250813T020000Z__sheet_registry.sql',
        '20250813T030000Z__sheet_registry_ranges.sql',
        '20250813T040000Z__transform_runs.sql',
        '20250813T041000Z__fact_meta_and_dim_uniques.sql',
        '20250813T042000Z__transform_rejects_lineage.sql',
        '20250813T050000Z__transform_jobs.sql',
        '20250813T010000Z__medallion_indexes_constraints.sql',
    ]:
        with open(root / 'db' / 'migrations' / mig) as f:
            conn.executescript(f.read())
    return conn


def _make_excel(tmp_path: Path) -> str:
    xlsx = tmp_path / 'twosheets.xlsx'
    with pd.ExcelWriter(xlsx) as writer:
        pd.DataFrame({'Person': ['u1'], 'Site': ['HQ'], 'Hours': [2.0], 'Date': ['2025-01-01']}).to_excel(writer, index=False, sheet_name='Volunteering')
        pd.DataFrame({'Site': ['HQ'], 'Amount': [100.0], 'Currency': ['USD'], 'Date': ['2025-01-02']}).to_excel(writer, index=False, sheet_name='Donations')
    return str(xlsx)


def test_enqueue_and_sync_execute(tmp_path):
    conn = setup_db(tmp_path)
    try:
        os.environ['AUTO_TRANSFORM'] = 'true'
        os.environ['TRANSFORM_MODE'] = 'sync'
        xlsx = _make_excel(tmp_path)
        # Ingest
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from bronze_ingest import ingest_bronze
        res = ingest_bronze(xlsx, os.environ['IMPACTOS_DB_PATH'])
        cur = conn.cursor()
        # Jobs
        cur.execute("SELECT COUNT(*) FROM transform_jobs")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT COUNT(*) FROM transform_runs WHERE sheet_registry_id IS NOT NULL AND transform_job_id IS NOT NULL")
        assert cur.fetchone()[0] == 2
        # Facts populated
        cur.execute("SELECT COUNT(*) FROM fact_volunteering")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM fact_donations")
        assert cur.fetchone()[0] == 1
        # Re-ingest unchanged: no new jobs
        res2 = ingest_bronze(xlsx, os.environ['IMPACTOS_DB_PATH'])
        cur.execute("SELECT COUNT(*) FROM transform_jobs")
        assert cur.fetchone()[0] == 2
    finally:
        conn.close()


def test_async_mode_pending(tmp_path):
    conn = setup_db(tmp_path)
    try:
        os.environ['AUTO_TRANSFORM'] = 'true'
        os.environ['TRANSFORM_MODE'] = 'async'
        xlsx = _make_excel(tmp_path)
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from bronze_ingest import ingest_bronze
        ingest_bronze(xlsx, os.environ['IMPACTOS_DB_PATH'])
        cur = conn.cursor()
        cur.execute("SELECT status FROM transform_jobs ORDER BY id")
        statuses = [r[0] for r in cur.fetchall()]
        assert all(s == 'pending' for s in statuses)
        # Facts untouched in async mode
        cur.execute("SELECT COUNT(*) FROM fact_volunteering")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT COUNT(*) FROM fact_donations")
        assert cur.fetchone()[0] == 0
    finally:
        conn.close()


