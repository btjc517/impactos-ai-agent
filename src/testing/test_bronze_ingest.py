import os
import sqlite3
from pathlib import Path
import sys
import pandas as pd


def _make_excel(tmp_path: Path) -> str:
    xlsx = tmp_path / 'multi.xlsx'
    with pd.ExcelWriter(xlsx) as writer:
        pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']}).to_excel(writer, index=False, sheet_name='Alpha')
        pd.DataFrame({'Date': pd.to_datetime(['2025-01-01', '2025-01-02']), 'Hours': [2.5, 3.0]}).to_excel(writer, index=False, sheet_name='Beta')
    return str(xlsx)


def test_bronze_multisheet_and_registry(tmp_path):
    os.environ['IMPACTOS_DB_PATH'] = str(tmp_path / 'impactos.db')
    file_path = _make_excel(tmp_path)

    # Ensure src on path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from bronze_ingest import ingest_bronze
    res1 = ingest_bronze(file_path, os.environ['IMPACTOS_DB_PATH'])
    assert len(res1['created_tables']) == 2

    # Re-ingest unchanged: expect no new registry version for same hash
    res2 = ingest_bronze(file_path, os.environ['IMPACTOS_DB_PATH'])
    assert res2['created_tables'] == res1['created_tables']

    # Check registry rows count per sheet/version
    with sqlite3.connect(os.environ['IMPACTOS_DB_PATH']) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sheet_registry")
        count = cur.fetchone()[0]
        # Two sheets, one version each
        assert count == 2

        # Assert sheet-level ranges exist
        cur.execute("SELECT data_range, header_range FROM sheet_registry WHERE sheet_name='Alpha'")
        dr, hr = cur.fetchone()
        assert isinstance(dr, str) and ':' in dr
        assert isinstance(hr, str) and ':' in hr
        # Assert per-row row_cell_ref in bronze
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bronze_multi__alpha'")
        assert cur.fetchone() is not None
        cur.execute("SELECT row_cell_ref FROM bronze_multi__alpha ORDER BY row_idx ASC LIMIT 3")
        refs = [r[0] for r in cur.fetchall()]
        assert all(isinstance(x, str) and ':' in x for x in refs)

    # Modify a sheet to bump hash/version
    with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='overlay') as writer:
        pd.DataFrame({'A': [3], 'B': ['z']}).to_excel(writer, index=False, sheet_name='Alpha')

    res3 = ingest_bronze(file_path, os.environ['IMPACTOS_DB_PATH'])
    with sqlite3.connect(os.environ['IMPACTOS_DB_PATH']) as conn:
        cur = conn.cursor()
        cur.execute("SELECT sheet_name, MAX(version) FROM sheet_registry GROUP BY sheet_name")
        versions = dict(cur.fetchall())
        assert versions.get('Alpha', 0) >= 2


