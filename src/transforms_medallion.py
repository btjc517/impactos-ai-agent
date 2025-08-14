"""
Medallion transforms and helpers for ImpactOS (Bronze → Silver → Gold).

This module provides:
- Bronze table creation helper for a given file/sheet with required columns
- Silver loaders: transform Bronze rows into canonical facts/dimensions
- Gold: utility to refresh materialized views (for SQLite, views are dynamic)

All functions are deterministic SQL-first; no free-text SQL.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, Any, Iterable, List, Tuple


def slugify(value: str) -> str:
    value = (value or '').strip().lower()
    repl = {
        ' ': '_',
        '/': '_',
        '\\': '_',
        '-': '_',
        '.': '_',
        ':': '_',
        ',': '_',
        '(': '',
        ')': '',
        '[': '',
        ']': '',
        '&': 'and',
        '%': 'percent',
        '£': 'gbp',
        '$': 'usd',
        '€': 'eur',
        '@': 'at',
        '#': 'num',
        '+': 'plus',
        '=': 'eq',
        '<': 'lt',
        '>': 'gt',
        '!': '',
        '?': '',
        '"': '',
        "'": '',
        '`': '',
        '~': '',
        '^': '',
        '*': '',
        '{': '',
        '}': '',
        '|': '',
    }
    for k, v in repl.items():
        value = value.replace(k, v)
    while '__' in value:
        value = value.replace('__', '_')
    return value.strip('_')


def ensure_bronze_table(conn: sqlite3.Connection, file_slug: str, sheet_slug: str, raw_columns: Dict[str, str]) -> str:
    """Create a Bronze table if not exists with required columns.

    Required columns: bronze_id PK, source_file, sheet_name, row_idx, cell_ref_range
    raw_columns: mapping of column_name -> sqlite type ('TEXT','REAL','INTEGER','DATE')
    Returns created table name.
    """
    table = f"bronze_{slugify(file_slug)}__{slugify(sheet_slug)}"
    cursor = conn.cursor()
    # Create registry entry if not exists
    cursor.execute(
        """
        INSERT OR IGNORE INTO bronze_registry (tenant_id, table_name, file_slug, sheet_slug, source_file, sheet_name)
        VALUES ('default', ?, ?, ?, ?, ?)
        """,
        (table, file_slug, sheet_slug, file_slug, sheet_slug),
    )

    # Build DDL (create if not exists; if exists, we'll patch missing columns below)
    cols = [
        "bronze_id INTEGER PRIMARY KEY AUTOINCREMENT",
        "tenant_id TEXT NOT NULL DEFAULT 'default'",
        "source_file TEXT NOT NULL",
        "sheet_name TEXT",
        "row_idx INTEGER NOT NULL",
        "row_cell_ref TEXT",
    ]
    for col, typ in raw_columns.items():
        safe_col = slugify(col)
        typ_sql = typ.upper()
        if typ_sql not in ("TEXT", "REAL", "INTEGER", "DATE"):
            typ_sql = "TEXT"
        cols.append(f"{safe_col} {typ_sql}")

    ddl = f"CREATE TABLE IF NOT EXISTS {table} (\n  " + ",\n  ".join(cols) + "\n)"
    cursor.execute(ddl)

    # If table pre-existed without tenant_id, add it (NOT NULL with default is allowed)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    if cursor.fetchone():
        cursor.execute(f"PRAGMA table_info({table})")
        existing_cols = {r[1] for r in cursor.fetchall()}  # col name at index 1
        if 'tenant_id' not in existing_cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
        if 'row_cell_ref' not in existing_cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN row_cell_ref TEXT")
    # Indexes for bronze
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_bronze_id ON {table}(bronze_id)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_tenant ON {table}(tenant_id)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_row_idx ON {table}(row_idx)")
    conn.commit()
    return table


def upsert_dim_date(conn: sqlite3.Connection, date_iso: str) -> int:
    """Ensure a dim_date row exists and return date_key (yyyymmdd)."""
    cursor = conn.cursor()
    # Compute key parts in SQL
    cursor.execute(
        """
        SELECT CAST(strftime('%Y%m%d', ?) AS INTEGER) AS date_key,
               CAST(strftime('%Y', ?) AS INTEGER) AS yy,
               CAST(((CAST(strftime('%m', ?) AS INTEGER) - 1) / 3) + 1 AS INTEGER) AS qq,
               CAST(strftime('%m', ?) AS INTEGER) AS mm,
               CAST(strftime('%d', ?) AS INTEGER) AS dd
        """,
        (date_iso, date_iso, date_iso, date_iso, date_iso),
    )
    row = cursor.fetchone()
    if not row:
        raise ValueError("Invalid date for dim_date")
    date_key, yy, qq, mm, dd = row
    cursor.execute(
        """
        INSERT OR IGNORE INTO dim_date (date_key, date_value, year, quarter, month, day)
        VALUES (?, date(?), ?, ?, ?, ?)
        """,
        (date_key, date_iso, yy, qq, mm, dd),
    )
    conn.commit()
    return date_key


def transform_bronze_to_fact_volunteering(
    conn: sqlite3.Connection,
    bronze_table: str,
    mappings: Dict[str, str],
    default_date: str,
) -> int:
    """Insert into fact_volunteering from a Bronze table using deterministic SQL.

    mappings: { 'hours': 'col_name_in_bronze', 'person_ref': 'optional', 'site_code': 'optional' }
    Returns number of rows inserted.
    """
    cursor = conn.cursor()
    date_key = upsert_dim_date(conn, default_date)

    hours_col = mappings.get('hours')
    person_col = mappings.get('person_ref')
    site_col = mappings.get('site_code')

    # Optional dim upserts
    site_join = ''
    site_select = 'NULL'
    if site_col:
        # Create sites from distinct values
        cursor.execute(
            f"""
            INSERT OR IGNORE INTO dim_site (tenant_id, site_code, site_name)
            SELECT DISTINCT 'default', {site_col}, {site_col}
            FROM {bronze_table}
            WHERE {site_col} IS NOT NULL AND TRIM({site_col}) <> ''
            """
        )
        site_select = f"(SELECT site_id FROM dim_site WHERE tenant_id='default' AND site_code = b.{site_col} LIMIT 1)"

    person_select = 'NULL'
    if person_col:
        cursor.execute(
            f"""
            INSERT OR IGNORE INTO dim_person (tenant_id, external_ref)
            SELECT DISTINCT 'default', {person_col}
            FROM {bronze_table}
            WHERE {person_col} IS NOT NULL AND TRIM({person_col}) <> ''
            """
        )
        person_select = f"(SELECT person_id FROM dim_person WHERE tenant_id='default' AND external_ref = b.{person_col} LIMIT 1)"

    # Deterministic insert
    insert_sql = f"""
        INSERT OR IGNORE INTO fact_volunteering (
            tenant_id, date_key, person_id, site_id, hours, unit, bronze_table, bronze_row_ids, transform_version
        )
        SELECT 
            'default' AS tenant_id,
            {date_key} AS date_key,
            {person_select} AS person_id,
            {site_select} AS site_id,
            CAST(b.{hours_col} AS REAL) AS hours,
            'hours' AS unit,
            ? AS bronze_table,
            CAST(b.bronze_id AS TEXT) AS bronze_row_ids,
            'v1' AS transform_version
        FROM {bronze_table} b
        WHERE b.{hours_col} IS NOT NULL AND TRIM(CAST(b.{hours_col} AS TEXT)) <> ''
    """

    cursor.execute(insert_sql, (bronze_table,))
    conn.commit()
    return cursor.rowcount


def create_gold_sample_view(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("DROP VIEW IF EXISTS gold_volunteer_hours_monthly")
    cursor.execute(
        """
        CREATE VIEW gold_volunteer_hours_monthly AS
        SELECT 
          v.tenant_id,
          substr(CAST(d.date_key AS TEXT), 1, 6) AS yyyymm,
          SUM(v.hours) AS total_hours
        FROM fact_volunteering v
        JOIN dim_date d ON d.date_key = v.date_key
        GROUP BY v.tenant_id, substr(CAST(d.date_key AS TEXT), 1, 6)
        ORDER BY yyyymm
        """
    )
    conn.commit()


