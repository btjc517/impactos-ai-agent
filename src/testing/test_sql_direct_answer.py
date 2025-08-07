import os
import sqlite3
import sys
from pathlib import Path

# Ensure src is on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from schema import DatabaseSchema
from query import QuerySystem


def setup_test_db(db_path: str):
    # Fresh DB
    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Initialize schema
    schema = DatabaseSchema(db_path)
    schema.initialize_database()

    # Insert one source
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sources (filename, file_type, file_size_bytes, processing_status)
            VALUES (?, ?, ?, ?)
            """,
            ("TakingCare_Benevity_Synthetic_Data.xlsx", "excel", 12345, "processed"),
        )
        source_id = cur.lastrowid

        # Insert multiple metrics with same name/unit/category to enable SQL SUM
        rows = [
            (source_id, "volunteering_hours", 100.0, "hours", "volunteering"),
            (source_id, "volunteering_hours", 50.0, "hours", "volunteering"),
            (source_id, "volunteering_hours", 37.5, "hours", "volunteering"),
        ]
        cur.executemany(
            """
            INSERT INTO impact_metrics (
                source_id, metric_name, metric_value, metric_unit, metric_category
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def run_test():
    db_path = "db/test_sql_direct.db"
    setup_test_db(db_path)

    qs = QuerySystem(db_path)

    question = "What is the total amount of volunteering hours?"
    answer = qs.query(question)

    print("Question:", question)
    print("Answer:")
    print(answer)

    # Expect deterministic SQL-first answer with 187.50 hours total
    expected_snippet = "187.50 hours"
    assert expected_snippet in answer, f"Expected '{expected_snippet}' in answer"
    assert answer.startswith("Total "), "Expected deterministic SQL-first answer starting with 'Total'"


if __name__ == "__main__":
    run_test() 