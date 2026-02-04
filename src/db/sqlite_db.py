import sqlite3
from pathlib import Path
import pandas as pd

from src.data_loader import load_hr_data

DB_PATH = Path("data") / "hr.sqlite"
TABLE_NAME = "employees"


def init_db(force_rebuild: bool = False) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists() and not force_rebuild:
        return

    df = load_hr_data()
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    finally:
        conn.close()


def get_conn() -> sqlite3.Connection:
    init_db()
    return sqlite3.connect(DB_PATH)


def run_sql(sql: str) -> pd.DataFrame:
    """
    MUST return pandas.DataFrame only.
    """
    conn = get_conn()
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def get_schema_sample() -> str:
    df = load_hr_data()
    cols = [f"- {c}" for c in df.columns]
    return f"Table: {TABLE_NAME}\nColumns:\n" + "\n".join(cols)
