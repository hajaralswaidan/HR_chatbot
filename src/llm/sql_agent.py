import re
from typing import Callable, Tuple, Optional, Dict, Any

from src.db.sqlite_db import run_sql, TABLE_NAME, get_schema_sample
from src.llm.prompt import build_sql_prompt, build_answer_from_sql_prompt


FORBIDDEN = [
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "pragma", "replace", "truncate", "vacuum"
]


def _clean_sql(text: str) -> str:
    t = (text or "").strip()

    # remove ```sql fences if present
    t = re.sub(r"^```sql\s*", "", t, flags=re.I)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # keep first SELECT occurrence
    m = re.search(r"(select[\s\S]+)", t, flags=re.I)
    t = (m.group(1).strip() if m else t).strip()

    # strip trailing semicolon
    if t.endswith(";"):
        t = t[:-1].strip()

    return t


def _is_safe_sql(sql: str) -> bool:
    s = (sql or "").strip().lower()
    if not s.startswith("select"):
        return False

    # prevent multi statements
    if ";" in s:
        return False

    for bad in FORBIDDEN:
        if re.search(rf"\b{re.escape(bad)}\b", s):
            return False

    return True


def _ensure_limit(sql: str, limit: int) -> str:
    if re.search(r"\blimit\b", sql, flags=re.I):
        return sql
    return f"{sql.rstrip()} LIMIT {int(limit)}"


def generate_sql(question: str, history_text: str, llm_call: Callable[[str], str]) -> str:
    schema = get_schema_sample()
    prompt = build_sql_prompt(question, history_text, schema, TABLE_NAME)
    raw = llm_call(prompt)
    return _clean_sql(raw)


def answer_with_sql(
    question: str,
    history_text: str,
    llm_call: Callable[[str], str],
    max_rows: int = 20,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:

    sql = generate_sql(question, history_text, llm_call)

    if not _is_safe_sql(sql):
        return None, None, "SQL_NOT_SAFE (model did not produce a safe SELECT query)"

    sql = _ensure_limit(sql, max_rows)

    try:
        df = run_sql(sql)
    except Exception as e:
        return None, {"sql": sql, "rows": []}, f"SQL_ERROR: {e}"

    if df is None or df.empty:
        evidence = {"sql": sql, "rows": []}
        return "No matching rows found for this question.", evidence, None

    preview_md = df.head(10).to_markdown(index=False)
    prompt2 = build_answer_from_sql_prompt(question, sql, preview_md)
    answer = (llm_call(prompt2) or "").strip()

    evidence = {
        "sql": sql,
        "rows": df.head(5).to_dict(orient="records"),
    }
    return answer, evidence, None
