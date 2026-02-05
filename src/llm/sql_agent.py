import re
from typing import Callable, Tuple, Optional, Dict, Any
import pandas as pd

from src.db.sqlite_db import run_sql, TABLE_NAME, get_schema_sample
from src.llm.prompt import build_sql_prompt, extract_first_select

# ================= Security Guardrails =================

FORBIDDEN_SQL = [
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "pragma", "replace", "truncate", "vacuum", "with"
]

FORBIDDEN_QUESTION = [
    "delete", "drop", "truncate", "update", "insert", "alter", "create",
    "remove all", "wipe", "destroy", "erase",
    "حذف", "امسح", "اسقاط", "دروب", "احذف", "افرمت", "فرمت"
]


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _question_is_unsafe(question: str) -> bool:
    q = _norm(question)
    return any(b in q for b in FORBIDDEN_QUESTION)


# ================= Formatting =================

def _format_number(n) -> str:
    try:
        val = int(float(n))
        return f"{val:,}"
    except Exception:
        return str(n)


def _format_df(df: pd.DataFrame, question: str) -> str:
    ar = _is_arabic(question)

    if df.empty:
        return "لا توجد بيانات." if ar else "No data found."

    if df.shape == (1, 1):
        val = df.iloc[0, 0]
        return (
            f"إجمالي عدد الموظفين: **{_format_number(val)}**"
            if ar
            else f"Total employees: **{_format_number(val)}**"
        )

    # attrition rate lists
    lines = []
    for name, value in df.values:
        lines.append(f"• {name}: **{value*100:.2f}%**")

    header = "نتائج التحليل:" if ar else "Analysis Results:"
    return f"**{header}**\n\n" + "\n".join(lines)


# ================= Follow-up =================

def _extract_last_rates(history: str) -> Optional[Dict[str, float]]:
    if not history:
        return None

    pattern = r"•\s*(.+?):\s*\*\*(\d+(\.\d+)?)%"
    matches = re.findall(pattern, history)
    if not matches:
        return None

    return {m[0]: float(m[1]) for m in matches}


def _followup_high_low(question: str, history: str) -> Optional[str]:
    q = _norm(question)
    data = _extract_last_rates(history)
    if not data:
        return None

    if re.search(r"highest|max|أعلى|الاعلى|اعلى", q):
        k = max(data, key=data.get)
        return f"Highest: **{k} ({data[k]:.2f}%)**"

    if re.search(r"lowest|min|أقل|الاقل|اقل", q):
        k = min(data, key=data.get)
        return f"Lowest: **{k} ({data[k]:.2f}%)**"

    return None


# ================= Rule-based SQL  =================

def _rule_based_sql(question: str) -> Optional[str]:
    q = _norm(question)

    #  عدّ الموظفين — بدون LLM
    if re.search(r"how many employees|total employees|عدد الموظفين|كم عدد الموظفين", q):
        return f"SELECT COUNT(*) FROM {TABLE_NAME}"

    if re.search(r"attrition rate by department|نسبة الاستقالات حسب القسم", q):
        return f"""
        SELECT Department,
               AVG(CASE WHEN Attrition='Yes' THEN 1.0 ELSE 0 END) AS attrition_rate
        FROM {TABLE_NAME}
        GROUP BY Department
        ORDER BY attrition_rate DESC
        """

    if re.search(r"attrition rate by gender|نسبة الاستقالات حسب الجنس", q):
        return f"""
        SELECT Gender,
               AVG(CASE WHEN Attrition='Yes' THEN 1.0 ELSE 0 END) AS attrition_rate
        FROM {TABLE_NAME}
        GROUP BY Gender
        ORDER BY attrition_rate DESC
        """

    return None


# ================= SQL Safety =================

def _is_safe_sql(sql: str) -> bool:
    if not sql:
        return False

    s = _norm(sql)

    # لازم يبدأ بـ SELECT فقط
    if not s.startswith("select"):
        return False

    # منع WITH أو أي شي خطير
    return not any(
        re.search(rf"\b{bad}\b", s)
        for bad in FORBIDDEN_SQL
    )


# ================= Main Entry =================

def answer_with_sql(
    question: str,
    history_text: str,
    llm_call: Callable[[str], str],
    max_rows: int = 20
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:

    if _question_is_unsafe(question):
        return None, None, "SQL_NOT_SAFE"

    # follow-up (highest / lowest)
    follow = _followup_high_low(question, history_text)
    if follow:
        return follow, {"sql": "follow-up logic", "rows": []}, None

    # rule-based first
    sql = _rule_based_sql(question)

    # fallback to LLM
    if not sql:
        schema = get_schema_sample()
        prompt = build_sql_prompt(question, history_text, schema, TABLE_NAME)
        llm_out = llm_call(prompt)
        sql = extract_first_select(llm_out)

    if not _is_safe_sql(sql):
        return None, None, "SQL_NOT_SAFE"

    try:
        df = run_sql(sql)
        answer = _format_df(df, question)
        return answer, {"sql": sql, "rows": df.head(max_rows).to_dict("records")}, None

    except Exception as e:
        return None, {"sql": sql}, f"SQL_ERROR: {e}"
