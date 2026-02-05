import re
import json

def extract_first_select(text: str) -> str:
    """
    Robust SQL extraction:
    1) JSON: {"sql":"..."}
    2) fenced code ```sql ... ```
    3) first SELECT or WITH..SELECT block in plain text
    Then cuts any multi-statement after first ';'
    """
    t = (text or "").strip()
    if not t:
        return ""

    # 1) JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and obj.get("sql"):
            sql = str(obj["sql"]).strip()
            return sql.split(";", 1)[0].strip()
        if isinstance(obj, dict) and obj.get("need_followup"):
            return ""
    except Exception:
        pass

    # 2) fenced code
    m = re.search(r"```(?:sql|sqlite)?\s*([\s\S]+?)```", t, flags=re.I)
    if m:
        sql = m.group(1).strip()
        return sql.split(";", 1)[0].strip()

    # 3) fallback: find first SELECT or WITH ... SELECT
    m2 = re.search(r"\b(with\b[\s\S]*?\bselect\b|select\b)[\s\S]*", t, flags=re.I)
    if not m2:
        return ""

    sql = m2.group(0).strip()
    sql = sql.split(";", 1)[0].strip()
    return sql


def build_sql_prompt(question: str, history: str, schema: str, table_name: str) -> str:
    """
    - Uses real schema input
    - Forces JSON output: {"sql":"..."} OR {"need_followup":true,"followup":"..."}
    - Arabic/English aware
    """
    ar = bool(re.search(r"[\u0600-\u06FF]", question or ""))

    system = (
        "أنت خبير SQLite. مهمتك توليد استعلام SQL واحد فقط للقراءة.\n"
        "مسموح فقط SELECT أو WITH..SELECT.\n"
        "ممنوع: INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA/ATTACH وأي multi-statement.\n"
        "إذا السؤال ناقص أو غامض: ارجع JSON بهذا الشكل فقط: "
        "{\"need_followup\": true, \"followup\": \"...\"}\n"
        "وإذا واضح: ارجع JSON بهذا الشكل فقط: {\"sql\": \"...\"}\n"
        "لا تخترع أعمدة. استخدم الـ schema.\n"
        if ar else
        "You are a SQLite expert. Generate ONE read-only SQL query.\n"
        "Only SELECT or WITH..SELECT is allowed.\n"
        "Forbidden: INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA/ATTACH and multi-statements.\n"
        "If ambiguous: return ONLY JSON: "
        "{\"need_followup\": true, \"followup\": \"...\"}\n"
        "If answerable: return ONLY JSON: {\"sql\": \"...\"}\n"
        "Do not invent columns. Use the schema.\n"
    )

    return f"""{system}
TABLE: {table_name}

SCHEMA (sample / columns):
{schema}

CHAT HISTORY (brief):
{history}

HINTS:
- Attrition uses 'Yes'/'No' for employees who left.
- Rate definition: AVG(CASE WHEN Attrition='Yes' THEN 1.0 ELSE 0 END)
- Use GROUP BY for breakdown questions (Department, Gender, JobRole, etc.).
- Prefer explicit column names.
- Add LIMIT if listing rows.

USER QUESTION:
{question}
"""
