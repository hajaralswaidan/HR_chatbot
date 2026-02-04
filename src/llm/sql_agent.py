import re
from typing import Callable, Tuple, Optional, Dict, Any
import pandas as pd
from src.db.sqlite_db import run_sql, TABLE_NAME, get_schema_sample
from src.llm.prompt import build_sql_prompt, extract_first_select

# ================= Security Guardrails =================

FORBIDDEN_SQL = [
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "pragma", "replace", "truncate", "vacuum"
]

FORBIDDEN_QUESTION = [
    "delete", "drop", "truncate", "update", "insert", "alter", "create",
    "remove all", "wipe", "destroy", "erase",
    "حذف", "امسح", "اسقاط", "دروب", "احذف", "افرمت", "فرمت"
]

def _norm(text: str) -> str:
    return (text or "").strip().lower()

def _is_arabic(text: str) -> bool:
    # التحقق مما إذا كان النص يحتوي على حروف عربية
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))

def _question_is_unsafe(question: str) -> bool:
    q = _norm(question)
    return any(b in q for b in FORBIDDEN_QUESTION)

# ================= Formatting Toolkit =================

def _format_rate_percent(v) -> str:
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return str(v)

def _format_number(n) -> str:
    try:
        val = float(n)
        # تنسيق الآلاف وتقريب الكسور
        if val == int(val):
            return f"{int(val):,}"
        return f"{val:,.2f}"
    except Exception:
        return str(n)

def _arabic_gender(name: str) -> str:
    n = (name or "").strip().lower()
    if n == "male": return "الذكور (Male)"
    if n == "female": return "الإناث (Female)"
    return name

def _arabic_dept(name: str) -> str:
    n = (name or "").strip().lower()
    mapping = {
        "sales": "المبيعات (Sales)",
        "human resources": "الموارد البشرية (Human Resources)",
        "research & development": "البحث والتطوير (Research & Development)",
    }
    return mapping.get(n, name)

def _format_df_direct(df: pd.DataFrame, question: str = "") -> str:
    ar = _is_arabic(question)
    if df is None or df.empty:
        return "لا توجد بيانات مطابقة." if ar else "No matching data found."

    # 1. النتائج الفردية (Single Value)
    if df.shape == (1, 1):
        col_raw = str(df.columns[0])
        col = col_raw.lower()
        val = df.iloc[0, 0]
        formatted_val = _format_number(val)

        if "attrition_count" in col:
            return (f"عدد الموظفين المستقيلين: **{formatted_val}**" if ar else f"Left Count: **{formatted_val}**")
        if "total_employees" in col or "count" in col:
            return (f"إجمالي عدد الموظفين: **{formatted_val}**" if ar else f"Total employees: **{formatted_val}**")
        if "avg_age" in col:
            return (f"متوسط العمر: **{float(val):.1f} سنة**" if ar else f"Average age: **{float(val):.1f} years**")
        if any(x in col for x in ["income", "salary", "wage"]):
            return (f"متوسط الدخل الشهري: **{formatted_val}**" if ar else f"Average Monthly Income: **{formatted_val}**")
        
        return f"{col_raw}: **{formatted_val}**"

    # 2. الجداول والتحليلات (تحويل الصفوف إلى أسطر رأسية مرتبة)
    if df.shape[1] >= 2:
        lines = []
        c1_low = str(df.columns[0]).lower()
        c2_low = str(df.columns[1]).lower()

        for row in df.values:
            name, value = row[0], row[1]
            if ar:
                if "gender" in c1_low: name = _arabic_gender(str(name))
                if "department" in c1_low: name = _arabic_dept(str(name))
            
            val_str = _format_rate_percent(value) if "rate" in c2_low else _format_number(value)
            # النقطة \n تضمن أن كل معلومة تظهر في سطر مستقل تماماً
            lines.append(f"• {name}: **{val_str}**")
            
        header = "**نتائج التحليل:**" if ar else "**Analysis Results:**"
        return header + "\n\n" + "\n".join(lines)

    return df.head(10).to_string(index=False)

# ================= Intelligence & Follow-up =================

def _extract_last_rates(history_text: str) -> Optional[Dict[str, float]]:
    if not history_text: return None
    # Regex محدث ليتوافق مع التنسيق الجديد بالنقاط والنجوم
    pattern = r"([^:•\n]+)\s*:\s*\**([0-9]+(?:\.[0-9]+)?)\s*%\**"
    matches = re.findall(pattern, history_text)
    if not matches: return None
    data = {}
    for name, val in matches:
        clean_name = re.sub(r"\(.*?\)|[\u0600-\u06FF]", "", name).strip()
        if not clean_name: clean_name = name.strip()
        data[clean_name] = float(val)
    return data

def _followup_high_low(question: str, history_text: str) -> Optional[str]:
    q = _norm(question)
    ar = _is_arabic(question)
    data = _extract_last_rates(history_text)
    if not data: return None
    
    if re.search(r"highest|max|الأعلى|اعلى|أكبر", q):
        k = max(data, key=data.get)
        return f"{'الأعلى' if ar else 'Highest'}: **{k}** ({data[k]:.2f}%)"
    if re.search(r"lowest|min|الأقل|اقل|أصغر", q):
        k = min(data, key=data.get)
        return f"{'الأقل' if ar else 'Lowest'}: **{k}** ({data[k]:.2f}%)"
    return None

# ================= Core SQL Logic =================

def _rule_based_sql(question: str) -> Optional[str]:
    q = _norm(question)
    
    # تحويل المسميات العربية إلى إنجليزية لقاعدة البيانات
    dept = None
    if any(k in q for k in ["بحث", "تطوير", "r&d", "research"]): dept = "Research & Development"
    elif any(k in q for k in ["موارد", "hr", "human"]): dept = "Human Resources"
    elif any(k in q for k in ["مبيعات", "sales"]): dept = "Sales"

    if re.search(r"نسبة الاستقالات حسب القسم|attrition.*department", q):
        return f"SELECT Department, AVG(CASE WHEN Attrition='Yes' THEN 1.0 ELSE 0 END) AS attrition_rate FROM {TABLE_NAME} GROUP BY Department ORDER BY attrition_rate DESC"
    
    if re.search(r"نسبة الاستقالات حسب الجنس|attrition.*gender", q):
        return f"SELECT Gender, AVG(CASE WHEN Attrition='Yes' THEN 1.0 ELSE 0 END) AS attrition_rate FROM {TABLE_NAME} GROUP BY Gender ORDER BY attrition_rate DESC"

    if dept:
        if re.search(r"left|غادر|استقال|attrition", q):
            return f"SELECT COUNT(*) AS attrition_count FROM {TABLE_NAME} WHERE Department='{dept}' AND Attrition='Yes'"
        return f"SELECT COUNT(*) AS employee_count FROM {TABLE_NAME} WHERE Department='{dept}'"

    if re.search(r"متوسط العمر|average age", q):
        return f"SELECT ROUND(AVG(Age), 2) AS avg_age FROM {TABLE_NAME}"

    if re.search(r"عدد الموظفين|total employees|count", q):
        return f"SELECT COUNT(*) AS total_employees FROM {TABLE_NAME}"

    return None

def _is_safe_sql(sql: str) -> bool:
    s = _norm(sql)
    if not s or not s.startswith("select"): return False
    return not any(re.search(rf"\b{re.escape(bad)}\b", s) for bad in FORBIDDEN_SQL)

def answer_with_sql(question: str, history_text: str, llm_call: Callable[[str], str], max_rows: int = 20) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    if _question_is_unsafe(question): return None, None, "SQL_NOT_SAFE"
    
    fu = _followup_high_low(question, history_text)
    if fu: return fu, {"sql": "Internal comparison logic", "rows": []}, None

    sql = _rule_based_sql(question)
    if not sql:
        schema = get_schema_sample()
        prompt = build_sql_prompt(question, history_text, schema, TABLE_NAME)
        sql = extract_first_select(llm_call(prompt))

    if not _is_safe_sql(sql): return None, None, "SQL_NOT_SAFE"
    
    try:
        df = run_sql(sql)
        answer = _format_df_direct(df, question)
        return answer, {"sql": sql, "rows": df.head(5).to_dict(orient="records")}, None
    except Exception as e:
        return None, {"sql": sql}, f"SQL_ERROR: {e}"