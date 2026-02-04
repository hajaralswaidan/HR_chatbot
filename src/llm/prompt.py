import re

def extract_first_select(text: str) -> str:
    t = (text or "").strip()
    # تنظيف الماركدوان والتعليقات
    t = re.sub(r"```sql|```", "", t, flags=re.I).strip()
    
    # البحث عن أول SELECT وتجاهل أي نص توضيحي قبله
    match = re.search(r"SELECT\b.*", t, flags=re.I | re.DOTALL)
    if not match: return ""
    
    sql = match.group(0).strip()
    # الحزم في أخذ أول جملة فقط حتى لو الموديل كتب عشر جمل
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
    else:
        # إذا لم يضع فاصلة منقوطة، نبحث عن كلمة SELECT ثانية ونقص عندها
        second_select = re.search(r".*?(?=\bSELECT\b)", sql[6:], flags=re.I | re.DOTALL)
        if second_select:
            sql = "SELECT" + second_select.group(0).strip()
            
    return sql

def build_sql_prompt(question: str, history: str, schema: str, table_name: str) -> str:
    return f"""
You are a SQLite expert. Output ONLY a single SQL query.
TABLE: {table_name}
COLUMNS: Age, Attrition (Yes/No), Department, Gender, MonthlyIncome, JobRole.
DEPARTMENTS: 'Sales', 'Research & Development', 'Human Resources'.

STRICT RULES:
1. Return ONLY the SQL string. No explanations.
2. If the user asks for a department not in the list (like IT), use the closest one or return 0.
3. Use Attrition='Yes' for people who left.
4. For rate: AVG(CASE WHEN Attrition='Yes' THEN 1.0 ELSE 0 END).

QUESTION: {question}
SQL:"""