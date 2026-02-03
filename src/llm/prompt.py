import re

EVIDENCE_KEYWORDS = [
    "evidence", "stats", "numbers", "percentage", "breakdown", "rates", "proof",
    "دليل", "إثبات", "اثبات", "أرقام", "ارقام", "نسبة", "تفصيل"
]


def wants_evidence(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in EVIDENCE_KEYWORDS)


def build_rag_prompt(question: str, context: str, show_evidence: bool) -> str:
    if show_evidence:
        fmt = "Return ONLY: (1) short answer, (2) evidence bullets with exact numbers from the context."
    else:
        fmt = "Return ONLY: a short clear answer (1–2 sentences) based strictly on the context."

    return f"""
You are an HR analytics assistant.

IMPORTANT RULES (STRICT):
- Answer ONLY using the provided CONTEXT.
- The context represents the HR dataset.
- DO NOT use any external knowledge.
- DO NOT explain concepts theoretically.
- DO NOT give examples, formulas, or general advice.
- If the answer is NOT explicitly supported by the context, respond EXACTLY with:
  "I can’t answer this from the HR dataset."

CONTEXT:
{context}

QUESTION:
{question}

FORMAT:
- {fmt}
- No introductions.
- No conclusions.
- No assumptions.
- No extra text.

FINAL:
""".strip()



def build_sql_prompt(question: str, history: str, schema: str, table_name: str) -> str:
    return f"""
You are a data analyst. Convert the user's question into ONE safe SQLite SQL query.

SCHEMA:
{schema}

RULES:
- Output SQL ONLY (no explanation, no markdown, no code fences).
- Use table name exactly: {table_name}
- Use SELECT only. Never write INSERT/UPDATE/DELETE/DROP/ALTER/PRAGMA/ATTACH/CREATE.
- If filtering Attrition, remember values are 'Yes'/'No'.
- If the user asks for "rate", compute it (e.g., AVG(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END)).
- Use exact column names from SCHEMA.
- If question is ambiguous, make a reasonable assumption.
- Always LIMIT 20 when returning rows.

CHAT HISTORY (for context):
{history}

QUESTION:
{question}
""".strip()


def build_answer_from_sql_prompt(question: str, sql: str, table_preview_md: str) -> str:
    return f"""
You are an HR analytics assistant.

STRICT RULES:
- Answer ONLY using the SQL result preview.
- DO NOT use external knowledge.
- DO NOT explain methodology.
- DO NOT add recommendations.
- If the result does not answer the question, respond EXACTLY with:
  "I can’t answer this from the HR dataset."

SQL:
{sql}

RESULT:
{table_preview_md}

QUESTION:
{question}

ANSWER FORMAT:
- Start with a direct answer.
- Use numbers exactly as shown.
- If table exists, summarize in 1–2 sentences only.
""".strip()


def smart_fallback(user_question: str, show_evidence: bool) -> str:
    return "I can’t answer this from the HR dataset."

def finalize_answer(user_question: str, model_text: str, show_evidence: bool) -> str:
    """
    Final safety layer:
    - Strip model artifacts
    - Return clean answer only
    """

    if not model_text:
        return "I can’t answer this from the HR dataset."

    t = model_text.strip()

    # Remove common model prefixes
    for marker in ["FINAL:", "FINAL_ANSWER:", "assistant:"]:
        if marker in t:
            t = t.split(marker, 1)[-1].strip()

    # Remove generation tokens (Qwen / HF)
    t = t.split("<|im_end|>")[0].strip()

    # Absolute safety
    if not t:
        return "I can’t answer this from the HR dataset."

    return t
