import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient

# ✅ Ensure project root is importable (fixes ModuleNotFoundError on Windows)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.retriever import Retriever
from src.data_loader import load_hr_data
from src.llm.prompt import (
    build_rag_prompt,
    finalize_answer,
    wants_evidence,
    smart_fallback,
)
from src.llm.local_qwen import LocalQwen


# -------------------- Page --------------------
st.set_page_config(page_title="HR Analytics Chatbot", layout="wide")

# ---------------- Session State ----------------
if "model_mode" not in st.session_state:
    st.session_state.model_mode = "Cloud"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# ✅ Store last computed metrics (scope-aware followups)
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {
        "scope": None,
        "values": [],
        "title": None,
    }


# -------------------- CSS --------------------
def load_css(file_name: str):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")


# ---------------- Cached resources ----------------
@st.cache_resource
def get_retriever():
    return Retriever()

@st.cache_resource
def get_local_llm():
    llm = LocalQwen(model_path="models/falcon-7b-instruct")
    llm.load()
    return llm

@st.cache_data
def get_df() -> pd.DataFrame:
    return load_hr_data()


# ✅ Sentiment pipeline (Hugging Face) + Feature Engineering
@st.cache_resource
def get_sentiment_pipeline():
    from transformers import pipeline
    # Lightweight English sentiment model (fast + stable)
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_data
def compute_sentiment_by_group(group_col: str) -> List[Tuple[str, float]]:
    df = get_df().copy()
    if group_col not in df.columns:
        return []

    # Feature Engineering: create a text column from structured fields
    df["text_for_sentiment"] = (
        "JobRole: " + df.get("JobRole", "").astype(str) +
        " | Department: " + df.get("Department", "").astype(str) +
        " | OverTime: " + df.get("OverTime", "").astype(str)
    )

    pipe = get_sentiment_pipeline()

    out: List[Tuple[str, float]] = []
    for name, g in df.groupby(group_col):
        # sample per group to keep it fast
        sample = g["text_for_sentiment"].dropna().astype(str).head(60).tolist()
        if not sample:
            continue

        preds = pipe(sample)
        pos = sum(1 for p in preds if str(p.get("label", "")).upper() == "POSITIVE")
        rate = (pos / len(preds)) * 100.0
        out.append((str(name), float(rate)))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


# ---------------- Messages ----------------
WELCOME_MSG = """
<div style="color:#666; font-size:14px; line-height:1.6;">
<b>HR Analytics Chatbot</b><br><br>
An AI- interface for exploring and analyzing HR data using both local and cloud-based language models.
</div>
"""


# ---------------- Helpers ----------------
def new_chat():
    st.session_state.messages = []
    st.session_state.is_generating = False
    st.session_state.last_metrics = {"scope": None, "values": [], "title": None}


def cloud_is_configured() -> bool:
    token = None
    try:
        token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        token = None
    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    return bool(token)

def get_cloud_token():
    try:
        tok = st.secrets.get("HF_TOKEN", None)
    except Exception:
        tok = None
    if not tok:
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    return tok


SYSTEM_PROMPT = """
You are a professional HR analytics assistant.

Light guardrails:
- Use the dataset results and the latest computed metrics in the chat.
- If the user asks highest/lowest/top/bottom/rank, answer from the latest computed metric set (scope-aware).
- Do NOT say "need more context" if the needed numbers exist.
- If truly ambiguous (no recent metric set), ask ONE short clarifying question.
- Never invent numbers.
- Keep answers concise and data-driven.
""".strip()


def clean_text_for_llm(text: str) -> str:
    t = re.sub(r"<[^>]+>", " ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_cloud_messages(user_input: str, max_turns: int = 8):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    history = st.session_state.messages[-max_turns:] if st.session_state.messages else []
    for m in history:
        role = m.get("role", "user")
        content = clean_text_for_llm(m.get("content", ""))
        if content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_input})
    return msgs


def run_cloud_chat(messages, model_id: str = "Qwen/Qwen2.5-7B-Instruct") -> str:
    token = get_cloud_token()
    if not token:
        return "Cloud token not found. Please set HF_TOKEN in secrets.toml or environment variables."

    client = InferenceClient(model=model_id, token=token)
    try:
        res = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
        )
        return (res.choices[0].message.content or "").strip()
    except Exception:
        try:
            last_user = messages[-1]["content"] if messages else ""
            out = client.text_generation(last_user, max_new_tokens=512, temperature=0.2)
            return (out or "").strip()
        except Exception as e:
            return f"Cloud error: {e}"


def looks_like_template_or_echo(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if t in ["undefined", "null", "none"]:
        return True
    return len(t) < 8


def is_analytic(user_input: str) -> bool:
    q = (user_input or "").lower()
    keywords = [
        "average", "mean", "count", "how many", "percentage", "rate", "attrition",
        "top", "lowest", "highest", "group by", "by department", "by job role",
        "sum", "median", "distribution", "overtime", "job satisfaction", "years",
        "sentiment"
    ]
    return any(k in q for k in keywords)


#  Interpretation / Recommendations / Consultant questions should go to LLM
def is_interpretation_question(user_input: str) -> bool:
    q = (user_input or "").lower()
    return any(k in q for k in [
        "what does", "indicate", "indicates", "mean", "means",
        "considered high", "is this high", "is it high",
        "why do you think", "why is", "reason", "reasons",
        "recommendation", "recommend", "reduce attrition",
        "prioritize", "retention efforts",
        "hr consultant", "as a consultant", "suggest"
    ])


def is_insight_question(user_input: str) -> bool:
    q = (user_input or "").lower()
    return any(k in q for k in ["short insight", "give me a short insight", "insight", "summarize insights"])


# ---------------- Pandas analytics ----------------
def _attrition_yes_mask(df: pd.DataFrame) -> pd.Series:
    if "Attrition" not in df.columns:
        return pd.Series([False] * len(df))
    return df["Attrition"].astype(str).str.lower().eq("yes")


def compute_attrition_rate(df: pd.DataFrame) -> float:
    yes = _attrition_yes_mask(df)
    return float(yes.mean() * 100.0)


def compute_attrition_by_group(df: pd.DataFrame, group_col: str) -> List[Tuple[str, float]]:
    if group_col not in df.columns or "Attrition" not in df.columns:
        return []

    temp = df.copy()
    temp[group_col] = temp[group_col].astype(str)

    rates = (
        temp.groupby(group_col, dropna=False)["Attrition"]
        .apply(lambda s: s.astype(str).str.lower().eq("yes").mean() * 100.0)
        .sort_values(ascending=False)
    )
    out = [(str(k), float(v)) for k, v in rates.items()]
    return out


def set_last_metrics(scope: str, title: str, values: List[Tuple[str, float]]):
    st.session_state.last_metrics = {"scope": scope, "title": title, "values": values}


def format_metric_list(values: List[Tuple[str, float]], top_n: Optional[int] = None) -> str:
    if not values:
        return "I can’t compute this from the dataset."
    items = values[:top_n] if top_n else values
    return "  ".join([f"**{k}:** {v:.1f}%" for k, v in items])


def compute_job_satisfaction_attrition(df: pd.DataFrame) -> List[Tuple[str, float]]:
    if "JobSatisfaction" not in df.columns or "Attrition" not in df.columns:
        return []
    temp = df.copy()
    temp["JobSatisfaction"] = temp["JobSatisfaction"].astype(str)
    vals = compute_attrition_by_group(temp, "JobSatisfaction")
    label_map = {"1": "Low (1)", "2": "Medium (2)", "3": "High (3)", "4": "Very High (4)"}
    return [(label_map.get(k, k), v) for k, v in vals]


def compute_years_bucket_attrition(df: pd.DataFrame) -> List[Tuple[str, float]]:
    if "YearsAtCompany" not in df.columns or "Attrition" not in df.columns:
        return []
    temp = df.copy()
    y = pd.to_numeric(temp["YearsAtCompany"], errors="coerce").fillna(-1)
    buckets = pd.cut(
        y,
        bins=[-1, 2, 5, 10, 1000],
        labels=["0–2 years", "3–5 years", "6–10 years", "11+ years"],
        include_lowest=True
    )
    temp["YearsBucket"] = buckets.astype(str)
    vals = compute_attrition_by_group(temp, "YearsBucket")
    return vals


# ---------------- Follow-up routing  ----------------
SCOPE_KEYWORDS = {
    "department": ["department", "dept"],
    "job_role": ["job role", "jobrole", "role"],
    "overtime": ["overtime", "over time"],
    "marital": ["marital", "single", "married", "divorced"],
    "jobsat": ["job satisfaction", "satisfaction"],
    "years": ["years at the company", "years at company", "years", "tenure"],
    "sentiment_department": ["sentiment", "department sentiment"],
    "sentiment_jobrole": ["sentiment", "job role sentiment"],
}

GROUP_COL_FOR_SCOPE = {
    "department": "Department",
    "job_role": "JobRole",
    "overtime": "OverTime",
    "marital": "MaritalStatus",
}

def infer_scope(user_input: str) -> Optional[str]:
    q = (user_input or "").lower()
    for scope, kws in SCOPE_KEYWORDS.items():
        if any(k in q for k in kws):
            return scope
    return None


def is_followup_extreme(user_input: str) -> bool:
    q = (user_input or "").lower().strip()
    return any(k in q for k in ["highest", "lowest", "top", "bottom", "maximum", "minimum", "rank", "ordered"])


def answer_followup_from_last(user_input: str) -> str:
    q = (user_input or "").lower().strip()
    last = st.session_state.last_metrics
    values = last.get("values", []) or []
    if not values:
        return ""

    if "rank" in q or "ordered" in q or "order" in q:
        lines = [f"- {k}: {v:.1f}%" for k, v in values]
        return f"**{last.get('title','Rank')} (highest → lowest):**\n" + "\n".join(lines)

    if any(k in q for k in ["highest", "top", "maximum"]):
        k, v = max(values, key=lambda x: x[1])
        return f"**Highest ({last.get('scope','')}):** {k} ({v:.1f}%)"

    if any(k in q for k in ["lowest", "bottom", "minimum"]):
        k, v = min(values, key=lambda x: x[1])
        return f"**Lowest ({last.get('scope','')}):** {k} ({v:.1f}%)"

    return ""


def insight_from_last_metrics() -> str:
    last = st.session_state.last_metrics
    values = last.get("values", []) or []
    scope = last.get("scope")
    title = last.get("title") or "latest analysis"
    if not values:
        return ""

    hi_k, hi_v = max(values, key=lambda x: x[1])
    lo_k, lo_v = min(values, key=lambda x: x[1])
    return (
        f"**Insight ({scope}):** Based on the **{title}**, the highest is **{hi_k} ({hi_v:.1f}%)**, "
        f"and the lowest is **{lo_k} ({lo_v:.1f}%)**."
    )


# ----------------  ----------------
def pandas_direct_answer(user_input: str) -> str:
    df = get_df()
    q = (user_input or "").lower()

    #  interpretation/recommendations- LLM
    if is_interpretation_question(user_input):
        return ""

    #  Insight only from last metrics 
    if is_insight_question(user_input):
        insight = insight_from_last_metrics()
        return insight if insight else ""

    # totals
    if "how many employees are there" in q or ("total employees" in q):
        return f"**Total Employees:** {len(df):,}"

    if "overall attrition rate" in q or ("attrition rate" in q and "overall" in q):
        rate = compute_attrition_rate(df)
        return f"**Overall Attrition Rate:** {rate:.2f}%"

    if "how many employees left" in q or ("employees left" in q) or ("left the company" in q):
        if "Attrition" in df.columns:
            left = int(_attrition_yes_mask(df).sum())
            return f"**Employees who left (Attrition=Yes):** {left:,}"
        return "I can’t compute this because 'Attrition' column is missing."

    #  Average Age 
    if "average age" in q or ("average" in q and "age" in q):
        if "Age" in df.columns:
            avg_age = pd.to_numeric(df["Age"], errors="coerce").mean()
            return f"**Average Age:** {avg_age:.1f}"
        return "I can’t compute this because 'Age' column is missing."

    if "average monthly income" in q or ("average" in q and "monthly income" in q):
        if "MonthlyIncome" in df.columns:
            avg_income = pd.to_numeric(df["MonthlyIncome"], errors="coerce").mean()
            return f"**Average Monthly Income:** ${avg_income:,.0f}"
        return "I can’t compute this because 'MonthlyIncome' column is missing."

    # group rates
    if "attrition rate by department" in q or ("attrition" in q and "by department" in q):
        vals = compute_attrition_by_group(df, "Department")
        set_last_metrics("department", "Attrition rate by department", vals)
        return format_metric_list(vals)

    if "attrition rate by job role" in q or ("attrition" in q and "by job role" in q):
        vals = compute_attrition_by_group(df, "JobRole")
        set_last_metrics("job_role", "Attrition rate by job role", vals)
        return format_metric_list(vals, top_n=None)

    if "does overtime affect attrition" in q or ("attrition" in q and "overtime" in q):
        vals = compute_attrition_by_group(df, "OverTime")
        set_last_metrics("overtime", "Attrition rate by overtime", vals)
        return format_metric_list(vals)

    if "compare attrition between employees who work overtime" in q:
        vals = compute_attrition_by_group(df, "OverTime")
        set_last_metrics("overtime", "Attrition rate by overtime", vals)
        d = dict(vals)
        if "Yes" in d and "No" in d:
            return f"Employees who work overtime have a higher attrition rate (**Yes: {d['Yes']:.1f}%**) compared to those who don’t (**No: {d['No']:.1f}%**)."
        return format_metric_list(vals)

    if "is attrition higher among employees with low job satisfaction" in q or ("job satisfaction" in q and "attrition" in q):
        vals = compute_job_satisfaction_attrition(df)
        set_last_metrics("jobsat", "Attrition rate by job satisfaction", vals)
        return format_metric_list(vals)

    if "is attrition higher for employees with fewer years" in q or ("years at the company" in q and "attrition" in q):
        vals = compute_years_bucket_attrition(df)
        set_last_metrics("years", "Attrition rate by tenure (YearsAtCompany)", vals)
        return format_metric_list(vals)

    # Alias
    if "attrition rate by years at company" in q or ("attrition" in q and "by years at company" in q):
        vals = compute_years_bucket_attrition(df)
        set_last_metrics("years", "Attrition rate by tenure (YearsAtCompany)", vals)
        return format_metric_list(vals)

    # Sentiment questions 
    if "overall sentiment" in q or ("sentiment" in q and "overall" in q):
        dept = compute_sentiment_by_group("Department")
        role = compute_sentiment_by_group("JobRole")

        # store last sentiment metrics (default: department)
        set_last_metrics("sentiment_department", "Sentiment by department (% positive)", dept)

        if dept:
            top_dept = dept[0]
            bottom_dept = dept[-1]
            return (
                f"**Sentiment (Positive %) — Department:** Highest: **{top_dept[0]} ({top_dept[1]:.1f}%)**, "
                f"Lowest: **{bottom_dept[0]} ({bottom_dept[1]:.1f}%)**."
            )
        return "I couldn’t compute sentiment from the engineered text."

    if "compare sentiment" in q and ("research" in q and "sales" in q):
        dept = compute_sentiment_by_group("Department")
        d = dict(dept)
        sales = d.get("Sales")
        rd = d.get("Research & Development")
        if sales is not None and rd is not None:
            set_last_metrics("sentiment_department", "Sentiment by department (% positive)", dept)
            return f"**Sentiment (Positive %):** Sales: **{sales:.1f}%** | R&D: **{rd:.1f}%**"
        return "I couldn’t compute sentiment comparison for Sales vs R&D."

    if "sentiment more positive" in q and ("technical" in q or "sales" in q):
        role = compute_sentiment_by_group("JobRole")
        r = dict(role)

        tech_names = ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Research Director"]
        sales_names = ["Sales Executive", "Sales Representative"]

        tech_vals = [r[n] for n in tech_names if n in r]
        sales_vals = [r[n] for n in sales_names if n in r]

        if tech_vals and sales_vals:
            tech = sum(tech_vals) / len(tech_vals)
            sales = sum(sales_vals) / len(sales_vals)
            more = "Technical roles" if tech > sales else "Sales roles"
            set_last_metrics("sentiment_jobrole", "Sentiment by job role (% positive)", role)
            return f"{more} are more positive (**Technical: {tech:.1f}%**, **Sales: {sales:.1f}%**)."

        return "I couldn’t compute the technical vs sales sentiment comparison."

    # compare sales vs r&d
    if "compare attrition between sales and r&d" in q or ("sales" in q and "r&d" in q and "attrition" in q):
        vals = compute_attrition_by_group(df, "Department")
        set_last_metrics("department", "Attrition rate by department", vals)
        d = dict(vals)
        sales = d.get("Sales")
        rd = d.get("Research & Development") or d.get("Research & Development ")
        if sales is not None and rd is not None:
            higher = "Sales" if sales > rd else "Research & Development"
            return f"{higher} has a higher attrition rate (**Sales: {sales:.1f}%**, **R&D: {rd:.1f}%**)."
        return format_metric_list(vals)

    # Follow-up extremes (scope-aware)
    if is_followup_extreme(user_input):
        intended = infer_scope(user_input)
        last_scope = st.session_state.last_metrics.get("scope")

        # recompute if user explicitly changes scope for structured groups
        if intended and intended in GROUP_COL_FOR_SCOPE and intended != last_scope:
            col = GROUP_COL_FOR_SCOPE[intended]
            vals = compute_attrition_by_group(df, col)
            set_last_metrics(intended, f"Attrition rate by {intended.replace('_',' ')}", vals)

        if not st.session_state.last_metrics.get("values"):
            return "Do you mean the highest/lowest **by department**, **job role**, **overtime**, or **sentiment**?"

        return answer_followup_from_last(user_input)

    return ""


# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Chat Settings")

    st.session_state.model_mode = st.radio(
        "Select model type",
        ["Local", "Cloud"],
        index=0 if st.session_state.model_mode == "Local" else 1,
        horizontal=True,
    )

    if st.session_state.model_mode == "Cloud":
        if cloud_is_configured():
            st.success("Cloud model is configured and ready.")
        else:
            st.warning("No cloud token found.")

    if st.button("New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown("### Chat History")
    user_questions = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    if user_questions:
        for qq in reversed(user_questions):
            st.markdown(f"- {qq}")
    else:
        st.caption("No questions yet.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#666; text-align:center;'>Developed by Hajar</p>",
        unsafe_allow_html=True
    )


# -------------------- Main --------------------
st.title("HR Analytics Chatbot")
st.caption("Ask questions about HR data.")

if not st.session_state.messages:
    st.markdown(WELCOME_MSG, unsafe_allow_html=True)

for msg in st.session_state.messages:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.markdown(content, unsafe_allow_html=True)

user_input = st.chat_input("Ask a question...")

if user_input and not st.session_state.is_generating:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.is_generating = True

    with st.chat_message("assistant"):
        try:
            # 1) Pandas direct answer 
            direct = pandas_direct_answer(user_input)
            if direct:
                answer_text = direct
            else:
                # 2) RAG + LLM 
                retriever = get_retriever()
                show_evidence = True if st.session_state.model_mode == "Cloud" else wants_evidence(user_input)

                context = retriever.get_context(user_input, top_k=4)
                final_prompt = build_rag_prompt(user_input, context, show_evidence) if context.strip() else user_input

                model_id = "Qwen/Qwen2.5-7B-Instruct"
                raw = ""

                use_cloud_now = (st.session_state.model_mode == "Cloud") or is_analytic(user_input)

                if not use_cloud_now:
                    try:
                        raw = get_local_llm().generate(final_prompt)
                    except Exception:
                        raw = ""

                    if not raw.strip() or looks_like_template_or_echo(raw):
                        msgs = build_cloud_messages(final_prompt, max_turns=8)
                        raw = run_cloud_chat(msgs, model_id=model_id)
                else:
                    msgs = build_cloud_messages(final_prompt, max_turns=8)
                    raw = run_cloud_chat(msgs, model_id=model_id)

                answer_text = finalize_answer(user_input, raw, show_evidence)

        except Exception as e:
            st.error(f"Error: {e}")
            answer_text = smart_fallback(user_input, True)

        st.markdown(answer_text)
        st.session_state.messages.append({"role": "assistant", "content": answer_text})
        st.session_state.is_generating = False
        st.rerun()
