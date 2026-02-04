import streamlit as st
import pandas as pd
from transformers import pipeline

from src.data_loader import load_hr_data

st.set_page_config(page_title="HR Data Analysis & Sentiment", layout="wide")

st.title("HR Data Analysis & Sentiment")
st.caption("Week 3 requirement: quick Pandas analysis + Hugging Face sentiment (separate from chatbot).")

# ----------------------------
# Load data (Pandas)
# ----------------------------
@st.cache_data
def get_df() -> pd.DataFrame:
    return load_hr_data()

df = get_df()

# ----------------------------
# Quick refresh analysis
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Employees", f"{len(df):,}")

with col2:
    if "Attrition" in df.columns:
        rate = (df["Attrition"].astype(str).str.lower().eq("yes").mean()) * 100
        st.metric("Attrition Rate", f"{rate:.1f}%")
    else:
        st.metric("Attrition Rate", "N/A")

with col3:
    if "Department" in df.columns:
        st.metric("Departments", df["Department"].nunique())
    else:
        st.metric("Departments", "N/A")

st.divider()

# ----------------------------
# Feature Engineering (create a text column)
# ----------------------------
st.subheader("Feature Engineering: Synthetic Text Column")

default_cols = ["JobRole", "Department", "OverTime"]
available = [c for c in default_cols if c in df.columns]

selected_cols = st.multiselect(
    "Choose columns to combine into a text field",
    options=list(df.columns),
    default=available if available else list(df.columns)[:2],
)

separator = st.text_input("Separator", value=" | ")

if not selected_cols:
    st.warning("Select at least one column to build the text field.")
    st.stop()

df_fe = df.copy()
df_fe["text_for_sentiment"] = df_fe[selected_cols].astype(str).agg(separator.join, axis=1)

st.write("Preview of the engineered text:")
st.dataframe(df_fe[["text_for_sentiment"]].head(10), use_container_width=True)

st.divider()

# ----------------------------
# Sentiment Analysis (Hugging Face)
# ----------------------------
st.subheader("Sentiment Analysis (Hugging Face)")
st.caption("We run sentiment on a small sample to keep it fast.")

# Lightweight model (English)
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_resource
def get_sentiment_pipe():
    return pipeline("sentiment-analysis", model=MODEL_ID)

pipe = get_sentiment_pipe()

sample_size = st.slider("Sample size", min_value=20, max_value=200, value=60, step=10)

# sample to keep runtime reasonable
texts = df_fe["text_for_sentiment"].dropna().astype(str).head(sample_size).tolist()

if st.button("Run Sentiment", type="primary"):
    with st.spinner("Running sentiment..."):
        preds = pipe(texts)

    out = pd.DataFrame(preds)
    out["text"] = texts

    # basic summary
    pos = (out["label"].str.upper() == "POSITIVE").sum()
    neg = (out["label"].str.upper() == "NEGATIVE").sum()
    total = len(out)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total analyzed", total)
    c2.metric("Positive", pos)
    c3.metric("Negative", neg)

    st.write("Sample results:")
    st.dataframe(out[["label", "score", "text"]].head(20), use_container_width=True)

    st.info(
        "Note: The dataset has no true text column, so we created a synthetic text field for experimentation "
        "(Week 3 requirement). This sentiment module is separate from the chatbot pipeline."
    )
