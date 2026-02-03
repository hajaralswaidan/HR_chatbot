# src/analysis/sentiment_analysis.py
from pathlib import Path
import pandas as pd
from transformers import pipeline

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # HR_chatbot/
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_text_column(df: pd.DataFrame) -> pd.Series:
    """
    Feature Engineering:
    Build a synthetic text column because HR dataset doesn't include free-text.
    This is only for testing a sentiment model integration.
    """
    # pick safe columns that exist in IBM HR dataset
    # (if any column missing, it will fallback to empty string)
    job_role = df.get("JobRole", "").astype(str)
    dept = df.get("Department", "").astype(str)
    overtime = df.get("OverTime", "").astype(str)
    wl_balance = df.get("WorkLifeBalance", "").astype(str)

    return (
        "Job role: " + job_role
        + ". Department: " + dept
        + ". OverTime: " + overtime
        + ". WorkLifeBalance: " + wl_balance
    )

def main():
    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    # 2) Create synthetic text
    df["text_summary"] = build_text_column(df)

    # 3) Load sentiment pipeline (English)
    # You can change model if you want, but this is stable and light.
    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # 4) Run sentiment on a sample (avoid running on full 1470 rows if slow)
    sample_size = min(200, len(df))
    sample_df = df.sample(sample_size, random_state=42).copy()

    preds = clf(sample_df["text_summary"].tolist(), batch_size=16, truncation=True)

    sample_df["sentiment_label"] = [p["label"] for p in preds]
    sample_df["sentiment_score"] = [float(p["score"]) for p in preds]

    # 5) Save results
    out_csv = OUT_DIR / "sentiment_results.csv"
    sample_df[[
        "EmployeeNumber", "JobRole", "Department", "OverTime", "WorkLifeBalance",
        "sentiment_label", "sentiment_score", "text_summary"
    ]].to_csv(out_csv, index=False)

    # 6) Make a quick summary for README
    summary = sample_df["sentiment_label"].value_counts().to_dict()
    avg_score = sample_df["sentiment_score"].mean()

    out_txt = OUT_DIR / "sentiment_summary.txt"
    out_txt.write_text(
        "Sentiment Analysis Summary (sample)\n"
        f"- Sample size: {sample_size}\n"
        f"- Label counts: {summary}\n"
        f"- Average confidence score: {avg_score:.3f}\n"
    )

    print(" Sentiment analysis completed.")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")

if __name__ == "__main__":
    main()
