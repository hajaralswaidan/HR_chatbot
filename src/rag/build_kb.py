# src/rag/build_kb.py
import json
from pathlib import Path
import pandas as pd

from src.data_loader import load_hr_data

# Output KB path ONLY from the HR dataset
KB_PATH = Path("data") / "kb.json"


def _safe_str(x) -> str:
    """Convert any value to a clean string."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return " ".join(s.split())


def build_kb_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convert the HR dataset into a list of text documents ("chunks") for RAG.
    These chunks are derived ONLY from the HR dataset.
    """
    docs: list[dict] = []

    # ---- Dataset summary 
    total = int(len(df))
    attr_yes = int((df["Attrition"] == "Yes").sum()) if "Attrition" in df.columns else 0
    attr_no = int((df["Attrition"] == "No").sum()) if "Attrition" in df.columns else 0
    attr_rate = (attr_yes / total) if total else 0.0

    summary_text = (
        f"Dataset Summary:\n"
        f"- Total employees: {total}\n"
        f"- Attrition Yes: {attr_yes}\n"
        f"- Attrition No: {attr_no}\n"
        f"- Attrition rate: {attr_rate:.3f}\n"
    )

    docs.append({
        "id": "summary_0",
        "type": "summary",
        "text": summary_text
    })

    # ---  Simple aggregated facts 
    def add_group_fact(col: str, doc_id: str, title: str):
        if col not in df.columns or "Attrition" not in df.columns:
            return

        grp = (
            df.groupby(col)["Attrition"]
              .apply(lambda s: (s == "Yes").mean())
              .sort_values(ascending=False)
        )

        lines = [f"{title} (Attrition rate by {col}):"]
        # Keep top 12 for compact KB
        for k, v in grp.head(12).items():
            lines.append(f"- {k}: {float(v):.3f}")

        docs.append({
            "id": doc_id,
            "type": "aggregate",
            "text": "\n".join(lines)
        })

    add_group_fact("Department", "agg_department", "HR Facts")
    add_group_fact("JobRole", "agg_jobrole", "HR Facts")
    add_group_fact("OverTime", "agg_overtime", "HR Facts")

    # ---  Row-level 
    key_cols = [c for c in [
        "Age", "BusinessTravel", "Department", "DistanceFromHome", "Education",
        "EducationField", "EnvironmentSatisfaction", "Gender", "JobInvolvement",
        "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome",
        "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
        "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
        "Attrition"
    ] if c in df.columns]

    chunk_size = 50  # if want smaller/larger chunks

    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        rows = df.iloc[start:end]

        lines = [f"Employee records {start + 1} to {end}:"]
        for idx, r in rows.iterrows():
            parts = []
            for c in key_cols:
                parts.append(f"{c}={_safe_str(r[c])}")
            lines.append(" | ".join(parts))

        docs.append({
            "id": f"rows_{start+1}_{end}",
            "type": "rows_chunk",
            "text": "\n".join(lines)
        })

    return docs


def save_kb(docs: list[dict], out_path: Path = KB_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "num_docs": len(docs),
        "docs": docs
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main():
    df = load_hr_data() 
    docs = build_kb_documents(df)
    path = save_kb(docs)
    print(f"[OK] KB saved to: {path.resolve()}")
    print(f"[OK] Documents: {len(docs)}")


if __name__ == "__main__":
    main()
