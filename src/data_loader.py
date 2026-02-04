import pandas as pd
from pathlib import Path

# ONLY official HR dataset
DATASET_PATH = Path("data") / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Columns with no analytical value
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours"]


def load_hr_data() -> pd.DataFrame:
    """
    Load the official HR dataset for INITIAL database setup only.

    NOTE:
    - Pandas is used here ONLY for loading/cleaning the dataset.
    - The chatbot answering pipeline uses SQLite + SQL only (no Pandas).

    Returns:
        pd.DataFrame: cleaned HR dataset
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"HR dataset not found at: {DATASET_PATH.resolve()}"
        )

    df = pd.read_csv(DATASET_PATH)

    # Drop non-informative columns
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Add numeric attrition flag (for SQL usage)
    if "Attrition" in df.columns and "AttritionFlag" not in df.columns:
        df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})

    return df
