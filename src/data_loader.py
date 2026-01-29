# src/data_loader.py
import pandas as pd
from pathlib import Path

# FIXED dataset path (ONLY this dataset is allowed)
DATASET_PATH = Path("data") / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# useless columns in this dataset
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours"]


def load_hr_data() -> pd.DataFrame:
    """
    Load ONLY the official HR dataset from /data.
    Any other dataset is not allowed by design.

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"HR dataset not found. Expected at: {DATASET_PATH.resolve()}"
        )

    df = pd.read_csv(DATASET_PATH)

    # Drop constant/useless columns if present
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Add numeric flag for analysis 
    if "Attrition" in df.columns and "AttritionFlag" not in df.columns:
        df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})

    return df
