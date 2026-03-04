"""
Quick test for missing_values module with CES interview data.

Run from project:
    cd /path/to/Personalized-Financial-Recommendation-System
    python EDA/test_missing_values.py

Runfrom EDA folder:
    cd EDA
    python test_missing_values.py
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure we can import missing_values
sys.path.insert(0, str(Path(__file__).parent))
from missing_values import handle_missing_values, analyze_missing

def main():
    data_dir = Path(__file__).parent.parent / "data" / "intrvw24"
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Ensure data/intrvw24 contains fmli242.csv, fmli243.csv, fmli244.csv")
        return

    # Load one quarter as sample
    sample_file = data_dir / "fmli242.csv"
    if not sample_file.exists():
        print(f"File not found: {sample_file}")
        return

    print("Loading fmli242.csv...")
    df = pd.read_csv(sample_file)
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Missing before: {df.isnull().sum().sum():,}")

    print("\n--- analyze_missing() ---")
    print(analyze_missing(df, top_n=10))

    print("\n--- handle_missing_values() ---")
    df_clean = handle_missing_values(df)
    print(f"Missing after: {df_clean.isnull().sum().sum():,}")
    print(f"Shape: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")

if __name__ == "__main__":
    main()
