"""
Missing Values Handler for CES Interview Data

Handles missing values in Consumer Expenditure Survey (CES) interview datasets
using domain-appropriate strategies:
- Expenditure variables: missing often means $0 (category not applicable)
- Income variables: median imputation
- Other numerical: median imputation  
- Categorical: fill with "MISSING"
"""

import pandas as pd


def handle_missing_values(
    df: pd.DataFrame,
    missing_threshold: float = 0.95,
    verbose: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Handle missing values in CES interview data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (e.g., FMLI interview data)
    missing_threshold : float, default 0.95
        Drop columns with more than this fraction of missing values (0-1)
    verbose : bool, default True
        Print summary of imputation steps
    inplace : bool, default False
        If True, modify df in place. Otherwise return a copy.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values handled
    """
    if not inplace:
        df = df.copy()

    # 1. Drop columns with excessive missingness
    cols_to_drop = (df.isnull().sum() / len(df)) > missing_threshold
    dropped_cols = cols_to_drop[cols_to_drop].index.tolist()
    df = df.drop(columns=dropped_cols)
    if verbose and dropped_cols:
        print(f"Dropped {len(dropped_cols)} columns with >{missing_threshold*100:.0f}% missing")

    # 2. Expenditure variables: missing often means $0
    expend_patterns = ['PQ', 'CQ', 'EXPPQ', 'EXPCQ', 'FD', 'HOUS', 'TRAN', 'HEALTH', 'ENTERT', 'EDUC', 'APPAR']
    expend_cols = [
        c for c in df.columns
        if any(p in c for p in expend_patterns)
        and df[c].dtype in ['float64', 'int64']
    ]
    df[expend_cols] = df[expend_cols].fillna(0)
    if verbose:
        print(f"Imputed {len(expend_cols)} expenditure columns with 0")

    # 3. Income variables: median imputation
    income_keywords = ['INC', 'INCOME', 'FINC', 'SALARY', 'WAGE', 'RETIR', 'INVEST', 'RENT', 'TAX']
    income_cols = [
        c for c in df.columns
        if any(k in c for k in income_keywords)
        and c not in expend_cols
        and df[c].dtype in ['float64', 'int64']
    ]
    income_imputed = sum(1 for c in income_cols if df[c].isnull().any())
    for col in income_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    if verbose:
        print(f"Imputed {income_imputed} income columns with median")

    # 4. Other numerical columns: median imputation
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # 5. Categorical columns: fill with "MISSING"
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna('MISSING').astype(str)

    remaining = df.isnull().sum().sum()
    if verbose:
        print(f"Missing value handling complete. Remaining missing values: {remaining}")

    return df


def analyze_missing(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """
    Analyze missing value patterns in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    top_n : int, default 20
        Number of top columns with missing data to return

    Returns
    -------
    pd.Series
        Percentage of missing values per column, sorted descending
    """
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    return missing_pct.head(top_n)
