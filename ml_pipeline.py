"""
ml_pipeline.py — Core ML: load models, predict risk + spending, match funds.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

DATA_DIR = os.environ.get("DATA_DIR", "data/processed")

_cache = {}


# ─────────────────────────────────────────────
# 1. LOAD ARTIFACTS
# ─────────────────────────────────────────────

def load_artifacts():
    """Load all saved models/data once and cache in memory."""
    if _cache:
        return _cache

    print("[ML] Loading artifacts …")

    _cache['fund_feat']         = pd.read_csv(f'{DATA_DIR}/fund_features.csv')
    _cache['census_clustered']  = pd.read_csv(f'{DATA_DIR}/census_clustered.csv')
    _cache['rec_map']           = pd.read_csv(f'{DATA_DIR}/cluster_recommendation_map.csv')
    _cache['ce_personas']       = pd.read_csv(f'{DATA_DIR}/ce_spending_personas.csv')

    _cache['classifier']        = joblib.load(f'{DATA_DIR}/best_classifier.pkl')
    _cache['label_encoder']     = joblib.load(f'{DATA_DIR}/label_encoder.pkl')
    _cache['feature_encoders']  = joblib.load(f'{DATA_DIR}/feature_encoders.pkl')
    _cache['ce_kmeans']         = joblib.load(f'{DATA_DIR}/ce_kmeans_model.pkl')
    _cache['ce_scaler']         = joblib.load(f'{DATA_DIR}/ce_kmeans_scaler.pkl')

    scaler_path = f'{DATA_DIR}/classifier_scaler.pkl'
    _cache['classifier_scaler'] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    _cache['raw_features'] = pd.read_csv(
        f'{DATA_DIR}/classifier_features.csv'
    ).iloc[:, 0].tolist()

    # Detect which CE cluster is high_saver vs low_saver
    _cache['spending_labels'] = _detect_spending_labels(_cache['ce_personas'])

    # Build preference vectors & fund matrix (one-time)
    _cache['pref_df'], _cache['pref_features'] = _build_preference_vectors()
    (
        _cache['fund_matrix'],
        _cache['fund_scaler'],
        _cache['fund_clean'],
        _cache['available_features'],
    ) = _prepare_fund_matrix(_cache['fund_feat'])

    # Align preference features to what's actually available in fund data
    avail = _cache['available_features']
    _cache['pref_features'] = [f for f in _cache['pref_features'] if f in avail]
    _cache['pref_df'] = _cache['pref_df'][_cache['pref_features']]

    print(f"[ML] Ready — {len(_cache['fund_clean']):,} funds loaded")
    return _cache


def _detect_spending_labels(ce_df):
    """Auto-detect which CE cluster = high_saver vs low_saver."""
    cluster_cols = [c for c in ce_df.columns if c.startswith('ratio_') or
                    c in ['savings_rate', 'needs_ratio', 'wants_ratio',
                           'budget_health_score', 'income_percentile']]
    profiles = ce_df.groupby('spending_cluster')[cluster_cols].mean()
    if 'savings_rate' in profiles.columns:
        hi = int(profiles['savings_rate'].idxmax())
        lo = int(profiles['savings_rate'].idxmin())
    elif 'budget_health_score' in profiles.columns:
        hi = int(profiles['budget_health_score'].idxmax())
        lo = int(profiles['budget_health_score'].idxmin())
    else:
        hi, lo = 0, 1
    return {hi: 'high_saver', lo: 'low_saver'}


# ─────────────────────────────────────────────
# 2. PREFERENCE VECTORS (6 composite profiles)
# ─────────────────────────────────────────────

MATCHING_FEATURES = [
    'fund_risk_score', 'expense_ratio', 'cost_score', 'avg_return',
    'return_consistency', 'sector_diversification',
    'top_sector_concentration', 'composite_score',
]

def _build_preference_vectors():
    """
    Define ideal fund preference vectors for each of the 6 composite profiles.
    (risk_tolerance × spending_behavior) = 3 × 2 = 6 profiles.
    Each value is 0-1 representing what this profile WANTS in a fund.
    """
    profiles = {
        # AGGRESSIVE + HIGH SAVER: max growth, can handle volatility
        ('aggressive', 'high_saver'):    [0.85, 0.50, 0.40, 0.95, 0.20, 0.60, 0.60, 0.80],
        # AGGRESSIVE + LOW SAVER: growth but needs cheap funds
        ('aggressive', 'low_saver'):     [0.75, 0.25, 0.75, 0.85, 0.30, 0.65, 0.45, 0.75],
        # MODERATE + HIGH SAVER: balanced with room for risk
        ('moderate',   'high_saver'):    [0.50, 0.40, 0.55, 0.65, 0.60, 0.75, 0.30, 0.80],
        # MODERATE + LOW SAVER: balanced but cost-sensitive
        ('moderate',   'low_saver'):     [0.45, 0.20, 0.85, 0.55, 0.70, 0.70, 0.25, 0.70],
        # CONSERVATIVE + HIGH SAVER: capital preservation
        ('conservative','high_saver'):   [0.20, 0.35, 0.60, 0.40, 0.90, 0.80, 0.15, 0.75],
        # CONSERVATIVE + LOW SAVER: ultra-safe, cheapest funds
        ('conservative','low_saver'):    [0.15, 0.10, 0.95, 0.30, 0.95, 0.85, 0.10, 0.65],
    }
    pref_df = pd.DataFrame(profiles, index=MATCHING_FEATURES).T
    pref_df.index = pd.MultiIndex.from_tuples(
        pref_df.index, names=['risk_tolerance', 'spending_behavior']
    )
    return pref_df, MATCHING_FEATURES


# ─────────────────────────────────────────────
# 3. FUND MATRIX (normalized for cosine similarity)
# ─────────────────────────────────────────────

def _prepare_fund_matrix(fund_feat):
    available = [f for f in MATCHING_FEATURES if f in fund_feat.columns]
    fund_clean = fund_feat.dropna(subset=available, thresh=len(available) - 2).copy()
    for col in available:
        fund_clean[col] = pd.to_numeric(fund_clean[col], errors='coerce')
        fund_clean[col] = fund_clean[col].fillna(fund_clean[col].median())
    scaler = MinMaxScaler()
    fund_matrix = pd.DataFrame(
        scaler.fit_transform(fund_clean[available]),
        columns=available, index=fund_clean.index,
    )
    return fund_matrix, scaler, fund_clean, available


# ─────────────────────────────────────────────
# 4. PREDICT USER PROFILE
# ─────────────────────────────────────────────

def predict_user_profile(user_input: dict) -> dict:
    """Raw user inputs → risk_tolerance + spending_behavior."""
    a = load_artifacts()

    # --- Risk tolerance (classifier) ---
    X_new = pd.DataFrame([user_input])[a['raw_features']].copy()
    for col in X_new.select_dtypes(include='object').columns:
        if col in a['feature_encoders']:
            X_new[col] = X_new[col].astype(str).str.strip()
            known = set(a['feature_encoders'][col].classes_)
            X_new[col] = X_new[col].apply(
                lambda x: x if x in known else a['feature_encoders'][col].classes_[0]
            )
            X_new[col] = a['feature_encoders'][col].transform(X_new[col])
    for col in X_new.select_dtypes(include=['int64', 'float64']).columns:
        X_new[col] = pd.to_numeric(X_new[col], errors='coerce').fillna(0)

    X_pred = (a['classifier_scaler'].transform(X_new)
              if a['classifier_scaler'] else X_new.values)
    risk_tolerance = a['label_encoder'].inverse_transform(
        a['classifier'].predict(X_pred)
    )[0]

    # --- Spending behaviour (CE K-Means) ---
    if hasattr(a['ce_scaler'], 'feature_names_in_'):
        ce_cols = list(a['ce_scaler'].feature_names_in_)
    else:
        ce_cols = [c for c in user_input if c.startswith('ratio_') or
                   c in ['savings_rate', 'needs_ratio', 'wants_ratio',
                          'budget_health_score', 'income_percentile']]

    X_ce = pd.DataFrame([{c: user_input.get(c, 0) for c in ce_cols}])
    spending_cluster = int(a['ce_kmeans'].predict(a['ce_scaler'].transform(X_ce))[0])
    spending_behavior = a['spending_labels'].get(spending_cluster, 'high_saver')

    return {
        'risk_tolerance': risk_tolerance,
        'spending_behavior': spending_behavior,
        'composite': f"{risk_tolerance} × {spending_behavior}",
    }


# ─────────────────────────────────────────────
# 5. MATCH FUNDS (cosine similarity)
# ─────────────────────────────────────────────

def match_funds(risk_tolerance: str, spending_behavior: str, top_n: int = 10) -> pd.DataFrame:
    """Return top-N funds for a given composite profile."""
    a = load_artifacts()
    key = (risk_tolerance, spending_behavior)
    features = a['pref_features']

    if key in a['pref_df'].index:
        pv = a['pref_df'].loc[key][features].values
    else:
        pv = a['pref_df'].loc[('moderate', 'high_saver')][features].values

    sims = cosine_similarity(pv.reshape(1, -1), a['fund_matrix'][features].values)[0]
    results = a['fund_clean'].copy()
    results['similarity_score'] = sims
    top = results.nlargest(top_n, 'similarity_score')

    # Build clean display dataframe
    display_cols = ['similarity_score']
    for col in ['fund_name', 'fund_symbol', 'fund_long_name', 'fund_short_name',
                'symbol', 'name', 'long_name', 'investment_type',
                'fund_risk_tier', 'fund_risk_score', 'expense_ratio',
                'avg_return', 'return_consistency', 'composite_score',
                'asset_class_derived', 'alloc_stocks', 'alloc_bonds']:
        if col in top.columns:
            display_cols.append(col)
    return top[[c for c in display_cols if c in top.columns]].reset_index(drop=True)


# ─────────────────────────────────────────────
# 6. PERSONA DESCRIPTIONS
# ─────────────────────────────────────────────

def get_persona_descriptions():
    return {
        ('aggressive', 'high_saver'):   "🚀 Max Growth Investor — High income, strong savings, seeks maximum returns",
        ('aggressive', 'low_saver'):    "📈 Growth Seeker — Growth-oriented but budget-conscious, needs low-cost options",
        ('moderate', 'high_saver'):     "⚖️ Balanced Builder — Steady saver who wants a mix of growth and stability",
        ('moderate', 'low_saver'):      "🛡️ Cautious Grower — Moderate risk with limited surplus, cost-sensitive",
        ('conservative', 'high_saver'): "🏦 Steady Preserver — Prioritises capital preservation, strong cash position",
        ('conservative', 'low_saver'):  "🔒 Safety First — Ultra-cautious, needs cheapest and most stable funds",
    }
