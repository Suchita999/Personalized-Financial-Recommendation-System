# ============================================================
# NOTEBOOK 3: Fund Matching (Cosine Similarity) + RAG Setup
#
# Input:  Trained models from Notebooks 1 & 2
# Output: Top-N fund recommendations per composite profile
#         + RAG document corpus for chatbot
#
# Run: python 03_Fund_Matching_and_RAG.py
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SECTION 1: LOAD SAVED ARTIFACTS FROM NOTEBOOKS 1 & 2
# ============================================================

def load_artifacts(data_dir='data/processed'):
    print("=" * 60)
    print("LOADING SAVED ARTIFACTS")
    print("=" * 60)

    a = {}
    a['fund_feat']         = pd.read_csv(f'{data_dir}/fund_features.csv')
    a['census_clustered']  = pd.read_csv(f'{data_dir}/census_clustered.csv')
    a['rec_map']           = pd.read_csv(f'{data_dir}/cluster_recommendation_map.csv')
    a['ce_personas']       = pd.read_csv(f'{data_dir}/ce_spending_personas.csv')
    a['classifier']        = joblib.load(f'{data_dir}/best_classifier.pkl')
    a['label_encoder']     = joblib.load(f'{data_dir}/label_encoder.pkl')
    a['feature_encoders']  = joblib.load(f'{data_dir}/feature_encoders.pkl')
    a['ce_kmeans']         = joblib.load(f'{data_dir}/ce_kmeans_model.pkl')
    a['ce_scaler']         = joblib.load(f'{data_dir}/ce_kmeans_scaler.pkl')

    scaler_path = f'{data_dir}/classifier_scaler.pkl'
    a['classifier_scaler'] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    a['raw_features'] = pd.read_csv(f'{data_dir}/classifier_features.csv').iloc[:, 0].tolist()

    print(f"  ✓ Fund features: {a['fund_feat'].shape}")
    print(f"  ✓ Census clustered: {a['census_clustered'].shape}")
    print(f"  ✓ CE personas: {a['ce_personas'].shape}")
    print(f"  ✓ Classifier loaded | Classes: {list(a['label_encoder'].classes_)}")
    print(f"  ✓ CE K-Means loaded (K={a['ce_kmeans'].n_clusters})")

    return a


# ============================================================
# SECTION 2: PROFILE CE SPENDING CLUSTERS (K=2)
# ============================================================

def profile_ce_clusters(ce_df):
    print("\n" + "=" * 60)
    print("CE SPENDING CLUSTER PROFILES (K=2)")
    print("=" * 60)

    cluster_cols = [c for c in ce_df.columns if c.startswith('ratio_') or
                    c in ['savings_rate', 'needs_ratio', 'wants_ratio',
                           'budget_health_score', 'income_percentile']]

    profiles = ce_df.groupby('spending_cluster')[cluster_cols].mean()
    print("\n  Mean feature values per spending cluster:")
    print(profiles.round(3).to_string())

    if 'savings_rate' in profiles.columns:
        hi = int(profiles['savings_rate'].idxmax())
        lo = int(profiles['savings_rate'].idxmin())
    elif 'budget_health_score' in profiles.columns:
        hi = int(profiles['budget_health_score'].idxmax())
        lo = int(profiles['budget_health_score'].idxmin())
    else:
        hi, lo = 0, 1

    spending_labels = {hi: 'high_saver', lo: 'low_saver'}

    print(f"\n  ★ Cluster {hi} → High Saver / Financially Disciplined")
    print(f"  ★ Cluster {lo} → Low Saver / Budget Stretched")

    counts = ce_df['spending_cluster'].value_counts().sort_index()
    for cid, count in counts.items():
        label = spending_labels.get(cid, 'unknown')
        print(f"    Cluster {cid} ({label}): {count:,} households")

    return spending_labels, profiles


# ============================================================
# SECTION 3: 6 COMPOSITE PREFERENCE VECTORS
# ============================================================

MATCHING_FEATURES = [
    'fund_risk_score', 'expense_ratio', 'cost_score', 'avg_return',
    'return_consistency', 'sector_diversification',
    'top_sector_concentration', 'composite_score',
]

def build_composite_preference_vectors():
    """
    6 composite profiles: (risk_tolerance × spending_behavior) = 3 × 2

    Each vector: what this profile WANTS in a fund (0-1 scale).
    """
    print("\n" + "=" * 60)
    print("BUILDING 6 COMPOSITE PREFERENCE VECTORS")
    print("=" * 60)

    profiles = {
        #                                   risk  exp   cost  ret   consist divers conc  comp
        ('aggressive', 'high_saver'):      [0.85, 0.50, 0.40, 0.95, 0.20, 0.60, 0.60, 0.80],
        ('aggressive', 'low_saver'):       [0.75, 0.25, 0.75, 0.85, 0.30, 0.65, 0.45, 0.75],
        ('moderate',   'high_saver'):      [0.50, 0.40, 0.55, 0.65, 0.60, 0.75, 0.30, 0.80],
        ('moderate',   'low_saver'):       [0.45, 0.20, 0.85, 0.55, 0.70, 0.70, 0.25, 0.70],
        ('conservative','high_saver'):     [0.20, 0.35, 0.60, 0.40, 0.90, 0.80, 0.15, 0.75],
        ('conservative','low_saver'):      [0.15, 0.10, 0.95, 0.30, 0.95, 0.85, 0.10, 0.65],
    }

    pref_df = pd.DataFrame(profiles, index=MATCHING_FEATURES).T
    pref_df.index = pd.MultiIndex.from_tuples(
        pref_df.index, names=['risk_tolerance', 'spending_behavior']
    )

    print("\n  Preference vectors:")
    print(pref_df.round(2).to_string())

    return pref_df


# ============================================================
# SECTION 4: PREPARE FUND FEATURE MATRIX
# ============================================================

def prepare_fund_matrix(fund_feat):
    print("\n" + "=" * 60)
    print("PREPARING FUND FEATURE MATRIX")
    print("=" * 60)

    available = [f for f in MATCHING_FEATURES if f in fund_feat.columns]
    missing = [f for f in MATCHING_FEATURES if f not in fund_feat.columns]
    if missing:
        print(f"  ⚠ Missing features (skipping): {missing}")
    print(f"  Using features: {available}")

    fund_clean = fund_feat.dropna(subset=available, thresh=len(available) - 2).copy()
    print(f"  Funds before cleaning: {len(fund_feat):,}")
    print(f"  Funds after cleaning:  {len(fund_clean):,}")

    for col in available:
        fund_clean[col] = pd.to_numeric(fund_clean[col], errors='coerce')
        fund_clean[col] = fund_clean[col].fillna(fund_clean[col].median())

    scaler = MinMaxScaler()
    fund_matrix = pd.DataFrame(
        scaler.fit_transform(fund_clean[available]),
        columns=available, index=fund_clean.index,
    )

    print(f"  Fund matrix shape: {fund_matrix.shape}")
    return fund_matrix, scaler, fund_clean, available


# ============================================================
# SECTION 5: COSINE SIMILARITY FUND MATCHING
# ============================================================

def match_funds_for_profile(profile_vector, fund_matrix, fund_clean, top_n=10):
    """Match a single profile to funds using cosine similarity."""
    pv = np.array(profile_vector).reshape(1, -1)
    sims = cosine_similarity(pv, fund_matrix.values)[0]

    results = fund_clean.copy()
    results['similarity_score'] = sims
    top = results.nlargest(top_n, 'similarity_score')

    display_cols = ['similarity_score']
    for col in ['fund_name', 'fund_symbol', 'fund_long_name', 'symbol', 'name',
                'investment_type', 'fund_risk_tier', 'fund_risk_score',
                'expense_ratio', 'avg_return', 'return_consistency',
                'composite_score', 'asset_class_derived', 'alloc_stocks', 'alloc_bonds']:
        if col in top.columns:
            display_cols.append(col)

    return top[[c for c in display_cols if c in top.columns]]


def match_all_profiles(pref_df, fund_matrix, fund_clean, features, top_n=10):
    """Run fund matching for ALL 6 composite profiles."""
    print("\n" + "=" * 60)
    print(f"FUND MATCHING — TOP {top_n} PER PROFILE")
    print("=" * 60)

    all_recs = {}

    for (risk, spend), row in pref_df.iterrows():
        label = f"({risk}, {spend})"
        print(f"\n  ── Profile: {label} ──")

        pv = row[features].values
        top_funds = match_funds_for_profile(pv, fund_matrix, fund_clean, top_n)
        all_recs[(risk, spend)] = top_funds

        avg_sim = top_funds['similarity_score'].mean()
        print(f"    Avg similarity: {avg_sim:.4f}")
        print(f"    Range: {top_funds['similarity_score'].min():.4f} — "
              f"{top_funds['similarity_score'].max():.4f}")

        if 'fund_risk_tier' in top_funds.columns:
            print(f"    Risk tiers: {top_funds['fund_risk_tier'].value_counts().to_dict()}")
        if 'investment_type' in top_funds.columns:
            print(f"    Fund types: {top_funds['investment_type'].value_counts().to_dict()}")

    return all_recs


# ============================================================
# SECTION 6: PREDICT & RECOMMEND FOR A NEW USER
# ============================================================

def predict_and_recommend(user_input, artifacts, pref_df, fund_matrix,
                          fund_clean, features, spending_labels, top_n=10):
    """
    End-to-end: raw user inputs → predict profile → match funds.
    This is what the Streamlit app calls.
    """
    print("\n" + "=" * 60)
    print("PREDICTING & RECOMMENDING FOR NEW USER")
    print("=" * 60)

    # Step 1: Risk tolerance
    classifier = artifacts['classifier']
    le = artifacts['label_encoder']
    encoders = artifacts['feature_encoders']
    clf_scaler = artifacts['classifier_scaler']
    raw_features = artifacts['raw_features']

    X_new = pd.DataFrame([user_input])[raw_features].copy()
    for col in X_new.select_dtypes(include='object').columns:
        if col in encoders:
            X_new[col] = X_new[col].astype(str).str.strip()
            known = set(encoders[col].classes_)
            X_new[col] = X_new[col].apply(lambda x: x if x in known else encoders[col].classes_[0])
            X_new[col] = encoders[col].transform(X_new[col])
    for col in X_new.select_dtypes(include=['int64', 'float64']).columns:
        X_new[col] = pd.to_numeric(X_new[col], errors='coerce').fillna(0)

    X_pred = clf_scaler.transform(X_new) if clf_scaler else X_new.values
    risk_tolerance = le.inverse_transform(classifier.predict(X_pred))[0]
    print(f"  ✓ Risk tolerance: {risk_tolerance}")

    # Step 2: Spending cluster
    ce_kmeans = artifacts['ce_kmeans']
    ce_scaler = artifacts['ce_scaler']
    if hasattr(ce_scaler, 'feature_names_in_'):
        ce_cols = list(ce_scaler.feature_names_in_)
    else:
        ce_cols = [c for c in user_input if c.startswith('ratio_') or
                   c in ['savings_rate', 'needs_ratio', 'wants_ratio',
                          'budget_health_score', 'income_percentile']]

    X_ce = pd.DataFrame([{c: user_input.get(c, 0) for c in ce_cols}])
    spending_cluster = int(ce_kmeans.predict(ce_scaler.transform(X_ce))[0])
    spending_behavior = spending_labels.get(spending_cluster, 'high_saver')
    print(f"  ✓ Spending: cluster {spending_cluster} → {spending_behavior}")

    # Step 3: Match funds
    key = (risk_tolerance, spending_behavior)
    print(f"  ✓ Composite: {key}")

    if key in pref_df.index:
        pv = pref_df.loc[key][features].values
    else:
        print(f"  ⚠ Falling back to (moderate, high_saver)")
        pv = pref_df.loc[('moderate', 'high_saver')][features].values

    top_funds = match_funds_for_profile(pv, fund_matrix, fund_clean, top_n)

    print(f"\n  ★ Top {top_n} recommendations:")
    print(top_funds.to_string())

    return {
        'risk_tolerance': risk_tolerance,
        'spending_behavior': spending_behavior,
        'composite_profile': key,
        'top_funds': top_funds,
    }


# ============================================================
# SECTION 7: SAVE OUTPUTS
# ============================================================

def save_outputs(all_recs, pref_df, data_dir='data/processed'):
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    pref_df.to_csv(f'{data_dir}/composite_preference_vectors.csv')
    print(f"  ✓ Preference vectors → composite_preference_vectors.csv")

    all_recs_list = []
    for (risk, spend), df in all_recs.items():
        rec = df.copy()
        rec['risk_tolerance'] = risk
        rec['spending_behavior'] = spend
        all_recs_list.append(rec)

    all_recs_df = pd.concat(all_recs_list, ignore_index=True)
    all_recs_df.to_csv(f'{data_dir}/fund_recommendations_all_profiles.csv', index=False)
    print(f"  ✓ All recommendations → fund_recommendations_all_profiles.csv")

    # Summary
    name_col = next((c for c in ['fund_name', 'fund_symbol', 'fund_long_name',
                                  'symbol', 'name'] if c in all_recs_df.columns), None)
    summary = []
    for (risk, spend), df in all_recs.items():
        top3 = df.head(3)
        names = top3[name_col].tolist() if name_col and name_col in top3.columns else ['N/A']
        summary.append({
            'risk_tolerance': risk,
            'spending_behavior': spend,
            'top_3_funds': ' | '.join(str(n) for n in names),
            'avg_similarity': df['similarity_score'].mean(),
        })

    pd.DataFrame(summary).to_csv(f'{data_dir}/profile_fund_summary.csv', index=False)
    print(f"  ✓ Summary → profile_fund_summary.csv")

    return all_recs_df


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs('data/processed', exist_ok=True)

    print("=" * 60)
    print("  NOTEBOOK 3: FUND MATCHING (COSINE SIMILARITY)")
    print("=" * 60)

    # A: Load everything
    artifacts = load_artifacts()

    # B: Profile CE clusters
    spending_labels, ce_profiles = profile_ce_clusters(artifacts['ce_personas'])

    # C: Build 6 preference vectors
    pref_df = build_composite_preference_vectors()

    # D: Prepare fund matrix
    fund_matrix, fund_scaler, fund_clean, available = prepare_fund_matrix(
        artifacts['fund_feat']
    )

    # Align preference vectors to available features
    features = [f for f in MATCHING_FEATURES if f in available]
    pref_aligned = pref_df[features]

    # E: Match all 6 profiles
    all_recs = match_all_profiles(pref_aligned, fund_matrix, fund_clean, features, top_n=10)

    # F: Save
    all_recs_df = save_outputs(all_recs, pref_df)

    print("\n" + "=" * 60)
    print("  ★ NOTEBOOK 3 COMPLETE")
    print("=" * 60)

    return {
        'artifacts': artifacts,
        'spending_labels': spending_labels,
        'pref_df': pref_aligned,
        'fund_matrix': fund_matrix,
        'fund_clean': fund_clean,
        'features': features,
        'all_recommendations': all_recs,
    }


if __name__ == '__main__':
    results = main()
