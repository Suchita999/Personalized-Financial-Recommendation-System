# FinWise - Personalized Financial Recommendation System

An AI-powered system that profiles users based on demographics and spending patterns, matches personalized fund recommendations from 26,000+ real mutual funds and ETFs, and generates natural-language explanations via a RAG pipeline — all with zero data persistence for privacy by design.

🔗 **Live Demo:** [FinWise on Streamlit Cloud](https://vmwg7knebklvbjmrwbggns.streamlit.app/)

---

## Table of Contents

- [About](#about)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Pipeline Details](#pipeline-details)
  - [Data Sources](#data-sources)
  - [Feature Engineering](#feature-engineering)
  - [Customer Segmentation (K-Means)](#customer-segmentation-k-means)
  - [Risk Profile Classification (XGBoost)](#risk-profile-classification-xgboost)
  - [Fund Matching Engine](#fund-matching-engine)
  - [Rule-Based Eligibility Engine](#rule-based-eligibility-engine)
  - [RAG Chatbot](#rag-chatbot)
- [Testing](#testing)
- [Team](#team)

---

## About

Most Americans lack access to personalized financial advice — 56% cannot cover a $1,000 emergency expense. Generic robo-advisors ignore individual demographics, spending behavior, and risk tolerance.

FinWise addresses this with a hybrid ML + rule-based + RAG pipeline that:

1. **Segments users** into risk profiles via K-Means clustering on demographic and spending data
2. **Classifies new users in real time** using an XGBoost ensemble trained on cluster-derived labels
3. **Matches users to real funds** (ETFs & mutual funds) through content-based cosine similarity on a 5-dimensional feature vector
4. **Determines tax-advantaged account eligibility** (401k, IRA, Roth IRA, HSA) via a rule-based IRS guideline engine
5. **Generates personalized explanations** via a RAG pipeline using ChromaDB + Sentence-Transformers + Google Gemini 2.5 Flash

All user data remains session-local — nothing is persisted beyond a single browser session.

---

## Tech Stack

| Layer | Tools |
|---|---|
| **ML & Data** | Python 3.12, Scikit-learn, XGBoost, Pandas, NumPy |
| **Clustering** | K-Means, StandardScaler, Silhouette Analysis |
| **NLP / RAG** | LangChain, ChromaDB, Sentence-Transformers (all-MiniLM-L6-v2), Google Gemini 2.5 Flash |
| **Fund Matching** | Cosine Similarity (5-dim feature vectors) |
| **Frontend** | Streamlit, Plotly |
| **Deployment** | Streamlit Community Cloud |

---

## Architecture

```
                         ┌──────────────┐
                         │  User Input  │
                         │ Streamlit UI │
                         └──────┬───────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│ UCI Census-Income │  │   CE Survey     │  │ Mutual Funds/ETFs│
│ 199K records     │  │ 23K households  │  │ 26K+ funds       │
└────────┬─────────┘  └────────┬────────┘  └────────┬─────────┘
         │                     │                     │
         └──────────┬──────────┘                     │
                    ▼                                 │
            ┌──────────────┐                          │
            │  ML Pipeline │                          │
            ├──────────────┤                          │
            │ K-Means      │                          │
            │ Clustering   │──► 6 Risk Profiles       │
            ├──────────────┤   (tolerance × spending) │
            │ XGBoost      │                          │
            │ Classifier   │   95.87% accuracy        │
            └──────┬───────┘                          │
                   │                                  │
     ┌─────────────┼──────────────────┐               │
     ▼             ▼                  ▼               ▼
┌─────────┐  ┌───────────┐  ┌──────────────────────────┐
│  Rule   │  │   RAG     │  │    Fund Matching          │
│  Engine │  │  Pipeline │  │  Cosine Similarity        │
│ 401k,   │  │ ChromaDB  │  │  90.8%–99.4% match scores│
│ IRA,HSA │  │ + Gemini  │  │                           │
└────┬────┘  └─────┬─────┘  └────────────┬─────────────┘
     │             │                     │
     └─────────────┼─────────────────────┘
                   ▼
        ┌─────────────────────┐
        │    Personalized     │
        │  Recommendations    │
        │ Fund picks + account│
        │ guidance + RAG chat │
        └─────────────────────┘
```

---

## Key Results

### Customer Segmentation (K-Means)

| Dataset | Optimal K | Silhouette Score | Records |
|---------|----------|-----------------|---------|
| UCI Census-Income | 10 | 0.483 | 199,523 |
| Consumer Expenditure Survey | 2 | 0.626 | 23,125 |

Six composite risk profiles were derived by crossing risk tolerance with spending behavior: Conservative Saver, Conservative Spender, Moderate Saver, Moderate Spender, Aggressive Saver, and Aggressive Spender.

### Risk Profile Classification (XGBoost Ensemble)

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Logistic Regression | 80.62% | 0.7868 | 0.8062 | 0.7868 |
| Random Forest | 95.82% | 0.9582 | 0.9582 | 0.9582 |
| **XGBoost** | **95.87%** | **0.9587** | **0.9587** | **0.9587** |

Per-class performance (XGBoost):

| Risk Profile | Precision | Recall | F1 |
|-------------|-----------|--------|-----|
| Aggressive | 0.91 | 0.92 | 0.92 |
| Conservative | 1.00 | 0.99 | 0.99 |
| Moderate | 0.97 | 0.97 | 0.97 |

### Fund Matching

Content-based cosine similarity on a 5-dimensional feature vector (expense ratio, cost score, return performance, return consistency, composite score) achieved match scores of **90.8%–99.4%** across all six risk profiles.

---

## Getting Started

### Prerequisites

- Python 3.12+
- A Google Gemini API key (for RAG responses)

### Installation

```bash
git clone https://github.com/Personalized-Financial-Recommendation-System.git
cd Personalized-Financial-Recommendation-System
pip install -r requirements.txt
```

### Set API Key

```bash
export GEMINI_API_KEY=your_key_here
```

### Run the Pipeline

```bash
python src/ml/feature_engineering.py           # Generate 75 engineered features
python src/ml/kmeans_clustering.py             # Customer segmentation
python src/ml/xgboost_ensemble_modeling.py     # Train risk profile classifier
```

### Launch the App

```bash
streamlit run app.py
```

---

## Project Structure

```
financial-recommender/
├── data/
│   ├── rag_knowledge_base/            # Financial education documents
│   └── vector_store/                  # ChromaDB persistent storage
├── src/
│   ├── ml/                            # Machine Learning components
│   │   ├── cluster_mapping.py         # ClusterMapper class
│   │   ├── etf_mf_integration.py      # Fund data integration
│   │   ├── feature_engineering.py
│   │   ├── feature_selection.py
│   │   ├── kmeans_clustering.py
│   │   ├── xgboost_ensemble_modeling.py
│   │   └── new_user_classifier.py
│   ├── Rules-RAG/                     # Rule engine + RAG pipeline
│   │   ├── rule_engine.py             # IRS eligibility rules
│   │   ├── ml_pipeline.py             # ML inference pipeline
│   │   ├── rag_pipeline.py            # RAG orchestration
│   │   └── fund_matching_rag.py       # Fund matching logic
│   ├── frontend/                      # Streamlit UI components
│   │   ├── streamlit_chatbot.py       # Main chatbot interface
│   │   ├── dashboard.py               # Savings dashboard
│   │   └── landing_page.py            # Landing page
│   └── rag_system.py                  # Core RAG implementation
├── app.py                             # Streamlit Cloud entry point
├── requirements.txt
└── README.md
```

---

## Pipeline Details

### Data Sources

| Dataset | Source | Records | Features |
|---------|--------|---------|----------|
| UCI Census-Income | UCI ML Repository | 199,523 | 42 |
| Consumer Expenditure Survey | BLS (FMLI, MEMI files) | 23,125 households | 75 (engineered) |
| US Mutual Funds & ETFs | Kaggle | 26,093 funds | 15 |

Merging these three datasets with different schemas, granularity levels, and missing value patterns required extensive alignment and custom join logic.

### Feature Engineering

75 features engineered across four categories from raw CE Survey and Census data:

- **Demographic (15):** age, family size, marital status, education, region, housing tenure, homeowner flag, family composition
- **Income (20):** total/log income, income rank & quintile, wage-to-income ratio, retirement income ratio, per-capita income, zero-income flag, high-income indicator (top 25%)
- **Expenditure (25):** total/log spending, category breakdowns (food, housing, transportation), spending ratios, essential vs. discretionary ratios, spending diversity index
- **Financial Health (15):** savings amount & rate, expenditure-to-income ratio, positive savings flag, high spender indicator, financial health tier

Key preprocessing: infinite values replaced with medians, savings rate clipped to [−2, 1], zero-income flagging for 927 households, 100% missing-value imputation. Feature selection via Random Forest importance reduced the set from 75 → 60 features (40% faster training, no accuracy loss).

### Customer Segmentation (K-Means)

K-Means clustering with StandardScaler preprocessing. UCI Census data yielded K=10 (silhouette = 0.483) and CE Survey data yielded K=2 (silhouette = 0.626). Six composite risk profiles were constructed by crossing risk tolerance (derived from age, income, wealth, employment stability) with spending behavior (essential vs. discretionary ratios): Conservative Saver, Conservative Spender, Moderate Saver, Moderate Spender, Aggressive Saver, and Aggressive Spender.

### Risk Profile Classification (XGBoost)

A 3-model ensemble (XGBoost, Random Forest, Logistic Regression) was benchmarked, with XGBoost achieving 95.87% accuracy (F1 = 0.9587). The classifier takes cluster-derived labels as training targets and predicts risk profiles for new users in real time. Validated with 80/20 stratified train/test split. The Conservative class is most separable (F1 = 0.99), while Aggressive profiles show slight boundary ambiguity with Moderate (F1 = 0.92).

### Fund Matching Engine

Content-based filtering via cosine similarity on a 5-dimensional feature vector: expense ratio, cost score, return performance, return consistency, and composite score. Achieved 90.8%–99.4% match scores across all six risk profiles, with no cold-start problem since matching is feature-based rather than interaction-based.

### Rule-Based Eligibility Engine

Applies IRS guidelines to determine user qualification for tax-advantaged accounts (401k, IRA, Roth IRA, HSA) based on income, employment status, and insurance coverage.

### RAG Chatbot

ChromaDB vector store with all-MiniLM-L6-v2 embeddings (384 dimensions, 500-token chunks, top-5 retrieval) feeding into Google Gemini 2.5 Flash for personalized explanation generation. The chatbot detects 20+ financial keywords, retrieves relevant documents via semantic similarity, and generates responses tailored to the user's financial profile. Knowledge base covers 13 categories: emergency funds, retirement, budgeting, investing, debt, taxes, insurance, real estate, education, ETFs, mutual funds, and ETF-vs-MF comparisons. No LLM fine-tuning — the RAG architecture grounds responses in retrieved fund data to minimize hallucination.

---

## Testing

Testing conducted using `unittest` and `pytest` across three levels: unit tests (data pipeline and ML functions), integration tests (end-to-end pipeline), and system tests (deployed Streamlit app).

| Module | Coverage |
|--------|---------|
| feature_engineering.py | 88% |
| kmeans_clustering.py | 86% |
| xgboost_ensemble.py | 90% |
| rag_system.py | 80% |
| streamlit_chatbot.py | 75% |
| **Overall** | **82%** |

---

## Team

- **Suchita Sharma**
- **Yogita Bisht** 
