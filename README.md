# 💰 Personalized Financial Recommendation System

An AI-powered system that analyzes customer demographics, income, and spending patterns to deliver personalized financial product recommendations — from customer segmentation to actionable insights.

🔗 **Live Demo:** [streamlit app](https://vmwg7knebklvbjmrwbggns.streamlit.app/)

---

## Table of Contents

- [About](#about)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Pipeline Details](#pipeline-details)
  - [Feature Engineering](#feature-engineering)
  - [Customer Segmentation](#customer-segmentation)
  - [Product Recommendation Models](#product-recommendation-models)
  - [RAG Chatbot](#rag-chatbot)
- [Team](#team)

---

## About

This project builds an end-to-end ML pipeline using **Consumer Expenditure (CE) Survey** data (13,886 households) to:

1. **Segment customers** into distinct financial profiles via K-Means clustering
2. **Predict product needs** (savings, investment, insurance, loans) using an XGBoost ensemble
3. **Match users to funds** (ETFs & mutual funds) through content-based cosine similarity
4. **Determine account eligibility** (IRA, 401k, HSA) with a rule-based engine
5. **Answer financial questions** via a RAG chatbot powered by ChromaDB + LangChain + Google Gemini 2.5 Flash

---

## Tech Stack

| Layer | Tools |
|---|---|
| **ML & Data** | Python, Scikit-learn, XGBoost, Pandas, NumPy |
| **Clustering** | K-Means, StandardScaler, Silhouette Analysis |
| **NLP / RAG** | LangChain, ChromaDB, Sentence-Transformers, Google Gemini 2.5 Flash |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Community Cloud |

---

## Architecture

```
CE Survey Data (FMLI, MEMI)
        │
        ▼
┌──────────────────────┐
│  Feature Engineering  │  75 engineered features
│  (imputation, ratios, │  across demographics,
│   flags, clipping)    │  income, expenditure,
└──────────┬───────────┘  and financial health
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────────┐
│ K-Means │  │ XGBoost Ensemble │
│ (3 seg) │  │ (6 product needs)│
└────┬────┘  └───────┬──────────┘
     │               │
     ▼               ▼
┌─────────────────────────────┐
│     Recommendation Layer     │
│  • Fund matching (cosine)    │
│  • Account eligibility rules │
│  • RAG chatbot (ChromaDB +   │
│    Gemini 2.5 Flash)         │
└──────────────┬───────────────┘
               ▼
        Streamlit App
```

---

## Key Results

### Customer Segmentation (K-Means, k=3)

| Cluster | Households | Avg Income | Savings Rate | Profile |
|---------|-----------|-----------|-------------|---------|
| 0 | 4,530 (32.6%) | $199,221 | 86.4% | High-Income Savers |
| 1 | 1,696 (12.2%) | $3,260 | −49.9% | Zero-Income Households |
| 2 | 7,660 (55.2%) | $49,963 | 79.7% | Middle-Income Families |

Silhouette score: **0.184**

### Product Need Prediction (XGBoost + RF + Logistic Regression Ensemble)

| Target | Positive Rate | Ensemble Accuracy | AUC |
|--------|--------------|-------------------|-----|
| Savings product | 5.6% | 100% | 1.000 |
| Investment product | 30.6% | 99.9% | 1.000 |
| Insurance product | 1.7% | 100% | 1.000 |
| Loan product | 2.7% | 100% | 1.000 |
| High spender | 25.0% | 100% | 1.000 |
| High income | 24.8% | 100% | 1.000 |

Validated with 5-fold stratified cross-validation.

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
cd EDA/
python feature_engineering_fixed.py      # Generate 75 engineered features
python kmeans_clustering.py              # Customer segmentation
python xgboost_ensemble_modeling.py      # Product need predictions
```

### Launch the App

```bash
streamlit run src/front_end/app.py
```

---

## Project Structure

```
src/
├── ml/                              # ML pipeline
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── kmeans_clustering.py
│   ├── xgboost_ensemble_modeling.py
│   ├── new_user_classifier.py
│   ├── cluster_mapping.py
│   ├── etf_mf_integration.py
│   ├── missing_values.py
│   ├── multi_k_clustering.py
│   └── skew_transform.py
├── Rules-RAG/                       # RAG + rule engine
│   ├── fund_matching_rag.py
│   ├── ml_pipeline.py
│   ├── rag_pipeline.py
│   └── rule_engine.py
├── front_end/                       # Streamlit UI
│   ├── app.py
│   ├── dashboard.py
│   ├── landing_page.py
│   └── streamlit_chatbot.py
├── rag_system.py                    # Core RAG implementation
data/
├── rag_knowledge_base/              # Financial knowledge docs
└── vector_store/                    # ChromaDB persistent storage
```

---

## Pipeline Details

### Feature Engineering

75 features across four categories, engineered from raw CE Survey data:

- **Demographic (15):** age, family size, marital status, education, region, housing tenure, etc.
- **Income (20):** total/log income, income rank & quintile, wage ratio, per-capita income, zero-income flag
- **Expenditure (25):** total/log spending, category breakdowns (food, housing, transport), spending ratios, diversity index
- **Financial Health (15):** savings amount & rate, expenditure-to-income ratio, financial health tier

Key preprocessing steps: infinite values replaced with medians, savings rate clipped to [−2, 1], zero-income flagging for 927 households, 100% missing-value imputation.

### Customer Segmentation

K-Means clustering with StandardScaler preprocessing. Optimal k=3 determined via silhouette analysis. Segments drive downstream recommendation logic — e.g., zero-income households are flagged for assistance programs rather than investment products.

### Product Recommendation Models

A 3-model ensemble (XGBoost + Random Forest + Logistic Regression) predicts need for 6 financial products per household. Business rules define the target labels (e.g., "needs insurance" = high healthcare spending ratio). Feature selection reduced input from 75 → 60 features with a 40% training speedup and no accuracy loss. Top driver for investment recommendations: `income_rank` at 75.3% importance.

### RAG Chatbot

The chatbot uses ChromaDB for vector storage, Sentence-Transformers for embeddings, and Google Gemini 2.5 Flash for response generation. It detects 20+ financial keywords, retrieves the top 3 relevant documents, and generates responses personalized to the user's financial profile. Knowledge base spans 13 categories: emergency funds, retirement, budgeting, investing, debt, taxes, insurance, real estate, education, ETFs, mutual funds, and ETF-vs-MF comparisons.

---

## Team

- **Suchita Sharma**
- **Yogita Bisht**
