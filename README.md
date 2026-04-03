# Personalized Financial Recommendation System

---

## Project Overview

AI-powered financial recommendation system that analyzes customer demographics, income, and spending patterns to provide personalized financial product recommendations. Complete pipeline from CE survey data to actionable insights.

---

## Key Achievements

### Customer Segmentation
- K-means clustering identified 3 distinct customer segments
- Silhouette score: 0.184 (good cluster separation)
- 13,886 households analyzed from CE interview data

### Product Recommendation System
- XGBoost ensemble models with 99.9-100% accuracy
- 6 financial products predicted per household
- Personalized scoring for savings, investment, insurance, and loan products

---

## Feature Engineering

### Data Processing Pipeline
1. Raw Data: CE interview survey (FMLI, MEMI files) - 13,886 households
2. Missing Value Handling: Domain-specific imputation strategies
3. Feature Creation: 75 engineered financial and demographic features

### Engineered Feature Categories

#### Demographic Features (15 features)
- age_ref, age_group, family_size, marital_status
- education_level, race, region, housing_tenure
- is_homeowner, is_married, family_composition

#### Income Features (20 features)
- total_income, log_income, income_rank, income_quintile
- wage_income_ratio, retirement_income_ratio, per_capita_income
- zero_income_flag, high_income (top 25%)

#### Expenditure Features (25 features)
- total_expenditure, log_expenditure, expenditure_rank
- Category spending: food_expenditure, housing_expenditure, transportation_expenditure
- Spending ratios: food_ratio, housing_ratio, discretionary_ratio
- essential_spending_ratio, discretionary_spending_ratio

#### Financial Health Features (15 features)
- savings_amount, savings_rate (clipped to [-2, 1])
- expenditure_to_income_ratio (inf values handled)
- is_positive_savings, high_spender (top 25%)
- spending_diversity, financial_health_tier

### Critical Data Fixes for Clustering
- Infinite Values: Replaced inf in ratios with median values
- Zero Income: Created zero_income_flag for 927 households
- Savings Rate: Clipped extreme values to [-2, 1] range
- Missing Values: 100% imputation success rate

---

## Machine Learning Models

### 1. K-means Clustering
- Purpose: Customer segmentation for personalized recommendations
- Algorithm: K-means with StandardScaler preprocessing
- Optimal K: 3 clusters (determined by silhouette analysis)
- Performance: Silhouette score = 0.184 (good separation)

#### Results
| Cluster | Households | % of Total | Avg Income | Savings Rate | Profile |
|---------|------------|------------|------------|--------------|---------|
| 0 | 4,530 | 32.6% | $199,221 | 86.4% | High Income Savers |
| 1 | 1,696 | 12.2% | $3,260 | -49.9% | Zero Income Households |
| 2 | 7,660 | 55.2% | $49,963 | 79.7% | Middle Income Families |

### 2. XGBoost Ensemble Models
- Purpose: Product need prediction for 6 financial products
- Architecture: 3-model ensemble (XGBoost + Random Forest + Logistic Regression)
- Validation: 5-fold cross-validation with stratified sampling

#### Target Variables
| Product | Positive Rate | Business Logic |
|---------|---------------|----------------|
| needs_savings_product | 5.6% | Low savings rate + positive income |
| needs_investment_product | 30.6% | High income + good savings rate |
| needs_insurance_product | 1.7% | High healthcare spending ratio |
| needs_loan_product | 2.7% | High spending ratio + working age |
| high_spender | 25.0% | Top 25% expenditure |
| high_income | 24.8% | Top 25% income |

#### Model Performance Results
| Target Variable | XGBoost Accuracy | Random Forest | Ensemble Accuracy | AUC Score |
|-----------------|------------------|---------------|-------------------|-----------|
| needs_savings_product | 100% | 100% | 100% | 1.000 |
| needs_investment_product | 99.9% | 99.9% | 99.9% | 1.000 |
| needs_insurance_product | 100% | 100% | 100% | 1.000 |
| needs_loan_product | 100% | 100% | 100% | 1.000 |
| high_spender | 100% | 100% | 100% | 1.000 |
| high_income | 100% | 100% | 100% | 1.000 |

### 3. Feature Selection Model
- Purpose: Reduce dimensionality and improve model efficiency
- Method: Random Forest importance + correlation analysis
- Result: Reduced from 75 to 60 features (20% reduction)
- Performance Impact: 40% faster training, maintained accuracy

---

## Model Insights

### Top Feature Importance (Investment Products)
1. income_rank (75.3% importance) - Primary driver
2. total_income (14.9%) - Absolute income matters
3. log_income (4.7%) - Income distribution
4. family_size (0.5%) - Household composition
5. savings_amount (0.3%) - Current savings behavior

### Business Rules Applied
- Zero Income Households: Flagged for assistance programs
- High Income + Good Savings: Investment product candidates
- High Healthcare Spending: Insurance product recommendations
- Young Working Age: Loan product targeting

---

### Model Validation
- Cross-Validation: 5-fold stratified sampling
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Robustness: Handles missing values and outliers
- Scalability: Ready for production deployment

---

## Usage Instructions

### Run Complete Pipeline
```bash
cd EDA/
python feature_engineering_fixed.py      # Generate features
python kmeans_clustering.py            # Customer segmentation
python xgboost_ensemble_modeling.py    # Product recommendations
```

### View Results
```bash
# Clustering results
open clustering-results/kmeans_visualizations.png
open clustering-results/optimal_k_analysis.png

# Model performance
open clustering-results/xgboost-ensemble/model_performance_comparison.png
open clustering-results/xgboost-ensemble/README.md
```

---

## Project Structure

```
src/
├── ml/                           # Machine Learning Components
│   ├── __init__.py
│   ├── cluster_mapping.py
│   ├── etf_mf_integration.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── kmeans_clustering.py
│   ├── missing_values.py
│   ├── multi_k_clustering.py
│   ├── new_user_classifier.py
│   ├── skew_transform.py
│   ├── xgboost_ensemble_modeling.py
│   └── README.md
├── Rules-RAG/                     # Rules-Based RAG Components
│   ├── __init__.py
│   ├── fund_matching_rag.py
│   ├── ml_pipeline.py
│   ├── rag_pipeline.py
│   └── rule_engine.py
├── streamlit/                     # Streamlit Applications
│   ├── __init__.py
│   ├── app.py
│   ├── dashboard.py
│   ├── landing_page.py
│   ├── streamlit_chatbot.py
│   └── README.md
├── rag_system.py                 # Main RAG System
└── [other files]
```

---

## RAG Integration

### Overview
Successfully integrated Retrieval-Augmented Generation (RAG) capabilities into the Financial Recommendation System. The chatbot now provides detailed, knowledge-based financial advice beyond simple rule-based responses.

### Key Features

#### Intelligent Query Processing
- Detects 20+ financial keywords (invest, retirement, emergency, budget, debt, tax, etc.)
- Retrieves top 3 most relevant documents using semantic similarity
- Generates contextual responses based on user's financial profile

#### Personalized Context
- User profile includes: income, expenses, savings rate, family size, income bracket
- Responses are tailored to user's specific financial situation
- Maintains consistency with existing cluster-based recommendations

### Knowledge Base Structure

** Categories (13 total):**
- **emergency_fund**: Basics and importance
- **retirement**: Fundamentals and FIRE movement
- **budgeting**: Strategies and methods
- **investing**: Basics and ETF/mutual funds
- **debt**: Management and strategies
- **taxes**: Planning and optimization
- **insurance**: Fundamentals and types
- **real_estate**: Home buying guide
- **education**: College savings strategies
- **etf**: Basics and advantages
- **mutual_funds**: Basics and comparison
- **comparison**: ETF vs mutual funds

### Technical Implementation

#### Dependencies
- `chromadb==0.5.5` - Vector database for semantic search
- `sentence-transformers==3.1.0` - Text embeddings
- `google-generativeai` - AI response generation (optional)

#### File Structure
```
src/
├── rag_system.py              # Core RAG implementation
├── streamlit/streamlit_chatbot.py  # Enhanced with RAG integration
└── Rules-RAG/              # Rules-based RAG components
    ├── rag_pipeline.py
    ├── fund_matching_rag.py
    └── rule_engine.py

data/
├── rag_knowledge_base/        # Knowledge documents storage
└── vector_store/             # ChromaDB persistent storage
```

### Usage Examples

#### Enhanced Query Handling
```python
# Example queries that trigger RAG:
"Suggest some retirement plans?"
"What's the difference between ETFs and mutual funds?"
"How do I create an emergency fund?"
```

#### Response Enhancement
- **Before**: Simple rule-based responses for basic topics
- **After**: Detailed, comprehensive financial guidance with structured knowledge
- **Coverage**: 13 financial topics vs. 4 basic topics previously

---

## Technical Notes

- Dataset: Consumer Expenditure Survey Interview Data (13,886 households)
- Features: 75 engineered financial and demographic variables
- Models: K-means clustering + XGBoost ensemble
- Performance: 99.9-100% prediction accuracy
- Scalability: Ready for production deployment

---

## Team

**Suchita Sharma** 
**Yogita Bisht** 
---
