# Machine Learning Components

This folder contains all machine learning models, data processing, and analytics components for the FinWise Financial Recommendation System.

## Files Structure

```
ml/
├── __init__.py               # Package initialization
├── cluster_mapping.py        # User profile clustering and mapping
├── etf_mf_integration.py     # ETF and Mutual Fund data integration
├── feature_engineering.py   # Feature extraction and engineering
├── feature_selection.py     # Feature selection algorithms
├── kmeans_clustering.py     # K-means clustering implementation
├── missing_values.py        # Missing value handling
├── multi_k_clustering.py    # Multiple K clustering analysis
├── new_user_classifier.py   # New user classification
├── skew_transform.py        # Data skewness transformation
├── xgboost_ensemble_modeling.py # XGBoost ensemble models
└── README.md                # This file
```

## Components

### Data Processing
- **missing_values.py**: Handles missing data in financial datasets
- **skew_transform.py**: Transforms skewed financial data
- **feature_engineering.py**: Creates meaningful features from raw data

### Clustering & Classification
- **kmeans_clustering.py**: K-means clustering for user segmentation
- **multi_k_clustering.py**: Multiple K clustering analysis
- **new_user_classifier.py**: Classification for new users
- **cluster_mapping.py**: Maps clusters to financial recommendations

### Feature Analysis
- **feature_selection.py**: Selects most important features
- **cluster_mapping.py**: Maps features to user profiles

### Modeling
- **xgboost_ensemble_modeling.py**: XGBoost ensemble models for prediction
- **etf_mf_integration.py**: Integration with ETF/Mutual Fund data

### Integration
- **etf_mf_integration.py**: Connects ML models with financial product data
- **cluster_mapping.py**: Maps ML outputs to actionable recommendations

## Usage

These components are used by:
- **Chatbot**: For user profiling and recommendations
- **Dashboard**: For analytics and visualizations
- **RAG System**: For contextual financial advice

## Dependencies

All components share the same dependencies as the main project:
- scikit-learn
- pandas
- numpy
- xgboost
- plotly (for visualization)

## Data Flow

1. **Input**: User financial data
2. **Processing**: Feature engineering and cleaning
3. **Modeling**: Clustering and classification
4. **Output**: Personalized recommendations

## Model Types

- **Clustering**: User segmentation (K-means, multiple K)
- **Classification**: User category prediction
- **Ensemble**: XGBoost for accurate predictions
- **Integration**: Mapping to financial products
