# Personalized Financial Recommendation System

## Overview
This branch handles **CE (Census Expenditure Data)** analysis and clustering for personalized financial recommendations. The system processes household expenditure data to identify spending patterns and provide tailored financial advice.

## Project Structure

```
├── data/                    # Raw and processed datasets
├── notebooks/               # Exploratory data analysis
├── src/                     # Source code for analysis
├── results/                 # Clustering results and outputs
└── figures/                 # Visualizations and plots
```

## Data Sources
- **Diary Data**: Household expenditure diaries (diary24/)
- **Interview Data**: Survey responses (intrvw24/)
- **Census Expenditure (CE)**: Primary dataset for analysis
- **ETF and Mutual Fund Data**: Financial product information

## Feature Engineering

### Key Processing Steps
1. **Data Cleaning**: Handle missing values and outliers
2. **Feature Transformation**: Apply skewness corrections and normalization
3. **Feature Selection**: Identify most predictive variables using statistical methods
4. **Feature Scaling**: Standardize features for clustering algorithms

### Main Features Created
- **Spending Patterns**: Categorized expenditure by type (food, housing, transportation, etc.)
- **Household Demographics**: Income brackets, family size, geographic location
- **Financial Ratios**: Savings rate, debt-to-income, discretionary spending percentage
- **Temporal Features**: Seasonal spending patterns, trend analysis

## Clustering Analysis

### Methodology
- **K-means Clustering**: Primary algorithm for household segmentation
- **Optimal K Selection**: Elbow method and silhouette analysis
- **XGBoost Ensemble**: Advanced modeling for cluster validation and prediction

### Cluster Profiles
The analysis identifies distinct household segments based on:
- **Spending Behavior**: Conservative vs. discretionary spenders
- **Income Levels**: Low, middle, and high-income households
- **Family Structure**: Single vs. multi-member households
- **Geographic Factors**: Urban vs. rural spending patterns

## Results Summary

### Key Findings
1. **Optimal Clusters**: Identified K=4 as the optimal number of household segments
2. **Cluster Characteristics**: Each segment shows unique spending patterns and financial needs
3. **Feature Importance**: Top predictors include income, housing costs, and food expenditure
4. **Model Performance**: XGBoost ensemble achieves high accuracy in cluster prediction

### Output Files
- `clustered_households.csv`: Household assignments to clusters
- `cluster_profiles.csv`: Detailed characteristics of each cluster
- `cluster_statistics.csv`: Statistical summaries for validation

## Visualizations

### Available Figures
- **K-means Analysis**: Cluster visualization and optimal K determination
- **Correlation Heatmap**: Feature relationships and dependencies
- **Feature Importance**: XGBoost feature ranking
- **Cluster Profiles**: Visual representation of spending patterns

## Usage

### Running the Analysis
1. **Data Preparation**: Place raw data in `data/` directory
2. **Feature Engineering**: Run `src/feature_engineering.py`
3. **Clustering**: Execute `src/kmeans_clustering.py`
4. **Results**: View outputs in `results/` and visualizations in `figures/`

### Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- xgboost
- jupyter (for notebooks)

## Next Steps
- Implement recommendation engine based on cluster profiles
- Add time-series analysis for spending trends
- Develop web interface for user interaction
- Integrate with financial product APIs

## Technical Notes
- Missing values handled through multiple imputation
- Skewed features transformed using log and Box-Cox methods
- Cross-validation used for model selection
- Results validated through statistical tests and domain expertise