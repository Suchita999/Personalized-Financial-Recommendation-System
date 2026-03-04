"""
XGBoost and Ensemble Modeling for Financial Product Prediction
Uses clustered CE interview data for product recommendation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class XGBoostEnsembleModeler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.feature_cols = None
        self.target_cols = None
        self.models = {}
        self.ensemble_model = None
        self.results = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load clustered data"""
        print("Loading clustered data...")
        self.data = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.data):,} records with {self.data.shape[1]} features")
        
        # Define target variables for product prediction
        self.target_cols = [
            'needs_savings_product',
            'needs_investment_product', 
            'needs_insurance_product',
            'needs_loan_product',
            'high_spender',
            'high_income'
        ]
        
        print(f"✓ Target variables: {self.target_cols}")
        
    def prepare_features(self, exclude_cols=None):
        """Prepare features for modeling"""
        print("Preparing features for modeling...")
        
        if exclude_cols is None:
            exclude_cols = ['NEWID', 'quarter', 'financial_health_tier', 
                          'primary_spending_category', 'age_group', 'region',
                          'housing_tenure', 'education_level', 'marital_status', 'race']
        
        # Select numeric features + cluster
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [col for col in numeric_cols if col not in exclude_cols + self.target_cols]
        
        print(f"Selected {len(self.feature_cols)} features for modeling")
        
        # Handle missing values
        X = self.data[self.feature_cols].copy()
        X = X.fillna(X.median())
        
        return X
    
    def train_xgboost_model(self, X, y, target_name):
        """Train XGBoost model for specific target"""
        print(f"Training XGBoost for {target_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.models[target_name] = xgb_model
        self.feature_importance[target_name] = feature_importance
        self.results[target_name] = metrics
        
        print(f"✓ XGBoost trained - Accuracy: {metrics['accuracy']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        
        return xgb_model, metrics, feature_importance
    
    def train_random_forest(self, X, y, target_name):
        """Train Random Forest model"""
        print(f"Training Random Forest for {target_name}...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"Random Forest trained - Accuracy: {metrics['accuracy']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        
        return rf_model, metrics
    
    def create_ensemble_model(self, X, y, target_name):
        """Create voting ensemble model"""
        print(f"Creating ensemble for {target_name}...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Base models
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        )
        
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('lr', lr_model)
            ],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"Ensemble trained - Accuracy: {metrics['accuracy']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        
        return ensemble, metrics
    
    def run_all_models(self):
        """Train models for all target variables"""
        print("Starting comprehensive modeling pipeline...")
        
        X = self.prepare_features()
        all_results = []
        
        for target in self.target_cols:
            print(f"\n{'='*60}")
            print(f"Modeling for: {target}")
            print(f"{'='*60}")
            
            y = self.data[target]
            positive_rate = y.mean() * 100
            print(f"Positive rate: {positive_rate:.1f}%")
            
            # Train XGBoost
            xgb_model, xgb_metrics, xgb_importance = self.train_xgboost_model(X, y, target)
            
            # Train Random Forest
            rf_model, rf_metrics = self.train_random_forest(X, y, target)
            
            # Train Ensemble
            ensemble_model, ensemble_metrics = self.create_ensemble_model(X, y, target)
            
            # Store ensemble model
            self.ensemble_model = ensemble_model
            
            # Compile results
            result_row = {
                'target': target,
                'positive_rate': positive_rate,
                'xgb_accuracy': xgb_metrics['accuracy'],
                'xgb_auc': xgb_metrics['roc_auc'],
                'rf_accuracy': rf_metrics['accuracy'],
                'rf_auc': rf_metrics['roc_auc'],
                'ensemble_accuracy': ensemble_metrics['accuracy'],
                'ensemble_auc': ensemble_metrics['roc_auc']
            }
            all_results.append(result_row)
        
        self.results_summary = pd.DataFrame(all_results)
        print("\nAll models trained successfully!")
        
        return self.results_summary
    
    def plot_model_comparison(self):
        """Plot model performance comparison"""
        if not hasattr(self, 'results_summary'):
            print("Run models first using run_all_models()")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy comparison
        x = np.arange(len(self.results_summary))
        width = 0.25
        
        axes[0, 0].bar(x - width, self.results_summary['xgb_accuracy'], width, 
                      label='XGBoost', color='skyblue')
        axes[0, 0].bar(x, self.results_summary['rf_accuracy'], width, 
                      label='Random Forest', color='lightgreen')
        axes[0, 0].bar(x + width, self.results_summary['ensemble_accuracy'], width, 
                      label='Ensemble', color='salmon')
        
        axes[0, 0].set_xlabel('Target Variable')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.results_summary['target'], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC comparison
        axes[0, 1].bar(x - width, self.results_summary['xgb_auc'], width, 
                      label='XGBoost', color='skyblue')
        axes[0, 1].bar(x, self.results_summary['rf_auc'], width, 
                      label='Random Forest', color='lightgreen')
        axes[0, 1].bar(x + width, self.results_summary['ensemble_auc'], width, 
                      label='Ensemble', color='salmon')
        
        axes[0, 1].set_xlabel('Target Variable')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].set_title('Model AUC Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.results_summary['target'], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Positive rate distribution
        axes[1, 0].bar(self.results_summary['target'], self.results_summary['positive_rate'], 
                      color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Target Variable')
        axes[1, 0].set_ylabel('Positive Rate (%)')
        axes[1, 0].set_title('Target Variable Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Best model by accuracy
        best_accuracy = self.results_summary[['xgb_accuracy', 'rf_accuracy', 'ensemble_accuracy']].max(axis=1)
        axes[1, 1].bar(self.results_summary['target'], best_accuracy, color='gold', alpha=0.7)
        axes[1, 1].set_xlabel('Target Variable')
        axes[1, 1].set_ylabel('Best Accuracy')
        axes[1, 1].set_title('Best Model Accuracy by Target')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./clustering-results/model_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, target_name, top_n=15):
        """Plot feature importance for specific target"""
        if target_name not in self.feature_importance:
            print(f"No feature importance available for {target_name}")
            return
        
        importance_df = self.feature_importance[target_name].head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {target_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(f'./clustering-results/feature_importance_{target_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_predictions(self, save_predictions=True):
        """Generate predictions for all households"""
        print("Generating predictions for all households...")
        
        X = self.prepare_features()
        predictions_df = self.data[['NEWID']].copy()
        
        for target in self.target_cols:
            if target in self.models:
                model = self.models[target]
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]
                
                predictions_df[f'{target}_pred'] = predictions
                predictions_df[f'{target}_prob'] = probabilities
        
        if save_predictions:
            predictions_df.to_csv('./clustering-results/household_predictions.csv', index=False)
            print("Predictions saved to clustering-results/household_predictions.csv")
        
        return predictions_df
    
    def save_results(self):
        """Save all modeling results"""
        print("Saving modeling results...")
        
        # Save model performance summary
        if hasattr(self, 'results_summary'):
            self.results_summary.to_csv('./clustering-results/model_performance_summary.csv', index=False)
        
        # Save feature importance for all targets
        for target, importance_df in self.feature_importance.items():
            importance_df.to_csv(f'./clustering-results/feature_importance_{target}.csv', index=False)
        
        print("All results saved to clustering-results/")

def main():
    """Main function to run XGBoost and ensemble modeling"""
    # Initialize modeler
    data_path = './clustering-results/clustered_households.csv'
    modeler = XGBoostEnsembleModeler(data_path)
    
    # Load data and run all models
    modeler.load_data()
    results_summary = modeler.run_all_models()
    
    # Display results
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(results_summary.to_string(index=False, float_format='%.3f'))
    
    # Generate visualizations
    modeler.plot_model_comparison()
    
    # Plot feature importance for key targets
    key_targets = ['needs_savings_product', 'needs_investment_product', 'needs_insurance_product']
    for target in key_targets:
        modeler.plot_feature_importance(target, top_n=15)
    
    # Generate predictions
    predictions_df = modeler.generate_predictions()
    
    # Save all results
    modeler.save_results()
    
    print("\nXGBoost and ensemble modeling complete!")
    return modeler, results_summary

if __name__ == "__main__":
    modeler, results_summary = main()
