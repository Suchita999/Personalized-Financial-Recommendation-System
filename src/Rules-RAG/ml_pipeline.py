"""
ML Pipeline for User Profile Prediction and Fund Matching
Machine learning components for personalized financial recommendations.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def load_artifacts():
    """
    Load ML model artifacts.
    
    Returns:
        dict: Dictionary containing loaded models and artifacts
    """
    artifacts = {}
    
    try:
        # Try to load user profile classifier
        model_path = Path(__file__).parent.parent.parent / "data" / "models"
        
        if (model_path / "user_classifier.pkl").exists():
            with open(model_path / "user_classifier.pkl", 'rb') as f:
                artifacts['user_classifier'] = pickle.load(f)
        
        if (model_path / "fund_matcher.pkl").exists():
            with open(model_path / "fund_matcher.pkl", 'rb') as f:
                artifacts['fund_matcher'] = pickle.load(f)
                
        if (model_path / "scaler.pkl").exists():
            with open(model_path / "scaler.pkl", 'rb') as f:
                artifacts['scaler'] = pickle.load(f)
        
        return artifacts if artifacts else None
        
    except Exception as e:
        print(f"Error loading ML artifacts: {e}")
        return None

def predict_user_profile(user_data, artifacts):
    """
    Predict user financial profile based on input data.
    
    Args:
        user_data (dict): User financial information
        artifacts (dict): Loaded ML artifacts
        
    Returns:
        dict: Predicted user profile
    """
    try:
        if not artifacts or 'user_classifier' not in artifacts:
            return fallback_profile_prediction(user_data)
        
        # Prepare features
        features = prepare_features(user_data)
        
        if 'scaler' in artifacts:
            features = artifacts['scaler'].transform([features])
        
        # Predict profile
        classifier = artifacts['user_classifier']
        prediction = classifier.predict(features)[0]
        
        return {
            'risk_profile': prediction,
            'investment_horizon': determine_horizon(user_data),
            'recommended_allocation': get_allocation(prediction)
        }
        
    except Exception as e:
        print(f"Error predicting user profile: {e}")
        return fallback_profile_prediction(user_data)

def match_funds(user_profile, available_funds, artifacts):
    """
    Match suitable funds based on user profile.
    
    Args:
        user_profile (dict): User profile information
        available_funds (list): Available investment funds
        artifacts (dict): Loaded ML artifacts
        
    Returns:
        list: Recommended funds
    """
    try:
        if not artifacts or 'fund_matcher' not in artifacts:
            return fallback_fund_matching(user_profile, available_funds)
        
        # Use ML model for fund matching
        fund_matcher = artifacts['fund_matcher']
        recommendations = fund_matcher.recommend(user_profile, available_funds)
        
        return recommendations[:5]  # Top 5 recommendations
        
    except Exception as e:
        print(f"Error matching funds: {e}")
        return fallback_fund_matching(user_profile, available_funds)

def prepare_features(user_data):
    """
    Prepare features for ML models.
    
    Args:
        user_data (dict): User financial data
        
    Returns:
        list: Feature vector
    """
    features = [
        user_data.get('income', 0) / 10000,  # Normalize income
        user_data.get('age', 30),
        user_data.get('family_size', 1),
        user_data.get('savings_rate', 0.1),
        user_data.get('risk_tolerance', 0.5)
    ]
    return features

def determine_horizon(user_data):
    """Determine investment horizon based on user data."""
    age = user_data.get('age', 30)
    if age < 35:
        return "long_term"
    elif age < 50:
        return "medium_term"
    else:
        return "short_term"

def get_allocation(risk_profile):
    """Get recommended asset allocation based on risk profile."""
    allocations = {
        'conservative': {'stocks': 40, 'bonds': 50, 'cash': 10},
        'moderate': {'stocks': 60, 'bonds': 30, 'cash': 10},
        'aggressive': {'stocks': 80, 'bonds': 15, 'cash': 5}
    }
    return allocations.get(risk_profile, allocations['moderate'])

def fallback_profile_prediction(user_data):
    """Fallback prediction when ML models are not available."""
    income = user_data.get('income', 0)
    age = user_data.get('age', 30)
    
    if age < 35 and income > 80000:
        risk_profile = 'aggressive'
    elif age > 50 or income < 40000:
        risk_profile = 'conservative'
    else:
        risk_profile = 'moderate'
    
    return {
        'risk_profile': risk_profile,
        'investment_horizon': determine_horizon(user_data),
        'recommended_allocation': get_allocation(risk_profile)
    }

def fallback_fund_matching(user_profile, available_funds):
    """Fallback fund matching when ML models are not available."""
    risk_profile = user_profile.get('risk_profile', 'moderate')
    
    # Simple rule-based matching
    if risk_profile == 'conservative':
        return [f for f in available_funds if 'bond' in f.get('name', '').lower()][:3]
    elif risk_profile == 'aggressive':
        return [f for f in available_funds if 'growth' in f.get('name', '').lower()][:3]
    else:
        return available_funds[:3]  # Moderate - balanced approach
