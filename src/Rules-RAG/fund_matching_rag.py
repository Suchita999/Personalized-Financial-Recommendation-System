"""
Fund Matching with RAG Integration
Combines fund matching algorithms with RAG for detailed fund analysis.
"""

import json
from pathlib import Path

def load_artifacts():
    """
    Load fund matching artifacts and data.
    
    Returns:
        dict: Fund data and matching artifacts
    """
    try:
        # Load fund database
        data_path = Path(__file__).parent.parent.parent / "data"
        
        if (data_path / "funds_database.json").exists():
            with open(data_path / "funds_database.json", 'r') as f:
                funds_data = json.load(f)
        else:
            funds_data = get_default_funds()
        
        return {
            'funds_database': funds_data,
            'fund_categories': categorize_funds(funds_data)
        }
        
    except Exception as e:
        print(f"Error loading fund artifacts: {e}")
        return {'funds_database': get_default_funds()}

def get_default_funds():
    """Get default fund database when file is not available."""
    return [
        {
            "ticker": "VFIAX",
            "name": "Vanguard 500 Index Admiral",
            "category": "Large Cap Growth",
            "expense_ratio": 0.04,
            "risk_level": "moderate",
            "min_investment": 3000,
            "description": "Tracks S&P 500 index"
        },
        {
            "ticker": "VTIAX",
            "name": "Vanguard Total Intl Stock Admiral",
            "category": "International",
            "expense_ratio": 0.11,
            "risk_level": "moderate",
            "min_investment": 3000,
            "description": "International stock market exposure"
        },
        {
            "ticker": "VBTLX",
            "name": "Vanguard Total Bond Market Admiral",
            "category": "Bond",
            "expense_ratio": 0.05,
            "risk_level": "conservative",
            "min_investment": 3000,
            "description": "Total bond market index fund"
        }
    ]

def categorize_funds(funds_data):
    """Categorize funds by type and risk."""
    categories = {
        'conservative': [],
        'moderate': [],
        'aggressive': []
    }
    
    for fund in funds_data:
        risk = fund.get('risk_level', 'moderate')
        categories[risk].append(fund)
    
    return categories

def recommend_funds(user_profile, funds_database, top_n=5):
    """
    Recommend funds based on user profile.
    
    Args:
        user_profile (dict): User risk profile and preferences
        funds_database (list): Available funds
        top_n (int): Number of recommendations to return
        
    Returns:
        list: Recommended funds with scores
    """
    risk_profile = user_profile.get('risk_profile', 'moderate')
    investment_amount = user_profile.get('investment_amount', 10000)
    
    # Score funds based on user profile
    scored_funds = []
    for fund in funds_database:
        score = calculate_fund_score(fund, user_profile, investment_amount)
        scored_funds.append({
            'fund': fund,
            'score': score,
            'match_reason': get_match_reason(fund, user_profile)
        })
    
    # Sort by score and return top recommendations
    scored_funds.sort(key=lambda x: x['score'], reverse=True)
    return scored_funds[:top_n]

def calculate_fund_score(fund, user_profile, investment_amount):
    """
    Calculate compatibility score for a fund.
    
    Args:
        fund (dict): Fund information
        user_profile (dict): User profile
        investment_amount (float): Available investment amount
        
    Returns:
        float: Compatibility score (0-100)
    """
    score = 50  # Base score
    
    # Risk profile matching
    fund_risk = fund.get('risk_level', 'moderate')
    user_risk = user_profile.get('risk_profile', 'moderate')
    
    risk_match = {
        ('conservative', 'conservative'): 30,
        ('moderate', 'moderate'): 25,
        ('aggressive', 'aggressive'): 30,
        ('conservative', 'moderate'): 15,
        ('moderate', 'conservative'): 10,
        ('moderate', 'aggressive'): 10,
        ('aggressive', 'moderate'): 15,
        ('conservative', 'aggressive'): 0,
        ('aggressive', 'conservative'): 0
    }
    
    score += risk_match.get((fund_risk, user_risk), 0)
    
    # Minimum investment check
    min_investment = fund.get('min_investment', 0)
    if investment_amount >= min_investment:
        score += 20
    else:
        score -= 30
    
    # Expense ratio bonus (lower is better)
    expense_ratio = fund.get('expense_ratio', 1.0)
    if expense_ratio < 0.1:
        score += 10
    elif expense_ratio < 0.5:
        score += 5
    
    return min(score, 100)

def get_match_reason(fund, user_profile):
    """Generate reason for fund recommendation."""
    reasons = []
    
    fund_risk = fund.get('risk_level', 'moderate')
    user_risk = user_profile.get('risk_profile', 'moderate')
    
    if fund_risk == user_risk:
        reasons.append(f"Matches your {user_risk} risk profile")
    
    expense_ratio = fund.get('expense_ratio', 1.0)
    if expense_ratio < 0.1:
        reasons.append("Very low expense ratio")
    
    category = fund.get('category', '')
    if 'Index' in fund.get('name', ''):
        reasons.append("Low-cost index fund")
    
    return " | ".join(reasons) if reasons else "Well-diversified option"
