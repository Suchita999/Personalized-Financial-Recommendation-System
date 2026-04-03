"""
Financial Rule Engine
Applies financial rules based on user profile and circumstances.
"""

def apply_rules(user_info, funds):
    """
    Apply financial rules based on user profile and available funds.
    
    Args:
        user_info (dict): User financial information
        funds (list): Available investment funds
        
    Returns:
        dict: Rule application results and recommendations
    """
    rules_results = {
        'recommendations': [],
        'warnings': [],
        'priority_actions': []
    }
    
    # Example rules based on user profile
    income = user_info.get('income', 0)
    expenses = user_info.get('expenses', 0)
    savings_rate = user_info.get('savings_rate', 0)
    family_size = user_info.get('family_size', 1)
    
    # Emergency fund rule
    monthly_expenses = expenses / 12 if expenses > 0 else 0
    emergency_fund_target = monthly_expenses * 6  # 6 months
    
    if savings_rate < 0.10:
        rules_results['warnings'].append("Low savings rate - aim for at least 10%")
        rules_results['priority_actions'].append("Build emergency fund first")
    
    # Investment allocation rules
    if income > 100000:
        rules_results['recommendations'].append("Consider tax-advantaged accounts")
    if family_size > 2:
        rules_results['recommendations'].append("Increase life insurance coverage")
    
    return rules_results
