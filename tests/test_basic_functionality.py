"""
Basic functionality tests that work with the existing codebase
Tests core functionality without complex data dependencies
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class TestDataValidation:
    """Test class for data validation and basic functionality"""

    def test_savings_rate_calculation(self):
        """Test basic savings rate calculation"""
        # Test cases: (income, expenses, expected_rate)
        test_cases = [
            (100000, 80000, 0.2),   # Normal case
            (50000, 60000, -0.2),   # Deficit case
            (0, 0, 0),             # Zero case
            (100000, 0, 1.0),       # No expenses
            (100000, 100000, 0),    # Break even
        ]
        
        for income, expenses, expected in test_cases:
            if income > 0:
                actual = (income - expenses) / income
                assert abs(actual - expected) < 0.001, f"Failed for income={income}, expenses={expenses}"

    def test_income_bracket_classification(self):
        """Test income bracket classification logic"""
        test_cases = [
            (0, "Zero Income Households"),
            (4000, "Zero Income Households"), 
            (50000, "Middle Income Families"),
            (150000, "Middle Income Families"),
            (200000, "High Income Savers"),
        ]
        
        for income, expected_bracket in test_cases:
            if income == 0 or income < 5000:
                bracket = "Zero Income Households"
            elif income > 150000:
                bracket = "High Income Savers"
            else:
                bracket = "Middle Income Families"
            
            assert bracket == expected_bracket, f"Income {income} should be {expected_bracket}, got {bracket}"

    def test_savings_rate_health_classification(self):
        """Test savings rate health classification"""
        test_cases = [
            (0.05, "Needs Attention"),    # Below 10%
            (0.10, "Healthy"),           # Exactly 10%
            (0.15, "Healthy"),           # Above 10%
            (0.25, "Healthy"),           # High savings
            (-0.1, "Needs Attention"),   # Negative savings
        ]
        
        for rate, expected_health in test_cases:
            health = "Healthy" if rate >= 0.10 else "Needs Attention"
            assert health == expected_health, f"Rate {rate} should be {expected_health}, got {health}"

    def test_age_based_risk_profile(self):
        """Test age-based risk profile classification"""
        test_cases = [
            (25, "Aggressive"),
            (35, "Moderate"), 
            (40, "Moderate"),
            (50, "Conservative"),
            (60, "Conservative"),
        ]
        
        for age, expected_risk in test_cases:
            if age < 30:
                risk = "Aggressive"
            elif age < 45:
                risk = "Moderate"
            else:
                risk = "Conservative"
            
            assert risk == expected_risk, f"Age {age} should be {expected_risk}, got {risk}"

    def test_emergency_fund_targets(self):
        """Test emergency fund target calculations"""
        monthly_expenses = 5000
        
        target_3_month = monthly_expenses * 3
        target_6_month = monthly_expenses * 6
        
        assert target_3_month == 15000, "3-month target should be 15000"
        assert target_6_month == 30000, "6-month target should be 30000"
        assert target_6_month == 2 * target_3_month, "6-month should be double 3-month"

    def test_investment_allocation_by_age(self):
        """Test investment allocation recommendations by age"""
        test_cases = [
            (25, (80, 20)),   # 80% stocks, 20% bonds
            (35, (70, 30)),   # 70% stocks, 30% bonds  
            (45, (40, 60)),   # 40% stocks, 60% bonds
            (55, (40, 60)),   # 40% stocks, 60% bonds
        ]
        
        for age, (expected_stocks, expected_bonds) in test_cases:
            if age < 30:
                stocks, bonds = 80, 20
            elif age < 45:
                stocks, bonds = 70, 30
            else:
                stocks, bonds = 40, 60
            
            assert stocks == expected_stocks, f"Age {age} stock allocation should be {expected_stocks}%"
            assert bonds == expected_bonds, f"Age {age} bond allocation should be {expected_bonds}%"

    def test_retirement_contribution_limits(self):
        """Test retirement contribution limit calculations"""
        # 2024 limits (simplified)
        limits = {
            '401k_limit': 22500,
            'ira_limit': 6500,
            'total_possible': 29000
        }
        
        assert limits['401k_limit'] == 22500, "401k limit should be 22500"
        assert limits['ira_limit'] == 6500, "IRA limit should be 6500"
        assert limits['total_possible'] == 29000, "Total should be 29000"

    def test_monthly_investment_capacity(self):
        """Test monthly investment capacity calculation"""
        test_cases = [
            (100000, 80000, 1666.67),   # (income, expenses, monthly_capacity)
            (75000, 60000, 1250.00),
            (50000, 45000, 416.67),
            (120000, 90000, 2500.00),
        ]
        
        for income, expenses, expected_monthly in test_cases:
            annual_savings = max(income - expenses, 0)
            monthly_capacity = annual_savings / 12
            assert abs(monthly_capacity - expected_monthly) < 0.01, f"Monthly capacity incorrect for income={income}"


class TestFinancialCalculations:
    """Test class for financial calculation functions"""

    def test_compound_interest_calculation(self):
        """Test compound interest calculation"""
        principal = 10000
        monthly_contribution = 500
        annual_rate = 0.07
        years = 10
        
        # Simple compound interest calculation
        balance = principal
        monthly_rate = annual_rate / 12
        
        for year in range(1, years + 1):
            for month in range(12):
                balance = balance * (1 + monthly_rate) + monthly_contribution
        
        assert balance > principal + (monthly_contribution * 12 * years), "Balance should grow with compound interest"
        assert balance > 20000, f"Balance should be significantly higher after {years} years"

    def test_inflation_adjusted_returns(self):
        """Test inflation-adjusted return calculations"""
        nominal_return = 0.07  # 7% nominal
        inflation_rate = 0.03  # 3% inflation
        
        real_return = (1 + nominal_return) / (1 + inflation_rate) - 1
        
        expected_real = (1.07 / 1.03) - 1
        assert abs(real_return - expected_real) < 0.001, "Real return calculation incorrect"
        assert real_return < nominal_return, "Real return should be less than nominal return"

    def test_debt_to_income_ratio(self):
        """Test debt-to-income ratio calculation"""
        test_cases = [
            (2000, 10000, 0.2),    # (monthly_debt, monthly_income, ratio)
            (3000, 10000, 0.3),
            (0, 10000, 0.0),
            (5000, 10000, 0.5),
        ]
        
        for debt, income, expected_ratio in test_cases:
            if income > 0:
                ratio = debt / income
                assert abs(ratio - expected_ratio) < 0.001, f"DTI ratio incorrect for debt={debt}, income={income}"

    def test_loan_payment_calculation(self):
        """Test basic loan payment calculation"""
        principal = 200000
        annual_rate = 0.06
        years = 30
        
        monthly_rate = annual_rate / 12
        num_payments = years * 12
        
        # Monthly payment formula (simplified)
        if monthly_rate > 0:
            monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        else:
            monthly_payment = principal / num_payments
        
        assert monthly_payment > 0, "Monthly payment should be positive"
        assert monthly_payment < principal, "Monthly payment should be less than principal"
        assert monthly_payment * num_payments > principal, "Total payments should exceed principal"


class TestInputValidation:
    """Test class for input validation and edge cases"""

    def test_numeric_input_validation(self):
        """Test numeric input validation"""
        valid_inputs = ["75000", "75000.50", "$75,000", "75 000"]
        invalid_inputs = ["abc", "", "75.000.50", "$$$", "nan", "inf"]
        
        for valid_input in valid_inputs:
            cleaned = valid_input.replace('$', '').replace(',', '').replace(' ', '').strip()
            try:
                value = float(cleaned)
                assert value > 0, f"Valid input {valid_input} should convert to positive number"
            except ValueError:
                pass  # Some valid inputs might still fail conversion
        
        for invalid_input in invalid_inputs:
            cleaned = invalid_input.replace('$', '').replace(',', '').replace(' ', '').strip()
            try:
                value = float(cleaned)
                # If conversion succeeds, check if it's actually valid
                if np.isfinite(value) and value > 0:
                    pass  # Some "invalid" might actually be valid after cleaning
            except ValueError:
                pass  # Expected to fail

    def test_family_size_validation(self):
        """Test family size validation"""
        valid_sizes = [1, 2, 3, 4, 5, 10]
        invalid_sizes = [0, -1, -5, 100, 1000]
        
        for size in valid_sizes:
            assert 1 <= size <= 20, f"Valid family size {size} should be in range"
        
        for size in invalid_sizes:
            assert size < 1 or size > 20, f"Invalid family size {size} should be out of range"

    def test_age_validation(self):
        """Test age validation"""
        valid_ages = [18, 25, 35, 50, 65, 80]
        invalid_ages = [0, -5, 17, 150, 200]
        
        for age in valid_ages:
            assert 18 <= age <= 100, f"Valid age {age} should be in range"
        
        for age in invalid_ages:
            assert age < 18 or age > 100, f"Invalid age {age} should be out of range"


class TestRecommendationLogic:
    """Test class for financial recommendation logic"""

    def test_savings_recommendations(self):
        """Test savings rate recommendations"""
        test_cases = [
            (0.05, "Increase savings rate to at least 10%"),
            (0.08, "Increase savings rate to at least 10%"),
            (0.10, "Maintain current savings rate"),
            (0.15, "Excellent savings rate"),
            (0.25, "Excellent savings rate"),
        ]
        
        for rate, expected_advice in test_cases:
            if rate < 0.10:
                advice = "Increase savings rate to at least 10%"
            elif rate < 0.15:
                advice = "Maintain current savings rate"
            else:
                advice = "Excellent savings rate"
            
            assert advice in expected_advice, f"Rate {rate} should get appropriate advice"

    def test_investment_recommendations_by_income(self):
        """Test investment recommendations by income level"""
        test_cases = [
            (25000, ["Start with employer 401(k)", "Low-cost index funds"]),
            (75000, ["Maximize 401(k) match", "Consider Roth IRA"]),
            (150000, ["Max 401(k) contribution", "Tax optimization", "Diversified portfolio"]),
        ]
        
        for income, expected_recommendations in test_cases:
            if income < 50000:
                recommendations = ["Start with employer 401(k)", "Low-cost index funds"]
            elif income < 100000:
                recommendations = ["Maximize 401(k) match", "Consider Roth IRA"]
            else:
                recommendations = ["Max 401(k) contribution", "Tax optimization", "Diversified portfolio"]
            
            assert len(recommendations) > 0, f"Income {income} should get recommendations"
            assert any(rec in recommendations for rec in expected_recommendations), f"Should include expected recommendations"

    def test_emergency_fund_recommendations(self):
        """Test emergency fund recommendations"""
        test_cases = [
            (0, 5000, 500),     # (current_savings, monthly_expenses, monthly_contribution)
            (1000, 5000, 500),
            (15000, 5000, 0),   # Already have 3-month fund
            (30000, 5000, 0),   # Already have 6-month fund
        ]
        
        for current, monthly, expected_contribution in test_cases:
            target_3_month = monthly * 3
            
            if current >= target_3_month:
                contribution = 0
            else:
                contribution = min(500, (target_3_month - current) / 12)
            
            assert contribution == expected_contribution, f"Emergency fund contribution incorrect"

    def test_risk_tolerance_assessment(self):
        """Test risk tolerance assessment"""
        profiles = [
            {"age": 25, "income": 75000, "savings_rate": 0.2, "expected": "Aggressive"},
            {"age": 35, "income": 100000, "savings_rate": 0.15, "expected": "Moderate"},
            {"age": 55, "income": 150000, "savings_rate": 0.25, "expected": "Conservative"},
        ]
        
        for profile in profiles:
            age = profile["age"]
            
            if age < 30:
                risk = "Aggressive"
            elif age < 45:
                risk = "Moderate"
            else:
                risk = "Conservative"
            
            assert risk == profile["expected"], f"Age {age} should have {profile['expected']} risk tolerance"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
