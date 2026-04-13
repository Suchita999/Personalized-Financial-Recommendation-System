"""
Pytest configuration and shared fixtures for the Financial Recommendation System
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.append(str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_financial_data():
    """Generate sample financial data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'total_income': np.random.lognormal(10.5, 0.8, n_samples),
        'total_expenditure': np.random.lognormal(10.2, 0.7, n_samples),
        'family_size': np.random.randint(1, 6, n_samples),
        'age': np.random.randint(18, 75, n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'housing_tenure': np.random.choice(['Owned', 'Rented'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], n_samples),
        'region': np.random.choice(['Northeast', 'South', 'Midwest', 'West'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic relationships
    df.loc[df['total_income'] > 150000, 'education_level'] = 'Master'
    df.loc[df['total_income'] > 200000, 'education_level'] = 'PhD'
    df.loc[df['age'] < 30, 'family_size'] = np.random.randint(1, 3, len(df.loc[df['age'] < 30]))
    df.loc[df['age'] > 50, 'family_size'] = np.random.randint(1, 4, len(df.loc[df['age'] > 50]))
    
    # Ensure some realistic constraints
    df['total_expenditure'] = np.minimum(df['total_expenditure'], df['total_income'] * 1.2)
    
    return df


@pytest.fixture(scope="session")
def sample_user_profiles():
    """Generate sample user profiles for testing"""
    return [
        {
            'name': 'Young Professional',
            'income': 85000,
            'family_size': 1,
            'expenses': 65000,
            'age': 28,
            'marital_status': 'Single',
            'education_level': 'Bachelor',
            'expected_cluster': 2,  # Middle Income Families
            'expected_bracket': 'Middle Income Families'
        },
        {
            'name': 'Family with Children',
            'income': 120000,
            'family_size': 5,
            'expenses': 100000,
            'age': 38,
            'marital_status': 'Married',
            'education_level': 'Master',
            'expected_cluster': 2,  # Middle Income Families
            'expected_bracket': 'Middle Income Families'
        },
        {
            'name': 'High Income Saver',
            'income': 250000,
            'family_size': 2,
            'expenses': 120000,
            'age': 45,
            'marital_status': 'Married',
            'education_level': 'PhD',
            'expected_cluster': 0,  # High Income Savers
            'expected_bracket': 'High Income Savers'
        },
        {
            'name': 'Low Income Household',
            'income': 25000,
            'family_size': 3,
            'expenses': 24000,
            'age': 32,
            'marital_status': 'Single',
            'education_level': 'High School',
            'expected_cluster': 2,  # Middle Income Families
            'expected_bracket': 'Middle Income Families'
        },
        {
            'name': 'Zero Income Household',
            'income': 0,
            'family_size': 1,
            'expenses': 12000,
            'age': 22,
            'marital_status': 'Single',
            'education_level': 'High School',
            'expected_cluster': 1,  # Zero Income Households
            'expected_bracket': 'Zero Income Households'
        }
    ]


@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state for testing"""
    return {
        'user_data': {},
        'messages': [],
        'current_step': 'income',
        'classification_done': False,
        'rag_enabled': True,
        'rag_ready': False
    }


@pytest.fixture
def financial_thresholds():
    """Financial thresholds for testing"""
    return {
        'high_income_threshold': 150000,
        'low_income_threshold': 25000,
        'healthy_savings_rate': 0.10,
        'excellent_savings_rate': 0.20,
        'max_family_size': 20,
        'min_age': 18,
        'max_age': 100,
        'emergency_fund_months_3': 3,
        'emergency_fund_months_6': 6
    }


@pytest.fixture
def expected_model_performance():
    """Expected model performance metrics"""
    return {
        'min_accuracy': 0.999,
        'min_silhouette_score': 0.15,
        'max_response_time': 2.0,  # seconds
        'max_chart_generation_time': 2.0,  # seconds
        'min_feature_count': 60,
        'max_feature_count': 75
    }


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "ml: mark test as machine learning related"
    )
    config.addinivalue_line(
        "markers", "ui: mark test as user interface related"
    )
    config.addinivalue_line(
        "markers", "rag: mark test as RAG system related"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on test file location
        if "test_feature_engineering" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.ml)
        elif "test_ml_models" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.ml)
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_rag_system" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.rag)
        elif "test_streamlit_ui" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.ui)
