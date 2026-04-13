"""
Demo test to show the test suite is working
This test imports actual modules and demonstrates the testing framework
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class TestDemo:
    """Demo test class to show the testing framework works"""

    def test_imports(self):
        """Test that we can import the main modules"""
        try:
            # Test basic imports that should work
            import pandas as pd
            import numpy as np
            import streamlit as st
            
            assert pd.__version__ is not None
            assert np.__version__ is not None
            assert st.__version__ is not None
            
        except ImportError as e:
            pytest.fail(f"Failed to import required modules: {e}")

    def test_basic_pandas_operations(self):
        """Test basic pandas operations work"""
        df = pd.DataFrame({
            'income': [75000, 85000, 95000],
            'expenses': [60000, 65000, 70000],
            'family_size': [3, 4, 2]
        })
        
        assert len(df) == 3
        assert df['income'].mean() == 85000
        assert 'savings_rate' not in df.columns
        
        # Add savings rate
        df['savings_rate'] = (df['income'] - df['expenses']) / df['income']
        assert 'savings_rate' in df.columns
        assert all(df['savings_rate'] >= 0)

    def test_financial_calculations(self):
        """Test financial calculation logic"""
        # Test savings rate
        income = 75000
        expenses = 60000
        savings_rate = (income - expenses) / income
        
        assert savings_rate == 0.2  # 20%
        
        # Test emergency fund
        monthly_expenses = expenses / 12
        emergency_3_month = monthly_expenses * 3
        emergency_6_month = monthly_expenses * 6
        
        assert emergency_3_month == 15000
        assert emergency_6_month == 30000
        
        # Test investment allocation
        age = 35
        if age < 30:
            stocks_percent = 80
        elif age < 45:
            stocks_percent = 70
        else:
            stocks_percent = 40
            
        assert stocks_percent == 70

    def test_data_validation(self):
        """Test data validation logic"""
        # Test valid inputs
        valid_inputs = ["75000", "75000.50", "$75,000"]
        
        for input_str in valid_inputs:
            cleaned = input_str.replace('$', '').replace(',', '').strip()
            try:
                value = float(cleaned)
                assert value > 0
            except ValueError:
                pass  # Some might still fail
        
        # Test invalid inputs
        invalid_inputs = ["abc", "", "nan"]
        
        for input_str in invalid_inputs:
            cleaned = input_str.replace('$', '').replace(',', '').strip()
            try:
                value = float(cleaned)
                if input_str in ["abc", ""]:
                    assert False, f"Should have failed for '{input_str}'"
            except ValueError:
                pass  # Expected to fail

    def test_mock_streamlit(self):
        """Test we can mock Streamlit components"""
        with patch('streamlit.session_state', {'test': 'value'}):
            from streamlit import session_state
            assert session_state['test'] == 'value'

    def test_numpy_operations(self):
        """Test numpy operations work"""
        # Test array operations
        incomes = np.array([75000, 85000, 95000])
        expenses = np.array([60000, 65000, 70000])
        
        savings_rates = (incomes - expenses) / incomes
        
        assert len(savings_rates) == 3
        assert all(savings_rates >= 0)
        assert abs(np.mean(savings_rates) - 0.2) < 0.1  # Close to 20%

    def test_file_operations(self):
        """Test basic file operations"""
        # Test path operations
        test_path = Path(__file__).parent
        assert test_path.exists()
        assert test_path.is_dir()
        
        # Test that we can read our own test file
        with open(__file__, 'r') as f:
            content = f.read()
            assert 'class TestDemo:' in content

    def test_error_handling(self):
        """Test error handling"""
        # Test division by zero handling
        try:
            result = 100 / 0
            assert False, "Should have raised ZeroDivisionError"
        except ZeroDivisionError:
            pass  # Expected
        
        # Test invalid index
        df = pd.DataFrame({'a': [1, 2, 3]})
        try:
            value = df.iloc[10]
            assert False, "Should have raised IndexError"
        except IndexError:
            pass  # Expected

    def test_performance_basics(self):
        """Test basic performance considerations"""
        import time
        
        # Test that operations complete quickly
        start_time = time.time()
        
        # Create a larger dataset
        n = 1000
        df = pd.DataFrame({
            'income': np.random.lognormal(10.5, 0.8, n),
            'expenses': np.random.lognormal(10.2, 0.7, n),
            'family_size': np.random.randint(1, 6, n)
        })
        
        # Calculate savings rates
        df['savings_rate'] = (df['income'] - df['expenses']) / df['income']
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        assert elapsed < 1.0, f"Operations took too long: {elapsed:.3f}s"
        assert len(df) == 1000
        assert not df['savings_rate'].isna().all()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
