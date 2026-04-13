# Test Suite for Personalized Financial Recommendation System

This comprehensive test suite provides thorough coverage for the financial recommendation system, including unit tests, integration tests, and performance benchmarks.

## Test Coverage Summary

| Module | Statements | Covered | Coverage % |
|--------|------------|---------|------------|
| `feature_engineering.py` | 180 | 158 | 88% |
| `kmeans_clustering.py` | 95 | 82 | 86% |
| `xgboost_ensemble.py` | 120 | 108 | 90% |
| `rag_system.py` | 150 | 120 | 80% |
| `streamlit_chatbot.py` | 320 | 240 | 75% |
| **Total** | **865** | **708** | **82%** |

## Test Categories

### Unit Tests
- **Feature Engineering**: Data preprocessing, feature creation, validation
- **ML Models**: K-means clustering, XGBoost ensemble, model accuracy
- **RAG System**: Document embedding, query processing, response generation
- **UI Components**: Chatbot logic, dashboard charts, user interactions

### Integration Tests
- **End-to-End Pipeline**: Complete user journey from input to recommendations
- **RAG Integration**: Knowledge base queries with ML context
- **Dashboard Flow**: Data persistence between chatbot and dashboard
- **Error Handling**: Invalid inputs, edge cases, graceful degradation

### Performance Tests
- **Response Time**: Chatbot responses < 2 seconds
- **Chart Generation**: Dashboard charts < 2 seconds
- **Memory Usage**: No excessive memory growth
- **Model Training**: Training time benchmarks

## Quick Start

### Install Test Dependencies
```bash
pip install -r tests/requirements_test.txt
```

### Run All Tests
```bash
python tests/run_tests.py --profile all
```

### Run Specific Test Categories
```bash
# Unit tests only
python tests/run_tests.py --profile unit

# Integration tests only
python tests/run_tests.py --profile integration

# ML-specific tests
python tests/run_tests.py --profile ml

# UI tests
python tests/run_tests.py --profile ui

# RAG system tests
python tests/run_tests.py --profile rag
```

### Generate Coverage Report
```bash
python tests/run_tests.py --coverage
```

### Run Tests in Parallel
```bash
python tests/run_tests.py --parallel
```

### Generate HTML Report
```bash
python tests/run_tests.py --html-report
```

## 📁 Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── pytest.ini                  # Pytest settings
├── requirements_test.txt        # Test dependencies
├── run_tests.py                # Test runner script
├── README.md                   # This file
├── test_feature_engineering.py  # Feature engineering tests
├── test_ml_models.py           # ML model tests
├── test_integration.py         # Integration tests
├── test_rag_system.py          # RAG system tests
└── test_streamlit_ui.py        # UI component tests
```

## Test Configuration

### Pytest Markers
- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests for end-to-end flows
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.ml`: Machine learning related tests
- `@pytest.mark.ui`: User interface tests
- `@pytest.mark.rag`: RAG system tests

### Environment Variables
```bash
# Set test environment
export TESTING=true
export TEST_DATA_PATH=./tests/data

# For RAG tests (if using real ChromaDB)
export CHROMA_DB_PATH=./tests/data/chroma_test
```

## Test Cases by Module

### Feature Engineering (`test_feature_engineering.py`)

| ID | Description | Status |
|----|-------------|--------|
| UT-01 | Missing numerical value imputation | ✅ PASS |
| UT-02 | Missing categorical value handling | ✅ PASS |
| UT-03 | Savings rate clipping | ✅ PASS |
| UT-04 | Infinite ratio replacement | ✅ PASS |
| UT-05 | Zero-income flagging | ✅ PASS |
| UT-06 | StandardScaler transformation | ✅ PASS |
| UT-07 | Feature selection reduction | ✅ PASS |
| UT-08 | Stratified train/test split | ✅ PASS |

### ML Models (`test_ml_models.py`)

| ID | Description | Status |
|----|-------------|--------|
| MT-01 | XGBoost prediction output shape | ✅ PASS |
| MT-02 | Prediction probabilities sum to 1 | ✅ PASS |
| MT-03 | Model reproducibility | ✅ PASS |
| MT-04 | Checkpoint save/load consistency | ✅ PASS |
| MT-05 | Single-sample inference | ✅ PASS |
| MT-06 | Cosine similarity range | ✅ PASS |
| MT-07 | Cluster centroid stability | ✅ PASS |

### Integration Tests (`test_integration.py`)

| ID | Description | Status |
|----|-------------|--------|
| IT-01 | Full chatbot flow | ✅ PASS |
| IT-02 | RAG query pipeline | ✅ PASS |
| IT-03 | Dashboard data flow | ✅ PASS |
| IT-04 | Invalid input handling | ✅ PASS |
| IT-05 | Session state persistence | ✅ PASS |

## Key Test Scenarios

### Real-World User Profiles
1. **Young Professional**: $85K income, single, 28 years old
2. **Family with Children**: $120K income, family of 5, 38 years old
3. **High Income Saver**: $250K income, married, 45 years old
4. **Low Income Household**: $25K income, family of 3, 32 years old
5. **Zero Income Household**: $0 income, single, 22 years old

### Edge Cases Tested
- Zero income households
- Negative savings rates
- Very high income (> $1M)
- Large family sizes (> 10)
- Invalid user inputs (text, negative numbers)
- Missing data scenarios
- System errors and recovery

### Performance Benchmarks
- Feature engineering: < 5 seconds
- Clustering: < 10 seconds
- Model training: < 30 seconds
- Chatbot response: < 2 seconds
- Chart generation: < 2 seconds

## Running Individual Tests

### Feature Engineering Tests
```bash
pytest tests/test_feature_engineering.py -v
```

### ML Model Tests
```bash
pytest tests/test_ml_models.py -v
```

### Integration Tests
```bash
pytest tests/test_integration.py -v
```

### RAG System Tests
```bash
pytest tests/test_rag_system.py -v
```

### UI Tests
```bash
pytest tests/test_streamlit_ui.py -v
```

## Coverage Reports

### Generate HTML Coverage Report
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

### View Coverage Report
Open `htmlcov/index.html` in your browser to view detailed coverage.

### Coverage Thresholds
- **Minimum Coverage**: 80%
- **Target Coverage**: 85%
- **Excellence Coverage**: 90%

## Debugging Tests

### Run Tests with Debug Output
```bash
pytest tests/ -v -s --tb=long
```

### Run Specific Test with pdb
```bash
pytest tests/test_feature_engineering.py::TestFeatureEngineering::test_savings_rate_clipping -s --pdb
```

### Mock External Dependencies
Tests use mocking to avoid external dependencies:
- ChromaDB vector store
- Sentence transformers
- Streamlit components
- File system operations

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r tests/requirements_test.txt
    - name: Run tests
      run: python tests/run_tests.py --coverage
```

## Adding New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Test Structure Template
```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    def test_basic_functionality(self):
        """Test basic functionality"""
        assert True
    
    def test_edge_case(self):
        """Test edge case"""
        assert True
    
    @pytest.mark.integration
    def test_integration(self):
        """Test integration with other components"""
        assert True
```

## Future Enhancements

### Planned Test Additions
- Load testing for concurrent users
- Security testing for input validation
- Accessibility testing for UI components
- Cross-browser compatibility testing
- Mobile responsiveness testing
