# Testing Guide

## Run All Tests Together

```bash
source .venv/bin/activate && pytest tests/ -v
```

## Run Individual Test Files

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run risk metrics tests (27 tests)
pytest tests/test_analytics_risk_metrics.py -v

# Run synthetic data tests (14 tests)
pytest tests/test_analytics_synthetic_data.py -v

# Run math utility tests (15 tests)
pytest tests/test_utils_math.py -v

# Run time utility tests (9 tests)
pytest tests/test_utils_time.py -v
```

## Run Specific Test Functions

```bash
source .venv/bin/activate

# Run a specific test by name
pytest tests/test_analytics_risk_metrics.py::test_compute_sharpe_ratio_constant_positive_returns -v

# Run tests matching a pattern
pytest tests/ -k "sharpe" -v
pytest tests/ -k "gbm" -v
```

## Test Coverage

```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage report
source .venv/bin/activate && pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```
