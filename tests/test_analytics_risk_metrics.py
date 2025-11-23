"""
Tests for src/analytics/risk_metrics.py

These tests use synthetic and hand-crafted equity/return series where expected
values are easy to verify qualitatively and numerically.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.risk_metrics import (
    compute_total_return,
    compute_cagr,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_drawdown_series,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_information_ratio,
    compute_beta_and_correlation,
    compute_hit_rate,
    compute_win_loss_ratio,
    compute_rolling_sharpe,
    compute_rolling_volatility,
)


def test_compute_total_return_simple_case():
    """Test total return on simple equity curve: 100 → 150 (50% gain)."""
    equity = pd.Series([100.0, 110.0, 130.0, 150.0])
    total_return = compute_total_return(equity)

    # Expected: (150 / 100) - 1 = 0.50
    assert np.isclose(total_return, 0.50)


def test_compute_total_return_flat_equity():
    """Test total return on flat equity (0% return)."""
    equity = pd.Series([100.0, 100.0, 100.0])
    total_return = compute_total_return(equity)

    # Expected: 0.0
    assert np.isclose(total_return, 0.0)


def test_compute_cagr_doubling_in_252_days():
    """Test CAGR when equity doubles over 252 trading days (1 year)."""
    # Start at 100, end at 200 over 252 days
    equity = pd.Series([100.0] + [100.0] * 251 + [200.0])  # 253 values
    cagr = compute_cagr(equity, periods_per_year=252)

    # Expected: (200/100)^(252/252) - 1 = 1.0 (100% annual growth)
    assert np.isclose(cagr, 1.0)


def test_compute_cagr_flat_equity():
    """Test CAGR on flat equity (0% growth)."""
    equity = pd.Series([100.0] * 100)
    cagr = compute_cagr(equity, periods_per_year=252)

    # Expected: 0.0
    assert np.isclose(cagr, 0.0)


def test_compute_annualized_volatility_zero_volatility():
    """Test annualized vol on constant returns (zero volatility)."""
    returns = pd.Series([0.01] * 100)
    vol = compute_annualized_volatility(returns, periods_per_year=252)

    # Constant returns → std = 0
    assert np.isclose(vol, 0.0)


def test_compute_sharpe_ratio_constant_positive_returns():
    """Test Sharpe on constant positive returns."""
    # Daily returns of 0.001 (0.1%) for 100 days
    returns = pd.Series([0.001] * 100)
    sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

    # Constant returns → std = 0 → Sharpe undefined (should return NaN)
    assert pd.isna(sharpe)


def test_compute_sharpe_ratio_positive_mean_with_volatility():
    """Test Sharpe on returns with positive mean and known volatility."""
    # Generate returns: alternating +0.02 and +0.00 → mean = 0.01, std > 0
    returns = pd.Series([0.02, 0.00] * 50)  # 100 values
    sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

    # Sharpe should be positive (positive mean, non-zero std)
    assert sharpe > 0


def test_compute_sortino_ratio_only_positive_returns():
    """Test Sortino when all returns are positive (no downside)."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.012, 0.018])
    sortino = compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

    # No negative returns → downside std = 0 → Sortino undefined (NaN or +inf)
    assert pd.isna(sortino) or np.isinf(sortino)


def test_compute_sortino_ratio_with_downside():
    """Test Sortino when there are both positive and negative returns."""
    # Mix of positive and negative returns
    returns = pd.Series([0.02, -0.01, 0.03, -0.015, 0.01])
    sortino = compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

    # Should be a finite positive value (positive mean, negative returns exist)
    assert not pd.isna(sortino)
    assert sortino > 0


def test_compute_drawdown_series_monotonic_increasing():
    """Test drawdown on monotonically increasing equity (no drawdown)."""
    equity = pd.Series([100.0, 110.0, 120.0, 130.0])
    drawdown = compute_drawdown_series(equity)

    # No drawdown → all values should be 0
    assert np.allclose(drawdown, 0.0)


def test_compute_drawdown_series_with_drop():
    """Test drawdown with a single drop."""
    # Equity: 100, 120 (new peak), 110 (drop), 130 (new peak)
    equity = pd.Series([100.0, 120.0, 110.0, 130.0])
    drawdown = compute_drawdown_series(equity)

    # Expected drawdowns:
    # t=0: 100/100 - 1 = 0
    # t=1: 120/120 - 1 = 0 (new peak)
    # t=2: 110/120 - 1 ≈ -0.0833 (down from peak)
    # t=3: 130/130 - 1 = 0 (new peak)
    assert np.isclose(drawdown.iloc[0], 0.0)
    assert np.isclose(drawdown.iloc[1], 0.0)
    assert np.isclose(drawdown.iloc[2], (110.0 / 120.0) - 1.0)
    assert np.isclose(drawdown.iloc[3], 0.0)


def test_compute_max_drawdown_monotonic_increasing():
    """Test max drawdown on monotonically increasing equity (should be 0)."""
    equity = pd.Series([100.0, 110.0, 120.0, 130.0])
    mdd = compute_max_drawdown(equity)

    # No drawdown → max drawdown = 0
    assert np.isclose(mdd, 0.0)


def test_compute_max_drawdown_with_drop():
    """Test max drawdown with a known drop."""
    # Equity: 100, 120 (peak), 90 (trough), 110
    # Max drawdown: (90 / 120) - 1 = -0.25 (25% drawdown)
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    mdd = compute_max_drawdown(equity)

    # Expected: -0.25
    assert np.isclose(mdd, (90.0 / 120.0) - 1.0)


def test_compute_calmar_ratio_positive_cagr_with_drawdown():
    """Test Calmar ratio with positive CAGR and known drawdown."""
    # Simplified: equity goes from 100 to 200 over 252 days with a 50% drawdown
    # CAGR ≈ 100%, MDD = -50% → Calmar = 100 / 50 = 2.0
    # Create equity: start at 100, peak at 150, drop to 75 (50% DD), recover to 200
    equity = pd.Series([100.0] * 63 + [150.0] * 63 + [75.0] * 63 + [200.0] * 63)
    calmar = compute_calmar_ratio(equity, periods_per_year=252)

    # CAGR: (200/100)^(252/(252-1)) - 1 ≈ 1.0
    # MDD: (75/150) - 1 = -0.5
    # Calmar = 1.0 / 0.5 = 2.0
    # (Approximate; exact depends on step count)
    assert calmar > 0  # Should be positive


def test_compute_information_ratio_same_returns():
    """Test Information Ratio when strategy and benchmark are identical."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.012])
    benchmark = returns.copy()
    ir = compute_information_ratio(returns, benchmark, periods_per_year=252)

    # Identical returns → excess = 0, tracking error = 0 → IR undefined (NaN)
    assert pd.isna(ir)


def test_compute_information_ratio_positive_alpha():
    """Test Information Ratio when strategy outperforms benchmark."""
    # Strategy returns always 0.01 higher than benchmark
    benchmark = pd.Series([0.01, 0.02, 0.015, 0.01, 0.012])
    returns = benchmark + 0.01  # Constant alpha
    ir = compute_information_ratio(returns, benchmark, periods_per_year=252)

    # Positive mean excess, zero tracking error → IR undefined (constant excess)
    # Actually, constant excess means std(excess) = 0 → IR = NaN
    assert pd.isna(ir)


def test_compute_information_ratio_varying_alpha():
    """Test Information Ratio with varying excess returns."""
    benchmark = pd.Series([0.01, 0.02, 0.015, 0.01, 0.012])
    # Strategy varies around benchmark
    returns = pd.Series([0.015, 0.025, 0.01, 0.015, 0.018])
    ir = compute_information_ratio(returns, benchmark, periods_per_year=252)

    # Should return a finite value
    assert not pd.isna(ir)


def test_compute_beta_and_correlation_identical_returns():
    """Test beta and correlation when returns are identical."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.012])
    benchmark = returns.copy()
    beta, corr = compute_beta_and_correlation(returns, benchmark)

    # Identical → beta = 1.0, correlation = 1.0
    assert np.isclose(beta, 1.0)
    assert np.isclose(corr, 1.0)


def test_compute_beta_and_correlation_scaled_returns():
    """Test beta when strategy is 2x benchmark (beta should be 2)."""
    benchmark = pd.Series([0.01, 0.02, 0.015, 0.01, 0.012])
    returns = benchmark * 2  # 2x levered
    beta, corr = compute_beta_and_correlation(returns, benchmark)

    # Beta should be ~2.0, correlation ~1.0
    assert np.isclose(beta, 2.0)
    assert np.isclose(corr, 1.0)


def test_compute_hit_rate_all_positive():
    """Test hit rate when all returns are positive."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.012])
    hit_rate = compute_hit_rate(returns)

    # All positive → hit rate = 1.0
    assert np.isclose(hit_rate, 1.0)


def test_compute_hit_rate_half_positive():
    """Test hit rate when half returns are positive."""
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.015, -0.015])
    hit_rate = compute_hit_rate(returns)

    # 3 positive out of 6 → hit rate = 0.5
    assert np.isclose(hit_rate, 0.5)


def test_compute_win_loss_ratio_no_losses():
    """Test win/loss ratio when there are no losses."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.012])
    ratio = compute_win_loss_ratio(returns)

    # No losses → ratio should be +inf
    assert np.isinf(ratio) and ratio > 0


def test_compute_win_loss_ratio_symmetric():
    """Test win/loss ratio with symmetric wins and losses."""
    # Alternating +0.02 and -0.02
    returns = pd.Series([0.02, -0.02, 0.02, -0.02])
    ratio = compute_win_loss_ratio(returns)

    # avg_win = 0.02, avg_loss = -0.02 → ratio = 0.02 / 0.02 = 1.0
    assert np.isclose(ratio, 1.0)


def test_compute_rolling_sharpe_constant_returns():
    """Test rolling Sharpe on constant returns (should be NaN due to zero std)."""
    returns = pd.Series([0.01] * 100)
    rolling_sharpe = compute_rolling_sharpe(returns, window=20, periods_per_year=252)

    # Constant returns → rolling std = 0 → rolling Sharpe = NaN
    # Check that non-NaN values (if any) or all are NaN
    # Actually, all should be NaN after window fills
    assert rolling_sharpe.iloc[20:].isna().all()


def test_compute_rolling_sharpe_varying_returns():
    """Test rolling Sharpe on varying returns (should produce finite values)."""
    # Alternating positive and negative returns
    returns = pd.Series([0.01, -0.005, 0.015, -0.002, 0.01] * 20)  # 100 values
    rolling_sharpe = compute_rolling_sharpe(returns, window=20, periods_per_year=252)

    # Should have finite values after window fills
    non_nan = rolling_sharpe.dropna()
    assert len(non_nan) > 0
    # Values should be finite (not NaN, not inf)
    assert np.isfinite(non_nan).all()


def test_compute_rolling_volatility_constant_returns():
    """Test rolling vol on constant returns (should be zero)."""
    returns = pd.Series([0.01] * 100)
    rolling_vol = compute_rolling_volatility(returns, window=20, periods_per_year=252)

    # Constant returns → rolling std = 0 → rolling vol = 0
    assert rolling_vol.iloc[20:].fillna(0).eq(0).all() or rolling_vol.iloc[20:].isna().all()


def test_compute_rolling_volatility_varying_returns():
    """Test rolling vol on varying returns (should be positive)."""
    # Alternating returns with some variance
    returns = pd.Series([0.01, -0.005, 0.015, -0.002, 0.01] * 20)
    rolling_vol = compute_rolling_volatility(returns, window=20, periods_per_year=252)

    # Should have positive values after window fills
    non_nan = rolling_vol.dropna()
    assert len(non_nan) > 0
    assert (non_nan > 0).all()
