"""
Tests for src/utils/math.py

These tests verify the correctness of math utilities using small, hand-crafted
datasets where expected values are easy to reason about.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.math import (
    compute_simple_returns,
    compute_log_returns,
    compute_moving_average_simple,
    compute_moving_average_exponential,
    compute_rolling_standard_deviation,
    compute_annualized_volatility,
    compute_velocity,
    compute_acceleration,
)


def test_compute_simple_returns_constant_prices():
    """Test simple returns on constant prices (should be all zeros)."""
    # Constant price of 100 for 5 periods
    prices = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
    returns = compute_simple_returns(prices)

    # First return is NaN, rest should be 0
    assert pd.isna(returns.iloc[0])
    assert np.allclose(returns.iloc[1:], 0.0)


def test_compute_simple_returns_linear_growth():
    """Test simple returns on linearly growing prices."""
    # Prices growing by 10 each period: 100, 110, 120, 130
    prices = pd.Series([100.0, 110.0, 120.0, 130.0])
    returns = compute_simple_returns(prices)

    # Expected returns: NaN, 0.10, ~0.0909, ~0.0833
    assert pd.isna(returns.iloc[0])
    assert np.isclose(returns.iloc[1], 0.10)
    assert np.isclose(returns.iloc[2], 10.0 / 110.0)
    assert np.isclose(returns.iloc[3], 10.0 / 120.0)


def test_compute_log_returns_constant_prices():
    """Test log returns on constant prices (should be all zeros)."""
    prices = pd.Series([100.0, 100.0, 100.0, 100.0])
    log_returns = compute_log_returns(prices)

    # First return is NaN, rest should be 0
    assert pd.isna(log_returns.iloc[0])
    assert np.allclose(log_returns.iloc[1:], 0.0)


def test_compute_log_returns_doubling_prices():
    """Test log returns when prices double."""
    # Prices: 100, 200 (doubling)
    prices = pd.Series([100.0, 200.0])
    log_returns = compute_log_returns(prices)

    # Expected: NaN, ln(200/100) = ln(2) ≈ 0.693
    assert pd.isna(log_returns.iloc[0])
    assert np.isclose(log_returns.iloc[1], np.log(2))


def test_compute_moving_average_simple_constant_prices():
    """Test SMA on constant prices (should equal the price)."""
    prices = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
    sma = compute_moving_average_simple(prices, window=3)

    # First 2 values are NaN, then all should be 100
    assert pd.isna(sma.iloc[0])
    assert pd.isna(sma.iloc[1])
    assert np.allclose(sma.iloc[2:], 100.0)


def test_compute_moving_average_simple_known_values():
    """Test SMA with known values."""
    # Prices: 10, 20, 30, 40, 50
    prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    sma = compute_moving_average_simple(prices, window=3)

    # Expected: NaN, NaN, 20 (mean of 10,20,30), 30 (mean of 20,30,40), 40 (mean of 30,40,50)
    assert pd.isna(sma.iloc[0])
    assert pd.isna(sma.iloc[1])
    assert np.isclose(sma.iloc[2], 20.0)
    assert np.isclose(sma.iloc[3], 30.0)
    assert np.isclose(sma.iloc[4], 40.0)


def test_compute_moving_average_exponential_constant_prices():
    """Test EMA on constant prices (should equal the price after warm-up)."""
    prices = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
    ema = compute_moving_average_exponential(prices, span=3)

    # EMA of constant series should converge to the constant value
    assert np.allclose(ema.dropna(), 100.0)


def test_compute_rolling_standard_deviation_constant_returns():
    """Test rolling std on constant returns (should be zero)."""
    # Constant returns of 0.01
    returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
    rolling_std = compute_rolling_standard_deviation(returns, window=3)

    # First 2 values are NaN, rest should be 0 (no variation)
    assert pd.isna(rolling_std.iloc[0])
    assert pd.isna(rolling_std.iloc[1])
    assert np.allclose(rolling_std.iloc[2:], 0.0)


def test_compute_rolling_standard_deviation_known_values():
    """Test rolling std with known simple values."""
    # Returns: 0.0, 0.0, 1.0 (std of these 3 is std([0,0,1]))
    returns = pd.Series([0.0, 0.0, 1.0])
    rolling_std = compute_rolling_standard_deviation(returns, window=3)

    # Expected std of [0, 0, 1] with ddof=1: sqrt((0^2 + 0^2 + 1^2) / 2 - (1/3)^2 * 3 / 2)
    # Easier: use numpy to verify
    expected_std = np.std([0.0, 0.0, 1.0], ddof=1)
    assert np.isclose(rolling_std.iloc[2], expected_std)


def test_compute_annualized_volatility_constant_returns():
    """Test annualized vol on constant returns (should be zero)."""
    returns = pd.Series([0.01] * 100)
    ann_vol = compute_annualized_volatility(returns, periods_per_year=252)

    # Constant returns → std = 0 → annualized vol = 0
    assert np.isclose(ann_vol, 0.0)


def test_compute_annualized_volatility_known_values():
    """Test annualized vol with known daily std."""
    # Create returns with known daily std = 0.01
    # Use a series of alternating +0.01 and -0.01 for simplicity
    returns = pd.Series([0.01, -0.01] * 50)  # 100 values
    daily_std = returns.std(ddof=1)

    # Annualized vol should be daily_std * sqrt(252)
    ann_vol = compute_annualized_volatility(returns, periods_per_year=252)
    expected_vol = daily_std * np.sqrt(252)
    assert np.isclose(ann_vol, expected_vol)


def test_compute_velocity_strictly_increasing_prices():
    """Test velocity on strictly increasing prices (positive slope)."""
    # Linear prices: 100, 101, 102, 103, 104, 105
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    velocity = compute_velocity(prices, window=3, use_log=False)

    # With linear growth of +1 per period, slope should be close to 1
    # (May vary slightly due to log transformation if use_log=True, but here use_log=False)
    # First 2 values are NaN (window=3)
    assert pd.isna(velocity.iloc[0])
    assert pd.isna(velocity.iloc[1])
    # Slope of [100, 101, 102] against [0, 1, 2] should be 1
    assert np.isclose(velocity.iloc[2], 1.0, atol=1e-6)
    assert np.isclose(velocity.iloc[3], 1.0, atol=1e-6)


def test_compute_velocity_flat_prices():
    """Test velocity on flat prices (slope should be zero)."""
    prices = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
    velocity = compute_velocity(prices, window=3, use_log=False)

    # Flat prices → slope = 0
    assert pd.isna(velocity.iloc[0])
    assert pd.isna(velocity.iloc[1])
    assert np.allclose(velocity.iloc[2:], 0.0, atol=1e-10)


def test_compute_acceleration_constant_velocity():
    """Test acceleration on constant velocity (should be zero)."""
    # Linear prices: slope is constant → acceleration = 0
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    acceleration = compute_acceleration(prices, window=3, use_log=False)

    # Constant slope → acceleration (diff of velocity) should be 0
    # First few values are NaN due to velocity window + diff
    # Check that non-NaN values are close to 0
    non_nan_accel = acceleration.dropna()
    assert len(non_nan_accel) > 0
    assert np.allclose(non_nan_accel, 0.0, atol=1e-6)


def test_compute_acceleration_quadratic_prices():
    """Test acceleration on quadratic prices (constant positive acceleration)."""
    # Prices: t^2 for t=0,1,2,3,4,5 → 0, 1, 4, 9, 16, 25
    # Velocity (slope) increases linearly → acceleration is positive constant
    prices = pd.Series([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
    acceleration = compute_acceleration(prices, window=3, use_log=False)

    # Acceleration should be positive (increasing slope)
    non_nan_accel = acceleration.dropna()
    assert len(non_nan_accel) > 0
    # All non-NaN accelerations should be positive (slope is increasing)
    assert (non_nan_accel > 0).all() or np.allclose(non_nan_accel, non_nan_accel[0], atol=0.5)
