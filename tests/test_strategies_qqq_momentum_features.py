"""
Tests for QQQ momentum feature engineering.

**Purpose**: Verify that feature building produces expected columns and values
on synthetic price data with known behavior (rising, falling, flat).

**Testing philosophy**: Use simple synthetic data where expected outcomes are
obvious (e.g., steadily rising prices → positive velocity, positive spreads).
Avoid testing on real historical data (brittle, non-deterministic).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.strategies.qqq_momentum_features import (
    QqqMomentumFeatureParams,
    build_qqq_momentum_features,
)


def make_synthetic_qqq_prices(
    n_days: int,
    initial_price: float,
    daily_return: float,
) -> pd.DataFrame:
    """
    Create synthetic QQQ prices with constant daily returns.

    Args:
        n_days: Number of trading days.
        initial_price: Starting price.
        daily_return: Daily percentage return (e.g., 0.01 = 1% per day).

    Returns:
        DataFrame with columns: timestamp, closing_price (sorted ASCENDING for feature calc).
    """
    # Generate dates in ASCENDING order (oldest to newest)
    # Pandas time series functions expect ascending order
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Generate prices with compounding returns
    # price[t] = initial_price * (1 + daily_return)^t
    prices = [initial_price * ((1 + daily_return) ** i) for i in range(n_days)]

    df = pd.DataFrame({
        'timestamp': dates,
        'closing_price': prices,
    })

    return df


def test_build_qqq_momentum_features_adds_expected_columns():
    """Test that feature builder adds all expected columns."""
    # Create simple synthetic data (10 days of flat prices)
    qqq = make_synthetic_qqq_prices(n_days=300, initial_price=100.0, daily_return=0.0)

    # Build features with default params
    params = QqqMomentumFeatureParams()
    features = build_qqq_momentum_features(qqq, params)

    # Check that all expected columns were added
    expected_cols = [
        'moving_average_20',
        'moving_average_50',
        'moving_average_100',
        'moving_average_250',
        'ma_spread_50_100',
        'ma_spread_50_250',
        'velocity_20d',
        'acceleration_20d',
    ]

    for col in expected_cols:
        assert col in features.columns, f"Missing expected column: {col}"


def test_build_qqq_momentum_features_flat_prices():
    """Test features on flat prices (no trend)."""
    # Flat prices: all returns = 0
    qqq = make_synthetic_qqq_prices(n_days=300, initial_price=100.0, daily_return=0.0)

    params = QqqMomentumFeatureParams()
    features = build_qqq_momentum_features(qqq, params)

    # After warm-up period, check feature values
    # All MAs should equal price (since price is constant)
    warm_up_done = features.iloc[-1]  # Most recent row (after 300 days)

    # MAs should all be 100.0 (constant price)
    assert np.isclose(warm_up_done['moving_average_20'], 100.0, atol=1e-6)
    assert np.isclose(warm_up_done['moving_average_50'], 100.0, atol=1e-6)
    assert np.isclose(warm_up_done['moving_average_250'], 100.0, atol=1e-6)

    # Spreads should be 0 (all MAs equal)
    assert np.isclose(warm_up_done['ma_spread_50_100'], 0.0, atol=1e-6)
    assert np.isclose(warm_up_done['ma_spread_50_250'], 0.0, atol=1e-6)

    # Velocity should be ~0 (no trend)
    assert np.isclose(warm_up_done['velocity_20d'], 0.0, atol=1e-6)

    # Acceleration should be ~0 (velocity constant)
    assert np.isclose(warm_up_done['acceleration_20d'], 0.0, atol=1e-6)


def test_build_qqq_momentum_features_rising_prices():
    """Test features on steadily rising prices (uptrend)."""
    # Rising prices: 1% daily return
    qqq = make_synthetic_qqq_prices(n_days=300, initial_price=100.0, daily_return=0.01)

    params = QqqMomentumFeatureParams()
    features = build_qqq_momentum_features(qqq, params)

    # After warm-up, check that features reflect uptrend
    # Data is in ascending order (oldest first), so iloc[-1] is most recent
    recent = features.iloc[-1]  # Most recent row (last in ascending order)

    # Shorter MAs should be above longer MAs (uptrend characteristic)
    # ma_20 > ma_50 > ma_100 > ma_250
    assert recent['moving_average_20'] > recent['moving_average_50']
    assert recent['moving_average_50'] > recent['moving_average_100']
    assert recent['moving_average_100'] > recent['moving_average_250']

    # Spreads should be positive (fast MA above slow MA)
    assert recent['ma_spread_50_100'] > 0.0, "50-day MA should be above 100-day MA in uptrend"
    assert recent['ma_spread_50_250'] > 0.0, "50-day MA should be above 250-day MA in uptrend"

    # Velocity should be positive (upward slope)
    assert recent['velocity_20d'] > 0.0, "Velocity should be positive in uptrend"

    # Acceleration can vary, but velocity should be consistently positive
    # Let's just check that velocity is positive for last 10 days
    last_10 = features.iloc[-10:]
    assert (last_10['velocity_20d'] > 0).all(), "Velocity should be positive throughout uptrend"


def test_build_qqq_momentum_features_falling_prices():
    """Test features on steadily falling prices (downtrend)."""
    # Falling prices: -1% daily return
    qqq = make_synthetic_qqq_prices(n_days=300, initial_price=100.0, daily_return=-0.01)

    params = QqqMomentumFeatureParams()
    features = build_qqq_momentum_features(qqq, params)

    # After warm-up, check that features reflect downtrend
    # Data is in ascending order (oldest first), so iloc[-1] is most recent
    recent = features.iloc[-1]  # Most recent row (last in ascending order)

    # Shorter MAs should be below longer MAs (downtrend characteristic)
    # ma_20 < ma_50 < ma_100 < ma_250
    assert recent['moving_average_20'] < recent['moving_average_50']
    assert recent['moving_average_50'] < recent['moving_average_100']

    # Spreads should be negative (fast MA below slow MA)
    assert recent['ma_spread_50_100'] < 0.0, "50-day MA should be below 100-day MA in downtrend"
    assert recent['ma_spread_50_250'] < 0.0, "50-day MA should be below 250-day MA in downtrend"

    # Velocity should be negative (downward slope)
    assert recent['velocity_20d'] < 0.0, "Velocity should be negative in downtrend"


def test_build_qqq_momentum_features_nans_at_start():
    """Test that NaNs appear only during warm-up period."""
    # Short series to make warm-up obvious
    qqq = make_synthetic_qqq_prices(n_days=100, initial_price=100.0, daily_return=0.01)

    params = QqqMomentumFeatureParams()
    features = build_qqq_momentum_features(qqq, params)

    # Count NaNs in each feature
    nan_counts = features.isna().sum()

    # Velocity and acceleration should have NaNs at the start (window requirement)
    # Note: Exact count can be window-1 due to how rolling operations work
    assert nan_counts['velocity_20d'] >= 19, "Velocity should have NaNs during 20-day warm-up"
    assert nan_counts['acceleration_20d'] >= 19, "Acceleration should have NaNs during warm-up"

    # Longer MAs should have more NaNs
    assert nan_counts['moving_average_250'] > nan_counts['moving_average_50']

    # Recent rows (after warm-up) should have no NaNs
    # Check last row (index -1 in ascending order is most recent)
    recent = features.iloc[-1]
    # For 100 days, 250-day MA will still be NaN, but shorter features should be valid
    assert not pd.isna(recent['velocity_20d']), "Velocity should be valid after warm-up"
    assert not pd.isna(recent['moving_average_20']), "20-day MA should be valid after warm-up"


def test_build_qqq_momentum_features_missing_closing_price():
    """Test that builder raises error if closing_price column is missing."""
    # Create DataFrame without closing_price
    bad_df = pd.DataFrame({
        'timestamp': [datetime(2024, 1, 1, tzinfo=timezone.utc)],
        'open_price': [100.0],
    })

    params = QqqMomentumFeatureParams()

    with pytest.raises(KeyError, match="closing_price"):
        build_qqq_momentum_features(bad_df, params)


def test_build_qqq_momentum_features_custom_params():
    """Test that custom feature parameters are respected."""
    qqq = make_synthetic_qqq_prices(n_days=300, initial_price=100.0, daily_return=0.01)

    # Custom params with different window sizes
    params = QqqMomentumFeatureParams(
        ma_short_window=10,
        ma_medium_window=30,
        velocity_window=15,
    )

    features = build_qqq_momentum_features(qqq, params)

    # Check that custom windows created correct column names
    assert 'moving_average_10' in features.columns
    assert 'moving_average_30' in features.columns
    assert 'velocity_15d' in features.columns


def test_build_qqq_momentum_features_preserves_original_columns():
    """Test that original columns (timestamp, closing_price) are preserved."""
    qqq = make_synthetic_qqq_prices(n_days=100, initial_price=100.0, daily_return=0.01)

    params = QqqMomentumFeatureParams()
    features = build_qqq_momentum_features(qqq, params)

    # Original columns should still be present
    assert 'timestamp' in features.columns
    assert 'closing_price' in features.columns

    # Values should be unchanged
    assert features['closing_price'].equals(qqq['closing_price'])
