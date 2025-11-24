"""
Tests for QQQ momentum regime classification and allocation logic.

**Purpose**: Verify that:
  1. Regime classification correctly maps feature values to regime labels.
  2. Regime-to-weights mapping assigns correct allocations per regime.
  3. Edge cases (NaN features, missing data) are handled gracefully.

**Testing philosophy**: Use synthetic feature DataFrames with known values
to test threshold logic. Each test should verify one regime or edge case.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.strategies.qqq_momentum_features import (
    QqqMomentumRegimeParams,
    QqqMomentumAllocationParams,
    QqqMomentumSymbols,
    MomentumRegime,
    classify_momentum_regime,
    regime_to_target_weights,
)


def make_feature_dataframe(
    n_rows: int,
    spread_50_250: float,
    velocity_20d: float,
    acceleration_20d: float,
) -> pd.DataFrame:
    """
    Create a synthetic feature DataFrame with constant feature values.

    Args:
        n_rows: Number of rows.
        spread_50_250: MA spread value (constant across all rows).
        velocity_20d: Velocity value (constant).
        acceleration_20d: Acceleration value (constant).

    Returns:
        DataFrame with columns: timestamp, ma_spread_50_250, velocity_20d, acceleration_20d.
    """
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'ma_spread_50_250': [spread_50_250] * n_rows,
        'velocity_20d': [velocity_20d] * n_rows,
        'acceleration_20d': [acceleration_20d] * n_rows,
    })
    return df


def test_classify_momentum_regime_strong_uptrend():
    """Test that strong uptrend is classified correctly."""
    # Strong uptrend: positive spread, positive velocity, positive acceleration
    features = make_feature_dataframe(
        n_rows=10,
        spread_50_250=0.05,  # 5% spread (> default 2% threshold)
        velocity_20d=0.01,  # Positive velocity
        acceleration_20d=0.001,  # Positive acceleration
    )

    params = QqqMomentumRegimeParams()
    regimes = classify_momentum_regime(features, params)

    # All rows should be classified as STRONG_UPTREND
    assert (regimes == MomentumRegime.STRONG_UPTREND).all()


def test_classify_momentum_regime_weakening_uptrend():
    """Test that weakening uptrend is classified correctly."""
    # Weakening uptrend: positive spread, positive velocity, but negative acceleration
    features = make_feature_dataframe(
        n_rows=10,
        spread_50_250=0.05,  # 5% spread
        velocity_20d=0.01,  # Positive velocity
        acceleration_20d=-0.001,  # Negative acceleration (trend slowing)
    )

    params = QqqMomentumRegimeParams()
    regimes = classify_momentum_regime(features, params)

    # All rows should be classified as WEAKENING_UPTREND
    assert (regimes == MomentumRegime.WEAKENING_UPTREND).all()


def test_classify_momentum_regime_downtrend_negative_spread():
    """Test that downtrend is classified when spread is negative."""
    # Downtrend: negative spread (fast MA below slow MA)
    features = make_feature_dataframe(
        n_rows=10,
        spread_50_250=-0.05,  # -5% spread (bearish)
        velocity_20d=-0.01,  # Negative velocity
        acceleration_20d=-0.001,  # Negative acceleration
    )

    params = QqqMomentumRegimeParams()
    regimes = classify_momentum_regime(features, params)

    # All rows should be classified as DOWNTREND
    assert (regimes == MomentumRegime.DOWNTREND).all()


def test_classify_momentum_regime_downtrend_negative_velocity():
    """Test that downtrend is classified when velocity is negative."""
    # Downtrend: negative velocity (even if spread is slightly positive)
    features = make_feature_dataframe(
        n_rows=10,
        spread_50_250=0.01,  # Small positive spread
        velocity_20d=-0.02,  # Strong negative velocity (overrides spread)
        acceleration_20d=0.0,
    )

    params = QqqMomentumRegimeParams()
    regimes = classify_momentum_regime(features, params)

    # All rows should be classified as DOWNTREND
    assert (regimes == MomentumRegime.DOWNTREND).all()


def test_classify_momentum_regime_neutral_small_spread():
    """Test that neutral is classified when spread is very small."""
    # Neutral: small spread, mixed velocity
    features = make_feature_dataframe(
        n_rows=10,
        spread_50_250=0.005,  # 0.5% spread (< default 1% neutral threshold)
        velocity_20d=0.005,  # Small positive velocity
        acceleration_20d=0.0,
    )

    params = QqqMomentumRegimeParams()
    regimes = classify_momentum_regime(features, params)

    # All rows should be classified as NEUTRAL
    assert (regimes == MomentumRegime.NEUTRAL).all()


def test_classify_momentum_regime_nan_features():
    """Test that NaN features default to NEUTRAL."""
    # Features with NaNs (e.g., warm-up period)
    features = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='D'),
        'ma_spread_50_250': [np.nan, np.nan, 0.05, 0.05, 0.05],
        'velocity_20d': [np.nan, np.nan, 0.01, 0.01, 0.01],
        'acceleration_20d': [np.nan, np.nan, 0.001, 0.001, 0.001],
    })

    params = QqqMomentumRegimeParams()
    regimes = classify_momentum_regime(features, params)

    # First two rows (NaNs) should be NEUTRAL
    assert regimes.iloc[0] == MomentumRegime.NEUTRAL
    assert regimes.iloc[1] == MomentumRegime.NEUTRAL

    # Last three rows (valid features) should be STRONG_UPTREND
    assert regimes.iloc[2] == MomentumRegime.STRONG_UPTREND
    assert regimes.iloc[3] == MomentumRegime.STRONG_UPTREND
    assert regimes.iloc[4] == MomentumRegime.STRONG_UPTREND


def test_classify_momentum_regime_custom_thresholds():
    """Test that custom thresholds are respected."""
    # Features just below standard thresholds
    features = make_feature_dataframe(
        n_rows=10,
        spread_50_250=0.015,  # 1.5% spread
        velocity_20d=0.005,
        acceleration_20d=0.001,
    )

    # Standard thresholds (min_spread_for_trend = 0.02 = 2%)
    # This should be NEUTRAL (spread < 2%)
    standard_params = QqqMomentumRegimeParams()
    regimes_standard = classify_momentum_regime(features, standard_params)
    assert (regimes_standard == MomentumRegime.NEUTRAL).all()

    # Custom thresholds (lower min_spread = 1%)
    # This should be STRONG_UPTREND (spread > 1%)
    custom_params = QqqMomentumRegimeParams(min_spread_for_trend=0.01)
    regimes_custom = classify_momentum_regime(features, custom_params)
    assert (regimes_custom == MomentumRegime.STRONG_UPTREND).all()


def test_regime_to_target_weights_strong_uptrend():
    """Test target weights for STRONG_UPTREND regime."""
    regime = MomentumRegime.STRONG_UPTREND
    symbols = QqqMomentumSymbols()
    alloc_params = QqqMomentumAllocationParams()

    weights = regime_to_target_weights(regime, symbols, alloc_params)

    # Should allocate 100% to TQQQ (default strong_uptrend_long_weight = 1.0)
    assert weights == {"TQQQ": 1.0}


def test_regime_to_target_weights_weakening_uptrend():
    """Test target weights for WEAKENING_UPTREND regime."""
    regime = MomentumRegime.WEAKENING_UPTREND
    symbols = QqqMomentumSymbols()
    alloc_params = QqqMomentumAllocationParams()

    weights = regime_to_target_weights(regime, symbols, alloc_params)

    # Should allocate 50% to TQQQ (default weakening_uptrend_long_weight = 0.5)
    assert weights == {"TQQQ": 0.5}


def test_regime_to_target_weights_neutral():
    """Test target weights for NEUTRAL regime."""
    regime = MomentumRegime.NEUTRAL
    symbols = QqqMomentumSymbols()
    alloc_params = QqqMomentumAllocationParams()

    weights = regime_to_target_weights(regime, symbols, alloc_params)

    # Should return empty dict = all cash
    assert weights == {}


def test_regime_to_target_weights_downtrend():
    """Test target weights for DOWNTREND regime."""
    regime = MomentumRegime.DOWNTREND
    symbols = QqqMomentumSymbols()
    alloc_params = QqqMomentumAllocationParams()

    weights = regime_to_target_weights(regime, symbols, alloc_params)

    # Should allocate 50% to SQQQ (default downtrend_short_weight = 0.5)
    assert weights == {"SQQQ": 0.5}


def test_regime_to_target_weights_downtrend_with_vol_overlay():
    """Test target weights for DOWNTREND with UVXY overlay enabled."""
    regime = MomentumRegime.DOWNTREND
    symbols = QqqMomentumSymbols(vol_symbol="UVXY")
    alloc_params = QqqMomentumAllocationParams(
        use_vol_overlay=True,
        vol_allocation_weight=0.3,
    )

    weights = regime_to_target_weights(regime, symbols, alloc_params)

    # Should allocate to UVXY instead of SQQQ when vol overlay is enabled
    assert weights == {"UVXY": 0.3}


def test_regime_to_target_weights_custom_allocations():
    """Test that custom allocation parameters are respected."""
    regime = MomentumRegime.STRONG_UPTREND
    symbols = QqqMomentumSymbols()

    # Custom allocation: 80% to TQQQ in strong uptrend
    custom_alloc = QqqMomentumAllocationParams(strong_uptrend_long_weight=0.8)

    weights = regime_to_target_weights(regime, symbols, custom_alloc)

    # Should respect custom weight
    assert weights == {"TQQQ": 0.8}


def test_classify_momentum_regime_missing_columns():
    """Test that classifier raises error when required columns are missing."""
    # DataFrame missing velocity column
    bad_features = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'ma_spread_50_250': [0.05] * 10,
        # Missing: velocity_20d, acceleration_20d
    })

    params = QqqMomentumRegimeParams()

    with pytest.raises(KeyError, match="Missing required feature columns"):
        classify_momentum_regime(bad_features, params)


def test_regime_to_target_weights_all_regimes_produce_valid_dicts():
    """Test that all regimes produce valid weight dictionaries."""
    symbols = QqqMomentumSymbols()
    alloc_params = QqqMomentumAllocationParams()

    # Test all regimes
    for regime in MomentumRegime:
        weights = regime_to_target_weights(regime, symbols, alloc_params)

        # All weights should be dicts
        assert isinstance(weights, dict)

        # All values should be floats
        for symbol, weight in weights.items():
            assert isinstance(symbol, str)
            assert isinstance(weight, (int, float))

        # Sum of absolute weights should be <= 1.0 (or slightly more for leverage)
        # Allow up to 1.5 for potential leverage strategies
        total_abs_weight = sum(abs(w) for w in weights.values())
        assert total_abs_weight <= 1.5, f"Total weight {total_abs_weight} exceeds 1.5 for regime {regime}"
