"""
Tests for QQQ momentum strategy regime classification and decision making.

**Purpose**: Verify that the QQQ momentum strategy correctly:
  1. Classifies market regimes based on features.
  2. Maps regimes to non-zero target weights.
  3. Handles timestamp normalization consistently.

**Testing philosophy**: Use synthetic price data with obvious trends to ensure
regime classification works as expected. This prevents silent regressions where
all regimes become "unknown" or all weights become zero.

**Why these tests are important**:
  - Prevents timestamp mismatch bugs (timezone-aware vs naive, date vs datetime).
  - Ensures regime classification actually produces different regimes (not all neutral).
  - Verifies that regime-to-weights mapping produces non-zero allocations.
  - Catches index alignment issues between features and regimes.
"""

import pytest
import pandas as pd
import numpy as np
from src.strategies.qqq_momentum_features import (
    QqqMomentumFeatureParams,
    QqqMomentumRegimeParams,
    QqqMomentumAllocationParams,
    QqqMomentumSymbols,
    MomentumRegime,
    build_qqq_momentum_features,
    classify_momentum_regime,
    regime_to_target_weights,
)
from src.strategies.qqq_momentum import QqqMomentumStrategy
from src.execution.paper_broker import PortfolioState


def make_synthetic_trend_data(start_date: str, n_days: int, trend: str = "up") -> pd.DataFrame:
    """
    Create synthetic price data with an obvious trend for testing.

    Args:
        start_date: Start date as YYYY-MM-DD string.
        n_days: Number of trading days to generate.
        trend: "up" for uptrend, "down" for downtrend, "flat" for neutral.

    Returns:
        DataFrame with timestamp and closing_price columns (UTC timezone-aware).
    """
    # Generate dates (business days only)
    dates = pd.date_range(start=start_date, periods=n_days, freq='B', tz='UTC')

    # Generate prices based on trend
    if trend == "up":
        # Strong uptrend: prices increase 0.5% per day on average
        daily_returns = np.random.normal(0.005, 0.01, n_days)  # Mean +0.5%, std 1%
        prices = 100 * np.exp(np.cumsum(daily_returns))
    elif trend == "down":
        # Downtrend: prices decrease 0.5% per day on average
        daily_returns = np.random.normal(-0.005, 0.01, n_days)  # Mean -0.5%, std 1%
        prices = 100 * np.exp(np.cumsum(daily_returns))
    else:  # flat
        # Neutral: prices oscillate around 100 with no trend
        daily_returns = np.random.normal(0.0, 0.005, n_days)  # Mean 0%, std 0.5%
        prices = 100 + np.cumsum(daily_returns)

    df = pd.DataFrame({
        'timestamp': dates,
        'closing_price': prices,
        # Add dummy OHLCV columns (required by schema)
        'open_price': prices * 0.99,
        'high_price': prices * 1.01,
        'low_price': prices * 0.98,
        'volume': [1_000_000] * n_days,
    })

    return df


def test_classify_momentum_regime_uptrend():
    """Test that strong uptrend is correctly classified."""
    # Generate synthetic uptrend data
    qqq = make_synthetic_trend_data(start_date='2023-01-01', n_days=300, trend='up')

    # Build features
    feature_params = QqqMomentumFeatureParams()
    qqq_features = build_qqq_momentum_features(qqq, feature_params)

    # Classify regimes
    regime_params = QqqMomentumRegimeParams()
    regimes = classify_momentum_regime(qqq_features, regime_params)

    # Verify that at least some dates are classified as uptrend (not all neutral)
    # After warm-up period (250 days for longest MA), we should see uptrend regimes
    regimes_after_warmup = regimes.iloc[250:]

    uptrend_count = (
        (regimes_after_warmup == MomentumRegime.STRONG_UPTREND) |
        (regimes_after_warmup == MomentumRegime.WEAKENING_UPTREND)
    ).sum()

    # At least 25% of post-warmup days should be classified as some form of uptrend
    # (This is a generous threshold; in practice it should be much higher for strong trends)
    assert uptrend_count > len(regimes_after_warmup) * 0.25, (
        f"Expected significant uptrend classification in synthetic uptrend data, "
        f"but only {uptrend_count}/{len(regimes_after_warmup)} days were uptrend. "
        f"Regime distribution: {regimes_after_warmup.value_counts()}"
    )

    # Verify that regimes are not all "unknown" or all neutral
    assert regimes_after_warmup.nunique() > 1, (
        f"All regimes are the same: {regimes_after_warmup.unique()}. "
        "Expected variety in uptrend data."
    )


def test_classify_momentum_regime_downtrend():
    """Test that downtrend is correctly classified."""
    # Generate synthetic downtrend data
    qqq = make_synthetic_trend_data(start_date='2023-01-01', n_days=300, trend='down')

    # Build features
    feature_params = QqqMomentumFeatureParams()
    qqq_features = build_qqq_momentum_features(qqq, feature_params)

    # Classify regimes
    regime_params = QqqMomentumRegimeParams()
    regimes = classify_momentum_regime(qqq_features, regime_params)

    # Verify that at least some dates are classified as downtrend
    regimes_after_warmup = regimes.iloc[250:]

    downtrend_count = (regimes_after_warmup == MomentumRegime.DOWNTREND).sum()

    # At least 25% of post-warmup days should be classified as downtrend
    assert downtrend_count > len(regimes_after_warmup) * 0.25, (
        f"Expected significant downtrend classification in synthetic downtrend data, "
        f"but only {downtrend_count}/{len(regimes_after_warmup)} days were downtrend. "
        f"Regime distribution: {regimes_after_warmup.value_counts()}"
    )


def test_regime_to_target_weights_strong_uptrend():
    """Test that STRONG_UPTREND maps to non-zero TQQQ weight."""
    symbols = QqqMomentumSymbols(
        reference_symbol="QQQ",
        long_symbol="TQQQ",
        short_symbol="SQQQ",
    )
    alloc_params = QqqMomentumAllocationParams(
        strong_uptrend_long_weight=1.0,
        downtrend_short_weight=0.5,
    )

    weights = regime_to_target_weights(MomentumRegime.STRONG_UPTREND, symbols, alloc_params)

    # Should allocate 100% to TQQQ
    assert weights.get('TQQQ', 0.0) == 1.0, (
        f"Expected TQQQ weight = 1.0 in STRONG_UPTREND, got {weights}"
    )
    assert weights.get('SQQQ', 0.0) == 0.0, (
        f"Expected no SQQQ in STRONG_UPTREND, got {weights}"
    )


def test_regime_to_target_weights_downtrend():
    """Test that DOWNTREND maps to non-zero SQQQ weight."""
    symbols = QqqMomentumSymbols(
        reference_symbol="QQQ",
        long_symbol="TQQQ",
        short_symbol="SQQQ",
    )
    alloc_params = QqqMomentumAllocationParams(
        strong_uptrend_long_weight=1.0,
        downtrend_short_weight=0.5,
    )

    weights = regime_to_target_weights(MomentumRegime.DOWNTREND, symbols, alloc_params)

    # Should allocate 50% to SQQQ
    assert weights.get('SQQQ', 0.0) == 0.5, (
        f"Expected SQQQ weight = 0.5 in DOWNTREND, got {weights}"
    )
    assert weights.get('TQQQ', 0.0) == 0.0, (
        f"Expected no TQQQ in DOWNTREND, got {weights}"
    )


def test_regime_to_target_weights_neutral():
    """Test that NEUTRAL maps to all cash (empty dict)."""
    symbols = QqqMomentumSymbols()
    alloc_params = QqqMomentumAllocationParams()

    weights = regime_to_target_weights(MomentumRegime.NEUTRAL, symbols, alloc_params)

    # Should be all cash (empty dict)
    assert len(weights) == 0, (
        f"Expected empty weights (all cash) in NEUTRAL, got {weights}"
    )


def test_strategy_generate_target_weights_with_uptrend():
    """
    Test that QqqMomentumStrategy generates non-zero weights for uptrend dates.

    **Purpose**: This is the critical integration test that verifies:
      1. Strategy can look up regimes from features (no timestamp mismatch).
      2. Regime classification produces non-neutral regimes in trending data.
      3. Regime-to-weights mapping produces non-zero allocations.
    """
    # Generate synthetic uptrend data
    qqq = make_synthetic_trend_data(start_date='2023-01-01', n_days=300, trend='up')

    # Build features
    feature_params = QqqMomentumFeatureParams()
    qqq_features = build_qqq_momentum_features(qqq, feature_params)

    # Create strategy
    regime_params = QqqMomentumRegimeParams()
    alloc_params = QqqMomentumAllocationParams(
        strong_uptrend_long_weight=1.0,
        weakening_uptrend_long_weight=0.5,
        downtrend_short_weight=0.5,
    )
    symbols = QqqMomentumSymbols()

    strategy = QqqMomentumStrategy(
        feature_params=feature_params,
        regime_params=regime_params,
        allocation_params=alloc_params,
        symbols=symbols,
        qqq_features=qqq_features,
    )

    # Test generate_target_weights for a date after warm-up
    # Use a date from the actual data range (after skipping warm-up period)
    # We generated 300 business days starting from 2023-01-01
    # After 250-day warm-up, we should have ~50 valid days
    # Pick a date in the middle of the valid range
    test_date = strategy.regimes.index[260]  # Skip 260 days for extra safety

    # Mock data dict (strategy doesn't use it, but engine passes it)
    data = {}

    # Call generate_target_weights
    weights = strategy.generate_target_weights(dt=test_date, data=data, portfolio_state=None)

    # Verify that weights are not empty (should have TQQQ allocation in uptrend)
    assert len(weights) > 0, (
        f"Expected non-empty weights for uptrend date {test_date}, got {weights}. "
        f"Regime: {strategy.get_regime_for_date(test_date)}"
    )

    # Verify that TQQQ weight is non-zero
    assert weights.get('TQQQ', 0.0) > 0, (
        f"Expected positive TQQQ weight in uptrend, got {weights}. "
        f"Regime: {strategy.get_regime_for_date(test_date)}"
    )


def test_strategy_timestamp_normalization():
    """
    Test that strategy handles both timezone-aware and timezone-naive timestamps.

    **Purpose**: Verify that timestamp normalization in strategy.__init__ and
    generate_target_weights works correctly, preventing index lookup failures.
    """
    # Generate synthetic data with timezone-aware timestamps
    qqq = make_synthetic_trend_data(start_date='2023-01-01', n_days=300, trend='up')

    # Build features (will have timezone-aware index from input data)
    feature_params = QqqMomentumFeatureParams()
    qqq_features = build_qqq_momentum_features(qqq, feature_params)

    # Create strategy (should normalize index internally)
    regime_params = QqqMomentumRegimeParams()
    alloc_params = QqqMomentumAllocationParams()
    symbols = QqqMomentumSymbols()

    strategy = QqqMomentumStrategy(
        feature_params=feature_params,
        regime_params=regime_params,
        allocation_params=alloc_params,
        symbols=symbols,
        qqq_features=qqq_features,
    )

    # Verify that strategy's features and regimes have normalized (timezone-naive) index
    assert strategy.qqq_features.index.tz is None, (
        "Strategy features index should be timezone-naive after normalization"
    )
    assert strategy.regimes.index.tz is None, (
        "Strategy regimes index should be timezone-naive after normalization"
    )

    # Use a date from the actual data range (after warm-up)
    test_date_from_index = strategy.regimes.index[260]

    # Test with timezone-aware timestamp (matching engine behavior)
    test_date_aware = pd.Timestamp(test_date_from_index).tz_localize('UTC')
    regime_aware = strategy.get_regime_for_date(test_date_aware)
    assert regime_aware is not None, (
        f"Strategy should handle timezone-aware timestamp {test_date_aware}"
    )

    # Test with timezone-naive timestamp
    test_date_naive = pd.Timestamp(test_date_from_index)
    regime_naive = strategy.get_regime_for_date(test_date_naive)
    assert regime_naive is not None, (
        f"Strategy should handle timezone-naive timestamp {test_date_naive}"
    )

    # Both should return the same regime (normalization should make them equivalent)
    assert regime_aware == regime_naive, (
        f"Timezone-aware and naive timestamps should return same regime. "
        f"Got {regime_aware} vs {regime_naive}"
    )


def test_strategy_regime_distribution():
    """
    Test that strategy's precomputed regimes match manual classification.

    **Purpose**: Verify that the strategy's internal regimes match what we'd get
    by calling classify_momentum_regime directly. This ensures __init__ doesn't
    accidentally break the regime assignment.
    """
    # Generate synthetic uptrend data
    qqq = make_synthetic_trend_data(start_date='2023-01-01', n_days=300, trend='up')

    # Build features
    feature_params = QqqMomentumFeatureParams()
    qqq_features = build_qqq_momentum_features(qqq, feature_params)

    # Classify regimes manually
    regime_params = QqqMomentumRegimeParams()
    regimes_manual = classify_momentum_regime(qqq_features, regime_params)

    # Create strategy (should compute same regimes internally)
    alloc_params = QqqMomentumAllocationParams()
    symbols = QqqMomentumSymbols()

    strategy = QqqMomentumStrategy(
        feature_params=feature_params,
        regime_params=regime_params,
        allocation_params=alloc_params,
        symbols=symbols,
        qqq_features=qqq_features,
    )

    # After normalization, indexes should match
    # (Manual regimes have original index, strategy regimes have normalized index)
    # Compare values for a few test dates
    test_dates = pd.date_range('2023-10-01', periods=10, freq='B')

    for test_date in test_dates:
        # Get manual regime (using original index)
        manual_date_key = pd.Timestamp(test_date).tz_localize('UTC')
        if manual_date_key in regimes_manual.index:
            manual_regime = regimes_manual.loc[manual_date_key]

            # Get strategy regime (using normalized lookup)
            strategy_regime = strategy.get_regime_for_date(manual_date_key)

            if strategy_regime is not None:
                # Should match
                assert strategy_regime == manual_regime, (
                    f"Regime mismatch for {test_date}: "
                    f"manual={manual_regime}, strategy={strategy_regime}"
                )
