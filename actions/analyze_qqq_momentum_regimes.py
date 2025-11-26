#!/usr/bin/env python3
"""
Analyze QQQ momentum regime classifications against actual QQQ returns.

**Purpose**: This diagnostic script helps understand why the QQQ momentum strategy
might be underperforming. It compares the strategy's regime classifications
(strong uptrend, weakening uptrend, neutral, downtrend) with actual QQQ returns
on those days.

**Key questions this script answers**:
  1. Are "strong uptrend" days actually followed by positive returns?
  2. Are "downtrend" days actually followed by negative returns?
  3. Is the strategy correctly identifying market regimes, or is it reacting
     to noise/lagging indicators?
  4. How much overlap is there between regimes (e.g., how often does the
     regime change)?

**Usage**:
    python actions/analyze_qqq_momentum_regimes.py

**Outputs**:
  - Terminal: Summary statistics of returns by regime.
  - CSV: `data/results/qqq_momentum_regime_analysis.csv` with per-date regime
    and forward returns.

**Teaching note**: This type of diagnostic is critical for strategy development.
Before tuning parameters or changing logic, we need to understand whether the
fundamental hypothesis (momentum predicts returns) holds in our data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Data loading
from src.data.loaders import load_qqq_history

# Feature engineering and strategy
from src.strategies.qqq_momentum_features import (
    QqqMomentumFeatureParams,
    QqqMomentumRegimeParams,
    build_qqq_momentum_features,
    classify_momentum_regime,
    MomentumRegime,
)


def compute_forward_returns(prices: pd.DataFrame, periods: list[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Compute forward returns for multiple holding periods.

    **Conceptual**: Forward returns measure how much the price moves AFTER
    a given date. For example, 1-day forward return = (price[t+1] - price[t]) / price[t].
    This tells us if a regime classification on day t predicted profitable trades.

    **Why multiple periods?**
      - 1-day: Tests immediate momentum (does strong uptrend mean next day is up?).
      - 5-day: Tests short-term persistence (does trend continue for a week?).
      - 10/20-day: Tests medium-term trends (does regime capture longer cycles?).

    Args:
        prices: DataFrame with 'timestamp' and 'closing_price' columns.
        periods: List of forward periods to compute (in trading days).

    Returns:
        DataFrame with original columns plus 'forward_return_Nd' columns for each period.
    """
    df = prices.copy()

    # Sort by timestamp ascending (oldest first) for forward computation
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    # Compute forward returns for each period
    for period in periods:
        # Shift prices backward by N periods to get future price
        future_price = df['closing_price'].shift(-period)
        current_price = df['closing_price']

        # Forward return = (future_price - current_price) / current_price
        forward_return = (future_price - current_price) / current_price
        df[f'forward_return_{period}d'] = forward_return

    # Sort back to descending (newest first) to match repo convention
    df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

    return df


def analyze_regimes_vs_returns(
    qqq_with_regimes: pd.DataFrame,
    feature_params: QqqMomentumFeatureParams,
    regime_params: QqqMomentumRegimeParams,
) -> None:
    """
    Analyze regime classifications against forward returns.

    **Conceptual**: Group days by regime and compute average forward returns.
    If the strategy works, we expect:
      - STRONG_UPTREND days → positive forward returns
      - DOWNTREND days → negative forward returns
      - NEUTRAL days → near-zero forward returns

    **If these expectations don't hold**:
      - The regime classification may be lagging (reacting to past trends).
      - The thresholds may be miscalibrated (too sensitive or too insensitive).
      - The market may not exhibit momentum (mean-reversion dominates).

    Args:
        qqq_with_regimes: DataFrame with 'regime' and 'forward_return_*' columns.
        feature_params: Feature parameters used (for logging/reference).
        regime_params: Regime parameters used (for logging/reference).
    """
    print("=" * 80)
    print("Regime Classification vs Forward Returns Analysis")
    print("=" * 80)
    print()

    print("Configuration:")
    print(f"  Feature params: MA windows {feature_params.ma_short_window}/{feature_params.ma_medium_window}/"
          f"{feature_params.ma_long_window}/{feature_params.ma_ultra_long_window}")
    print(f"  Regime params: min_spread={regime_params.min_spread_for_trend:.3f}, "
          f"min_velocity={regime_params.min_velocity_for_trend:.3f}")
    print()

    # Regime distribution
    print("Regime Distribution:")
    print("-" * 80)
    regime_counts = qqq_with_regimes['regime'].value_counts()
    total_days = len(qqq_with_regimes)
    for regime, count in regime_counts.items():
        pct = 100.0 * count / total_days
        print(f"  {regime.value:20s}: {count:5d} days ({pct:5.1f}%)")
    print()

    # Forward returns by regime
    print("Average Forward Returns by Regime:")
    print("-" * 80)
    print(f"{'Regime':20s} {'Count':>6s} {'1-day':>8s} {'5-day':>8s} {'10-day':>8s} {'20-day':>8s}")
    print("-" * 80)

    for regime in [MomentumRegime.STRONG_UPTREND, MomentumRegime.WEAKENING_UPTREND,
                   MomentumRegime.NEUTRAL, MomentumRegime.DOWNTREND]:
        regime_data = qqq_with_regimes[qqq_with_regimes['regime'] == regime]
        if len(regime_data) == 0:
            continue

        count = len(regime_data)
        returns_1d = regime_data['forward_return_1d'].mean() * 100  # Convert to percentage
        returns_5d = regime_data['forward_return_5d'].mean() * 100
        returns_10d = regime_data['forward_return_10d'].mean() * 100
        returns_20d = regime_data['forward_return_20d'].mean() * 100

        print(f"{regime.value:20s} {count:6d} {returns_1d:7.2f}% {returns_5d:7.2f}% "
              f"{returns_10d:7.2f}% {returns_20d:7.2f}%")

    print("-" * 80)
    print()

    # Hit rate analysis (% of positive forward returns by regime)
    print("Hit Rate (% Positive Forward Returns) by Regime:")
    print("-" * 80)
    print(f"{'Regime':20s} {'Count':>6s} {'1-day':>8s} {'5-day':>8s} {'10-day':>8s} {'20-day':>8s}")
    print("-" * 80)

    for regime in [MomentumRegime.STRONG_UPTREND, MomentumRegime.WEAKENING_UPTREND,
                   MomentumRegime.NEUTRAL, MomentumRegime.DOWNTREND]:
        regime_data = qqq_with_regimes[qqq_with_regimes['regime'] == regime]
        if len(regime_data) == 0:
            continue

        count = len(regime_data)
        hit_1d = (regime_data['forward_return_1d'] > 0).mean() * 100
        hit_5d = (regime_data['forward_return_5d'] > 0).mean() * 100
        hit_10d = (regime_data['forward_return_10d'] > 0).mean() * 100
        hit_20d = (regime_data['forward_return_20d'] > 0).mean() * 100

        print(f"{regime.value:20s} {count:6d} {hit_1d:7.1f}% {hit_5d:7.1f}% "
              f"{hit_10d:7.1f}% {hit_20d:7.1f}%")

    print("-" * 80)
    print()

    # Interpretation guidance
    print("Interpretation:")
    print("-" * 80)
    print("For a working momentum strategy, we expect:")
    print("  - STRONG_UPTREND: Positive forward returns, >50% hit rate")
    print("  - DOWNTREND: Negative forward returns, <50% hit rate")
    print("  - NEUTRAL: Near-zero returns, ~50% hit rate")
    print()
    print("If these patterns don't hold:")
    print("  - Regime classification may be lagging (reacting to past moves)")
    print("  - Thresholds may need tuning (too sensitive or too insensitive)")
    print("  - Market may not exhibit momentum in this time period")
    print("=" * 80)


def main():
    """
    Main entrypoint for regime analysis.

    Steps:
      1. Load QQQ historical data.
      2. Build QQQ momentum features.
      3. Classify regimes using default parameters.
      4. Compute forward returns for multiple periods.
      5. Analyze regime effectiveness.
      6. Save detailed results to CSV.
    """
    print("=" * 80)
    print("QQQ Momentum Regime Analysis")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Load QQQ data
    # ========================================================================
    print("Step 1: Loading QQQ historical data...")
    try:
        qqq = load_qqq_history()
        print(f"  ✓ Loaded QQQ: {len(qqq)} rows from {qqq['timestamp'].min()} to {qqq['timestamp'].max()}")
    except FileNotFoundError:
        print("  ✗ QQQ data not found at data/raw/QQQ.csv")
        print("    Please run: python actions/fetch_massive_price_history.py QQQ --start 2020-01-01")
        return
    print()

    # ========================================================================
    # Step 2: Build features
    # ========================================================================
    print("Step 2: Building QQQ momentum features...")
    feature_params = QqqMomentumFeatureParams()
    qqq_features = build_qqq_momentum_features(qqq, feature_params)
    print(f"  ✓ Built features: {len(qqq_features)} rows, {len(qqq_features.columns)} columns")
    print()

    # ========================================================================
    # Step 3: Classify regimes
    # ========================================================================
    print("Step 3: Classifying momentum regimes...")
    regime_params = QqqMomentumRegimeParams()
    regimes = classify_momentum_regime(qqq_features, regime_params)
    print(f"  ✓ Classified {len(regimes)} regimes")
    print()

    # ========================================================================
    # Step 4: Compute forward returns
    # ========================================================================
    print("Step 4: Computing forward returns...")
    qqq_with_returns = compute_forward_returns(qqq, periods=[1, 5, 10, 20])
    print(f"  ✓ Computed forward returns for 1, 5, 10, 20-day periods")
    print()

    # ========================================================================
    # Step 5: Merge regimes with returns
    # ========================================================================
    print("Step 5: Merging regimes with forward returns...")

    # Regimes is a Series with integer index (same as qqq_features)
    # We need to align it with timestamps using the features DataFrame
    # Create a DataFrame with timestamp and regime
    regimes_df = pd.DataFrame({
        'timestamp': qqq_features['timestamp'],
        'regime': regimes.values,
    })

    # Normalize timestamp columns to ensure consistent format for merging
    qqq_with_returns = qqq_with_returns.copy()
    qqq_with_returns['timestamp'] = pd.to_datetime(qqq_with_returns['timestamp'])

    # If timezone-aware, convert to UTC and remove timezone
    if qqq_with_returns['timestamp'].dt.tz is not None:
        qqq_with_returns['timestamp'] = qqq_with_returns['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

    # Normalize to date only (midnight timestamp)
    qqq_with_returns['timestamp'] = qqq_with_returns['timestamp'].dt.normalize()

    # Ensure regimes_df timestamp is also normalized
    regimes_df['timestamp'] = pd.to_datetime(regimes_df['timestamp'])
    if regimes_df['timestamp'].dt.tz is not None:
        regimes_df['timestamp'] = regimes_df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    regimes_df['timestamp'] = regimes_df['timestamp'].dt.normalize()

    # Merge on timestamp
    qqq_analysis = qqq_with_returns.merge(regimes_df, on='timestamp', how='left')

    # Drop rows with NaN forward returns (last N days have no future data)
    qqq_analysis = qqq_analysis.dropna(subset=['forward_return_1d'])

    print(f"  ✓ Merged data: {len(qqq_analysis)} rows with regimes and forward returns")
    print()

    # ========================================================================
    # Step 6: Analyze regime effectiveness
    # ========================================================================
    analyze_regimes_vs_returns(qqq_analysis, feature_params, regime_params)

    # ========================================================================
    # Step 7: Save detailed results
    # ========================================================================
    print("Step 7: Saving detailed results...")
    results_dir = repo_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "qqq_momentum_regime_analysis.csv"

    # Select and order columns for output
    output_columns = [
        'timestamp',
        'closing_price',
        'regime',
        'forward_return_1d',
        'forward_return_5d',
        'forward_return_10d',
        'forward_return_20d',
        'ma_spread_50_250',
        'velocity_20d',
        'acceleration_20d',
    ]

    # Filter to available columns
    available_columns = [col for col in output_columns if col in qqq_analysis.columns]

    # Use write_normalized_csv to ensure canonical timestamp format
    from src.data.io import write_normalized_csv
    write_normalized_csv(qqq_analysis[available_columns], output_path)

    print(f"  ✓ Saved detailed analysis: {output_path}")
    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
