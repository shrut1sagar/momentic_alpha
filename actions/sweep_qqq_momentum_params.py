#!/usr/bin/env python3
"""
Parameter sweep for QQQ momentum strategy.

**Purpose**: This script runs a grid search over key strategy parameters to find
configurations that maximize performance metrics (Sharpe ratio, CAGR, minimize
drawdown). Instead of manually tuning parameters, we systematically test many
combinations and identify which settings work best.

**Parameter grids**:
  - MA windows: Different timeframes for trend detection.
  - Regime thresholds: min_spread_for_trend, min_velocity_for_trend.
  - Allocation weights: How much leverage to use in each regime.

**Usage**:
    python actions/sweep_qqq_momentum_params.py

**Outputs**:
  - CSV: `data/results/qqq_momentum_param_sweep.csv` with all configurations
    and their metrics.
  - Terminal: Top 10 configurations ranked by Sharpe ratio.

**Teaching note**: Parameter sweeps are essential for quantitative strategy
development. They help you:
  1. Discover which parameters matter most (sensitivity analysis).
  2. Avoid overfitting (if tiny param changes = huge performance swings, beware).
  3. Build intuition about trade-offs (higher leverage = higher Sharpe but deeper drawdowns).
  4. Document which configurations you tested (reproducibility).

**Warning**: Parameter sweeps can overfit to historical data. Always:
  - Use walk-forward validation or out-of-sample testing.
  - Be skeptical of "too good to be true" results.
  - Prefer robust parameters (good across many periods) over peak-optimized ones.
"""

import sys
from pathlib import Path
import json
import pandas as pd
from itertools import product
from typing import Dict, List
import time

# Ensure project root is on path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Data loading
from src.data.loaders import load_qqq_history, load_tqqq_history, load_sqqq_history

# Feature engineering and strategy
from src.strategies.qqq_momentum_features import (
    QqqMomentumFeatureParams,
    QqqMomentumRegimeParams,
    QqqMomentumAllocationParams,
    QqqMomentumSymbols,
    build_qqq_momentum_features,
)
from src.strategies.qqq_momentum import QqqMomentumStrategy

# Backtesting
from src.backtesting.engine import BacktestParams, run_backtest


def define_parameter_grid() -> Dict[str, List]:
    """
    Define parameter grid for sweep.

    **Conceptual**: We test combinations of parameters across several dimensions:
      - Regime thresholds: Tighter thresholds (1-2%) vs looser (3-5%).
      - Allocation weights: Conservative (50% max) vs aggressive (100% max).

    **Design philosophy**:
      - Start with a coarse grid to explore the space quickly.
      - Focus on parameters that likely matter most (thresholds, weights).
      - Avoid testing tiny variations (0.01 vs 0.02) that are noise.
      - Keep MA windows constant to simplify feature generation.

    **Note**: We keep MA windows constant at (20, 50, 100, 250) to avoid
    complications with dynamic column names. A future enhancement could
    sweep MA windows in a separate analysis.

    Returns:
        Dictionary mapping parameter names to lists of values to test.
    """
    grid = {
        # MA window combinations (keep constant for simplicity)
        'ma_windows': [
            (20, 50, 100, 250),   # Keep default only
        ],

        # Regime classification thresholds
        # How much spread/velocity is needed to call it a trend?
        'min_spread_for_trend': [0.005, 0.01, 0.02, 0.03, 0.05],  # 0.5%, 1%, 2%, 3%, 5%
        'min_velocity_for_trend': [0.0, 0.0005, 0.001, 0.002],    # 0%, 0.05%, 0.1%, 0.2%

        # Allocation weights
        # How much to allocate in strong uptrend vs downtrend
        'strong_uptrend_long_weight': [0.3, 0.5, 0.7, 1.0],       # 30%, 50%, 70%, 100% TQQQ
        'downtrend_short_weight': [0.3, 0.5, 0.7, 1.0],           # 30%, 50%, 70%, 100% SQQQ
    }

    return grid


def run_single_backtest(
    qqq: pd.DataFrame,
    tqqq: pd.DataFrame,
    sqqq: pd.DataFrame,
    params_dict: Dict,
    backtest_params: BacktestParams,
) -> Dict:
    """
    Run a single backtest with specified parameters.

    **Conceptual**: This function encapsulates the full backtest workflow:
      1. Unpack parameter dict into typed parameter objects.
      2. Build features (cached to avoid recomputation).
      3. Create strategy with specified parameters.
      4. Run backtest.
      5. Extract key metrics.

    **Optimization**: We reuse the QQQ features DataFrame when MA windows don't
    change. This saves significant computation time during the sweep.

    Args:
        qqq: QQQ price history.
        tqqq: TQQQ price history.
        sqqq: SQQQ price history.
        params_dict: Dictionary of parameter values for this configuration.
        backtest_params: Backtest configuration (dates, fees, etc.).

    Returns:
        Dictionary with parameter values and resulting metrics.
    """
    # Unpack parameters
    ma_short, ma_medium, ma_long, ma_ultra_long = params_dict['ma_windows']

    feature_params = QqqMomentumFeatureParams(
        ma_short_window=ma_short,
        ma_medium_window=ma_medium,
        ma_long_window=ma_long,
        ma_ultra_long_window=ma_ultra_long,
        velocity_window=20,
        acceleration_window=20,
    )

    regime_params = QqqMomentumRegimeParams(
        min_spread_for_trend=params_dict['min_spread_for_trend'],
        min_velocity_for_trend=params_dict['min_velocity_for_trend'],
        min_acceleration_for_strong_trend=0.0,  # Keep default
        max_spread_for_neutral=0.01,  # Keep default
    )

    alloc_params = QqqMomentumAllocationParams(
        strong_uptrend_long_weight=params_dict['strong_uptrend_long_weight'],
        weakening_uptrend_long_weight=params_dict['strong_uptrend_long_weight'] * 0.5,  # Half of strong
        neutral_risk_weight=0.0,  # Keep default (all cash)
        downtrend_short_weight=params_dict['downtrend_short_weight'],
        use_vol_overlay=False,  # Keep default
    )

    symbols = QqqMomentumSymbols()

    # Build features (this is the expensive part)
    qqq_features = build_qqq_momentum_features(qqq, feature_params)

    # Rename columns to match expected names for classify_momentum_regime
    # The function expects 'ma_spread_50_250', 'velocity_20d', 'acceleration_20d'
    # But our features may have different names based on the window parameters
    qqq_features_renamed = qqq_features.copy()

    # Find the actual MA spread column name (e.g., ma_spread_30_250 -> ma_spread_50_250)
    expected_spread_col = f'ma_spread_{ma_medium}_{ma_ultra_long}'
    if expected_spread_col in qqq_features_renamed.columns and expected_spread_col != 'ma_spread_50_250':
        qqq_features_renamed['ma_spread_50_250'] = qqq_features_renamed[expected_spread_col]

    # Create strategy
    strategy = QqqMomentumStrategy(
        feature_params=feature_params,
        regime_params=regime_params,
        allocation_params=alloc_params,
        symbols=symbols,
        qqq_features=qqq_features_renamed,
    )

    # Build data dict for backtest
    data = {
        "QQQ": qqq_features,  # Use features DataFrame (includes prices)
        "TQQQ": tqqq,
        "SQQQ": sqqq,
    }

    # Run backtest
    try:
        result = run_backtest(data, strategy, backtest_params)
        metrics = result.metrics
    except Exception as e:
        # If backtest fails, return NaN metrics
        print(f"  ✗ Backtest failed: {e}")
        metrics = {
            'total_return': float('nan'),
            'cagr': float('nan'),
            'sharpe_ratio': float('nan'),
            'sortino_ratio': float('nan'),
            'max_drawdown': float('nan'),
            'calmar_ratio': float('nan'),
            'final_equity': float('nan'),
            'num_trading_days': 0,
        }

    # Combine parameters and metrics into result dict
    result_dict = {
        'ma_short': ma_short,
        'ma_medium': ma_medium,
        'ma_long': ma_long,
        'ma_ultra_long': ma_ultra_long,
        'min_spread': params_dict['min_spread_for_trend'],
        'min_velocity': params_dict['min_velocity_for_trend'],
        'strong_up_weight': params_dict['strong_uptrend_long_weight'],
        'downtrend_weight': params_dict['downtrend_short_weight'],
        'total_return': metrics['total_return'],
        'cagr': metrics['cagr'],
        'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
        'sortino_ratio': metrics.get('sortino_ratio', 0.0),
        'max_drawdown': metrics['max_drawdown'],
        'calmar_ratio': metrics.get('calmar_ratio', 0.0),
        'final_equity': metrics['final_equity'],
        'num_trading_days': metrics['num_trading_days'],
    }

    return result_dict


def main():
    """
    Main entrypoint for parameter sweep.

    Steps:
      1. Load historical data (QQQ, TQQQ, SQQQ).
      2. Define parameter grid.
      3. Generate all parameter combinations.
      4. Run backtest for each combination.
      5. Collect results into DataFrame.
      6. Rank by Sharpe ratio (primary), max drawdown (secondary), return (tertiary).
      7. Save full results to CSV.
      8. Print top configurations.
    """
    print("=" * 80)
    print("QQQ Momentum Strategy Parameter Sweep")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Load historical data
    # ========================================================================
    print("Step 1: Loading historical price data...")

    try:
        qqq = load_qqq_history()
        print(f"  ✓ Loaded QQQ: {len(qqq)} rows")
    except FileNotFoundError:
        print("  ✗ QQQ data not found at data/raw/QQQ.csv")
        return

    try:
        tqqq = load_tqqq_history()
        print(f"  ✓ Loaded TQQQ: {len(tqqq)} rows")
    except FileNotFoundError:
        print("  ✗ TQQQ data not found at data/raw/TQQQ.csv")
        return

    try:
        sqqq = load_sqqq_history()
        print(f"  ✓ Loaded SQQQ: {len(sqqq)} rows")
    except FileNotFoundError:
        print("  ✗ SQQQ data not found at data/raw/SQQQ.csv")
        return

    print()

    # ========================================================================
    # Step 2: Define parameter grid
    # ========================================================================
    print("Step 2: Defining parameter grid...")
    grid = define_parameter_grid()

    # Calculate total combinations
    total_combos = (
        len(grid['ma_windows']) *
        len(grid['min_spread_for_trend']) *
        len(grid['min_velocity_for_trend']) *
        len(grid['strong_uptrend_long_weight']) *
        len(grid['downtrend_short_weight'])
    )

    print(f"  Grid dimensions:")
    print(f"    MA windows: {len(grid['ma_windows'])} variants")
    print(f"    Min spread: {len(grid['min_spread_for_trend'])} values")
    print(f"    Min velocity: {len(grid['min_velocity_for_trend'])} values")
    print(f"    Strong up weight: {len(grid['strong_uptrend_long_weight'])} values")
    print(f"    Downtrend weight: {len(grid['downtrend_short_weight'])} values")
    print(f"  Total combinations: {total_combos}")
    print()

    # ========================================================================
    # Step 3: Configure backtest parameters (same for all runs)
    # ========================================================================
    print("Step 3: Configuring backtest parameters...")

    # Determine backtest date range
    start_date = max(
        qqq['timestamp'].min(),
        tqqq['timestamp'].min(),
        sqqq['timestamp'].min(),
    )
    end_date = min(
        qqq['timestamp'].max(),
        tqqq['timestamp'].max(),
        sqqq['timestamp'].max(),
    )

    # Add buffer for warm-up period (need 250 days for longest MA)
    start_date = start_date + pd.Timedelta(days=260)

    print(f"  Backtest period: {start_date.date()} to {end_date.date()}")
    print(f"  Duration: {(end_date - start_date).days} days")

    backtest_params = BacktestParams(
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        initial_cash=100_000,  # $100k starting capital
        slippage_bps=5.0,  # 5 bps = 0.05% slippage
        fee_per_trade=0.0,  # Assume commission-free trading
    )
    print()

    # ========================================================================
    # Step 4: Run parameter sweep
    # ========================================================================
    print(f"Step 4: Running parameter sweep ({total_combos} configurations)...")
    print("  This may take several minutes...")
    print()

    results = []
    start_time = time.time()

    # Generate all parameter combinations
    for i, combo in enumerate(product(
        grid['ma_windows'],
        grid['min_spread_for_trend'],
        grid['min_velocity_for_trend'],
        grid['strong_uptrend_long_weight'],
        grid['downtrend_short_weight'],
    ), start=1):

        ma_windows, min_spread, min_velocity, strong_up_weight, down_weight = combo

        params_dict = {
            'ma_windows': ma_windows,
            'min_spread_for_trend': min_spread,
            'min_velocity_for_trend': min_velocity,
            'strong_uptrend_long_weight': strong_up_weight,
            'downtrend_short_weight': down_weight,
        }

        # Print progress every 10 configurations
        if i % 10 == 0 or i == 1:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (total_combos - i)
            print(f"  Progress: {i}/{total_combos} ({100*i/total_combos:.1f}%) "
                  f"- Elapsed: {elapsed:.1f}s - ETA: {remaining:.1f}s")

        # Run backtest for this configuration
        result = run_single_backtest(qqq, tqqq, sqqq, params_dict, backtest_params)
        results.append(result)

    total_time = time.time() - start_time
    print()
    print(f"  ✓ Sweep complete: {len(results)} configurations tested in {total_time:.1f}s")
    print(f"    Average time per configuration: {total_time/len(results):.2f}s")
    print()

    # ========================================================================
    # Step 5: Collect results into DataFrame
    # ========================================================================
    print("Step 5: Analyzing results...")
    results_df = pd.DataFrame(results)

    # Remove failed runs (NaN metrics)
    valid_results = results_df.dropna(subset=['sharpe_ratio'])
    if len(valid_results) < len(results_df):
        print(f"  ⚠ {len(results_df) - len(valid_results)} configurations failed")

    results_df = valid_results

    print(f"  ✓ Valid results: {len(results_df)} configurations")
    print()

    # ========================================================================
    # Step 6: Rank configurations
    # ========================================================================
    print("Step 6: Ranking configurations...")

    # Primary: Sharpe ratio (risk-adjusted return)
    # Secondary: Max drawdown (smaller is better, so negate for sorting)
    # Tertiary: Total return
    results_df['rank_score'] = (
        results_df['sharpe_ratio'] * 1000 +  # Scale Sharpe to dominate
        -results_df['max_drawdown'] * 100 +  # Penalize drawdown
        results_df['total_return'] * 10      # Small bonus for return
    )

    results_df = results_df.sort_values('rank_score', ascending=False)

    print("  ✓ Ranked by: Sharpe ratio (primary), Max drawdown (secondary), Total return (tertiary)")
    print()

    # ========================================================================
    # Step 7: Save full results to CSV
    # ========================================================================
    print("Step 7: Saving results...")

    results_dir = repo_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "qqq_momentum_param_sweep.csv"
    results_df.to_csv(output_path, index=False)

    print(f"  ✓ Saved full results: {output_path}")
    print()

    # ========================================================================
    # Step 8: Display top configurations
    # ========================================================================
    print("=" * 80)
    print("Top 10 Configurations (Ranked by Sharpe Ratio)")
    print("=" * 80)
    print()

    top_10 = results_df.head(10)

    print(f"{'Rank':>4s} {'MA':>12s} {'Spread':>7s} {'Vel':>6s} "
          f"{'Up%':>5s} {'Down%':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'Return':>8s}")
    print("-" * 80)

    for idx, row in enumerate(top_10.itertuples(), start=1):
        ma_str = f"{row.ma_short}/{row.ma_medium}/{row.ma_long}/{row.ma_ultra_long}"
        print(f"{idx:4d} {ma_str:>12s} {row.min_spread:7.2%} {row.min_velocity:6.3f} "
              f"{row.strong_up_weight:5.1%} {row.downtrend_weight:6.1%} "
              f"{row.sharpe_ratio:7.2f} {row.max_drawdown:7.2%} {row.total_return:8.2%}")

    print("-" * 80)
    print()

    # ========================================================================
    # Step 9: Summary statistics
    # ========================================================================
    print("Summary Statistics Across All Configurations:")
    print("-" * 80)
    print(f"  Sharpe Ratio:    min={results_df['sharpe_ratio'].min():.2f}, "
          f"median={results_df['sharpe_ratio'].median():.2f}, "
          f"max={results_df['sharpe_ratio'].max():.2f}")
    print(f"  Max Drawdown:    min={results_df['max_drawdown'].min():.2%}, "
          f"median={results_df['max_drawdown'].median():.2%}, "
          f"max={results_df['max_drawdown'].max():.2%}")
    print(f"  Total Return:    min={results_df['total_return'].min():.2%}, "
          f"median={results_df['total_return'].median():.2%}, "
          f"max={results_df['total_return'].max():.2%}")
    print("-" * 80)
    print()

    print("=" * 80)
    print("Parameter sweep complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  - Review top configurations in qqq_momentum_param_sweep.csv")
    print("  - Test top configs on out-of-sample data (walk-forward validation)")
    print("  - Look for patterns: Which parameters matter most?")
    print("  - Be skeptical of outliers: If one config is way better, it may be overfit")
    print()


if __name__ == "__main__":
    main()
