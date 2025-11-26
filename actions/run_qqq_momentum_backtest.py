#!/usr/bin/env python3
"""
Run QQQ momentum strategy backtest and save results.

**Purpose**: This script demonstrates how to:
  1. Load historical price data using Phase 3 loaders.
  2. Build features using Phase 5 feature engineering.
  3. Configure and run a backtest using Phase 4 engine.
  4. Save results (equity curve, metrics, reasoning trace) to data/results/.

**Usage**:
    From project root:
    ```bash
    python actions/run_qqq_momentum_backtest.py
    ```

**Outputs** (saved to data/results/):
  - qqq_momentum_equity_curve.csv: Time series of portfolio value.
  - qqq_momentum_metrics.json: Summary metrics (Sharpe, max drawdown, etc.).
  - qqq_momentum_reasoning_trace.csv: Per-date regime and target weights.
  - qqq_momentum_equity_curve.png: Plot of equity vs QQQ benchmark (optional).

**Teaching note**: This script is a "runnable example" showing how all Phase 1-5
components fit together. In production, you'd:
  - Add command-line arguments for dates, params, etc.
  - Run this in a scheduled job (cron, Airflow, etc.).
  - Send alerts if backtest fails or metrics degrade.
  - Archive results with version-tagged params for reproducibility.
"""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timezone

# Ensure project root is on path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Phase 3: Data loading
from src.data.loaders import load_qqq_history, load_tqqq_history, load_sqqq_history

# Phase 5: Feature engineering and strategy
from src.strategies.qqq_momentum_features import (
    QqqMomentumFeatureParams,
    QqqMomentumRegimeParams,
    QqqMomentumAllocationParams,
    QqqMomentumSymbols,
    build_qqq_momentum_features,
)
from src.strategies.qqq_momentum import QqqMomentumStrategy

# Phase 4: Backtesting
from src.backtesting.engine import BacktestParams, run_backtest

# Phase 3: Data I/O for saving results
from src.data.io import write_processed_data_csv


def main():
    """
    Main entrypoint for QQQ momentum backtest.

    Steps:
      1. Load data (QQQ, TQQQ, SQQQ).
      2. Build QQQ features.
      3. Configure strategy parameters.
      4. Run backtest.
      5. Save results to data/results/.
    """
    print("=" * 80)
    print("QQQ Momentum Strategy Backtest")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Load historical price data
    # ========================================================================
    print("Step 1: Loading historical price data...")

    try:
        qqq = load_qqq_history()
        print(f"  ✓ Loaded QQQ: {len(qqq)} rows from {qqq['timestamp'].min()} to {qqq['timestamp'].max()}")
    except FileNotFoundError:
        print("  ✗ QQQ data not found at data/raw/QQQ.csv")
        print("    Please download QQQ historical data and save to data/raw/QQQ.csv")
        return

    try:
        tqqq = load_tqqq_history()
        print(f"  ✓ Loaded TQQQ: {len(tqqq)} rows from {tqqq['timestamp'].min()} to {tqqq['timestamp'].max()}")
    except FileNotFoundError:
        print("  ✗ TQQQ data not found at data/raw/TQQQ.csv")
        print("    Please download TQQQ historical data and save to data/raw/TQQQ.csv")
        return

    try:
        sqqq = load_sqqq_history()
        print(f"  ✓ Loaded SQQQ: {len(sqqq)} rows from {sqqq['timestamp'].min()} to {sqqq['timestamp'].max()}")
    except FileNotFoundError:
        print("  ✗ SQQQ data not found at data/raw/SQQQ.csv")
        print("    Please download SQQQ historical data and save to data/raw/SQQQ.csv")
        return

    print()

    # ========================================================================
    # Step 2: Build QQQ features
    # ========================================================================
    print("Step 2: Building QQQ momentum features...")

    # Configure feature parameters
    feature_params = QqqMomentumFeatureParams(
        ma_short_window=20,
        ma_medium_window=50,
        ma_long_window=100,
        ma_ultra_long_window=250,
        velocity_window=20,
        acceleration_window=20,
        normalize_features=False,  # Use raw features (thresholds calibrated for raw values)
    )

    # Build features
    qqq_features = build_qqq_momentum_features(qqq, feature_params)
    print(f"  ✓ Built features: {qqq_features.shape[1]} columns")
    print(f"    Feature columns: {[col for col in qqq_features.columns if col not in qqq.columns]}")

    # Count NaNs (warm-up period)
    nan_count = qqq_features[['velocity_20d', 'acceleration_20d']].isna().any(axis=1).sum()
    print(f"    Warm-up period: {nan_count} days with NaN features (expected at start)")
    print()

    # ========================================================================
    # Step 3: Configure strategy parameters
    # ========================================================================
    print("Step 3: Configuring strategy parameters...")

    # Regime classification parameters
    regime_params = QqqMomentumRegimeParams(
        min_spread_for_trend=0.02,  # 2% spread required for trend
        min_velocity_for_trend=0.0,  # Positive velocity for uptrend
        min_acceleration_for_strong_trend=0.0,  # Positive accel for "strong" uptrend
        max_spread_for_neutral=0.01,  # < 1% spread = neutral
    )

    # Allocation parameters
    allocation_params = QqqMomentumAllocationParams(
        strong_uptrend_long_weight=1.0,  # 100% TQQQ in strong uptrend
        weakening_uptrend_long_weight=0.5,  # 50% TQQQ in weakening uptrend
        neutral_risk_weight=0.0,  # 0% = all cash in neutral
        downtrend_short_weight=0.5,  # 50% SQQQ in downtrend
        use_vol_overlay=False,  # Don't use UVXY (not loaded)
    )

    # Instrument symbols
    symbols = QqqMomentumSymbols(
        reference_symbol="QQQ",
        long_symbol="TQQQ",
        short_symbol="SQQQ",
        vol_symbol=None,  # Not using UVXY
    )

    print(f"  ✓ Regime params: min_spread={regime_params.min_spread_for_trend}, "
          f"min_velocity={regime_params.min_velocity_for_trend}")
    print(f"  ✓ Allocation params: strong_up={allocation_params.strong_uptrend_long_weight}, "
          f"down={allocation_params.downtrend_short_weight}")
    print()

    # ========================================================================
    # Step 4: Create strategy and prepare backtest data
    # ========================================================================
    print("Step 4: Creating strategy...")

    # Create strategy with precomputed features
    strategy = QqqMomentumStrategy(
        feature_params=feature_params,
        regime_params=regime_params,
        allocation_params=allocation_params,
        symbols=symbols,
        qqq_features=qqq_features,
    )
    print(f"  ✓ Strategy created with {len(strategy.regimes)} regime classifications")

    # Print regime distribution
    regime_counts = strategy.regimes.value_counts()
    print("    Regime distribution:")
    for regime, count in regime_counts.items():
        pct = 100 * count / len(strategy.regimes)
        print(f"      {regime.value:20s}: {count:4d} days ({pct:5.1f}%)")
    print()

    # Prepare data dict for backtest engine
    # Engine needs dict mapping symbol -> DataFrame
    data = {
        "QQQ": qqq_features,  # Include features for potential future use
        "TQQQ": tqqq,
        "SQQQ": sqqq,
    }

    # ========================================================================
    # Step 5: Configure and run backtest
    # ========================================================================
    print("Step 5: Running backtest...")

    # Determine backtest date range
    # Use overlapping period where all instruments have data
    start_date = max(
        qqq_features['timestamp'].min(),
        tqqq['timestamp'].min(),
        sqqq['timestamp'].min(),
    )
    end_date = min(
        qqq_features['timestamp'].max(),
        tqqq['timestamp'].max(),
        sqqq['timestamp'].max(),
    )

    # Add buffer for warm-up period (need 250 days for ultra-long MA)
    start_date = start_date + pd.Timedelta(days=260)

    print(f"  Backtest period: {start_date.date()} to {end_date.date()}")
    print(f"  Duration: {(end_date - start_date).days} days")

    # Configure backtest params
    backtest_params = BacktestParams(
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        initial_cash=100_000,  # $100k starting capital
        slippage_bps=5.0,  # 5 bps = 0.05% slippage (realistic for TQQQ/SQQQ)
        fee_per_trade=0.0,  # Assume commission-free trading
    )

    # Run backtest
    print("  Running backtest (this may take a minute)...")
    result = run_backtest(data, strategy, backtest_params)

    print(f"  ✓ Backtest complete: {len(result.equity_curve)} trading days")
    print()

    # ========================================================================
    # Step 6: Display metrics
    # ========================================================================
    print("Step 6: Backtest metrics:")
    print("-" * 80)
    metrics = result.metrics
    print(f"  Total Return:       {metrics['total_return']:>10.2%}")
    print(f"  CAGR:              {metrics['cagr']:>10.2%}")
    print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):>10.2f}")
    print(f"  Sortino Ratio:     {metrics.get('sortino_ratio', 0):>10.2f}")
    print(f"  Max Drawdown:      {metrics['max_drawdown']:>10.2%}")
    print(f"  Calmar Ratio:      {metrics.get('calmar_ratio', 0):>10.2f}")
    print(f"  Final Equity:      ${metrics['final_equity']:>10,.0f}")
    print(f"  Trading Days:      {metrics['num_trading_days']:>10,}")
    print("-" * 80)
    print()

    # ========================================================================
    # Step 7: Save results to data/results/
    # ========================================================================
    print("Step 7: Saving results to data/results/...")

    results_dir = repo_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save equity curve as CSV
    equity_curve_path = results_dir / "qqq_momentum_equity_curve.csv"
    equity_df = result.equity_curve.reset_index()
    equity_df.columns = ['timestamp', 'equity']

    # Use write_normalized_csv to ensure canonical timestamp format
    from src.data.io import write_normalized_csv
    write_normalized_csv(equity_df, equity_curve_path)
    print(f"  ✓ Saved equity curve: {equity_curve_path}")

    # Save metrics as JSON
    metrics_path = results_dir / "qqq_momentum_metrics.json"
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"  ✓ Saved metrics: {metrics_path}")

    # Build reasoning trace (per-date regime and target weights)
    # IMPORTANT: This captures what the STRATEGY DECIDED, not what was executed
    # This is crucial for debugging: "What did the strategy think vs what happened?"
    print("  Building reasoning trace...")
    reasoning_trace = []

    for state in result.portfolio_history:
        dt = state.timestamp

        # Get regime for this date (using normalized timestamp matching)
        regime = strategy.get_regime_for_date(dt)
        regime_label = regime.value if regime else "unknown"

        # Get the ACTUAL target weights the strategy generated for this date
        # This is what the strategy WANTED, before broker execution
        # We pass minimal data_slice (just current date) since strategy uses precomputed features
        data_slice = {}
        for symbol, df in data.items():
            # Filter to dates <= dt (matching engine behavior)
            # Handle both timezone-aware and timezone-naive timestamps
            df_timestamps = pd.to_datetime(df['timestamp'])
            if df_timestamps.dt.tz is not None:
                # Timezone-aware: convert to UTC, remove timezone, normalize
                normalized_timestamps = df_timestamps.dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
            else:
                # Timezone-naive: just normalize
                normalized_timestamps = df_timestamps.dt.normalize()
            df_slice = df[normalized_timestamps <= dt].copy()
            data_slice[symbol] = df_slice

        target_weights = strategy.generate_target_weights(dt=dt, data=data_slice, portfolio_state=state)

        # Calculate actual executed weights from final positions (for comparison)
        executed_weights = {sym: pos / state.equity if state.equity > 0 else 0
                           for sym, pos in state.positions.items()}

        reasoning_trace.append({
            'timestamp': dt,
            'regime': regime_label,
            'target_tqqq_weight': target_weights.get('TQQQ', 0.0),
            'target_sqqq_weight': target_weights.get('SQQQ', 0.0),
            'target_cash_weight': 1.0 - sum(target_weights.values()),
            'executed_tqqq_weight': executed_weights.get('TQQQ', 0.0),
            'executed_sqqq_weight': executed_weights.get('SQQQ', 0.0),
            'executed_cash_weight': 1.0 - sum(executed_weights.values()),
            'equity': state.equity,
        })

    reasoning_df = pd.DataFrame(reasoning_trace)
    reasoning_path = results_dir / "qqq_momentum_reasoning_trace.csv"

    # Use write_normalized_csv to ensure canonical timestamp format
    write_normalized_csv(reasoning_df, reasoning_path)
    print(f"  ✓ Saved reasoning trace: {reasoning_path}")

    # Optional: Plot equity curve
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot strategy equity
        result.equity_curve.plot(ax=ax, label='QQQ Momentum Strategy', linewidth=2)

        # Plot QQQ benchmark for comparison (if data aligns)
        # Extract QQQ prices that overlap with backtest period
        qqq_backtest = qqq[
            (qqq['timestamp'] >= start_date) &
            (qqq['timestamp'] <= end_date)
        ].copy()

        if len(qqq_backtest) > 0:
            # Normalize QQQ to start at same equity as strategy
            qqq_backtest = qqq_backtest.sort_values('timestamp')
            qqq_backtest = qqq_backtest.set_index('timestamp')
            qqq_equity = (qqq_backtest['closing_price'] / qqq_backtest['closing_price'].iloc[0]) * 100_000
            qqq_equity.plot(ax=ax, label='QQQ Buy & Hold', linewidth=2, alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.set_title('QQQ Momentum Strategy vs Buy & Hold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = results_dir / "qqq_momentum_equity_curve.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  ✓ Saved equity curve plot: {plot_path}")

    except ImportError:
        print("  ⚠ matplotlib not installed, skipping plot generation")
    except Exception as e:
        print(f"  ⚠ Could not generate plot: {e}")

    print()

    # ========================================================================
    # Done
    # ========================================================================
    print("=" * 80)
    print("Backtest complete! Results saved to data/results/")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  - Review metrics and equity curve plot")
    print("  - Inspect reasoning trace to understand regime decisions")
    print("  - Try different parameters (modify this script and re-run)")
    print("  - Run parameter sweeps to optimize Sharpe or minimize drawdown")
    print()


if __name__ == "__main__":
    main()
