#!/usr/bin/env python3
"""
Run "always long QQQ" baseline backtest.

**Purpose**: This script runs a simple buy-and-hold backtest on QQQ to establish
a performance baseline. The QQQ momentum strategy should ideally outperform this
baseline—if it doesn't, the strategy is adding negative value through its
allocation decisions.

**Strategy**: AlwaysLongStrategy allocates 100% to QQQ at all times (no timing,
no leverage, no hedging). This is the simplest possible "equity exposure" strategy.

**Usage**:
    python actions/run_always_long_qqq_baseline.py

**Outputs** (saved to data/results/):
  - always_long_qqq_equity_curve.csv: Time series of portfolio value.
  - always_long_qqq_metrics.json: Summary metrics (Sharpe, max drawdown, etc.).

**Teaching note**: Every active strategy should be compared to a passive baseline.
If your momentum strategy has a Sharpe of 0.5 but buy-and-hold QQQ has a Sharpe
of 0.8, you're better off just holding QQQ. Baselines keep strategy developers
honest—they force you to prove that complexity adds value.
"""

import sys
from pathlib import Path
import json
import pandas as pd

# Ensure project root is on path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Data loading
from src.data.loaders import load_qqq_history

# Strategy
from src.strategies.base import AlwaysLongStrategy

# Backtesting
from src.backtesting.engine import BacktestParams, run_backtest


def main():
    """
    Main entrypoint for always-long QQQ baseline backtest.

    Steps:
      1. Load QQQ historical data.
      2. Create AlwaysLongStrategy("QQQ").
      3. Run backtest over the same period as momentum strategy.
      4. Save results to data/results/ for comparison.
    """
    print("=" * 80)
    print("Always Long QQQ Baseline Backtest")
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
    # Step 2: Create always-long strategy
    # ========================================================================
    print("Step 2: Creating AlwaysLongStrategy...")
    strategy = AlwaysLongStrategy(symbol="QQQ")
    print("  ✓ Strategy created: 100% allocation to QQQ at all times")
    print()

    # ========================================================================
    # Step 3: Configure and run backtest
    # ========================================================================
    print("Step 3: Running backtest...")

    # Determine backtest date range
    # Use full QQQ history, but add buffer for warm-up (250 days for longest MA)
    # This matches the momentum strategy's backtest period for fair comparison
    start_date = qqq['timestamp'].min() + pd.Timedelta(days=260)
    end_date = qqq['timestamp'].max()

    print(f"  Backtest period: {start_date.date()} to {end_date.date()}")
    print(f"  Duration: {(end_date - start_date).days} days")

    # Configure backtest params
    # Use same params as momentum strategy for fair comparison
    backtest_params = BacktestParams(
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        initial_cash=100_000,  # $100k starting capital
        slippage_bps=5.0,  # 5 bps = 0.05% slippage (same as momentum strategy)
        fee_per_trade=0.0,  # Assume commission-free trading
    )

    # Build data dict (strategy only needs QQQ)
    data = {"QQQ": qqq}

    # Run backtest
    print("  Running backtest...")
    result = run_backtest(data, strategy, backtest_params)

    print(f"  ✓ Backtest complete: {len(result.equity_curve)} trading days")
    print()

    # ========================================================================
    # Step 4: Display metrics
    # ========================================================================
    print("Step 4: Backtest metrics:")
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
    # Step 5: Save results to data/results/
    # ========================================================================
    print("Step 5: Saving results to data/results/...")

    results_dir = repo_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save equity curve as CSV
    equity_curve_path = results_dir / "always_long_qqq_equity_curve.csv"
    equity_df = result.equity_curve.reset_index()
    equity_df.columns = ['timestamp', 'equity']

    # Use write_normalized_csv to ensure canonical timestamp format
    from src.data.io import write_normalized_csv
    write_normalized_csv(equity_df, equity_curve_path)
    print(f"  ✓ Saved equity curve: {equity_curve_path}")

    # Save metrics as JSON
    metrics_path = results_dir / "always_long_qqq_metrics.json"
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"  ✓ Saved metrics: {metrics_path}")
    print()

    # ========================================================================
    # Step 6: Comparison guidance
    # ========================================================================
    print("=" * 80)
    print("Backtest complete! Results saved to data/results/")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  - Compare these metrics to qqq_momentum_metrics.json")
    print("  - If momentum strategy has lower Sharpe or higher drawdown, investigate:")
    print("    - Is regime classification lagging? (run analyze_qqq_momentum_regimes.py)")
    print("    - Are allocation weights too aggressive/conservative?")
    print("    - Does the strategy trade too frequently? (check reasoning trace)")
    print("  - Baseline Sharpe and return provide context for evaluating momentum strategy")
    print()


if __name__ == "__main__":
    main()
