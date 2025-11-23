"""
Tests for the backtest engine.

This module tests the backtest engine's ability to:
  - Iterate through time correctly.
  - Call strategies and brokers in the right sequence.
  - Respect date ranges.
  - Build equity curves and compute metrics.

All tests use trivial strategies (AlwaysCashStrategy, AlwaysLongStrategy) and
simple synthetic data with known expected outcomes.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.backtesting.engine import run_backtest, BacktestParams, BacktestResult
from src.strategies.base import AlwaysCashStrategy, AlwaysLongStrategy, BalancedStrategy


def make_simple_price_data(
    symbol: str,
    start_date: datetime,
    n_days: int,
    initial_price: float,
    daily_return: float = 0.0,
) -> pd.DataFrame:
    """
    Helper to create simple synthetic price data.

    Creates a DataFrame with timestamps and closing prices for a single symbol.
    Prices follow a simple pattern: price[t] = price[t-1] * (1 + daily_return).

    Args:
        symbol: Symbol name (for clarity in tests).
        start_date: Starting date (datetime).
        n_days: Number of trading days.
        initial_price: Starting closing price.
        daily_return: Daily percentage return (e.g., 0.01 = 1% daily growth).

    Returns:
        DataFrame with columns: timestamp, closing_price (sorted descending, newest first).
    """
    # Generate dates
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Generate prices (compounding daily returns)
    prices = [initial_price * ((1 + daily_return) ** i) for i in range(n_days)]

    # Create DataFrame (sorted descending, newest first, as per Phase 3 schema)
    df = pd.DataFrame({
        'timestamp': dates,
        'closing_price': prices,
    })

    # Sort descending (newest first)
    df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

    return df


def test_backtest_always_cash_strategy():
    """
    Test that an always-cash strategy maintains constant equity.

    Scenario:
      - Start with $100,000
      - Always hold 100% cash (no positions)
      - Backtest over 5 days

    Expected:
      - Equity remains $100,000 at every step
      - No positions ever opened
      - Total return = 0%
    """
    # Create simple data (prices don't matter since we're not trading)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data = {
        "TEST": make_simple_price_data("TEST", start, n_days=5, initial_price=100.0)
    }

    # Backtest params
    params = BacktestParams(
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(start + timedelta(days=4)),
        initial_cash=100_000,
        slippage_bps=0.0,
        fee_per_trade=0.0,
    )

    # Run backtest with always-cash strategy
    strategy = AlwaysCashStrategy()
    result = run_backtest(data, strategy, params)

    # Equity should be constant at $100,000
    assert len(result.equity_curve) == 5
    assert all(result.equity_curve == 100_000.0)

    # Metrics
    assert result.metrics['total_return'] == pytest.approx(0.0, abs=1e-6)
    assert result.metrics['final_equity'] == pytest.approx(100_000.0, abs=1e-6)


def test_backtest_always_long_monotonic_increase():
    """
    Test that an always-long strategy tracks price increases.

    Scenario:
      - Start with $100,000
      - Always hold 100% in TEST
      - TEST price increases from $100 to $121 over 3 days (+10% each day)

    Expected:
      - Day 0: Buy at $100 → 1000 shares → equity $100,000
      - Day 1: Price $110 → equity $110,000 (+10%)
      - Day 2: Price $121 → equity $121,000 (+10%)
      - Total return ≈ 21%
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Price increases 10% daily: 100, 110, 121
    data = {
        "TEST": make_simple_price_data("TEST", start, n_days=3, initial_price=100.0, daily_return=0.10)
    }

    params = BacktestParams(
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(start + timedelta(days=2)),
        initial_cash=100_000,
        slippage_bps=0.0,
        fee_per_trade=0.0,
    )

    # Always-long strategy
    strategy = AlwaysLongStrategy("TEST")
    result = run_backtest(data, strategy, params)

    # Check equity curve length
    assert len(result.equity_curve) == 3

    # Check equity values (should track price: 100k * (1.1^day))
    expected_equity = [100_000.0, 110_000.0, 121_000.0]
    for i, expected in enumerate(expected_equity):
        assert result.equity_curve.iloc[i] == pytest.approx(expected, abs=1.0)

    # Check metrics
    assert result.metrics['total_return'] == pytest.approx(0.21, abs=0.001)  # 21% gain
    assert result.metrics['final_equity'] == pytest.approx(121_000.0, abs=1.0)


def test_backtest_always_long_monotonic_decrease():
    """
    Test that an always-long strategy tracks price decreases.

    Scenario:
      - Always hold 100% in TEST
      - TEST price decreases from $100 to $81 over 3 days (-10% each day)

    Expected:
      - Equity decreases in line with price
      - Total return ≈ -19%
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Price decreases 10% daily: 100, 90, 81
    data = {
        "TEST": make_simple_price_data("TEST", start, n_days=3, initial_price=100.0, daily_return=-0.10)
    }

    params = BacktestParams(
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(start + timedelta(days=2)),
        initial_cash=100_000,
    )

    strategy = AlwaysLongStrategy("TEST")
    result = run_backtest(data, strategy, params)

    # Check final equity
    expected_final = 100_000 * (0.9 ** 2)  # 100k * 0.9 * 0.9 = 81k
    assert result.metrics['final_equity'] == pytest.approx(expected_final, abs=1.0)

    # Check total return
    assert result.metrics['total_return'] == pytest.approx(-0.19, abs=0.001)  # -19%


def test_backtest_date_range_filtering():
    """
    Test that backtest respects start_date and end_date.

    Scenario:
      - Data available for 10 days
      - Backtest only days 3-7 (5 days)

    Expected:
      - Equity curve has 5 data points (days 3-7)
      - First equity point uses day 3 price
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # 10 days of data
    data = {
        "TEST": make_simple_price_data("TEST", start, n_days=10, initial_price=100.0)
    }

    # Backtest only days 3-7 (indices 3, 4, 5, 6, 7 = 5 days)
    params = BacktestParams(
        start_date=pd.Timestamp(start + timedelta(days=3)),
        end_date=pd.Timestamp(start + timedelta(days=7)),
        initial_cash=100_000,
    )

    strategy = AlwaysLongStrategy("TEST")
    result = run_backtest(data, strategy, params)

    # Should have exactly 5 data points
    assert len(result.equity_curve) == 5
    assert result.metrics['num_trading_days'] == 5


def test_backtest_balanced_strategy():
    """
    Test a balanced strategy across two symbols.

    Scenario:
      - Two symbols: TEST_A and TEST_B
      - Allocate 50% to each
      - Both start at $100
      - TEST_A increases 10%, TEST_B decreases 10%

    Expected:
      - Portfolio gains 5% from TEST_A, loses 5% from TEST_B
      - Net effect: ≈ 0% return
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    data = {
        "TEST_A": make_simple_price_data("TEST_A", start, n_days=2, initial_price=100.0, daily_return=0.10),
        "TEST_B": make_simple_price_data("TEST_B", start, n_days=2, initial_price=100.0, daily_return=-0.10),
    }

    params = BacktestParams(
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(start + timedelta(days=1)),
        initial_cash=100_000,
        slippage_bps=0.0,
        fee_per_trade=0.0,
    )

    # 50/50 balanced strategy
    strategy = BalancedStrategy(["TEST_A", "TEST_B"])
    result = run_backtest(data, strategy, params)

    # Day 0: Buy 50% each at $100
    # - TEST_A: $50k / $100 = 500 shares
    # - TEST_B: $50k / $100 = 500 shares
    #
    # Day 1:
    # - TEST_A: 500 * $110 = $55k
    # - TEST_B: 500 * $90 = $45k
    # - Total: $100k (no net change)

    assert result.metrics['final_equity'] == pytest.approx(100_000.0, abs=1.0)
    assert result.metrics['total_return'] == pytest.approx(0.0, abs=0.001)


def test_backtest_with_slippage_and_fees():
    """
    Test that slippage and fees reduce returns.

    Scenario:
      - Always-long strategy on TEST
      - Price constant at $100
      - Slippage = 10 bps, fee = $100

    Expected:
      - Day 0: Buy 100% at $100 with slippage + fee
        - Trade: 1000 shares at effective price $100.10
        - Cost: 1000 * $100.10 + $100 fee = $100,200
        - Cash: -$200
        - Position value: 1000 * $100 = $100,000
        - Equity: $99,800
      - Day 1: Price still $100, rebalance (no change needed)
        - No trade (already at target weight), so no additional costs
        - Equity: $99,800 (but then we rebalance to maintain 100% weight)
        - Actually, we rebalance every day, which creates more trades
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Constant price (no market return)
    data = {
        "TEST": make_simple_price_data("TEST", start, n_days=2, initial_price=100.0, daily_return=0.0)
    }

    params = BacktestParams(
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(start + timedelta(days=1)),
        initial_cash=100_000,
        slippage_bps=10.0,  # 10 bps = 0.1%
        fee_per_trade=100.0,  # $100 flat fee
    )

    strategy = AlwaysLongStrategy("TEST")
    result = run_backtest(data, strategy, params)

    # With constant price, we lose slippage + fees on each rebalance
    # Day 0: Initial buy costs $100 slippage + $100 fee = -$200
    # Day 1: Rebalance to 100% of new equity ($99,800)
    #        Need to adjust position, which incurs more slippage + fees
    # Final equity < $99,800 due to day 1 rebalancing costs

    assert result.metrics['final_equity'] < 100_000.0
    # With daily rebalancing, expect ~$99,700 (slippage + fees on 2 days)
    assert result.metrics['final_equity'] == pytest.approx(99_700.0, abs=50.0)


def test_backtest_error_empty_data():
    """Test that backtest raises an error when data is empty."""
    params = BacktestParams(
        start_date=pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc)),
        end_date=pd.Timestamp(datetime(2024, 1, 5, tzinfo=timezone.utc)),
        initial_cash=100_000,
    )

    strategy = AlwaysCashStrategy()

    with pytest.raises(ValueError, match="Data dictionary is empty"):
        run_backtest({}, strategy, params)


def test_backtest_error_missing_columns():
    """Test that backtest raises an error when data is missing required columns."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Missing 'closing_price' column
    bad_data = {
        "TEST": pd.DataFrame({
            'timestamp': [start, start + timedelta(days=1)],
            'open_price': [100.0, 110.0],  # Has open but not closing
        })
    }

    params = BacktestParams(
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(start + timedelta(days=1)),
        initial_cash=100_000,
    )

    strategy = AlwaysCashStrategy()

    with pytest.raises(ValueError, match="missing 'closing_price' column"):
        run_backtest(bad_data, strategy, params)


def test_backtest_metrics_computed():
    """
    Test that backtest result includes expected metrics.

    Expected metrics:
      - total_return
      - cagr
      - sharpe_ratio
      - sortino_ratio
      - max_drawdown
      - calmar_ratio
      - final_equity
      - num_trading_days
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    data = {
        "TEST": make_simple_price_data("TEST", start, n_days=10, initial_price=100.0, daily_return=0.01)
    }

    params = BacktestParams(
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(start + timedelta(days=9)),
        initial_cash=100_000,
    )

    strategy = AlwaysLongStrategy("TEST")
    result = run_backtest(data, strategy, params)

    # Check that all expected metrics are present
    expected_metrics = [
        'total_return',
        'cagr',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'final_equity',
        'num_trading_days',
    ]

    for metric in expected_metrics:
        assert metric in result.metrics, f"Missing metric: {metric}"

    # All metrics should be numeric
    for metric, value in result.metrics.items():
        assert isinstance(value, (int, float, np.number)), f"Metric {metric} is not numeric: {value}"
