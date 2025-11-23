"""
Daily-bar backtest engine for strategy evaluation.

**Conceptual**: The backtest engine is the orchestrator that brings together
strategies, data, and the broker to simulate portfolio evolution over time.
It iterates through historical data day by day, calling the strategy for
allocation decisions and the broker for execution simulation. The output is
an equity curve and performance metrics.

**Why separate engine from broker and strategy?**
  - Separation of concerns: The engine handles time iteration and data management,
    the broker handles execution, and the strategy handles allocation decisions.
  - Testability: Each component can be tested independently.
  - Extensibility: We can swap strategies, brokers, or data sources without
    changing the engine logic.

**Teaching note**: In production quant systems, the backtest engine is critical
infrastructure. It must be:
  - Correct: No time-travel bugs (looking into the future).
  - Fast: Able to run hundreds of backtests in parameter sweeps.
  - Observable: Produce detailed logs and diagnostics.
  - Reproducible: Same inputs → same outputs (no hidden state).

This implementation prioritizes correctness and clarity over speed. Optimizations
can come later (vectorization, caching, parallel execution).
"""

from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

from src.strategies.base import Strategy
from src.execution.paper_broker import PaperBroker, PortfolioState
from src.analytics.risk_metrics import (
    compute_total_return,
    compute_cagr,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
)


@dataclass
class BacktestParams:
    """
    Parameters for a backtest run.

    **Conceptual**: BacktestParams encapsulates all the settings needed to run
    a backtest: date range, initial capital, cost models. Keeping these in a
    dataclass makes it easy to:
      - Run multiple backtests with different parameters (parameter sweeps).
      - Serialize parameters for reproducibility (save to JSON/YAML).
      - Pass parameters around without long argument lists.

    **Teaching note**: In production, you'd also include:
      - Rebalancing frequency (e.g., daily, weekly, monthly).
      - Universe filters (which symbols to trade).
      - Risk limits (max position size, max drawdown stop-loss).
      - Benchmark for comparison.

    Attributes:
        start_date: Start date for the backtest (inclusive).
                   Data before this date is available for lookback but not traded.
        end_date: End date for the backtest (inclusive).
                 This is the last day the strategy can trade.
        initial_cash: Starting capital in base currency (e.g., 100000 for $100k).
                     Must be positive.
        slippage_bps: Slippage in basis points (default 0.0 = no slippage).
                     Applied symmetrically to buys and sells.
        fee_per_trade: Flat fee per trade in dollars (default 0.0 = commission-free).
                      Applied to each non-zero trade.
    """
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_cash: float
    slippage_bps: float = 0.0
    fee_per_trade: float = 0.0


@dataclass
class BacktestResult:
    """
    Results from a backtest run.

    **Conceptual**: BacktestResult packages everything you need to analyze
    backtest performance: equity curve, detailed state history, and summary metrics.
    This makes it easy to:
      - Plot equity curves and drawdowns.
      - Compare multiple backtests.
      - Store results to disk (CSV for equity, JSON for metrics).

    **Teaching note**: In production, you'd also include:
      - Trade ledger (every buy/sell with timestamp, symbol, quantity, price, fees).
      - Detailed diagnostics (turnover, concentration, exposure over time).
      - Attribution (which positions contributed to P&L).

    Attributes:
        equity_curve: Time series of total equity at each backtest step.
                     Index is timestamps, values are equity in dollars.
                     This is the primary output for performance analysis.
        portfolio_history: List of PortfolioState snapshots (one per backtest step).
                          Useful for detailed inspection of positions and exposure over time.
        metrics: Dictionary of summary metrics (Sharpe, max drawdown, etc.).
                Keys are metric names (e.g., "sharpe_ratio", "max_drawdown").
                Values are floats.
        params: The BacktestParams that generated this result (for reproducibility).
    """
    equity_curve: pd.Series
    portfolio_history: List[PortfolioState] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    params: BacktestParams | None = None


def run_backtest(
    data: Dict[str, pd.DataFrame],
    strategy: Strategy,
    params: BacktestParams,
) -> BacktestResult:
    """
    Run a daily-bar backtest.

    **Conceptual**: This is the main entrypoint for backtesting. It:
      1. Sets up a paper broker with the specified initial cash and cost models.
      2. Determines the backtest date range from params and available data.
      3. Iterates forward through time (day by day).
      4. At each step:
         - Extracts current prices (closing prices for the day).
         - Builds a data slice for the strategy (historical data up to current day).
         - Gets current portfolio state.
         - Calls the strategy to generate target weights.
         - Passes targets to the broker for execution.
         - Records the updated portfolio state.
      5. Builds an equity curve from the recorded states.
      6. Computes performance metrics.
      7. Returns a BacktestResult with all outputs.

    **Why iterate forward (ascending dates)?**
      - Simulates real-time behavior: you can only trade based on past data.
      - Prevents time-travel bugs: ensures strategy can't peek into the future.
      - Natural for event-driven logic (future extension to live trading).

    **Data alignment and time-travel prevention**:
      - All DataFrames in `data` must have a 'timestamp' column.
      - Data is filtered to only include rows with timestamp <= current_date.
      - Closing prices for current_date are used for trading (end-of-day execution).

    **Teaching note**: The most common backtest bug is time travel—accidentally
    using future data to make past decisions. This engine prevents that by:
      - Iterating forward in time.
      - Filtering data strictly to <= current_date.
      - Using only the closing price of current_date for trades (no intraday lookahead).

    Args:
        data: Dictionary mapping symbol -> DataFrame.
             Each DataFrame must have at minimum:
               - 'timestamp' column (datetime, sorted descending, newest first as per Phase 3).
               - 'closing_price' column (float, used for trading).
             Optional: other columns like volume, features, etc. (passed to strategy).
             Data should cover a range including start_date and end_date.
        strategy: Strategy object implementing the Strategy protocol.
                 Will be called once per backtest day with (dt, data_slice, portfolio_state).
        params: BacktestParams specifying date range, initial capital, and costs.

    Returns:
        BacktestResult with equity curve, portfolio history, and performance metrics.

    Raises:
        ValueError: If data is empty, missing required columns, or doesn't cover date range.
        ValueError: If params are invalid (e.g., negative initial_cash).
    """
    # Validate inputs
    if not data:
        raise ValueError("Data dictionary is empty. Need at least one symbol.")

    # Create paper broker with specified parameters
    broker = PaperBroker(
        initial_cash=params.initial_cash,
        slippage_bps=params.slippage_bps,
        fee_per_trade=params.fee_per_trade,
    )

    # Determine backtest date range
    # Extract all unique dates from all symbols (union of available dates)
    all_dates = set()
    for symbol, df in data.items():
        if 'timestamp' not in df.columns:
            raise ValueError(
                f"DataFrame for symbol '{symbol}' missing 'timestamp' column. "
                f"All data must have timestamps."
            )
        if 'closing_price' not in df.columns:
            raise ValueError(
                f"DataFrame for symbol '{symbol}' missing 'closing_price' column. "
                f"All data must have closing prices for trading."
            )
        # Extract dates (handle both datetime and date)
        dates = pd.to_datetime(df['timestamp']).dt.normalize()
        all_dates.update(dates.tolist())

    # Filter to backtest date range and sort ascending
    backtest_dates = sorted([
        dt for dt in all_dates
        if params.start_date <= dt <= params.end_date
    ])

    if not backtest_dates:
        raise ValueError(
            f"No data available in backtest date range "
            f"({params.start_date} to {params.end_date}). "
            f"Check that your data covers this period."
        )

    # Initialize tracking
    portfolio_history: List[PortfolioState] = []

    # Main backtest loop: iterate forward through time
    for current_date in backtest_dates:
        # Step 1: Extract current closing prices for all symbols
        # These are the prices at which we can trade at the end of current_date
        current_prices = {}
        for symbol, df in data.items():
            # Find the row for current_date (or most recent date <= current_date)
            # Data is sorted descending (newest first), so we need to find the right row
            df_filtered = df[pd.to_datetime(df['timestamp']).dt.normalize() <= current_date]
            if not df_filtered.empty:
                # Get the most recent row (first row after filtering, since sorted descending)
                latest_row = df_filtered.iloc[0]
                current_prices[symbol] = latest_row['closing_price']

        # Skip this date if no prices available (shouldn't happen if data is clean)
        if not current_prices:
            continue

        # Step 2: Update broker with current prices
        broker.update_prices(current_prices, current_date)

        # Step 3: Build data slice for strategy
        # Strategy receives historical data UP TO AND INCLUDING current_date
        # (but not future dates - this prevents time travel)
        data_slice = {}
        for symbol, df in data.items():
            # Filter to dates <= current_date
            df_slice = df[pd.to_datetime(df['timestamp']).dt.normalize() <= current_date].copy()
            data_slice[symbol] = df_slice

        # Step 4: Get current portfolio state (before trading)
        portfolio_state = broker.get_portfolio_state()

        # Step 5: Call strategy to generate target weights
        target_weights = strategy.generate_target_weights(
            dt=current_date,
            data=data_slice,
            portfolio_state=portfolio_state,
        )

        # Step 6: Pass target weights to broker for execution
        broker.set_target_weights(target_weights)

        # Step 7: Record updated portfolio state (after trading)
        updated_state = broker.get_portfolio_state()
        portfolio_history.append(updated_state)

    # Post-processing: build equity curve and compute metrics
    if not portfolio_history:
        raise ValueError(
            "No portfolio history recorded. Backtest may have failed silently."
        )

    # Build equity curve (time series of equity values)
    equity_curve = pd.Series(
        data=[state.equity for state in portfolio_history],
        index=[state.timestamp for state in portfolio_history],
        name='equity',
    )

    # Compute returns for metrics (daily simple returns)
    returns = equity_curve.pct_change().dropna()

    # Compute summary metrics using Phase 2 risk_metrics functions
    metrics = {}

    # Core performance metrics
    metrics['total_return'] = compute_total_return(equity_curve)
    metrics['cagr'] = compute_cagr(equity_curve, periods_per_year=252)

    # Risk-adjusted metrics (if we have enough data)
    if len(returns) > 1:
        metrics['sharpe_ratio'] = compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        metrics['sortino_ratio'] = compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

    # Drawdown metrics
    metrics['max_drawdown'] = compute_max_drawdown(equity_curve)

    # Calmar ratio (only if we have enough data and max_drawdown != 0)
    if len(equity_curve) > 1 and metrics['max_drawdown'] != 0:
        metrics['calmar_ratio'] = compute_calmar_ratio(equity_curve, periods_per_year=252)

    # Additional metrics
    metrics['final_equity'] = equity_curve.iloc[-1]
    metrics['num_trading_days'] = len(equity_curve)

    # Create and return result
    result = BacktestResult(
        equity_curve=equity_curve,
        portfolio_history=portfolio_history,
        metrics=metrics,
        params=params,
    )

    return result
