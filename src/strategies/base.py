"""
Strategy interface and base implementations for backtesting.

**Conceptual**: This module defines the contract between strategies and the
backtest engine. A Strategy is responsible for making allocation decisions
(target weights per instrument) based on current market data and portfolio state.
The engine iterates through time, calls the strategy at each step, and passes
the resulting targets to the broker for execution.

**Why a strategy interface?**
  - Decoupling: The backtest engine doesn't need to know strategy-specific logic.
  - Extensibility: New strategies plug in without modifying the engine.
  - Testability: We can test engines with trivial strategies and test strategies
    independently of engines.
  - Composability: Future work could combine multiple strategies (ensemble, rotation).

**Teaching note**: In a production quant system, the strategy interface is the
most critical design decision. It defines what information strategies receive,
what decisions they can make, and how they communicate with the execution layer.
A clean interface prevents strategies from accidentally looking into the future
(time travel bugs) and makes backtests reproducible.
"""

from typing import Protocol
import pandas as pd

# Forward reference to avoid circular import with paper_broker
# PortfolioState will be defined in src/execution/paper_broker.py
# We use string annotation here and import TYPE_CHECKING for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.execution.paper_broker import PortfolioState


class Strategy(Protocol):
    """
    Strategy interface for backtesting.

    **Conceptual**: A Strategy is a callable object that generates target
    portfolio allocations (weights per instrument) based on current market
    conditions and optional portfolio state. Strategies are stateless from
    the engine's perspective—they receive all necessary context as arguments
    and return a decision.

    **Why target weights instead of orders?**
      - Simpler interface: weight semantics are intuitive (0.5 = 50% of equity).
      - Rebalancing is automatic: the broker computes required trades from weights.
      - Execution details (slippage, fees, fills) are handled by the broker,
        not the strategy.
      - Later phases can extend to explicit orders if needed for granular control.

    **Teaching note**: This is a Protocol (structural typing), not an ABC.
    Any object with a `generate_target_weights` method matching this signature
    can be used as a Strategy. This is more flexible than inheritance and aligns
    with modern Python typing practices.
    """

    def generate_target_weights(
        self,
        dt: pd.Timestamp,
        data: dict[str, pd.DataFrame],
        portfolio_state: "PortfolioState | None" = None,
    ) -> dict[str, float]:
        """
        Generate target portfolio weights for a given timestamp.

        **Conceptual**: This method is called once per backtest step (e.g., daily
        close). The strategy receives the current date, historical price/feature
        data up to that date, and current portfolio state. It returns target
        weights expressing how much of total equity should be allocated to each
        instrument.

        **Target weight semantics**:
          - weights are fractions of total equity (0.0 = no allocation, 1.0 = 100%)
          - Positive weights = long exposure
          - Negative weights = short exposure (if shorting is enabled)
          - Sum of abs(weights) can be <= 1.0 (cash remainder) or > 1.0 (leverage, if allowed)
          - Example: {"QQQ": 0.5, "TQQQ": 0.3} = 50% in QQQ, 30% in TQQQ, 20% cash

        **Why pass full DataFrames instead of just current row?**
          - Strategies often need historical context (e.g., moving averages require
            lookback windows).
          - Passing full history up to `dt` gives strategies maximum flexibility.
          - The engine ensures no future data leaks (data is filtered to <= dt).

        **Why pass portfolio_state?**
          - Some strategies are path-dependent (e.g., "don't trade if position
            is already X% of equity").
          - Tax-aware or transaction-cost-aware strategies need current holdings
            to minimize turnover.
          - For stateless strategies (pure price signals), portfolio_state can be ignored.

        **Teaching note**: This signature balances simplicity (target weights are
        easy to reason about) with flexibility (strategies can access full history
        and portfolio state if needed). More complex order types (limit orders,
        iceberg orders) can be added later without breaking this interface—just
        create a new method or extend the return type.

        Args:
            dt: Current timestamp (e.g., end-of-day close for daily bars).
                This is the "as-of" time—data should not include anything after dt.
            data: Dictionary mapping symbol -> DataFrame.
                  Each DataFrame has historical OHLCV and/or features up to and
                  including dt (timestamps <= dt).
                  Typically has columns like: timestamp, closing_price, volume,
                  plus optional features (moving_average_50, velocity, etc.).
                  Rows are sorted in descending order by timestamp (newest first).
            portfolio_state: Current portfolio state (cash, positions, equity).
                            May be None on the first step or for stateless strategies.

        Returns:
            Dictionary mapping symbol -> target weight.
            Example: {"QQQ": 0.6, "TQQQ": 0.4} means 60% in QQQ, 40% in TQQQ, 0% cash.
            Empty dict {} means no positions (100% cash).
            Symbols not in the return dict default to 0.0 weight (liquidate if held).

        Raises:
            Strategies may raise exceptions for invalid data, but should generally
            handle edge cases gracefully (e.g., missing features → fallback to cash).
        """
        ...


# ============================================================================
# Simple strategy implementations for testing and demonstration
# ============================================================================

class AlwaysCashStrategy:
    """
    Trivial strategy that always holds 100% cash (no positions).

    **Conceptual**: This is the simplest possible strategy—never trade, never
    take risk. Useful for testing that the backtest engine and broker correctly
    handle zero-exposure scenarios.

    **Expected behavior in backtest**:
      - Equity remains constant at initial capital (minus any fees if enabled).
      - No positions are ever opened.
      - Useful as a baseline to verify engine plumbing.

    **Teaching note**: In production, a "cash strategy" might be a defensive
    fallback during high uncertainty or market stress. Here it's just a test fixture.
    """

    def generate_target_weights(
        self,
        dt: pd.Timestamp,
        data: dict[str, pd.DataFrame],
        portfolio_state: "PortfolioState | None" = None,
    ) -> dict[str, float]:
        """
        Always return empty target weights (100% cash).

        Args:
            dt: Current timestamp (ignored).
            data: Market data (ignored).
            portfolio_state: Portfolio state (ignored).

        Returns:
            Empty dict (no allocations, all cash).
        """
        return {}


class AlwaysLongStrategy:
    """
    Trivial strategy that allocates 100% to a single symbol.

    **Conceptual**: This strategy is fully invested in one instrument at all
    times. Useful for testing that the backtest engine correctly handles:
      - Full allocation (no cash remainder).
      - Position updates as prices change.
      - Equity growth/decline tracking the underlying instrument.

    **Expected behavior in backtest**:
      - Equity should track the instrument's price movements (scaled by initial capital).
      - If the instrument goes up 10%, equity goes up ~10% (minus fees/slippage).
      - No rebalancing unless initial capital changes.

    **Teaching note**: This is analogous to a "buy and hold" strategy on a single ETF.
    It's deterministic and easy to verify manually, making it ideal for testing.
    """

    def __init__(self, symbol: str):
        """
        Initialize the always-long strategy.

        Args:
            symbol: The instrument symbol to allocate 100% to (e.g., "QQQ").
        """
        self.symbol = symbol

    def generate_target_weights(
        self,
        dt: pd.Timestamp,
        data: dict[str, pd.DataFrame],
        portfolio_state: "PortfolioState | None" = None,
    ) -> dict[str, float]:
        """
        Always return 100% weight on the configured symbol.

        Args:
            dt: Current timestamp (ignored).
            data: Market data (ignored—we always allocate regardless of price).
            portfolio_state: Portfolio state (ignored).

        Returns:
            Dictionary with single entry: {self.symbol: 1.0}.
        """
        return {self.symbol: 1.0}


class BalancedStrategy:
    """
    Simple strategy that allocates evenly across multiple symbols.

    **Conceptual**: This strategy maintains equal weights across a fixed set
    of instruments. It's a diversification baseline—no market timing, no selection,
    just rebalance to equal weights at every step.

    **Expected behavior in backtest**:
      - If given ["QQQ", "SPY"], allocates 50% to each.
      - Automatically rebalances at every step to maintain equal weights.
      - Equity tracks a simple average of the instruments.

    **Teaching note**: Equal-weight portfolios are a common benchmark in finance.
    They're easy to understand and implement, and they serve as a reference for
    more sophisticated strategies. In production, you'd add rebalancing thresholds
    to reduce turnover costs.
    """

    def __init__(self, symbols: list[str]):
        """
        Initialize the balanced strategy.

        Args:
            symbols: List of symbols to allocate across evenly (e.g., ["QQQ", "SPY"]).
                    Must have at least one symbol.
        """
        if not symbols:
            raise ValueError("BalancedStrategy requires at least one symbol.")
        self.symbols = symbols
        self.weight_per_symbol = 1.0 / len(symbols)

    def generate_target_weights(
        self,
        dt: pd.Timestamp,
        data: dict[str, pd.DataFrame],
        portfolio_state: "PortfolioState | None" = None,
    ) -> dict[str, float]:
        """
        Return equal weights across all configured symbols.

        Args:
            dt: Current timestamp (ignored).
            data: Market data (ignored).
            portfolio_state: Portfolio state (ignored).

        Returns:
            Dictionary mapping each symbol to 1/N weight.
            Example: ["QQQ", "SPY"] -> {"QQQ": 0.5, "SPY": 0.5}.
        """
        return {symbol: self.weight_per_symbol for symbol in self.symbols}
