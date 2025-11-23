"""
Paper broker and portfolio simulation for backtesting.

**Conceptual**: This module implements a simplified broker that simulates portfolio
evolution during backtests. The paper broker accepts target weight allocations
from strategies, converts them into trades, applies simple cost models (slippage,
fees), and maintains cash and position balances. It's called "paper" because
no real money changes hands—it's purely simulation.

**Why a paper broker?**
  - Backtesting requires simulating execution without hitting real markets.
  - The broker encapsulates all execution logic (fills, costs, portfolio accounting),
    keeping strategies clean and focused on allocation decisions.
  - By centralizing execution, we can easily swap cost models or add complexity
    (partial fills, market impact) without touching strategy code.

**Financial assumptions** (Phase 4 baseline—document clearly):
  - Trading happens at the daily close price (the last price passed to update_prices).
  - All orders are market orders (instant fill at known price).
  - Slippage is modeled as basis points on traded notional (buy: pay more, sell: receive less).
  - Fees are flat per trade (could be extended to percentage of notional).
  - Shorting is allowed (negative positions) with no explicit margin modeling.
  - No borrowing costs or margin interest (these can be added later).
  - Fractional shares are allowed (simplifies rebalancing math).

**Teaching note**: In a production system, execution modeling would be far more
complex (limit orders, partial fills, market impact, real-time feeds, margin calls).
For backtesting, we trade off realism for simplicity and reproducibility. The key
is to document assumptions clearly so users know what behavior to expect.
"""

from dataclasses import dataclass, field
from typing import Dict
import pandas as pd


@dataclass
class Position:
    """
    Represents a position in a single instrument.

    **Conceptual**: A position tracks how many shares/units of an instrument
    we own (positive = long, negative = short) and the last known price for
    valuation purposes.

    **Teaching note**: In a real broker, positions would also track cost basis
    (for tax reporting), realized P&L, entry timestamps, etc. Here we keep it
    minimal for backtesting.

    Attributes:
        symbol: Instrument symbol (e.g., "QQQ", "TQQQ").
        quantity: Number of shares held. Positive = long, negative = short,
                 zero = no position. Fractional shares are allowed.
        last_price: Last known price for this instrument (used to value position).
                   Updated whenever prices are refreshed.
    """
    symbol: str
    quantity: float
    last_price: float

    @property
    def market_value(self) -> float:
        """
        Market value of this position at last_price.

        **Financial concept**: Market value = quantity * price.
        For a long position, this is the amount you'd receive if you sold.
        For a short position, this is the liability (amount you'd pay to cover).

        Returns:
            Market value in dollars (or base currency).
            Positive for long positions, negative for short positions.
        """
        return self.quantity * self.last_price


@dataclass
class PortfolioState:
    """
    Snapshot of portfolio state at a given timestamp.

    **Conceptual**: PortfolioState captures everything needed to understand
    the portfolio's condition: how much cash, which positions, total equity,
    and exposure metrics. The backtest engine records a PortfolioState at each
    time step to build the equity curve.

    **Why snapshot immutability?**
      - Each PortfolioState is a point-in-time record. Once created, it shouldn't
        change (makes time-series analysis easier).
      - The broker creates a new PortfolioState at each step rather than mutating
        a single object.

    **Teaching note**: In production, you'd also track unrealized P&L, realized P&L,
    margin usage, buying power, etc. Here we focus on the essentials for backtesting.

    Attributes:
        timestamp: The date/time this snapshot was taken (typically daily close).
        cash: Cash balance in base currency. Can be negative if shorting without
             enough cash (though Phase 4 doesn't model margin calls).
        positions: Dictionary mapping symbol -> quantity. Only includes non-zero positions.
                  Example: {"QQQ": 250.5, "TQQQ": -100.0} = long 250.5 QQQ, short 100 TQQQ.
        prices: Dictionary mapping symbol -> last_price. Needed to value positions.
        equity: Total portfolio value = cash + sum(position_market_values).
                This is the primary metric tracked over time (equity curve).
        gross_exposure: Sum of absolute position values. Measures total risk regardless
                       of direction. Example: $50k long + $30k short = $80k gross exposure.
        net_exposure: Sum of position values (long - short). Measures directional risk.
                     Example: $50k long - $30k short = $20k net long exposure.
    """
    timestamp: pd.Timestamp
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0


class PaperBroker:
    """
    Paper (simulated) broker for backtesting.

    **Conceptual**: The PaperBroker simulates a brokerage account. It:
      - Tracks cash and positions.
      - Accepts target weight allocations from strategies.
      - Computes required trades to achieve those targets.
      - Applies cost models (slippage, fees).
      - Updates portfolio state after each step.

    **Financial assumptions** (clearly documented for reproducibility):
      - All trades execute at the daily close price (no intraday dynamics).
      - Slippage is symmetric: pay slippage_bps on buys, lose slippage_bps on sells.
      - Fees are charged per trade (flat dollar amount per transaction).
      - Shorting is allowed without margin requirements or borrow costs (simplified).
      - Fractional shares allowed (avoids rounding complications).

    **Why separate update_prices and set_target_weights?**
      - Flexibility: The engine can update prices independently of rebalancing.
      - Clarity: Price updates are read-only (no trades), weight changes trigger trades.
      - Testing: Easier to test price movements vs allocation changes separately.

    **Teaching note**: In a real broker, you'd submit orders and get fill confirmations
    asynchronously. Here, we simplify to synchronous updates since backtests are deterministic.
    """

    def __init__(
        self,
        initial_cash: float,
        slippage_bps: float = 0.0,
        fee_per_trade: float = 0.0,
    ):
        """
        Initialize the paper broker.

        **Financial setup**:
          - Start with all cash, no positions.
          - Configure cost models (slippage, fees).

        Args:
            initial_cash: Starting cash balance (e.g., 100000 for $100k).
                         Must be positive.
            slippage_bps: Slippage in basis points (1 bp = 0.01%).
                         Applied to traded notional. Example: 5.0 = 0.05% slippage.
                         When buying, pay (price * (1 + slippage_bps/10000)).
                         When selling, receive (price * (1 - slippage_bps/10000)).
            fee_per_trade: Flat fee charged per trade (e.g., 0.0 for free, 1.0 for $1/trade).
                          Applied to each non-zero trade (buy or sell).

        Raises:
            ValueError: If initial_cash <= 0.
        """
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")

        # Portfolio state
        self._cash = initial_cash
        self._positions: Dict[str, float] = {}  # symbol -> quantity
        self._prices: Dict[str, float] = {}  # symbol -> last known price
        self._timestamp: pd.Timestamp | None = None

        # Cost model parameters
        self._slippage_bps = slippage_bps
        self._fee_per_trade = fee_per_trade

    def update_prices(self, prices: dict[str, float], dt: pd.Timestamp) -> None:
        """
        Update the broker's price map without triggering trades.

        **Conceptual**: This method is called by the backtest engine at each time
        step to provide the latest closing prices. The broker stores these prices
        to value positions and compute equity, but doesn't trade until
        set_target_weights is called.

        **Why separate price updates from trading?**
          - Reflect reality: Prices update continuously, but strategies decide
            when to trade.
          - Testability: Can test price movements without triggering rebalancing.
          - Flexibility: Engine can update prices multiple times (e.g., intraday)
            before strategy makes a decision.

        Args:
            prices: Dictionary mapping symbol -> price (e.g., {"QQQ": 403.5}).
                   Only symbols present in this dict will be updated.
                   Missing symbols retain their previous price.
            dt: Timestamp for this price update (typically end-of-day close).

        Returns:
            None (side effect: updates internal price map and timestamp).
        """
        # Store timestamp for this price snapshot
        self._timestamp = dt

        # Update prices (merge new prices into existing map)
        self._prices.update(prices)

    def set_target_weights(self, target_weights: dict[str, float]) -> None:
        """
        Set target portfolio weights and execute trades to achieve them.

        **Conceptual**: This method accepts target allocations from the strategy
        (e.g., {"QQQ": 0.6, "TQQQ": 0.4} = 60% in QQQ, 40% in TQQQ). It computes
        the required trades to move from the current portfolio to the target,
        applies slippage and fees, and updates cash and positions.

        **Financial logic** (step by step):
          1. Compute current equity = cash + sum(position_values).
          2. For each symbol in target_weights:
             - target_value = target_weight * equity
             - current_value = current_position * price (or 0 if no position)
             - trade_value = target_value - current_value
             - trade_quantity = trade_value / price
          3. For each trade:
             - Apply slippage: adjust effective price based on direction.
             - Compute cash impact: -(trade_quantity * effective_price).
             - Charge fees: -fee_per_trade if trade_quantity != 0.
          4. Update positions and cash.

        **Why target weights instead of explicit orders?**
          - Simplicity: Strategies think in terms of allocation percentages,
            not dollar amounts or share counts.
          - Rebalancing is automatic: Broker computes minimal trades to achieve targets.
          - Extensible: Later can support explicit orders alongside weights.

        **Teaching note**: Converting weights to trades is non-trivial because equity
        changes as you trade (spend cash to buy → equity decreases). The correct
        approach is to compute all trades first based on pre-trade equity, then
        apply them atomically. Iterative rebalancing (trade one symbol, recalculate
        equity, trade next) can lead to errors.

        Args:
            target_weights: Dictionary mapping symbol -> target weight.
                           Weights are fractions of total equity (0.0 to 1.0 for long-only,
                           can be negative for shorts, can sum to > 1.0 for leverage).
                           Example: {"QQQ": 0.5, "TQQQ": 0.3} = 50% QQQ, 30% TQQQ, 20% cash.
                           Symbols not in this dict will be liquidated (set to 0 weight).

        Returns:
            None (side effect: updates cash, positions).

        Raises:
            ValueError: If prices haven't been updated yet (no prices available).
            ValueError: If target symbol has no price (can't value position).
        """
        if not self._prices:
            raise ValueError("Cannot set target weights before updating prices.")

        # Step 1: Compute current equity (cash + market value of all positions)
        current_equity = self._compute_equity()

        # Step 2: Determine which symbols to trade
        # Include: symbols in target_weights + symbols we currently hold
        # (need to liquidate positions not in target_weights)
        all_symbols = set(target_weights.keys()) | set(self._positions.keys())

        # Step 3: Compute required trades for each symbol
        trades: Dict[str, float] = {}  # symbol -> quantity to trade (+ = buy, - = sell)

        for symbol in all_symbols:
            # Get target weight (default 0.0 if not specified → liquidate)
            target_weight = target_weights.get(symbol, 0.0)

            # Get current position quantity (default 0.0 if no position)
            current_quantity = self._positions.get(symbol, 0.0)

            # Get price for this symbol
            if symbol not in self._prices:
                raise ValueError(
                    f"No price available for symbol '{symbol}'. "
                    f"Call update_prices with this symbol first."
                )
            price = self._prices[symbol]

            # Compute target dollar value for this symbol
            target_value = target_weight * current_equity

            # Compute current dollar value
            current_value = current_quantity * price

            # Compute required trade in dollar terms
            trade_value = target_value - current_value

            # Convert to quantity (shares)
            # Handle zero price edge case (should not happen in practice)
            if price == 0:
                raise ValueError(f"Price for symbol '{symbol}' is zero. Cannot trade.")

            trade_quantity = trade_value / price

            # Only record non-trivial trades (avoid noise from floating-point errors)
            if abs(trade_quantity) > 1e-9:  # Threshold for negligible trades
                trades[symbol] = trade_quantity

        # Step 4: Apply trades with slippage and fees
        for symbol, trade_quantity in trades.items():
            self._execute_trade(symbol, trade_quantity)

    def get_portfolio_state(self) -> PortfolioState:
        """
        Get a snapshot of the current portfolio state.

        **Conceptual**: This method packages the broker's internal state into
        a PortfolioState dataclass that can be:
          - Passed to strategies (for path-dependent decisions).
          - Recorded by the backtest engine (for equity curve tracking).
          - Logged for diagnostics.

        **Financial calculations**:
          - Equity = cash + sum(position_values)
          - Gross exposure = sum(abs(position_values))
          - Net exposure = sum(position_values) [long - short]

        **Teaching note**: In production, you'd also compute metrics like:
          - Leverage ratio (gross_exposure / equity)
          - Margin usage
          - Buying power
        Here we keep it simple for backtesting.

        Returns:
            PortfolioState snapshot with current timestamp, cash, positions, equity,
            and exposure metrics.
        """
        # Compute position values and exposure metrics
        total_position_value = 0.0
        gross_exposure = 0.0

        for symbol, quantity in self._positions.items():
            price = self._prices.get(symbol, 0.0)  # Default 0 if price unknown (shouldn't happen)
            position_value = quantity * price
            total_position_value += position_value
            gross_exposure += abs(position_value)

        # Total equity
        equity = self._cash + total_position_value

        # Net exposure (long - short)
        net_exposure = total_position_value

        # Create snapshot
        return PortfolioState(
            timestamp=self._timestamp or pd.Timestamp.now(),  # Fallback if no timestamp set
            cash=self._cash,
            positions=self._positions.copy(),  # Copy to prevent external mutation
            prices=self._prices.copy(),
            equity=equity,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
        )

    # ========================================================================
    # Internal helper methods
    # ========================================================================

    def _compute_equity(self) -> float:
        """
        Compute total portfolio equity (cash + market value of positions).

        Returns:
            Total equity in dollars.
        """
        total_position_value = sum(
            self._positions.get(symbol, 0.0) * self._prices.get(symbol, 0.0)
            for symbol in self._positions
        )
        return self._cash + total_position_value

    def _execute_trade(self, symbol: str, quantity: float) -> None:
        """
        Execute a single trade with slippage and fees.

        **Financial logic**:
          - If buying (quantity > 0): pay (price * (1 + slippage)) per share.
          - If selling (quantity < 0): receive (price * (1 - slippage)) per share.
          - Charge flat fee per trade.

        **Why asymmetric slippage?**
          - Models market impact and bid-ask spread.
          - Buying: you pay the ask (higher), selling: you get the bid (lower).
          - In basis points: 5 bps = 0.05% = you lose 0.05% on each trade.

        Args:
            symbol: Symbol to trade.
            quantity: Quantity to trade (positive = buy, negative = sell).
                     Fractional shares allowed.

        Side effects:
            - Updates self._positions[symbol]
            - Updates self._cash
        """
        # Get base price
        price = self._prices[symbol]

        # Compute effective price with slippage
        if quantity > 0:
            # Buying: pay more (price * (1 + slippage_bps/10000))
            effective_price = price * (1 + self._slippage_bps / 10000)
        else:
            # Selling: receive less (price * (1 - slippage_bps/10000))
            effective_price = price * (1 - self._slippage_bps / 10000)

        # Compute cash impact (negative = spend, positive = receive)
        # When buying: cash_impact = -(quantity * effective_price) < 0
        # When selling: cash_impact = -(-|quantity| * effective_price) > 0
        cash_impact = -(quantity * effective_price)

        # Apply flat fee
        cash_impact -= self._fee_per_trade

        # Update cash
        self._cash += cash_impact

        # Update position
        current_quantity = self._positions.get(symbol, 0.0)
        new_quantity = current_quantity + quantity

        # Store new position (remove if zero to keep dict clean)
        if abs(new_quantity) < 1e-9:  # Threshold for zero position
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = new_quantity
