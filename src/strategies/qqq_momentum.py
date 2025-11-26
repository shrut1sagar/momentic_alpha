"""
QQQ momentum strategy implementation (Strategy 1).

**Conceptual**: This module implements a complete momentum strategy that:
  1. Reads QQQ prices as the signal source.
  2. Computes technical features (MAs, spreads, velocity, acceleration).
  3. Classifies the current market regime (strong uptrend, weakening uptrend, neutral, downtrend).
  4. Maps regimes to target allocations in TQQQ (3x long), SQQQ (3x inverse), or cash.
  5. Integrates with the Phase 4 backtest engine via the Strategy protocol.

**Strategy 1 rationale**:
  - QQQ (Nasdaq-100 ETF) is liquid, well-covered, and has clear trends.
  - TQQQ/SQQQ provide 3x leverage, amplifying returns (and drawdowns).
  - Momentum works when trends persist; fails during whipsaws.
  - This implementation aims to avoid whipsaws by:
    - Using multiple timeframes (20/50/100/250-day MAs).
    - Requiring positive acceleration for "strong" uptrend (not just positive velocity).
    - Staying in cash during neutral/choppy regimes.

**Teaching note**: This is a teaching implementation, not production-ready.
Real strategies would add:
  - Stop-losses (exit TQQQ if drawdown exceeds threshold).
  - Volatility targeting (reduce exposure when volatility spikes).
  - Macro overlays (avoid leverage during recessions, Fed tightening).
  - Transaction cost awareness (avoid overtrading).
  - Walk-forward validation (avoid overfitting to historical data).

This code prioritizes clarity, testability, and educational value.
"""

import pandas as pd
from typing import Dict

from src.strategies.base import Strategy
from src.execution.paper_broker import PortfolioState
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


class QqqMomentumStrategy(Strategy):
    """
    QQQ momentum strategy using trend, velocity, and acceleration signals.

    **Conceptual**: This strategy implements a rules-based momentum approach:
      - Signal source: QQQ price history.
      - Execution: TQQQ (3x long), SQQQ (3x inverse), or cash.
      - Logic: Classify daily regime based on MA spreads, velocity, acceleration.
      - Allocation: Strong uptrend → max TQQQ, downtrend → SQQQ, neutral → cash.

    **How it works**:
      1. At initialization: Precompute QQQ features from historical prices.
      2. At each backtest step (daily):
         - Look up current date in feature DataFrame.
         - Classify regime for that date.
         - Map regime to target weights.
         - Return weights to broker for execution.

    **Teaching note**: The key design decision is to precompute features rather than
    recompute them at each step. This is much faster for backtesting (features only
    computed once) and matches real-world usage (features are computed nightly, then
    used for next-day trading).

    **Why separate signal source (QQQ) from execution (TQQQ/SQQQ)?**
      - QQQ has longer history and more stable data.
      - TQQQ/SQQQ are leveraged, so they have higher slippage and fees.
      - Separating signal from execution lets us:
        - Backtest on longer QQQ history.
        - Easily swap execution instruments (e.g., use UPRO/SPXU for S&P500).
        - Adjust leverage ratios without changing signal logic.

    Attributes:
        feature_params: Parameters for building QQQ features (MA windows, etc.).
        regime_params: Parameters for regime classification (thresholds).
        allocation_params: Parameters for mapping regimes to target weights.
        symbols: Instrument symbols (QQQ, TQQQ, SQQQ, UVXY).
        qqq_features: Precomputed QQQ feature DataFrame (indexed by date).
        regimes: Precomputed regime labels (Series indexed by date).
    """

    def __init__(
        self,
        feature_params: QqqMomentumFeatureParams,
        regime_params: QqqMomentumRegimeParams,
        allocation_params: QqqMomentumAllocationParams,
        symbols: QqqMomentumSymbols,
        qqq_features: pd.DataFrame,
    ):
        """
        Initialize the QQQ momentum strategy with precomputed features.

        **Conceptual**: This constructor takes precomputed QQQ features rather than
        raw prices. This design choice:
          - Separates feature engineering from strategy logic (testability).
          - Allows reusing the same features for multiple strategy variants.
          - Matches production usage (features computed once, reused many times).

        **Teaching note**: An alternative design would compute features internally:
          ```python
          def __init__(self, qqq_prices, feature_params, ...):
              self.qqq_features = build_qqq_momentum_features(qqq_prices, feature_params)
          ```
          This is more convenient but couples feature engineering to strategy init.
          We prefer the explicit approach for clarity.

        Args:
            feature_params: QqqMomentumFeatureParams (stored for reference/logging).
            regime_params: QqqMomentumRegimeParams (thresholds for regime classification).
            allocation_params: QqqMomentumAllocationParams (weights per regime).
            symbols: QqqMomentumSymbols (which instruments to trade).
            qqq_features: Precomputed QQQ feature DataFrame with columns:
                         - timestamp, closing_price, moving_average_*, ma_spread_*,
                           velocity_*, acceleration_*.
                         Should be indexed by date or have a 'timestamp' column.

        Raises:
            ValueError: If qqq_features is missing required columns.
        """
        # Store parameters for reference (useful for logging/debugging)
        self.feature_params = feature_params
        self.regime_params = regime_params
        self.allocation_params = allocation_params
        self.symbols = symbols

        # Store precomputed features
        # Ensure features are indexed by NORMALIZED (timezone-naive, date-only) timestamps
        # This must match the format the backtest engine uses when calling generate_target_weights
        if "timestamp" in qqq_features.columns and not isinstance(qqq_features.index, pd.DatetimeIndex):
            # Set timestamp as index for fast date lookups
            qqq_features = qqq_features.set_index("timestamp")

        # Normalize index to timezone-naive dates (drop time component and timezone)
        # The backtest engine normalizes dates via dt.normalize() which creates timezone-naive dates
        # We must match that format for lookups to work
        if isinstance(qqq_features.index, pd.DatetimeIndex):
            # Handle timezone conversion: if tz-aware, convert to UTC then remove tz; if naive, keep as-is
            if qqq_features.index.tz is not None:
                # Timezone-aware: convert to UTC, then strip timezone, then normalize to midnight
                normalized_index = qqq_features.index.tz_convert('UTC').tz_localize(None).normalize()
            else:
                # Timezone-naive: just normalize to midnight
                normalized_index = qqq_features.index.normalize()
            qqq_features.index = normalized_index

        self.qqq_features = qqq_features

        # Precompute regimes for all dates
        # This saves computation during backtesting (classify once, use many times)
        # Regimes will have the same normalized index as qqq_features
        self.regimes = classify_momentum_regime(self.qqq_features, regime_params)

        # Validate that features have required columns
        required_cols = ["closing_price", "ma_spread_50_250", "velocity_20d", "acceleration_20d"]
        missing = [col for col in required_cols if col not in self.qqq_features.columns]
        if missing:
            raise ValueError(
                f"QQQ features missing required columns: {missing}. "
                "Use build_qqq_momentum_features to create valid feature DataFrame."
            )

    def generate_target_weights(
        self,
        dt: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        portfolio_state: PortfolioState | None = None,
    ) -> Dict[str, float]:
        """
        Generate target portfolio weights for the current date.

        **Conceptual**: This is the core Strategy method called by the backtest engine
        at each time step. It:
          1. Looks up the regime for the current date (from precomputed regimes).
          2. Maps the regime to target weights (via regime_to_target_weights).
          3. Returns the weights dict to the broker for execution.

        **Mathematically**: No math here—this is pure lookup and mapping:
          - regime = self.regimes[dt]
          - weights = regime_to_target_weights(regime, symbols, alloc_params)

        **Functionally**:
          - Input dt: Current timestamp from backtest engine.
          - Input data: Dict of symbol -> DataFrame (unused; we use precomputed features).
          - Input portfolio_state: Current portfolio state (unused for stateless strategy).
          - Output: Dict of symbol -> target weight (e.g., {"TQQQ": 1.0}).

        **Edge cases**:
          - If dt is not in qqq_features (warm-up period or missing data):
            Return empty dict {} = all cash (safe fallback).
          - If regime is NaN (shouldn't happen after classification, but defensive):
            Return empty dict {} = all cash.

        **Teaching note**: This strategy is stateless—it only looks at current features,
        not past portfolio state. A stateful strategy might use portfolio_state to:
          - Implement stop-losses (exit if current position has lost >10%).
          - Avoid overtrading (don't rebalance if target is within 5% of current).
          - Track unrealized P&L for tax-loss harvesting.

        Args:
            dt: Current timestamp (pd.Timestamp) from backtest engine.
            data: Dict mapping symbol -> DataFrame with historical prices up to dt.
                 Provided by engine but unused here (we use precomputed features).
            portfolio_state: Current portfolio state (cash, positions, equity).
                            Unused for this stateless strategy.

        Returns:
            Dict mapping symbol (str) -> target weight (float).
            Example: {"TQQQ": 1.0} = allocate 100% to TQQQ.
            Empty dict {} = allocate 100% to cash.

        Raises:
            None (returns safe default on errors).
        """
        # Normalize dt to match the format of our feature/regime index
        # The backtest engine passes normalized timestamps (timezone-naive, time=00:00:00)
        # Our features are indexed by the same format (set up in __init__)
        # However, we still normalize here for defensive consistency
        dt_ts = pd.Timestamp(dt)
        if dt_ts.tz is not None:
            # Timezone-aware: convert to UTC, strip timezone, normalize
            dt_normalized = dt_ts.tz_convert('UTC').tz_localize(None).normalize()
        else:
            # Timezone-naive: just normalize
            dt_normalized = dt_ts.normalize()

        # Look up regime for current date
        # If dt is not in regimes (warm-up period or missing data), fall back to NEUTRAL
        if dt_normalized not in self.regimes.index:
            # Warm-up period: not enough data to compute features
            # Safe default: return empty dict = all cash
            # Note: This should be rare after warm-up; if it's common, check timestamp alignment
            return {}

        regime = self.regimes.loc[dt_normalized]

        # Handle NaN regime (shouldn't happen, but defensive programming)
        if pd.isna(regime):
            # Missing or invalid regime: safe default = all cash
            return {}

        # Map regime to target weights using allocation params
        weights = regime_to_target_weights(
            regime,
            self.symbols,
            self.allocation_params,
        )

        return weights

    def get_regime_for_date(self, dt: pd.Timestamp) -> MomentumRegime | None:
        """
        Get the classified regime for a specific date (helper for logging/debugging).

        **Conceptual**: This is a convenience method for inspecting regime decisions
        during backtests or for generating "reasoning traces" (which regime was active
        on each date).

        **Timestamp handling**: Normalizes the input timestamp to match the regime index format
        (timezone-naive, date-only). This ensures consistent lookups regardless of input format.

        Args:
            dt: Date to look up (can be timezone-aware or have time component).

        Returns:
            MomentumRegime enum value, or None if date not found.
        """
        # Normalize to match regime index format
        dt_ts = pd.Timestamp(dt)
        if dt_ts.tz is not None:
            dt_normalized = dt_ts.tz_convert('UTC').tz_localize(None).normalize()
        else:
            dt_normalized = dt_ts.normalize()

        if dt_normalized not in self.regimes.index:
            return None
        regime = self.regimes.loc[dt_normalized]
        return regime if not pd.isna(regime) else None

    def get_feature_snapshot(self, dt: pd.Timestamp) -> pd.Series | None:
        """
        Get feature values for a specific date (helper for logging/debugging).

        **Conceptual**: This method returns the feature values that led to a regime
        classification. Useful for:
          - Debugging: "Why did the strategy go to cash on 2023-05-15?"
          - Reporting: "QQQ velocity was +0.02 and acceleration was -0.01 on this date."
          - Visualization: Plot features alongside equity curve to understand performance.

        **Timestamp handling**: Normalizes the input timestamp to match the feature index format.

        Args:
            dt: Date to look up (can be timezone-aware or have time component).

        Returns:
            pd.Series with feature values (closing_price, MAs, spreads, velocity, accel),
            or None if date not found.
        """
        # Normalize to match feature index format
        dt_ts = pd.Timestamp(dt)
        if dt_ts.tz is not None:
            dt_normalized = dt_ts.tz_convert('UTC').tz_localize(None).normalize()
        else:
            dt_normalized = dt_ts.normalize()

        if dt_normalized not in self.qqq_features.index:
            return None
        return self.qqq_features.loc[dt_normalized]
