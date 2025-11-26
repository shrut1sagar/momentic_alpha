"""
QQQ momentum strategy feature engineering and regime classification.

**Conceptual**: This module implements Strategy 1's feature pipeline:
  1. Build technical features from QQQ prices (MAs, spreads, velocity, acceleration).
  2. Classify trend regimes based on feature thresholds.
  3. Map regimes to target weights for TQQQ/SQQQ execution.

**Strategy 1 overview**:
  - Signal source: QQQ (unleveraged Nasdaq-100 ETF) price history.
  - Execution instruments: TQQQ (3x long QQQ), SQQQ (3x inverse QQQ), optionally UVXY (2x VIX).
  - Core idea: Use trend strength (MA spreads), momentum (velocity), and acceleration
    to identify strong uptrends (lean into TQQQ leverage) vs downtrends (use SQQQ)
    vs choppy markets (stay in cash to avoid whipsaw).

**Teaching note**: Momentum strategies profit when trends persist but suffer during
regime changes and sideways markets. This implementation is educational—real production
would add:
  - Macro overlays (Fed policy, recession signals).
  - Dynamic position sizing (volatility targeting).
  - Stop-losses and risk limits.
  - Transaction cost awareness (avoid overtrading in choppy regimes).

This code prioritizes clarity and teaching value over optimization.
"""

from dataclasses import dataclass
from enum import Enum
import pandas as pd

from src.analytics.features import (
    add_moving_averages,
    add_ma_spreads,
    add_velocity_and_acceleration,
)


@dataclass
class QqqMomentumFeatureParams:
    """
    Parameters for building QQQ momentum features.

    **Conceptual**: These params control which features are computed and how.
    Tuning these windows is part of strategy development (parameter sweeps).

    Attributes:
        ma_short_window: Short-term moving average window (e.g., 20 days).
        ma_medium_window: Medium-term moving average window (e.g., 50 days).
        ma_long_window: Long-term moving average window (e.g., 100 days).
        ma_ultra_long_window: Ultra-long moving average window (e.g., 250 days ~ 1 year).
        velocity_window: Window for velocity (trend slope) computation (e.g., 20 days).
        acceleration_window: Window for acceleration (trend change) computation (e.g., 20 days).
        normalize_features: If True, z-score normalize velocity and acceleration for comparability.
    """
    ma_short_window: int = 20
    ma_medium_window: int = 50
    ma_long_window: int = 100
    ma_ultra_long_window: int = 250
    velocity_window: int = 20
    acceleration_window: int = 20
    normalize_features: bool = False


def build_qqq_momentum_features(
    qqq_prices: pd.DataFrame,
    params: QqqMomentumFeatureParams,
) -> pd.DataFrame:
    """
    Enrich QQQ price data with momentum features for Strategy 1.

    **Conceptual**: This function transforms raw QQQ prices into a feature-rich
    DataFrame ready for regime classification. Features capture:
      - Trend direction and strength (MAs and spreads).
      - Momentum (velocity = slope of price).
      - Trend quality (acceleration = change in slope).

    **Mathematically**: Uses Phase 2 math helpers wrapped by Phase 5 feature helpers:
      - MAs: Simple moving averages over multiple windows.
      - Spreads: (fast_MA - slow_MA) / slow_MA (percentage divergence).
      - Velocity: Linear regression slope on log(price) over rolling window.
      - Acceleration: First difference of velocity.

    **Functionally**: Adds columns to qqq_prices:
      - moving_average_20, moving_average_50, moving_average_100, moving_average_250
      - ma_spread_50_100, ma_spread_50_250 (normalized percentage spreads)
      - velocity_20d, acceleration_20d
      - Optionally: normalized versions of velocity/acceleration

    Args:
        qqq_prices: DataFrame with at least "timestamp" and "closing_price" columns.
                   Should be sorted in descending order (newest first) per Phase 3 schema.
        params: QqqMomentumFeatureParams controlling window sizes and normalization.

    Returns:
        Modified DataFrame with all original columns plus new feature columns.
        NaNs appear at the start of the series (warm-up period) where windows don't
        have enough data.

    Raises:
        KeyError: If required columns ("closing_price") are missing.

    Usage example:
        >>> from src.data.loaders import load_qqq_history
        >>> qqq = load_qqq_history()
        >>> params = QqqMomentumFeatureParams()
        >>> qqq_features = build_qqq_momentum_features(qqq, params)
        >>> print(qqq_features[["closing_price", "moving_average_50", "velocity_20d"]])
    """
    # Validate input
    if "closing_price" not in qqq_prices.columns:
        raise KeyError(
            "QQQ prices DataFrame must have 'closing_price' column. "
            "Use Phase 3 loaders (load_qqq_history) to get validated data."
        )

    # Make a copy to avoid modifying input (defensive)
    df = qqq_prices.copy()

    # CRITICAL FIX: Sort to ascending order (oldest first) for correct rolling calculations
    # Phase 3 schema specifies descending order (newest first), but rolling window
    # calculations need ascending order to avoid look-ahead bias.
    # We'll sort here, compute features, then sort back to descending.
    original_order = df.index.copy()
    is_descending = df['timestamp'].iloc[0] > df['timestamp'].iloc[-1]

    if is_descending:
        # Sort to ascending (oldest first) by timestamp
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    # Step 1: Add moving averages
    # We'll use string labels matching window sizes for column naming
    ma_windows = {
        str(params.ma_short_window): params.ma_short_window,
        str(params.ma_medium_window): params.ma_medium_window,
        str(params.ma_long_window): params.ma_long_window,
        str(params.ma_ultra_long_window): params.ma_ultra_long_window,
    }
    df = add_moving_averages(df, ma_windows, price_column="closing_price", use_ema=False)

    # Step 2: Add MA spreads
    # Compare medium-term to long-term, and medium-term to ultra-long-term
    # These capture "is the 50-day trend above the 100-day trend?" (bullish) or below (bearish)
    spread_pairs = [
        (str(params.ma_medium_window), str(params.ma_long_window)),  # 50 vs 100
        (str(params.ma_medium_window), str(params.ma_ultra_long_window)),  # 50 vs 250
    ]
    df = add_ma_spreads(df, spread_pairs, normalize=True)

    # Step 3: Add velocity and acceleration
    # Velocity = trend slope over last 20 days (or configured window)
    # Acceleration = change in velocity (is trend accelerating or decelerating?)
    df = add_velocity_and_acceleration(
        df,
        velocity_window=params.velocity_window,
        acceleration_window=params.acceleration_window,
        price_column="closing_price",
        use_log=True,  # Use log prices for compounding-friendly velocity
    )

    # Step 4: Optional normalization
    # If enabled, z-score normalize velocity and acceleration for comparability
    # This makes thresholds more stable across different market regimes
    if params.normalize_features:
        from src.analytics.features import normalize_features
        feature_cols = [
            f"velocity_{params.velocity_window}d",
            f"acceleration_{params.acceleration_window}d",
        ]
        df = normalize_features(df, feature_cols, method="zscore")

    # CRITICAL FIX: Sort back to descending order (newest first) to match Phase 3 schema
    if is_descending:
        df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

    return df


class MomentumRegime(Enum):
    """
    Trend regime labels for QQQ momentum strategy.

    **Conceptual**: Each regime represents a qualitative market state:
      - STRONG_UPTREND: Prices rising, positive momentum, accelerating (lean into leverage).
      - WEAKENING_UPTREND: Prices rising but momentum slowing (reduce exposure, prepare for reversal).
      - NEUTRAL: Mixed signals or low momentum (stay in cash, avoid whipsaw).
      - DOWNTREND: Prices falling, negative momentum (use inverse leverage SQQQ or cash).

    **Teaching note**: Regime classification is the heart of discretionary trading
    translated into rules. These labels are subjective (no "true" regime), but
    consistent rules allow backtesting and parameter tuning.
    """
    STRONG_UPTREND = "strong_uptrend"
    WEAKENING_UPTREND = "weakening_uptrend"
    NEUTRAL = "neutral"
    DOWNTREND = "downtrend"


@dataclass
class QqqMomentumRegimeParams:
    """
    Parameters for classifying momentum regimes.

    **Conceptual**: These thresholds define the boundaries between regimes.
    Example:
      - min_spread_for_trend > 0.0 → fast MA must be above slow MA to call it uptrend.
      - min_velocity_for_trend > 0.0 → velocity must be positive for uptrend.
      - min_acceleration_for_strong_trend > 0.0 → acceleration must be positive for "strong" uptrend.

    **Teaching note**: These are tunable hyperparameters. In practice, you'd run
    parameter sweeps to find thresholds that maximize Sharpe or minimize drawdown.
    Start with intuitive values (e.g., spreads > 2%, velocity > 0) and iterate.

    Attributes:
        min_spread_for_trend: Minimum MA spread (50 vs 250) to call it a trend (e.g., 0.02 = 2%).
        min_velocity_for_trend: Minimum velocity to call it trending (e.g., 0.0 = positive slope).
        min_acceleration_for_strong_trend: Minimum acceleration for "strong" vs "weakening" (e.g., 0.0).
        max_spread_for_neutral: Spreads below this are considered neutral/choppy (e.g., 0.01 = 1%).
    """
    min_spread_for_trend: float = 0.02  # 2% spread = medium MA 2% above long MA
    min_velocity_for_trend: float = 0.0  # Positive velocity required for uptrend
    min_acceleration_for_strong_trend: float = 0.0  # Positive accel for "strong" uptrend
    max_spread_for_neutral: float = 0.01  # |spread| < 1% = neutral/choppy


def classify_momentum_regime(
    features: pd.DataFrame,
    params: QqqMomentumRegimeParams,
) -> pd.Series:
    """
    Classify momentum regime for each date based on feature thresholds.

    **Conceptual**: This function implements the regime classification logic:
      1. Check MA spreads: Are we in a trend or choppy?
      2. Check velocity: Is the trend up or down?
      3. Check acceleration: Is the trend strengthening or weakening?

    **Mathematically**: Simple threshold-based rules applied row-wise:
      - STRONG_UPTREND: spread > min_spread AND velocity > min_velocity AND accel > min_accel
      - WEAKENING_UPTREND: spread > min_spread AND velocity > min_velocity AND accel <= min_accel
      - DOWNTREND: spread < -min_spread OR (velocity < -min_velocity)
      - NEUTRAL: Everything else (low spreads, mixed signals)

    **Functionally**: Returns a Series of regime labels (MomentumRegime enum values)
    indexed by the same dates as the input features DataFrame.

    Args:
        features: DataFrame containing feature columns from build_qqq_momentum_features.
                 Required columns: "ma_spread_50_250", "velocity_20d", "acceleration_20d"
                 (or configured window equivalents).
        params: QqqMomentumRegimeParams defining threshold values.

    Returns:
        pd.Series of MomentumRegime enum values, one per row in features.
        NaN rows in features (warm-up period) will map to NEUTRAL by default.

    Raises:
        KeyError: If required feature columns are missing.

    Usage example:
        >>> qqq_features = build_qqq_momentum_features(qqq, feature_params)
        >>> regime_params = QqqMomentumRegimeParams()
        >>> regimes = classify_momentum_regime(qqq_features, regime_params)
        >>> print(regimes.value_counts())
    """
    # Validate required columns
    required_cols = ["ma_spread_50_250", "velocity_20d", "acceleration_20d"]
    missing = [col for col in required_cols if col not in features.columns]
    if missing:
        raise KeyError(
            f"Missing required feature columns: {missing}. "
            "Run build_qqq_momentum_features first."
        )

    # Extract feature series for readability
    spread = features["ma_spread_50_250"]
    velocity = features["velocity_20d"]
    acceleration = features["acceleration_20d"]

    # Initialize regime series (default to NEUTRAL)
    regime = pd.Series(MomentumRegime.NEUTRAL, index=features.index)

    # Regime classification logic (applied row-wise via boolean indexing)

    # STRONG_UPTREND: positive spread, positive velocity, positive acceleration
    strong_uptrend_mask = (
        (spread > params.min_spread_for_trend) &
        (velocity > params.min_velocity_for_trend) &
        (acceleration > params.min_acceleration_for_strong_trend)
    )
    regime.loc[strong_uptrend_mask] = MomentumRegime.STRONG_UPTREND

    # WEAKENING_UPTREND: positive spread, positive velocity, but negative/zero acceleration
    weakening_uptrend_mask = (
        (spread > params.min_spread_for_trend) &
        (velocity > params.min_velocity_for_trend) &
        (acceleration <= params.min_acceleration_for_strong_trend)
    )
    regime.loc[weakening_uptrend_mask] = MomentumRegime.WEAKENING_UPTREND

    # DOWNTREND: negative spread OR negative velocity (bearish divergence)
    downtrend_mask = (
        (spread < -params.min_spread_for_trend) |
        (velocity < -params.min_velocity_for_trend)
    )
    regime.loc[downtrend_mask] = MomentumRegime.DOWNTREND

    # NEUTRAL: everything else (small spreads, mixed signals)
    # Note: NEUTRAL is already the default, so we don't need to explicitly set it again.
    # The neutral_mask was causing issues by overwriting other classifications.
    # NaNs are handled by the default NEUTRAL initialization.

    return regime


@dataclass
class QqqMomentumSymbols:
    """
    Instrument symbols for QQQ momentum strategy.

    **Conceptual**: Separates signal source (QQQ) from execution instruments
    (TQQQ, SQQQ, UVXY). This allows the strategy to:
      - Read QQQ prices for signals.
      - Trade leveraged instruments for higher returns (and higher risk).

    **Teaching note**: In production, you'd also configure:
      - Leverage ratios (TQQQ = 3x, SQQQ = -3x, UVXY = 2x VIX).
      - Correlation/beta to QQQ for risk management.
      - Liquidity constraints (can you actually fill orders?).

    Attributes:
        reference_symbol: Symbol for signal generation (typically "QQQ").
        long_symbol: Symbol for long exposure (typically "TQQQ" = 3x long QQQ).
        short_symbol: Symbol for short exposure (typically "SQQQ" = 3x inverse QQQ).
        vol_symbol: Optional symbol for volatility exposure (typically "UVXY" = 2x VIX).
                   Can be None if not using volatility overlay.
    """
    reference_symbol: str = "QQQ"
    long_symbol: str = "TQQQ"
    short_symbol: str = "SQQQ"
    vol_symbol: str | None = None  # Optional UVXY


@dataclass
class QqqMomentumAllocationParams:
    """
    Parameters for mapping regimes to target weights.

    **Conceptual**: These weights define the strategy's risk appetite in each regime.
    Example:
      - strong_uptrend_long_weight = 1.0 → Allocate 100% to TQQQ (max leverage).
      - weakening_uptrend_long_weight = 0.5 → Reduce to 50% TQQQ, 50% cash.
      - downtrend_short_weight = -1.0 → Short 100% via SQQQ (or sell TQQQ).
      - neutral_risk_weight = 0.0 → Stay 100% cash.

    **Teaching note**: Conservative parameters (lower weights) reduce drawdowns but
    also reduce upside. Aggressive parameters (full leverage) maximize returns but
    risk large losses during whipsaws. Finding the right balance is the art of
    strategy tuning.

    Attributes:
        strong_uptrend_long_weight: Target weight for long_symbol in STRONG_UPTREND (e.g., 1.0 = 100%).
        weakening_uptrend_long_weight: Target weight for long_symbol in WEAKENING_UPTREND (e.g., 0.5).
        neutral_risk_weight: Target weight in NEUTRAL regime (typically 0.0 = all cash).
        downtrend_short_weight: Target weight for short_symbol in DOWNTREND (e.g., 0.5 for 50% SQQQ).
                                Can be negative to indicate short position, or positive for inverse ETF.
        use_vol_overlay: If True and vol_symbol is available, allocate to UVXY in downtrends.
        vol_allocation_weight: Target weight for vol_symbol when use_vol_overlay is True.
    """
    strong_uptrend_long_weight: float = 1.0  # 100% TQQQ in strong uptrend
    weakening_uptrend_long_weight: float = 0.5  # 50% TQQQ in weakening uptrend
    neutral_risk_weight: float = 0.0  # 0% allocation = all cash in neutral
    downtrend_short_weight: float = 0.5  # 50% SQQQ in downtrend
    use_vol_overlay: bool = False  # Whether to use UVXY
    vol_allocation_weight: float = 0.2  # 20% UVXY if enabled


def regime_to_target_weights(
    regime: MomentumRegime,
    symbols: QqqMomentumSymbols,
    alloc_params: QqqMomentumAllocationParams,
) -> dict[str, float]:
    """
    Map a momentum regime to target portfolio weights.

    **Conceptual**: This function translates qualitative regime labels into
    quantitative position sizes. It's the final step before passing weights to
    the broker for execution.

    **Mathematically**: Simple lookup/mapping:
      - STRONG_UPTREND → {TQQQ: 1.0} (or configured weight)
      - WEAKENING_UPTREND → {TQQQ: 0.5} (reduced exposure)
      - NEUTRAL → {} (empty dict = all cash)
      - DOWNTREND → {SQQQ: 0.5} (or UVXY if vol overlay enabled)

    **Functionally**: Returns a dict of symbol -> target weight that conforms to
    the Phase 4 Strategy interface (used by run_backtest).

    Args:
        regime: MomentumRegime enum value for current date.
        symbols: QqqMomentumSymbols defining which instruments to trade.
        alloc_params: QqqMomentumAllocationParams defining target weights per regime.

    Returns:
        Dict mapping symbol (str) -> target weight (float).
        Weights are fractions of total equity: 1.0 = 100%, 0.5 = 50%, etc.
        Empty dict {} means 100% cash.
        Remaining weight not allocated to symbols is implicitly cash.

    Usage example:
        >>> regime = MomentumRegime.STRONG_UPTREND
        >>> symbols = QqqMomentumSymbols()
        >>> alloc_params = QqqMomentumAllocationParams()
        >>> weights = regime_to_target_weights(regime, symbols, alloc_params)
        >>> print(weights)  # {"TQQQ": 1.0}
    """
    # Initialize empty weights (default to all cash)
    weights: dict[str, float] = {}

    # Map regime to target weights based on allocation params
    if regime == MomentumRegime.STRONG_UPTREND:
        # Strong uptrend: max exposure to leveraged long
        weights[symbols.long_symbol] = alloc_params.strong_uptrend_long_weight

    elif regime == MomentumRegime.WEAKENING_UPTREND:
        # Weakening uptrend: reduced exposure to leveraged long
        weights[symbols.long_symbol] = alloc_params.weakening_uptrend_long_weight

    elif regime == MomentumRegime.NEUTRAL:
        # Neutral: stay in cash (empty dict)
        # Could also explicitly set all symbols to 0.0, but empty dict is cleaner
        pass

    elif regime == MomentumRegime.DOWNTREND:
        # Downtrend: allocate to inverse leveraged ETF (SQQQ)
        # Or optionally to volatility ETF (UVXY) if enabled
        if alloc_params.use_vol_overlay and symbols.vol_symbol is not None:
            # Use volatility overlay (UVXY) in downtrends
            weights[symbols.vol_symbol] = alloc_params.vol_allocation_weight
        else:
            # Use inverse leveraged ETF (SQQQ)
            weights[symbols.short_symbol] = alloc_params.downtrend_short_weight

    return weights
