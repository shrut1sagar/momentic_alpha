"""
Generic feature engineering helpers for price-based strategies.

**Conceptual**: This module provides reusable feature engineering functions that
transform raw price data into trading signals. These helpers wrap Phase 2 math
utilities in a more "feature-oriented" API, adding columns to DataFrames in place
or returning modified DataFrames.

**Why generic helpers?**
  - Reusability: Multiple strategies can use the same MA, spread, velocity logic
    without duplicating code.
  - Testability: Features can be tested independently of strategies.
  - Clarity: Feature engineering is separate from regime classification and
    allocation decisions.

**Teaching note**: In production quant systems, feature engineering is often the
most time-consuming part of strategy development. Good feature engineering:
  - Uses descriptive, human-readable column names (e.g., "ma_spread_50_250" not "f1").
  - Documents assumptions (daily frequency, no missing data, etc.).
  - Returns DataFrames that preserve index alignment (no accidental date shifts).
  - Handles edge cases (NaNs at window starts, empty data, etc.).

This module focuses on momentum/trend features for leveraged ETF strategies (QQQ/TQQQ/SQQQ),
but the patterns are general-purpose.
"""

import pandas as pd
from src.utils.math import (
    compute_moving_average_simple,
    compute_moving_average_exponential,
    compute_velocity,
    compute_acceleration,
)


def add_moving_averages(
    df: pd.DataFrame,
    ma_windows: dict[str, int],
    price_column: str = "closing_price",
    use_ema: bool = False,
) -> pd.DataFrame:
    """
    Add simple or exponential moving averages to a DataFrame.

    **Conceptual**: Moving averages smooth price noise and reveal trend direction.
    Traders use multiple MA windows to gauge trend strength: price above long-term
    MA = uptrend, price below = downtrend. Crossovers (short MA crossing long MA)
    signal regime changes.

    **Mathematically**: For each window N, computes SMA or EMA over last N prices.
    See `src.utils.math` for detailed formulas.

    **Functionally**: Adds new columns to `df` named `moving_average_{window}` or
    `ema_{window}`. Modifies `df` in place and also returns it for chaining.

    Args:
        df: DataFrame with at least a price column (e.g., "closing_price").
           Index should be timestamps in descending order (newest first) or ascending
           (function handles both; just preserve original sort order).
        ma_windows: Dict mapping label -> window size. Example:
           {"ma_20": 20, "ma_50": 50, "ma_250": 250}
           Labels are used for column names (e.g., "moving_average_20").
        price_column: Column name to compute MAs from (default "closing_price").
        use_ema: If True, use exponential MA; if False, use simple MA.

    Returns:
        Modified DataFrame with new MA columns. NaNs appear at window starts where
        insufficient data exists.

    Raises:
        KeyError: If price_column is missing from df.

    Usage example:
        >>> df = load_qqq_history()
        >>> df = add_moving_averages(df, {"20": 20, "50": 50, "250": 250})
        >>> print(df[["closing_price", "moving_average_20", "moving_average_250"]])
    """
    # Validate input
    if price_column not in df.columns:
        raise KeyError(
            f"Column '{price_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Get price series
    prices = df[price_column]

    # Compute each MA and add as a column
    for label, window in ma_windows.items():
        # Determine column name (e.g., "moving_average_50" or "ema_50")
        col_name = f"{'ema' if use_ema else 'moving_average'}_{label}"

        # Compute MA using Phase 2 helper
        if use_ema:
            ma_series = compute_moving_average_exponential(prices, span=window)
        else:
            ma_series = compute_moving_average_simple(prices, window=window)

        # Add to DataFrame (aligned by index)
        df[col_name] = ma_series

    return df


def add_ma_spreads(
    df: pd.DataFrame,
    spread_pairs: list[tuple[str, str]],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Add moving average spreads (fast MA - slow MA) to a DataFrame.

    **Conceptual**: MA spreads measure the gap between short- and long-term trends.
    Positive spread = short-term momentum exceeds long-term trend (bullish divergence).
    Negative spread = short-term weakening vs long-term (bearish divergence).
    Normalizing by the slow MA makes spreads comparable across different price levels.

    **Mathematically**:
      - Raw spread: spread = MA_fast - MA_slow
      - Normalized spread: spread = (MA_fast - MA_slow) / MA_slow
        (This is a percentage: +0.05 = fast is 5% above slow)

    **Functionally**: Adds columns named `ma_spread_{fast}_{slow}` to df.

    Args:
        df: DataFrame containing moving average columns (e.g., "moving_average_50").
        spread_pairs: List of (fast_label, slow_label) tuples. Example:
           [("50", "100"), ("50", "250")]
           Will create columns "ma_spread_50_100" and "ma_spread_50_250".
        normalize: If True, divide spread by slow MA (percentage spread).
                  If False, use raw difference.

    Returns:
        Modified DataFrame with new spread columns.

    Raises:
        KeyError: If any required MA column is missing.

    Usage example:
        >>> df = add_moving_averages(df, {"50": 50, "100": 100, "250": 250})
        >>> df = add_ma_spreads(df, [("50", "100"), ("50", "250")])
        >>> # Now df has "ma_spread_50_100" and "ma_spread_50_250"
    """
    for fast_label, slow_label in spread_pairs:
        # Construct column names (assume MAs were added with add_moving_averages)
        fast_col = f"moving_average_{fast_label}"
        slow_col = f"moving_average_{slow_label}"

        # Validate columns exist
        if fast_col not in df.columns:
            raise KeyError(f"Column '{fast_col}' not found. Add MAs first using add_moving_averages().")
        if slow_col not in df.columns:
            raise KeyError(f"Column '{slow_col}' not found. Add MAs first using add_moving_averages().")

        # Compute spread
        fast_ma = df[fast_col]
        slow_ma = df[slow_col]

        if normalize:
            # Normalized spread: (fast - slow) / slow
            # Handle division by zero (slow_ma could be zero in pathological cases)
            spread = (fast_ma - slow_ma) / slow_ma.replace(0, pd.NA)
        else:
            # Raw spread: fast - slow
            spread = fast_ma - slow_ma

        # Add spread column
        spread_col = f"ma_spread_{fast_label}_{slow_label}"
        df[spread_col] = spread

    return df


def add_velocity_and_acceleration(
    df: pd.DataFrame,
    velocity_window: int,
    acceleration_window: int | None = None,
    price_column: str = "closing_price",
    use_log: bool = True,
) -> pd.DataFrame:
    """
    Add velocity (price trend slope) and acceleration (change in slope) to a DataFrame.

    **Conceptual**: Velocity measures how steeply prices are rising or falling
    (trend strength). Acceleration measures whether the trend is speeding up or
    slowing down. For leveraged strategies:
      - High positive velocity + positive acceleration = strong trend, lean into leverage.
      - High velocity + negative acceleration = trend weakening, reduce exposure.
      - Low velocity = choppy/sideways, avoid leverage.

    **Mathematically**:
      - Velocity: slope of linear regression on (log) prices over rolling window.
        Units: price change per day (or log-price change per day if use_log=True).
      - Acceleration: first difference of velocity series.
        Units: change in slope per day.

    **Functionally**: Adds columns:
      - `velocity_{window}d`: Velocity over specified window.
      - `acceleration_{window}d`: Acceleration (if acceleration_window is not None).

    Args:
        df: DataFrame with price column.
        velocity_window: Rolling window for velocity computation (e.g., 20 for 20-day slope).
        acceleration_window: Rolling window for acceleration (typically same as velocity_window).
                            If None, acceleration is not computed.
        price_column: Column name for prices (default "closing_price").
        use_log: If True, compute velocity on log(prices) (recommended for compounding returns).
                If False, use raw prices.

    Returns:
        Modified DataFrame with velocity and acceleration columns.

    Usage example:
        >>> df = add_velocity_and_acceleration(df, velocity_window=20, acceleration_window=20)
        >>> # Now df has "velocity_20d" and "acceleration_20d"
    """
    # Validate input
    if price_column not in df.columns:
        raise KeyError(f"Column '{price_column}' not found in DataFrame.")

    # Get price series
    prices = df[price_column]

    # Compute velocity using Phase 2 helper
    velocity_series = compute_velocity(prices, window=velocity_window, use_log=use_log)

    # Add velocity column
    velocity_col = f"velocity_{velocity_window}d"
    df[velocity_col] = velocity_series

    # Compute acceleration if requested
    if acceleration_window is not None:
        acceleration_series = compute_acceleration(
            prices, window=acceleration_window, use_log=use_log
        )

        # Add acceleration column
        accel_col = f"acceleration_{acceleration_window}d"
        df[accel_col] = acceleration_series

    return df


def normalize_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    method: str = "zscore",
    window: int | None = None,
) -> pd.DataFrame:
    """
    Normalize feature columns for comparability across different regimes.

    **Conceptual**: Raw features (velocity, spreads) have different scales and
    distributions. Normalization makes them comparable:
      - Z-score: (x - mean) / std → dimensionless, centered at 0.
      - Logistic: 1 / (1 + exp(-x)) → maps to [0, 1], highlighting extremes.
      - Rolling z-score: Uses rolling mean/std instead of global (adaptive normalization).

    **Mathematically**:
      - Z-score: z = (x - μ) / σ
      - Logistic: sigmoid(x) = 1 / (1 + exp(-x))
      - Rolling z-score: z_t = (x_t - μ_rolling) / σ_rolling

    **Functionally**: Adds new columns named `{feature}_normalized` to df.

    Args:
        df: DataFrame containing feature columns.
        feature_columns: List of column names to normalize (e.g., ["velocity_20d", "ma_spread_50_250"]).
        method: Normalization method: "zscore", "rolling_zscore", or "logistic".
        window: Rolling window for "rolling_zscore" method (ignored for global methods).

    Returns:
        Modified DataFrame with normalized feature columns.

    Usage example:
        >>> df = normalize_features(df, ["velocity_20d", "acceleration_20d"], method="zscore")
        >>> # Now df has "velocity_20d_normalized" and "acceleration_20d_normalized"
    """
    for col in feature_columns:
        if col not in df.columns:
            raise KeyError(f"Feature column '{col}' not found in DataFrame.")

        feature = df[col]

        if method == "zscore":
            # Global z-score normalization
            mean = feature.mean()
            std = feature.std(ddof=1)
            normalized = (feature - mean) / std if std > 0 else feature * 0

        elif method == "rolling_zscore":
            # Rolling z-score normalization
            if window is None:
                raise ValueError("window must be provided for rolling_zscore method.")
            rolling_mean = feature.rolling(window=window).mean()
            rolling_std = feature.rolling(window=window).std(ddof=1)
            # Avoid division by zero
            normalized = (feature - rolling_mean) / rolling_std.replace(0, pd.NA)

        elif method == "logistic":
            # Logistic (sigmoid) normalization: maps to [0, 1]
            import numpy as np
            normalized = 1 / (1 + np.exp(-feature))

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Add normalized column
        normalized_col = f"{col}_normalized"
        df[normalized_col] = normalized

    return df
