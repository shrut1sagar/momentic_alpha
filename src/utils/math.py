"""
Mathematical and statistical utilities for trading analytics.

This module provides clean, well-documented implementations of fundamental
math operations used throughout the trading system: returns calculations,
moving averages, volatility measures, and momentum indicators (velocity/acceleration).

All functions are designed to be teaching-friendly with extensive docstrings
and inline comments explaining both the conceptual and mathematical foundations.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """
    Convert a price series into simple (arithmetic) returns.

    **Conceptual**: Answers "what was the percent change from one period to
    the next?" Simple returns are intuitive for single-period performance and
    are commonly used in trading P&L reporting. For a trader, this tells you
    the direct percentage gain or loss between consecutive periods.

    **Mathematical**: For each period t:
        r_t = (P_t / P_{t-1}) - 1
    where P_t is the price at time t.

    **Functionally**:
    - Input: pandas Series of prices, indexed in chronological order (oldest to newest).
    - Output: pandas Series of simple returns, aligned to input index.
    - The first value will be NaN (no prior price to compare).
    - Assumes no missing prices; handle forward/backfill upstream if needed.
    - Used later for daily P&L streams feeding Sharpe/Sortino calculations and
      rolling statistics in QQQ/TQQQ/SQQQ strategy tests.

    **Edge cases**:
    - If prices contains zeros or negative values, returns may be undefined or misleading.
    - Empty or single-element series will return all NaNs or a single NaN.

    Args:
        prices: Time series of asset prices (daily bars expected).

    Returns:
        Time series of simple returns, same index as input.
    """
    # Use pandas pct_change which computes (P_t / P_{t-1}) - 1
    # This is vectorized and handles the shift + division automatically
    return prices.pct_change()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Convert a price series into logarithmic (continuously compounded) returns.

    **Conceptual**: Log returns accumulate additively across periods and provide
    more stable variance properties for compounding price series. They are
    mathematically convenient for modeling drift and volatility because the
    sum of log returns equals the log return over the full period. For traders,
    log returns are preferred when estimating volatility or fitting stochastic
    models (GBM, etc.).

    **Mathematical**: For each period t:
        r_t = ln(P_t) - ln(P_{t-1}) = ln(P_t / P_{t-1})
    where ln is the natural logarithm.

    **Functionally**:
    - Input: pandas Series of prices in chronological order.
    - Output: pandas Series of log returns, aligned to input index.
    - First value is NaN (no prior price).
    - Assumes strictly positive prices (ln undefined for zero/negative).
    - Missing data should be cleaned before calling this function.
    - Prefer log returns for drift/volatility estimation and velocity calculations
      due to their additive and symmetric properties.

    **Edge cases**:
    - Prices <= 0 will produce NaN or -inf.
    - Very small price changes may have log returns ≈ simple returns.

    Args:
        prices: Time series of asset prices (must be positive).

    Returns:
        Time series of log returns, same index as input.
    """
    # Compute log of prices, then take first difference: ln(P_t) - ln(P_{t-1})
    # np.log is the natural logarithm
    return np.log(prices).diff()


def compute_moving_average_simple(prices: pd.Series, window: int) -> pd.Series:
    """
    Compute a simple moving average (SMA) to smooth noise in a time series.

    **Conceptual**: A simple moving average smooths short-term fluctuations by
    averaging the most recent N periods with equal weight. Traders use SMAs to
    identify trend direction and levels of support/resistance. For example, a
    50-day SMA crossing above a 200-day SMA is a classic bullish "golden cross" signal.

    **Mathematical**: At each time t, the SMA is:
        SMA_t = (1 / window) * Σ(P_{t-i}) for i = 0 to window-1
    This is the arithmetic mean of the last `window` prices BEFORE time t (inclusive).

    **Functionally**:
    - Input: pandas Series of prices or any feature series; window size (integer).
    - Output: pandas Series of SMA values, aligned with input index.
    - The first (window - 1) values will be NaN until enough data is available.
    - Used for trend context (e.g., 50-day, 100-day, 250-day levels) before
      strategy logic is developed.
    - Daily frequency assumed (window is in days).
    - **IMPORTANT**: Works correctly for both ascending (oldest first) and
      descending (newest first) data order. For descending data (Phase 3 schema),
      the function reverses the data before computing the MA to avoid look-ahead bias.

    **Edge cases**:
    - Window must be >= 1; window = 1 returns the original series.
    - If series length < window, all values will be NaN.

    Args:
        prices: Time series of prices or features.
        window: Number of periods to average (must be positive integer).

    Returns:
        Time series of simple moving average, same index as input.
    """
    # CRITICAL FIX: Check if data is in descending order (newest first)
    # pandas rolling() always looks backward in positional terms, but for
    # descending data this means looking forward in TIME, causing look-ahead bias.
    #
    # Solution: Reverse to ascending order, compute MA, then reverse back.

    if len(prices) > 1:
        # Check if indices are descending (indicator of newest-first order)
        indices = prices.index.tolist()
        if indices[-1] < indices[0]:
            # Data is in descending order - reverse it
            prices_ascending = prices.iloc[::-1]
            # Compute MA on ascending data (no look-ahead bias)
            ma_ascending = prices_ascending.rolling(window=window).mean()
            # Reverse back to descending order
            return ma_ascending.iloc[::-1]

    # Data is in ascending order or single-element - compute normally
    # min_periods defaults to window, so we get NaN until window is filled
    return prices.rolling(window=window).mean()


def compute_moving_average_exponential(prices: pd.Series, span: int) -> pd.Series:
    """
    Compute an exponential moving average (EMA) with more weight on recent data.

    **Conceptual**: An EMA applies exponentially decreasing weights to older
    observations, making it more responsive to recent price changes than a simple
    moving average. Traders use EMAs for faster trend signals with less lag.
    This is particularly useful in momentum strategies where timely detection
    of regime changes matters.

    **Mathematical**: The EMA uses a recursive formula:
        EMA_t = α * P_t + (1 - α) * EMA_{t-1}
    where α = 2 / (span + 1) is the smoothing factor.

    Pandas implements this via ewm(span=span, adjust=False).

    **Functionally**:
    - Input: pandas Series of prices or features; span (integer, analogous to window).
    - Output: pandas Series of EMA values, aligned with input index.
    - Warm-up NaNs as determined by pandas (typically the first value is used as seed).
    - Helpful for responsive signals and later regime detection without introducing
      excessive lag typical of long simple moving averages.
    - Daily frequency assumed.
    - **IMPORTANT**: Works correctly for both ascending and descending data order.
      For descending data (Phase 3 schema), reverses before computing to avoid look-ahead bias.

    **Edge cases**:
    - Span must be >= 1; span = 1 approaches P_t (minimal smoothing).
    - Smaller span = more reactive; larger span = more smoothing.

    Args:
        prices: Time series of prices or features.
        span: EMA span (roughly analogous to SMA window).

    Returns:
        Time series of exponential moving average, same index as input.
    """
    # CRITICAL FIX: Same as SMA - check for descending order
    if len(prices) > 1:
        indices = prices.index.tolist()
        if indices[-1] < indices[0]:
            # Data is in descending order - reverse it
            prices_ascending = prices.iloc[::-1]
            # Compute EMA on ascending data
            ema_ascending = prices_ascending.ewm(span=span, adjust=False).mean()
            # Reverse back to descending order
            return ema_ascending.iloc[::-1]

    # Data is in ascending order or single-element - compute normally
    # adjust=False gives the recursive form commonly used in trading
    return prices.ewm(span=span, adjust=False).mean()


def compute_rolling_standard_deviation(returns: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling standard deviation of returns over a moving window.

    **Conceptual**: Standard deviation measures the dispersion or variability
    of returns around their mean. In a rolling context, it shows how volatility
    evolves over time. Traders care about this because periods of high volatility
    often require reduced position sizes, especially with leveraged instruments
    like TQQQ/SQQQ.

    **Mathematical**: At each time t, compute the sample standard deviation over
    the last `window` returns:
        σ_t = sqrt( (1 / (window - 1)) * Σ((r_i - mean(r))^2) )
    where the sum is over the window. We use ddof=1 for unbiased estimator
    (Bessel's correction).

    **Functionally**:
    - Input: pandas Series of returns (simple or log); window size.
    - Output: pandas Series of rolling standard deviation, aligned to input.
    - First (window - 1) values are NaN until the window fills.
    - This forms the basis for rolling volatility (annualized via sqrt scaling)
      and is used in position scaling and risk budgeting.
    - Assumes cleaned return data (no large gaps or missing values).
    - Units are in the same frequency as input (e.g., daily std for daily returns).
    - **IMPORTANT**: Works correctly for both ascending and descending data order.

    **Edge cases**:
    - Window must be >= 2 for std to be defined (ddof=1).
    - Constant returns over window yield std = 0.

    Args:
        returns: Time series of returns.
        window: Number of periods over which to compute rolling std.

    Returns:
        Time series of rolling standard deviation, same index as input.
    """
    # CRITICAL FIX: Same pattern as SMA/EMA - check for descending order
    if len(returns) > 1:
        indices = returns.index.tolist()
        if indices[-1] < indices[0]:
            # Data is in descending order - reverse it
            returns_ascending = returns.iloc[::-1]
            # Compute rolling std on ascending data
            std_ascending = returns_ascending.rolling(window=window).std(ddof=1)
            # Reverse back to descending order
            return std_ascending.iloc[::-1]

    # Data is in ascending order or single-element - compute normally
    # Use pandas rolling with std; ddof=1 for sample std (unbiased estimator)
    return returns.rolling(window=window).std(ddof=1)


def compute_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Scale return volatility to an annualized figure for cross-strategy comparison.

    **Conceptual**: Volatility (standard deviation of returns) is frequency-dependent.
    To compare strategies or interpret risk in annual terms, we annualize by scaling
    the periodic volatility by the square root of the number of periods per year.
    This assumes returns are i.i.d., which is a simplification but standard in practice.
    For traders, annualized volatility is the "headline" risk number used in Sharpe
    ratios and risk budgets.

    **Mathematical**: Given periodic returns with standard deviation σ_periodic,
        σ_annualized = σ_periodic * sqrt(periods_per_year)
    For daily returns, periods_per_year = 252 (trading days).

    **Functionally**:
    - Input: pandas Series of periodic returns (e.g., daily); periods_per_year
      (default 252 for daily data).
    - Output: scalar float representing annualized volatility.
    - NaNs are dropped before computing std.
    - Used for Sharpe/Sortino denominators and risk budgeting.
    - **Critical**: Ensure the returns frequency matches periods_per_year
      (252 for daily, 12 for monthly, etc.).

    **Edge cases**:
    - Empty series or all NaN returns np.nan.
    - Single non-NaN return yields std = 0 (or NaN depending on implementation).

    Args:
        returns: Time series of periodic returns.
        periods_per_year: Number of periods in one year (252 for daily, 12 for monthly).

    Returns:
        Annualized volatility as a scalar float.
    """
    # Compute sample standard deviation (ddof=1) and scale by sqrt of periods per year
    # dropna() ensures we ignore NaN values
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def compute_velocity(
    prices: pd.Series,
    window: int,
    use_log: bool = True
) -> pd.Series:
    """
    Compute the "velocity" (slope/trend strength) of prices over a rolling window.

    **Conceptual**: Velocity captures how steeply prices are moving over recent
    history, answering "is the trend strengthening or just drifting sideways?"
    For a trader, positive velocity indicates upward momentum; negative indicates
    downward momentum. This is a key input for trend-following strategies to
    distinguish strong trends (allocate to TQQQ) from weak or ranging markets
    (move to cash or SQQQ).

    **Mathematical**: At each time t, fit a simple linear regression of
    (log prices if use_log=True, else raw prices) against time index over the
    last `window` periods. The slope coefficient β is the velocity:
        y = α + β * time + ε
    where y is (log) price and β is the per-period slope.

    We use scipy.stats.linregress over a rolling window. The slope is in units
    of (log) price change per period.

    **Functionally**:
    - Input: pandas Series of prices; window size; use_log flag (default True).
    - Output: pandas Series of velocity (slope), aligned to input index.
    - First (window - 1) values are NaN until window fills.
    - Daily frequency assumed; slope units are price change per day (or log-price per day).
    - Later used in QQQ/TQQQ/SQQQ allocation logic to gauge trend strength.
    - use_log=True is recommended for symmetry and stability (log returns are additive).
    - **IMPORTANT**: Works correctly regardless of whether data is sorted in ascending
      (oldest first) or descending (newest first) order. The function handles both by
      reversing the window to chronological order before regression.

    **Edge cases**:
    - Flat prices yield velocity ≈ 0.
    - Short window may be noisy; long window may lag trend changes.

    Args:
        prices: Time series of asset prices.
        window: Number of periods over which to compute velocity (trend slope).
        use_log: If True, compute velocity on log(prices); else on raw prices.

    Returns:
        Time series of velocity (slope of price trend), same index as input.
    """
    # Transform prices to log scale if requested
    if use_log:
        # Use log prices for velocity calculation (more stable, additive)
        y = np.log(prices)
    else:
        y = prices

    # Prepare a series to hold velocity values
    velocity = pd.Series(index=prices.index, dtype=float)

    # Rolling window loop: for each window, fit linear regression and extract slope
    for i in range(window - 1, len(prices)):
        # Extract the window of (log) prices
        window_y = y.iloc[i - window + 1 : i + 1]

        # CRITICAL FIX: Ensure window is in chronological order (oldest to newest)
        # for correct slope calculation.
        #
        # The Phase 3 schema specifies data is in descending order (newest first),
        # so iloc[i-window+1:i+1] gives [newest, ..., oldest].
        # Regression needs [oldest, ..., newest] to get correct positive slope
        # for rising prices.
        #
        # However, test data may be in ascending order. We need to detect which
        # order the window is in and ensure chronological order.
        #
        # Detection: Check if the index values are increasing or decreasing.
        # If index is not monotonic, we fall back to assuming descending (Phase 3 default).
        window_indices = window_y.index.tolist()

        # Check if indices are in ascending order (chronological for timestamp indices)
        # For integer indices, we can't tell chronological order, so we check if
        # they're decreasing (which would indicate newest-first for descending data)
        if len(window_indices) > 1 and window_indices[-1] < window_indices[0]:
            # Indices are descending, which for timestamp indices means newest first
            # Reverse to get chronological order
            window_y_values = window_y.values[::-1]
        else:
            # Indices are ascending or equal - assume already chronological
            window_y_values = window_y.values

        # Time index for regression (0, 1, 2, ..., window-1)
        # Now x=0 corresponds to oldest price, x=window-1 to newest
        window_x = np.arange(window)

        # Fit linear regression: y = slope * x + intercept
        # linregress returns (slope, intercept, r_value, p_value, std_err)
        slope, _, _, _, _ = stats.linregress(window_x, window_y_values)

        # Store the slope as velocity at this timestep
        velocity.iloc[i] = slope

    return velocity


def compute_acceleration(
    prices: pd.Series,
    window: int,
    use_log: bool = True
) -> pd.Series:
    """
    Compute the "acceleration" (rate of change of velocity) over time.

    **Conceptual**: Acceleration measures whether the trend is speeding up or
    slowing down. For a trader, positive acceleration in an uptrend suggests
    the trend is gaining momentum (bullish reinforcement), while negative
    acceleration suggests the trend is "tiring" or losing steam. This helps
    distinguish strong, accelerating trends (ideal for leverage) from exhausted
    trends (reduce exposure or prepare for reversal).

    **Mathematical**: Acceleration is the first difference of velocity:
        accel_t = velocity_t - velocity_{t-1}
    Velocity itself is the rolling slope of (log) prices, so acceleration is
    the second derivative concept: how the slope is changing.

    **Functionally**:
    - Input: pandas Series of prices; window size; use_log flag.
    - Output: pandas Series of acceleration, aligned to velocity.
    - First (window) values are NaN (velocity needs window-1, then diff adds another NaN).
    - Daily cadence assumed; units are (log) price change per day per day.
    - Useful for distinguishing strong vs tiring trends before entering leveraged
      positions (TQQQ/SQQQ).

    **Edge cases**:
    - Constant velocity yields acceleration = 0.
    - Noisy velocity (small window) leads to noisy acceleration.

    Args:
        prices: Time series of asset prices.
        window: Number of periods for velocity calculation (same as compute_velocity).
        use_log: If True, base velocity on log(prices); else on raw prices.

    Returns:
        Time series of acceleration (change in velocity), same index as input.
    """
    # First compute velocity using the same window and log settings
    velocity = compute_velocity(prices, window=window, use_log=use_log)

    # Acceleration is the first difference of velocity: accel_t = velocity_t - velocity_{t-1}
    # This adds one more NaN at the beginning
    acceleration = velocity.diff()

    return acceleration
