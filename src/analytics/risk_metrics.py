"""
Risk and performance metrics for trading strategy evaluation.

This module implements a curated set of performance metrics designed specifically
for evaluating leveraged trend-following strategies (e.g., QQQ/TQQQ/SQQQ).
Metrics are grouped into:
  - Core performance: risk/return framing (Sharpe, Sortino, CAGR, etc.)
  - Drawdown/pain: protecting against sharp losses and whipsaws
  - Relative-to-benchmark: judging value-add vs the underlying index
  - Distribution/trade-style: understanding hit patterns and path dependency
  - Rolling/time-varying: detecting regime shifts and strategy degradation

All functions use full type hints and are extensively documented for teaching purposes.
"""

import numpy as np
import pandas as pd


def compute_total_return(equity_curve: pd.Series) -> float:
    """
    Compute the overall return on investment (ROI) from start to end.

    **Conceptual**: Total return is the simplest performance measure, answering
    "how much did $1 grow (or shrink) over the entire period?" For a trader,
    this is the headline number, but it ignores time and risk. In leveraged
    strategies (TQQQ/SQQQ), total return can be large but must be contextualized
    with volatility and drawdown.

    **Mathematical**: Given an equity curve with initial value E_0 and final value E_T:
        Total Return = (E_T / E_0) - 1

    **Functionally**:
    - Input: pandas Series of equity values indexed by date.
    - Output: scalar float (e.g., 0.25 for 25% return).
    - Assumes positive starting equity (E_0 > 0).
    - Does not account for time (a 25% return over 1 year vs 10 years is very different).
    - Baseline for interpreting leveraged performance vs simple QQQ buy-and-hold.

    **Interpretation for TQQQ/SQQQ**:
    - A high total return with high drawdown may not be sustainable.
    - Compare against QQQ total return to assess if leverage added value.

    **Edge cases**:
    - If equity_curve is empty or single value, return may be undefined (NaN or 0).
    - Negative starting equity is undefined for ROI.

    Args:
        equity_curve: Time series of portfolio equity values (must be positive).

    Returns:
        Total return as a decimal (e.g., 0.50 = 50% gain).
    """
    # Extract first and last equity values (iloc for positional indexing)
    initial_equity = equity_curve.iloc[0]
    final_equity = equity_curve.iloc[-1]

    # ROI formula: (final / initial) - 1
    return (final_equity / initial_equity) - 1.0


def compute_cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute the Compound Annual Growth Rate (CAGR) accounting for compounding.

    **Conceptual**: CAGR is the annualized rate of return assuming profits are
    reinvested. It smooths out volatility to show "what constant annual return
    would produce this total return over this time period?" For traders, CAGR
    is more interpretable than total return because it normalizes for time,
    making it easy to compare strategies of different durations.

    **Mathematical**: Given equity curve with E_0, E_T over n periods:
        CAGR = (E_T / E_0)^(periods_per_year / n) - 1
    where n is the number of periods (e.g., trading days).

    **Functionally**:
    - Input: pandas Series of daily equity values; periods_per_year (default 252 for daily).
    - Output: scalar float representing annualized growth rate.
    - Sensitive to start and end dates (endpoint bias).
    - Useful for Calmar ratio (CAGR / max drawdown) and comparing strategies
      against QQQ buy-and-hold over the same period.
    - Assumes continuous compounding via geometric mean.

    **Interpretation for TQQQ/SQQQ**:
    - High CAGR with manageable drawdown suggests successful leverage usage.
    - Compare CAGR to QQQ to assess alpha from trend-following.

    **Edge cases**:
    - If equity_curve has length < 2, CAGR is undefined.
    - Negative final equity yields complex/undefined CAGR.

    Args:
        equity_curve: Time series of portfolio equity values.
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly).

    Returns:
        CAGR as a decimal (e.g., 0.15 = 15% annualized).
    """
    # Extract initial and final equity
    initial_equity = equity_curve.iloc[0]
    final_equity = equity_curve.iloc[-1]

    # Number of periods in the equity curve
    n_periods = len(equity_curve) - 1  # -1 because we measure intervals, not points

    # CAGR formula: (final / initial) ^ (periods_per_year / n) - 1
    # Handle edge case where n_periods = 0
    if n_periods == 0:
        return 0.0

    return (final_equity / initial_equity) ** (periods_per_year / n_periods) - 1.0


def compute_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized volatility (standard deviation of returns).

    **Conceptual**: Volatility measures the variability of returns, which traders
    interpret as risk. Annualizing allows comparison across strategies and asset
    classes. For leveraged ETFs like TQQQ/SQQQ, volatility is inherently higher
    due to 3x leverage, so this metric must be contextualized with return
    (hence Sharpe ratio).

    **Mathematical**: Given periodic returns with standard deviation σ:
        σ_annualized = σ * sqrt(periods_per_year)
    This assumes i.i.d. returns (a simplification, but standard practice).

    **Functionally**:
    - Input: pandas Series of periodic returns (e.g., daily); periods_per_year.
    - Output: scalar float representing annualized volatility (e.g., 0.20 = 20% annual vol).
    - NaNs are dropped before computing std.
    - Used as the denominator in Sharpe and Sortino ratios.
    - **Critical**: Ensure returns frequency matches periods_per_year.

    **Interpretation for TQQQ/SQQQ**:
    - Expect ~3x the volatility of QQQ due to leverage.
    - High volatility isn't necessarily bad if returns are proportionately higher (check Sharpe).

    **Edge cases**:
    - Empty series or all NaNs return NaN.
    - Constant returns yield vol = 0.

    Args:
        returns: Time series of periodic returns.
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        Annualized volatility as a decimal.
    """
    # Compute sample standard deviation (ddof=1 for unbiased estimator)
    # and scale by sqrt(periods_per_year) to annualize
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute the Sharpe ratio: excess return per unit of total volatility.

    **Conceptual**: The Sharpe ratio measures risk-adjusted performance, answering
    "how much return am I getting for each unit of risk I'm taking?" Higher is
    better. For leveraged strategies like TQQQ/SQQQ, Sharpe contextualizes whether
    the higher returns justify the higher volatility. A strategy with lower
    absolute returns but higher Sharpe may be preferable for risk-averse traders.

    **Mathematical**: Given returns r_t and risk-free rate r_f:
        Sharpe = (mean(r_t) - r_f / periods_per_year) / std(r_t) * sqrt(periods_per_year)
    The numerator is annualized excess return; the denominator is annualized volatility.

    **Functionally**:
    - Input: pandas Series of periodic returns; risk_free_rate (annualized, default 0);
      periods_per_year.
    - Output: scalar Sharpe ratio.
    - NaNs are dropped before calculation.
    - Typical "good" Sharpe: > 1.0 (varies by market/strategy type).

    **Interpretation for TQQQ/SQQQ**:
    - Leveraged ETFs may show higher volatility, so Sharpe helps compare fairly.
    - Trend-following may have lower hit rate but higher Sharpe if winners are large.
    - Compare strategy Sharpe to QQQ Sharpe to assess alpha.

    **Edge cases**:
    - If std = 0 (constant returns), Sharpe is undefined (inf or NaN).
    - Negative Sharpe indicates returns below risk-free rate.

    Args:
        returns: Time series of periodic returns.
        risk_free_rate: Annualized risk-free rate (e.g., 0.03 for 3%).
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        Sharpe ratio as a scalar.
    """
    # Drop NaNs from returns
    clean_returns = returns.dropna()

    # Compute mean return per period
    mean_return = clean_returns.mean()

    # Risk-free rate per period
    rf_per_period = risk_free_rate / periods_per_year

    # Excess return per period
    excess_return = mean_return - rf_per_period

    # Volatility per period
    vol_per_period = clean_returns.std(ddof=1)

    # Sharpe ratio: annualize both numerator and denominator
    # Annualized excess return = excess_return * periods_per_year
    # Annualized vol = vol_per_period * sqrt(periods_per_year)
    # Sharpe = (excess_return * periods_per_year) / (vol_per_period * sqrt(periods_per_year))
    #        = (excess_return / vol_per_period) * sqrt(periods_per_year)
    # Use tolerance check instead of exact equality for floating-point safety
    if vol_per_period < 1e-10 or np.isnan(vol_per_period):
        return np.nan  # Undefined if no volatility

    return (excess_return / vol_per_period) * np.sqrt(periods_per_year)


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute the Sortino ratio: excess return per unit of downside volatility.

    **Conceptual**: The Sortino ratio is like Sharpe but penalizes only downside
    volatility (negative returns), not upside. This is valuable for leveraged
    strategies where upside volatility is desirable. For a trader, Sortino
    answers "am I being rewarded for the downside risk I'm taking?" and is often
    more intuitive than Sharpe for asymmetric return distributions.

    **Mathematical**: Given returns r_t and risk-free rate r_f:
        Sortino = (mean(r_t) - r_f / periods_per_year) / downside_std * sqrt(periods_per_year)
    where downside_std = std(r_t[r_t < 0]).

    **Functionally**:
    - Input: pandas Series of periodic returns; risk_free_rate (annualized); periods_per_year.
    - Output: scalar Sortino ratio.
    - NaNs are dropped.
    - If there are no negative returns, downside_std = 0, and Sortino is undefined
      (conventionally return +inf or NaN; here we return NaN).

    **Interpretation for TQQQ/SQQQ**:
    - Leveraged strategies often have upside volatility from large wins; Sortino
      captures this better than Sharpe.
    - A high Sortino with manageable Sharpe suggests good risk/reward asymmetry.

    **Edge cases**:
    - No negative returns → downside_std = 0 → Sortino undefined (NaN or +inf).
    - Constant negative returns → defined but negative Sortino.

    Args:
        returns: Time series of periodic returns.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Number of periods per year.

    Returns:
        Sortino ratio as a scalar.
    """
    # Drop NaNs
    clean_returns = returns.dropna()

    # Mean return per period
    mean_return = clean_returns.mean()

    # Risk-free rate per period
    rf_per_period = risk_free_rate / periods_per_year

    # Excess return per period
    excess_return = mean_return - rf_per_period

    # Downside returns: only negative returns
    downside_returns = clean_returns[clean_returns < 0]

    # Downside standard deviation (use ddof=1)
    downside_std = downside_returns.std(ddof=1)

    # Handle case where there are no downside returns
    if len(downside_returns) == 0 or downside_std == 0:
        return np.nan  # Undefined

    # Sortino ratio: annualized excess return / annualized downside vol
    return (excess_return / downside_std) * np.sqrt(periods_per_year)


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Compute the drawdown time series showing % drop from each prior peak.

    **Conceptual**: Drawdown measures the pain of being in a losing position
    relative to your best historical performance. It answers "how far am I down
    from my peak?" at every point in time. For traders, especially with leverage,
    drawdown is critical because large drawdowns require even larger gains to
    recover (e.g., a 50% loss requires a 100% gain to break even).

    **Mathematical**: At each time t:
        drawdown_t = (equity_t / cumulative_peak_t) - 1
    where cumulative_peak_t = max(equity_0, ..., equity_t).

    **Functionally**:
    - Input: pandas Series of equity values indexed by date.
    - Output: pandas Series of drawdowns (values <= 0), same index as input.
    - The series starts at 0 (no drawdown at inception).
    - Highlights whipsaw risk and recovery periods in leveraged trend strategies.

    **Interpretation for TQQQ/SQQQ**:
    - Leveraged ETFs can have deep drawdowns during adverse periods.
    - Drawdown series helps visualize when the strategy is "underwater" and by how much.
    - Use with max_drawdown to understand worst-case pain.

    **Edge cases**:
    - Monotonically increasing equity → all drawdowns = 0.
    - Equity = 0 at any point → drawdown undefined (or -100%).

    Args:
        equity_curve: Time series of portfolio equity values.

    Returns:
        Time series of drawdowns (values <= 0), same index as input.
    """
    # Compute the running maximum (cumulative peak) up to each point
    cumulative_peak = equity_curve.cummax()

    # Drawdown at each point: (current / peak) - 1
    # This will be <= 0 (zero when at a new peak, negative when below peak)
    drawdown = (equity_curve / cumulative_peak) - 1.0

    return drawdown


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute the maximum drawdown: worst peak-to-trough loss over the period.

    **Conceptual**: Max drawdown (MDD) is the single worst loss from a peak,
    measuring the largest "pain" experienced. For traders, MDD is critical for
    position sizing and risk management, especially with leverage. A 30% MDD
    means at worst, you lost 30% from your peak—psychologically and financially
    significant.

    **Mathematical**: MDD is the minimum value of the drawdown series:
        MDD = min(drawdown_t) for all t
    where drawdown_t = (equity_t / cumulative_peak_t) - 1.

    **Functionally**:
    - Input: pandas Series of equity values.
    - Output: scalar float (value <= 0, e.g., -0.30 for 30% drawdown).
    - Used in Calmar ratio (CAGR / |MDD|) and risk budgeting.
    - Critical for sizing leverage: deeper MDD → lower acceptable leverage.

    **Interpretation for TQQQ/SQQQ**:
    - Leveraged strategies can have MDD > 50% even if underlying index is < 20%.
    - Compare MDD to QQQ to assess cost of leverage.
    - A strategy with high CAGR but very deep MDD may not be tradable for most investors.

    **Edge cases**:
    - Monotonically increasing equity → MDD = 0.
    - Equity goes to zero → MDD = -100% (-1.0).

    Args:
        equity_curve: Time series of portfolio equity values.

    Returns:
        Maximum drawdown as a scalar (value <= 0).
    """
    # Compute the full drawdown series
    drawdown_series = compute_drawdown_series(equity_curve)

    # Max drawdown is the minimum (most negative) value
    return drawdown_series.min()


def compute_calmar_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute the Calmar ratio: CAGR divided by absolute max drawdown.

    **Conceptual**: The Calmar ratio contrasts return (CAGR) against pain (max drawdown),
    answering "how much annualized return am I getting per unit of worst-case loss?"
    Higher is better. For traders using leverage (TQQQ/SQQQ), Calmar is particularly
    valuable because it directly penalizes strategies with deep drawdowns, which
    can be psychologically unbearable and hard to recover from.

    **Mathematical**:
        Calmar = CAGR / |max_drawdown|
    Both CAGR and MDD are computed from the equity curve.

    **Functionally**:
    - Input: pandas Series of equity values; periods_per_year.
    - Output: scalar float (higher is better).
    - If MDD = 0 (no drawdown), Calmar is undefined (returns +inf or NaN; here NaN).
    - Typical good Calmar: > 1.0, but varies by strategy type.

    **Interpretation for TQQQ/SQQQ**:
    - A leveraged strategy with CAGR = 20% and MDD = 40% → Calmar = 0.5.
    - Compare to QQQ's Calmar to see if leverage improved risk-adjusted returns.
    - High Calmar suggests good return-to-pain trade-off.

    **Edge cases**:
    - MDD = 0 → Calmar undefined (NaN).
    - Negative CAGR → Calmar will be negative.

    Args:
        equity_curve: Time series of portfolio equity values.
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        Calmar ratio as a scalar.
    """
    # Compute CAGR
    cagr = compute_cagr(equity_curve, periods_per_year=periods_per_year)

    # Compute max drawdown (will be <= 0)
    mdd = compute_max_drawdown(equity_curve)

    # Calmar = CAGR / |MDD|
    # Handle edge case where MDD = 0
    if mdd == 0:
        return np.nan

    return cagr / abs(mdd)


def compute_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute the Information Ratio: excess return vs benchmark per unit of tracking error.

    **Conceptual**: The Information Ratio (IR) measures active return (return above
    benchmark) per unit of active risk (tracking error). It answers "am I being
    rewarded for deviating from the benchmark?" For a TQQQ/SQQQ trend strategy
    vs QQQ benchmark, IR shows whether the strategy's leverage and timing add
    value beyond simple buy-and-hold.

    **Mathematical**: Given strategy returns r_s and benchmark returns r_b:
        IR = mean(r_s - r_b) / std(r_s - r_b) * sqrt(periods_per_year)
    The numerator is average excess return (alpha); denominator is tracking error.

    **Functionally**:
    - Input: pandas Series of strategy returns; pandas Series of benchmark returns;
      periods_per_year.
    - Output: scalar IR.
    - Series are aligned by index; NaNs are dropped after computing excess.
    - Typical benchmark: QQQ for a QQQ-based strategy.

    **Interpretation for TQQQ/SQQQ**:
    - Positive IR: strategy outperforms benchmark on a risk-adjusted basis.
    - Negative IR: strategy underperforms despite higher complexity/costs.
    - High IR with high tracking error: strategy takes significant active bets.

    **Edge cases**:
    - Identical returns → IR undefined (std = 0).
    - Misaligned indices will drop non-overlapping dates.

    Args:
        returns: Time series of strategy returns.
        benchmark_returns: Time series of benchmark returns (e.g., QQQ).
        periods_per_year: Number of periods per year.

    Returns:
        Information ratio as a scalar.
    """
    # Align the two series by index and compute excess returns
    # Use inner join to ensure matching dates
    aligned = pd.DataFrame({
        'strategy': returns,
        'benchmark': benchmark_returns
    }).dropna()

    # Compute excess returns (strategy - benchmark)
    excess_returns = aligned['strategy'] - aligned['benchmark']

    # Mean excess return per period
    mean_excess = excess_returns.mean()

    # Tracking error (std of excess returns)
    tracking_error = excess_returns.std(ddof=1)

    # Handle case where tracking error = 0 (identical returns)
    # Use tolerance check instead of exact equality for floating-point safety
    if tracking_error < 1e-10 or np.isnan(tracking_error):
        return np.nan

    # Information Ratio: annualized mean excess / annualized tracking error
    return (mean_excess / tracking_error) * np.sqrt(periods_per_year)


def compute_beta_and_correlation(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> tuple[float, float]:
    """
    Compute beta and correlation of strategy returns relative to benchmark.

    **Conceptual**: Beta measures systematic risk (sensitivity to benchmark moves),
    answering "if the benchmark moves 1%, how much does my strategy move?"
    Correlation measures the strength of co-movement. For TQQQ/SQQQ strategies,
    beta >> 1 is expected due to 3x leverage. Correlation shows how tightly the
    strategy tracks regime changes in QQQ.

    **Mathematical**:
        Beta = Cov(r_s, r_b) / Var(r_b)
        Correlation = Corr(r_s, r_b)
    where r_s = strategy returns, r_b = benchmark returns.

    **Functionally**:
    - Input: pandas Series of strategy returns; pandas Series of benchmark returns.
    - Output: tuple (beta, correlation), both scalars.
    - Series are aligned and deduplicated by index; NaNs dropped.
    - Beta > 1: strategy is more volatile than benchmark (expected for leveraged).
    - Correlation near 1: tight coupling; near 0: independent; negative: inverse.

    **Interpretation for TQQQ/SQQQ**:
    - Beta ~3: expected for TQQQ if always long; beta ~-3 for SQQQ if always short.
    - Beta between -3 and +3 with switching: trend-following is working.
    - High correlation: strategy is regime-coupled to QQQ (good for trend-following).

    **Edge cases**:
    - Constant benchmark returns → Var(r_b) = 0 → beta undefined.
    - Constant strategy returns → correlation undefined.

    Args:
        returns: Time series of strategy returns.
        benchmark_returns: Time series of benchmark returns (e.g., QQQ).

    Returns:
        Tuple of (beta, correlation) as floats.
    """
    # Align the two series by index
    aligned = pd.DataFrame({
        'strategy': returns,
        'benchmark': benchmark_returns
    }).dropna()

    # Extract aligned columns
    strat_ret = aligned['strategy']
    bench_ret = aligned['benchmark']

    # Compute covariance and variance
    covariance = strat_ret.cov(bench_ret)
    benchmark_variance = bench_ret.var(ddof=1)

    # Compute correlation
    correlation = strat_ret.corr(bench_ret)

    # Compute beta
    if benchmark_variance == 0:
        beta = np.nan  # Undefined if benchmark has no variance
    else:
        beta = covariance / benchmark_variance

    return beta, correlation


def compute_hit_rate(returns: pd.Series) -> float:
    """
    Compute the hit rate: fraction of periods with positive returns.

    **Conceptual**: Hit rate (or win rate) is the percentage of winning periods,
    answering "how often do I make money?" For traders, this is a simple but
    important psychological metric. However, it must be interpreted alongside
    win/loss ratio: a strategy can have low hit rate but high win/loss ratio
    (few big wins, many small losses) and still be profitable.

    **Mathematical**:
        Hit Rate = count(r_t > 0) / count(non-null r_t)

    **Functionally**:
    - Input: pandas Series of periodic returns.
    - Output: scalar float between 0 and 1 (e.g., 0.55 = 55% win rate).
    - NaNs are dropped before calculation.

    **Interpretation for TQQQ/SQQQ**:
    - Leveraged trend-following may have hit rate < 0.5 (many small losses in chop).
    - Interpret with win/loss ratio: if winners are much larger than losers,
      low hit rate can still be profitable.
    - Compare to benchmark hit rate to assess strategy style.

    **Edge cases**:
    - All positive returns → hit rate = 1.0.
    - All negative returns → hit rate = 0.0.
    - Empty series → undefined (NaN).

    Args:
        returns: Time series of periodic returns.

    Returns:
        Hit rate as a scalar between 0 and 1.
    """
    # Drop NaNs
    clean_returns = returns.dropna()

    # Count positive returns
    num_wins = (clean_returns > 0).sum()

    # Total number of periods
    total_periods = len(clean_returns)

    # Handle empty series
    if total_periods == 0:
        return np.nan

    # Hit rate = wins / total
    return num_wins / total_periods


def compute_win_loss_ratio(returns: pd.Series) -> float:
    """
    Compute the win/loss ratio: average win magnitude vs average loss magnitude.

    **Conceptual**: The win/loss ratio (or reward/risk ratio) compares the size
    of winning periods to losing periods, answering "when I win, how much do I
    win relative to when I lose?" A ratio > 1 means wins are larger than losses
    on average. For traders, this is crucial for understanding whether a low hit
    rate strategy can still be profitable (big wins offset many small losses).

    **Mathematical**:
        Win/Loss Ratio = mean(r_t | r_t > 0) / |mean(r_t | r_t < 0)|

    **Functionally**:
    - Input: pandas Series of periodic returns.
    - Output: scalar float (e.g., 1.5 means wins are 1.5x larger than losses).
    - NaNs are dropped.
    - If there are no losses, conventionally return +inf or NaN (here +inf).
    - If there are no wins, ratio = 0.

    **Interpretation for TQQQ/SQQQ**:
    - Win/loss ratio > 1 with hit rate < 0.5: classic trend-following (cut losses, let winners run).
    - Win/loss ratio < 1 with hit rate > 0.5: many small wins, few big losses (mean-reversion style).
    - Use alongside hit rate to understand strategy behavior.

    **Edge cases**:
    - No losses → win/loss ratio = +inf.
    - No wins → win/loss ratio = 0.
    - All returns = 0 → undefined.

    Args:
        returns: Time series of periodic returns.

    Returns:
        Win/loss ratio as a scalar.
    """
    # Drop NaNs
    clean_returns = returns.dropna()

    # Separate wins and losses
    wins = clean_returns[clean_returns > 0]
    losses = clean_returns[clean_returns < 0]

    # Compute average win and average loss
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    # Win/loss ratio
    if avg_loss == 0:
        # No losses or avg loss is zero
        if avg_win > 0:
            return np.inf  # All wins, no losses
        else:
            return np.nan  # No wins, no losses (flat)
    else:
        return avg_win / abs(avg_loss)


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Compute rolling Sharpe ratio over a moving window.

    **Conceptual**: Rolling Sharpe shows how risk-adjusted performance evolves
    over time, answering "is my strategy's Sharpe improving or degrading?"
    For traders, this helps detect regime shifts, parameter drift, or strategy
    degradation (e.g., a previously high-Sharpe strategy may degrade as markets change).

    **Mathematical**: At each time t, compute Sharpe over the last `window` periods:
        Sharpe_t = (mean(r_{t-window+1:t}) - rf/periods_per_year) / std(r_{t-window+1:t}) * sqrt(periods_per_year)

    **Functionally**:
    - Input: pandas Series of returns; window size; risk_free_rate; periods_per_year.
    - Output: pandas Series of rolling Sharpe values, aligned to input index.
    - First (window - 1) values are NaN until window fills.
    - Useful for detecting when to halt or re-optimize a strategy.

    **Interpretation for TQQQ/SQQQ**:
    - Declining rolling Sharpe: strategy may be facing unfavorable regime.
    - Stable or increasing rolling Sharpe: strategy is adapting well.
    - Use to trigger re-optimization or position size adjustments.

    **Edge cases**:
    - Window too small → noisy Sharpe.
    - Constant returns over window → Sharpe undefined (NaN).

    Args:
        returns: Time series of periodic returns.
        window: Number of periods for rolling window.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Number of periods per year.

    Returns:
        Time series of rolling Sharpe ratios, same index as input.
    """
    # Risk-free rate per period
    rf_per_period = risk_free_rate / periods_per_year

    # Rolling mean of returns
    rolling_mean = returns.rolling(window=window).mean()

    # Rolling std of returns (ddof=1 for sample std)
    rolling_std = returns.rolling(window=window).std(ddof=1)

    # Excess return per period (mean - rf_per_period)
    excess_return = rolling_mean - rf_per_period

    # Rolling Sharpe: (excess_return / rolling_std) * sqrt(periods_per_year)
    # Handle division by zero by replacing with NaN where std is close to zero
    # Use np.where to avoid division by zero warnings
    rolling_sharpe = np.where(
        rolling_std < 1e-10,
        np.nan,
        (excess_return / rolling_std) * np.sqrt(periods_per_year)
    )

    return pd.Series(rolling_sharpe, index=returns.index)


def compute_rolling_volatility(
    returns: pd.Series,
    window: int,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Compute rolling annualized volatility over a moving window.

    **Conceptual**: Rolling volatility shows how risk (return variability) evolves
    over time, answering "is the market getting choppier or calmer?" For traders
    using leverage, rising volatility often signals the need to reduce position
    size to maintain constant risk exposure (volatility targeting).

    **Mathematical**: At each time t, compute annualized std over last `window` periods:
        Vol_t = std(r_{t-window+1:t}, ddof=1) * sqrt(periods_per_year)

    **Functionally**:
    - Input: pandas Series of returns; window size; periods_per_year.
    - Output: pandas Series of rolling volatility, aligned to input index.
    - First (window - 1) values are NaN.
    - Helps decide dynamic sizing or de-leveraging triggers.

    **Interpretation for TQQQ/SQQQ**:
    - Rising rolling vol: markets are volatile, consider reducing leverage.
    - Falling rolling vol: markets are calm, potentially safe to increase leverage.
    - Use for volatility-targeting position sizing.

    **Edge cases**:
    - Constant returns → vol = 0.
    - Window too small → noisy vol estimates.

    Args:
        returns: Time series of periodic returns.
        window: Number of periods for rolling window.
        periods_per_year: Number of periods per year.

    Returns:
        Time series of rolling annualized volatility, same index as input.
    """
    # Rolling standard deviation (sample std with ddof=1)
    rolling_std = returns.rolling(window=window).std(ddof=1)

    # Annualize by multiplying by sqrt(periods_per_year)
    rolling_vol = rolling_std * np.sqrt(periods_per_year)

    return rolling_vol
