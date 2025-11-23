# Developer Notes

## Phase 1 – Scaffolding & Docs
We start with structure and documentation to lock in boundaries early, make imports predictable, and avoid accidental coupling as logic arrives. A clear skeleton de-risks later refactors, keeps core functions stateless, and smooths migration to AWS batch/containers by separating I/O, orchestration, and pure computation from day one.

### Decisions and Assumptions
- Use the spec’s directory layout verbatim (docs, config, data/{raw,processed,results}, state, src subpackages, actions, tests).
- Initialize Python packages with `__init__.py` so modules resolve cleanly across src/*.
- Keep `main.py`/action stubs minimal; no strategy, backtest engine, or venue logic in Phase 1.
- Favor CSV contracts and explicit state folders even before logic exists to guide future I/O patterns.
- AWS readiness is architectural: stateless core + explicit state files; no cloud-specific code yet.
- Testing folder present but empty; real tests arrive with utilities/logic in later phases.

## Phase 2 – Core utilities design

Phase 2 goal (core utilities):
- Establish math, risk, time, and synthetic data primitives that are correct, teachable, and reusable so later engines/strategies plug into stable APIs.
- Getting these right now reduces downstream refactors, keeps backtests reproducible, and ensures risk/metrics are trustworthy for leveraged QQQ/TQQQ/SQQQ workflows before any strategy code exists.

### src/utils/math.py – API design (NumPy/Pandas friendly, blog-ready docstrings)

```python
def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """
    Concept: Converts a price series into simple returns, answering “what was the percent change from one period to the next?”
    Math: r_t = (P_t / P_{t-1}) - 1 for each period t.
    Usage: Input is prices indexed in time order (daily bars expected). Outputs a return series aligned to prices (first value is NaN). Assumes no missing prices; forward/backfill should be handled upstream. Used later for daily PnL streams feeding Sharpe/Sortino and rolling stats in QQQ/TQQQ/SQQQ tests.
    """

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Concept: Log returns accumulate additively and stabilize variance for compounding series.
    Math: r_t = ln(P_t) - ln(P_{t-1}).
    Usage: Input daily prices; output log return series (NaN at first index). Prefer log returns for drift/volatility estimation and velocity calculations. Missing data should be cleaned before calling.
    """

def compute_moving_average_simple(prices: pd.Series, window: int) -> pd.Series:
    """
    Concept: Smooths noise with an equal-weighted moving average.
    Math: SMA_t = (1/window) * sum_{i=0}^{window-1} P_{t-i}.
    Usage: Input price or feature series; output SMA aligned with input (NaNs until window fills). Used for trend context (e.g., 50/100/250-day levels) before strategy logic exists.
    """

def compute_moving_average_exponential(prices: pd.Series, span: int) -> pd.Series:
    """
    Concept: Smooths with more weight on recent data for faster responsiveness.
    Math: EMA via pandas ewm(span=span, adjust=False), equivalent to recursive EMA_t = α*P_t + (1-α)*EMA_{t-1}, α=2/(span+1).
    Usage: Input prices/features; output EMA (warm-up NaNs as per pandas). Helpful for responsive signals and later regime detection without introducing lag-heavy behavior.
    """

def compute_rolling_standard_deviation(returns: pd.Series, window: int) -> pd.Series:
    """
    Concept: Measures dispersion of returns over a rolling window.
    Math: Std dev over last `window` returns (pandas .rolling(window).std(ddof=1)).
    Usage: Input return series; output rolling std (NaNs until window fills). Forms the basis for rolling volatility and position scaling later. Assumes cleaned return data.
    """

def compute_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Concept: Scales return variability to an annualized figure for comparability.
    Math: vol_annualized = std(returns, ddof=1) * sqrt(periods_per_year).
    Usage: Input periodic returns (daily by default). Outputs a scalar float. Used for Sharpe/Sortino denominators and risk budgeting. Ensure returns frequency matches periods_per_year.
    """

def compute_velocity(prices: pd.Series, window: int, use_log: bool = True) -> pd.Series:
    """
    Concept: Captures slope/trend strength over a lookback, answering “how steeply are prices moving?”
    Math: Fit a simple linear regression of (log_prices if use_log else prices) vs. time index over each rolling window; velocity is the slope coefficient per period.
    Usage: Input price series; output velocity series (NaNs until window fills). Daily frequency assumed; slope units are price-change per day (or log-price per day). Later used to gauge trend strength for TQQQ/SQQQ allocation logic.
    """

def compute_acceleration(prices: pd.Series, window: int, use_log: bool = True) -> pd.Series:
    """
    Concept: Captures change in velocity over time, answering “is the trend speeding up or slowing down?”
    Math: Compute velocity over the same window, then take first differences: accel_t = velocity_t - velocity_{t-1}.
    Usage: Input price series; output acceleration aligned to velocity (extra initial NaN). Daily cadence assumed. Useful for distinguishing strong vs tiring trends before entering leverage.
    """
```

### src/analytics/risk_metrics.py – curated metrics (leveraged trend-following focus)

Metric groups matter because:
- Core performance: risk/return framing for leveraged ETFs.
- Drawdown/pain: protects against sharp losses and whipsaws common in trend strategies.
- Relative: benchmark vs QQQ to judge value-add beyond the underlying.
- Distributional: assess hit patterns for path-dependency and behavior under leverage.
- Rolling: time-varying health checks to catch regime shifts.
Excluded for now: VaR/CVaR (needs heavier distributional assumptions), skew/kurtosis (lower priority for initial QQQ/TQQQ/SQQQ validation), MAR beyond Calmar (Calmar suffices for CAGR vs drawdown).

```python
def compute_total_return(equity_curve: pd.Series) -> float:
    """
    Concept: Overall ROI from start to end of the equity curve.
    Math: (equity_final / equity_initial) - 1.
    Usage: Input equity indexed by date; output scalar float. Assumes positive starting equity. Baseline for interpreting leveraged performance vs QQQ.
    """

def compute_cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Concept: Annualized growth rate accounting for compounding.
    Math: CAGR = (equity_final / equity_initial) ** (periods_per_year / n_periods) - 1.
    Usage: Input daily equity; output scalar. Sensitive to start/end selection; useful for Calmar and ROI comparisons against QQQ buy-and-hold.
    """

def compute_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Concept: Annualized return variability; reused from math helpers.
    Math: std(returns, ddof=1) * sqrt(periods_per_year).
    Usage: Input periodic returns; output scalar. Align periods_per_year with data frequency. Key for Sharpe in leveraged contexts.
    """

def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Concept: Excess return per unit of total volatility.
    Math: Sharpe = ((mean(returns) - risk_free_rate/periods_per_year) / std(returns)) * sqrt(periods_per_year).
    Usage: Input periodic returns; output scalar. NaNs dropped. Higher is better; leveraged ETFs may show higher volatility so Sharpe contextualizes risk-adjusted edge.
    """

def compute_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Concept: Excess return per unit of downside volatility (penalizes only negative periods).
    Math: Sortino = ((mean(returns) - risk_free_rate/periods_per_year) / std(returns[returns < 0])) * sqrt(periods_per_year).
    Usage: Input returns; output scalar. NaNs dropped; if no negative returns, conventionally return +inf or handle gracefully. Useful when upside volatility from leverage should not be punished.
    """

def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Concept: Pathwise pain measurement showing % drop from each prior peak.
    Math: drawdown_t = (equity_t / cumulative_peak_t) - 1, where cumulative_peak_t = max(equity_0..t).
    Usage: Input equity series; output drawdown series (starts at 0). Highlights whipsaw risk in leveraged trends.
    """

def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Concept: Worst peak-to-trough loss over the period.
    Math: min(drawdown_series).
    Usage: Input equity; output scalar (<=0). Critical for sizing leverage and for Calmar denominator.
    """

def compute_calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Concept: Return-to-pain ratio contrasting CAGR vs max drawdown.
    Math: Calmar = CAGR(equity) / abs(max_drawdown).
    Usage: Input equity; output scalar. Higher is better; valuable for leveraged ETFs where drawdowns can be sharp.
    """

def compute_information_ratio(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Concept: Excess return vs benchmark per unit of tracking error.
    Math: IR = mean(returns - benchmark_returns) / std(returns - benchmark_returns), often annualized via sqrt(periods_per_year).
    Usage: Align and drop NaNs on both series. Benchmark typically QQQ. Shows value-add beyond the base ETF despite leverage costs.
    """

def compute_beta_and_correlation(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    """
    Concept: Sensitivity and co-movement vs benchmark.
    Math: Beta = cov(returns, benchmark) / var(benchmark); corr = correlation coefficient.
    Usage: Align/dedup indices; output (beta, corr). For leveraged strategies, beta >> 1 is expected; corr shows regime coupling.
    """

def compute_hit_rate(returns: pd.Series) -> float:
    """
    Concept: Fraction of periods that are profitable.
    Math: hit_rate = count(returns > 0) / count(non-null returns).
    Usage: Input returns; output scalar between 0 and 1. High leverage can have lower hit rate but larger winners; interpret with win/loss ratio.
    """

def compute_win_loss_ratio(returns: pd.Series) -> float:
    """
    Concept: Average win magnitude vs average loss magnitude.
    Math: mean(returns[returns > 0]) / abs(mean(returns[returns < 0])).
    Usage: Input returns; output scalar. If no losses, handle safely (e.g., return +inf). Helps see whether fewer but bigger wins justify drawdowns.
    """

def compute_rolling_sharpe(returns: pd.Series, window: int, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> pd.Series:
    """
    Concept: Time-varying risk-adjusted performance.
    Math: Sharpe over each rolling window; annualized by sqrt(periods_per_year) using window mean/std.
    Usage: Input returns; output series aligned to window end. NaNs until window fills. Useful to detect regime drift or degradation.
    """

def compute_rolling_volatility(returns: pd.Series, window: int, periods_per_year: int = 252) -> pd.Series:
    """
    Concept: Time-varying volatility profile.
    Math: rolling std * sqrt(periods_per_year).
    Usage: Input returns; output series aligned to window end. Helps decide dynamic sizing or de-leveraging triggers later.
    """
```

### src/utils/time.py – clock abstraction

- Approach: A lightweight `Clock` protocol/class with two constructors: `RealClock()` and `FrozenClock(fixed_now: datetime)`. Consumers depend on `clock.now()` instead of `datetime.utcnow()`.
- Signatures and docstrings:

```python
from datetime import datetime, timezone

class Clock:
    def now(self) -> datetime:
        """
        Concept: Abstract time source so code can ask “what time is it?” without binding to system time.
        Usage: Subclasses implement now(); callers inject a clock (real or frozen) into workflows, tests, and backtests.
        """

class RealClock(Clock):
    def now(self) -> datetime:
        """
        Returns the current UTC time (datetime.now(timezone.utc)).
        """

class FrozenClock(Clock):
    def __init__(self, fixed_now: datetime):
        """
        Store a fixed timestamp to return on every now() call.
        """
    def now(self) -> datetime:
        """
        Returns the configured fixed timestamp, enabling deterministic tests/backtests (e.g., simulating “today” in 2015).
        """

def get_real_clock() -> Clock:
    """Factory for a RealClock instance."""

def get_frozen_clock(fixed_now: datetime) -> Clock:
    """Factory for a FrozenClock with the given fixed timestamp."""
```

- Usage example:
```python
clock = get_frozen_clock(datetime(2015, 1, 5, tzinfo=timezone.utc))
timestamp = clock.now()  # always 2015-01-05T00:00:00Z
```
Rationale: Deterministic timestamps for metrics, filenames, and logs in backtests; easy swap for real time in production.

### src/analytics/synthetic_data.py – synthetic market data generators

```python
def generate_gbm_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    n_steps: int,
    dt: float = 1/252,
    seed: int | None = None,
) -> pd.Series:
    """
    Concept: Geometric Brownian Motion for trending, compounding price behavior (classic equity model).
    Math: dS = μS dt + σS dW; discrete update S_{t+1} = S_t * exp((μ - 0.5σ^2)dt + σ sqrt(dt) * Z_t), Z~N(0,1).
    Usage: Returns a pd.Series of length n_steps+1 (including initial) indexed by step number. Useful for testing returns/volatility metrics and path-dependent logic under controlled drift/vol regimes.
    """

def generate_ou_paths(
    initial_price: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
    n_steps: int,
    dt: float = 1/252,
    seed: int | None = None,
) -> pd.Series:
    """
    Concept: Ornstein–Uhlenbeck mean-reverting process for choppy/sideways regimes.
    Math: dX = κ(θ - X) dt + σ dW; discrete update X_{t+1} = X_t + κ(θ - X_t)dt + σ sqrt(dt) * Z_t.
    Usage: Returns a pd.Series of length n_steps+1 indexed by step. Useful to stress mean-reversion vs trend metrics and to verify drawdown/vol calculations in non-trending conditions.
    """
```
Note: These generators let us validate backtesting plumbing and risk metrics under known regimes (pure trend vs mean reversion) before touching live data.

### Tests for Phase 2

- `tests/test_utils_math.py`: Verify simple/log returns on tiny price arrays; SMA/EMA vs hand-calculated; rolling std/annualized vol on constant and known series; velocity/acceleration on synthetic linear (positive slope) and quadratic (increasing slope) price paths.
- `tests/test_analytics_risk_metrics.py`: Constant-return series (Sharpe/Sortino predictable), monotonic equity (drawdown 0), single drop then recovery (max drawdown known), benchmark comparisons where returns differ by fixed spread (information ratio, beta), hit rate/win-loss on crafted returns.
- `tests/test_utils_time.py`: RealClock returns near `datetime.now(timezone.utc)`; FrozenClock returns fixed timestamp; swapping clocks leaves dependent code deterministic.
- `tests/test_analytics_synthetic_data.py`: GBM with zero volatility stays flat; GBM with known seed matches expected length and positive drift trend; OU with strong mean reversion stays near mean; both honor seed reproducibility.

### Summary for implementation

- Modules to implement: `src/utils/math.py`, `src/analytics/risk_metrics.py`, `src/utils/time.py`, `src/analytics/synthetic_data.py`.
- Test files to implement: `tests/test_utils_math.py`, `tests/test_analytics_risk_metrics.py`, `tests/test_utils_time.py`, `tests/test_analytics_synthetic_data.py`.
- Key design choices: blog-ready docstrings for teaching; velocity via rolling linear slope on (log) prices; acceleration as delta of velocity; curated metrics focused on leveraged trend ETFs (excluded VaR/CVaR for simplicity now); clock abstraction via injectable class for determinism; synthetic generators (GBM, OU) to validate metrics/vol/drawdown behavior before engines/strategies exist.
