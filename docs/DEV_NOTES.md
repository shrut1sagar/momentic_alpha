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

## Phase 3 – Data IO and canonical schemas

### 1. Phase 3 goal

- Implement robust CSV I/O and schema validation for raw, processed, and results data so every data touchpoint is consistent and enforceable.
- Strict, explicit schemas with actionable errors are essential for a CSV-first backtesting system and for AWS portability, where deterministic contracts and clear remediation keep pipelines reliable.

### Documentation and commenting standard for Phase 3

- All `src/data/...` functions must be teaching material, as in Phase 2:
  - Docstrings explain conceptually (e.g., “load a QQQ price history CSV and enforce schema”), functionally (inputs, outputs, types, assumptions such as daily frequency), and what errors can be raised.
  - Inline comments for non-trivial logic, especially schema checks, timestamp parsing, and sort-order enforcement.
- Error messages must be human-readable and actionable, e.g., “missing required column `closing_price` in data/raw/QQQ.csv; expected columns: …”.

### src/data/schemas.py – canonical schemas and validation helpers

- Purpose: define canonical CSV schemas and validate DataFrames against them.
- Schema concepts:
  1) Raw price schema (`data/raw/QQQ.csv`, `TQQQ.csv`, `SQQQ.csv`, `UVXY.csv`):
     - Required columns: `timestamp`, `open_price`, `high_price`, `low_price`, `closing_price`, `volume`.
     - `timestamp`: ISO 8601 date-time strings.
     - Rows must be strictly decreasing by `timestamp` (newest first).
  2) Processed feature schema (`data/processed/...`):
     - Must include `timestamp` and `closing_price` at minimum.
     - Additional feature columns (e.g., `daily_return`, `moving_average_50`, `volatility_20d`, `velocity`, `acceleration`) are human-readable, snake_case, descriptive per spec.
- Proposed structures: simple dicts or TypedDict/dataclasses capturing required columns and types.
- Helper functions (signatures only):
  - `validate_raw_price_schema(df: pd.DataFrame, context: str) -> None`
  - `validate_processed_schema(df: pd.DataFrame, required_features: list[str] | None = None, context: str | None = None) -> None`
- Helper behavior:
  - Check required columns present.
  - Ensure `timestamp` exists and is parseable as datetime.
  - Ensure strict descending order by `timestamp` (raise if broken).
  - Raise clear, actionable errors including `context` (file/instrument path/name).
- Usage: called by `io.py` readers to fail fast with remediation guidance.

### src/data/io.py – CSV readers/writers with schema enforcement

- Functions:
  - `read_raw_price_csv(path: Path | str, instrument_name: str | None = None) -> pd.DataFrame`
  - `write_raw_price_csv(df: pd.DataFrame, path: Path | str) -> None`
  - `read_processed_data_csv(path: Path | str, instrument_name: str | None = None) -> pd.DataFrame`
  - `write_processed_data_csv(df: pd.DataFrame, path: Path | str) -> None`
- Reading behavior:
  - Parse `timestamp` to datetime.
  - Enforce strict descending order by `timestamp` (either sort or validate/raise per design choice; aim to validate and raise unless explicitly allowed to sort).
  - Call schema validators from `schemas.py`.
  - Raise clear exceptions on failure with file/instrument context.
- Writing behavior:
  - Convert `timestamp` back to ISO strings.
  - Ensure strict descending order by `timestamp`.
  - Ensure column naming/ordering follows conventions.
- All CSV access points should use these functions instead of ad-hoc `pd.read_csv`.
- Consider a custom `SchemaValidationError` (either in `schemas.py` or shared `src/utils/errors.py`) for clarity.

### src/data/loaders.py – instrument-specific convenience loaders

- Functions such as:
  - `load_instrument_history(symbol: str) -> pd.DataFrame`
  - Specific helpers: `load_qqq_history()`, `load_tqqq_history()`, `load_sqqq_history()`, `load_uvxy_history()`.
- Behavior:
  - Resolve paths under `data/raw/` (e.g., `data/raw/QQQ.csv`) without hardcoding paths in strategies.
  - Delegate to `read_raw_price_csv` for validation/sorting.
  - Optionally: `load_default_universe() -> dict[str, pd.DataFrame]` for common instruments.
- Later use: orchestration/backtesting (e.g., Strategy 1 QQQ long/short) can import clean, validated DataFrames without duplicating path logic.

### Strategy 1 data requirements (QQQ/TQQQ/SQQQ/UVXY)

- Strategy 1 uses QQQ price history as the reference for features (velocity, acceleration, moving averages) and TQQQ/SQQQ/UVXY for execution.
- Data alignment:
  - All series share the same `timestamp` column and strict descending order.
  - Missing dates/gaps should be detected and handled explicitly (join/intersection/forward-fill decisions in later phase).
- Phase 3 IO + schema enforcement should make it easy for Strategy 1 and the backtest engine to obtain clean, aligned DataFrames and surface missing-day issues early.

### Tests for Phase 3

- `tests/test_data_io.py`:
  - Round-trip read/write of small synthetic DataFrame following raw schema.
  - Enforcement of strict descending timestamps.
  - Detection of missing required columns (e.g., missing `closing_price`).
  - Detection of non-ISO `timestamp` values or duplicate timestamps.
  - Clear error messages including file/instrument context.
- Future: tests against sample files in `data/raw`; behavior when writing processed feature CSVs.

### Summary

- Modules to implement: `src/data/schemas.py`, `src/data/io.py`, `src/data/loaders.py`.
- Test file: `tests/test_data_io.py`.
- Key decisions: all CSVs must have timestamps in strict descending order; all CSV access must go through `src/data/io.py` to guarantee schema enforcement; instrument-specific loaders in `loaders.py` keep strategy code clean and venue-agnostic.

## Phase 4 – Backtesting engine and paper broker

### 1. Phase 4 goal

- Introduce a daily-bar backtest engine that iterates through time, calls a strategy, and uses a broker to simulate fills and portfolio evolution.
- Implement a paper broker responsible for cash, positions, fills, slippage, and fees.
- Keep the design generic enough for multiple strategies and future venues, initially tested with toy strategies and CSV data from Phase 3.

### Documentation and commenting standard for Phase 4

- Every public class and function must have a docstring explaining:
  - Conceptually: the role in the backtest.
  - Mathematically/financially: assumptions (PnL computation, slippage model).
  - Functionally: inputs, outputs, side effects, invariants (portfolio consistency).
- Inline comments for non-trivial logic: order application, position updates, equity curve computation.
- Code must be “blog-ready” so the engine and broker can be educational artifacts.

### src/strategies/base.py – strategy interface

- Define a minimal Strategy protocol/ABC invoked by the engine.
- Proposed interface:
  ```python
  class Strategy(Protocol):
      def generate_signals(
          self,
          data: pd.DataFrame,
          dt: pd.Timestamp,
          portfolio_state: PortfolioState | None = None,
      ) -> dict[str, float]:
          """
          Return target weights per symbol (or signals) for the current dt.
          data: row/window of features/prices for dt.
          portfolio_state: current holdings/cash (may be None for stateless strategies).
          """
  ```
- Inputs: current date, current data row/window, current portfolio state, config as needed.
- Outputs (Phase 4): target weights per instrument (simpler than explicit orders); trivial test strategies (always-cash, always-long) for validation.

### src/execution/paper_broker.py – paper broker and portfolio model

- Paper broker API:
  - Accept target weights per instrument (Phase 4 default; explicit orders can come later).
  - Maintain cash, positions, portfolio market value, equity over time.
- Core components:
  - `Position`/`Holding` dataclass.
  - `PortfolioState` dataclass (cash, positions, equity, timestamp, optional exposure metrics).
  - `PaperBroker` methods:
    - `update_prices(prices: dict[str, float], dt: pd.Timestamp) -> None`
    - `set_target_weights(targets: dict[str, float], dt: pd.Timestamp) -> None`
    - `step() -> PortfolioState` (apply trades implied by target weights at current prices, update cash/positions, record snapshot).
- Assumptions:
  - Trading at daily close.
  - Slippage/fees configurable (bps of notional); can be stubbed initially.
  - Short selling optional; if allowed, define simple rules (e.g., allow negatives with no margin modeling in Phase 4 or start with long-only).

### src/backtesting/engine.py – daily-bar backtest engine

- Engine responsibilities:
  - Inputs: data (dict[symbol -> DataFrame]), strategy, broker, backtest params (start/end, initial capital, config).
  - Iterate dates in ascending order.
  - For each step: build context (dt, data slice, portfolio state), call strategy for target weights, pass to broker, receive updated `PortfolioState`.
  - Record equity curve and optional diagnostics (gross/net exposure).
- Proposed entrypoint:
  ```python
  def run_backtest(
      data: dict[str, pd.DataFrame],
      strategy: Strategy,
      broker: PaperBroker,
      params: BacktestParams,
  ) -> BacktestResult:
      ...
  ```
  - `BacktestParams` dataclass: start/end, initial_capital, fees/slippage settings, maybe rebalance frequency.
  - `BacktestResult` dataclass: equity curve (pd.Series), trades/allocations log, diagnostics, metrics.
- Post-run: call `src/analytics/risk_metrics.py` to compute metrics; writing outputs to `data/results/` can be added here or in later orchestration.

### Tests for Phase 4

- `tests/test_execution_paper_broker.py`:
  - From all cash, target 100% single symbol → expected position size and cash after one step.
  - Equity updates with price moves as expected.
  - Optional: simple fee/slippage handling works.
- `tests/test_backtesting_engine.py`:
  - Trivial strategy always 0% exposure → equity flat.
  - Trivial strategy always 100% long on monotonic price series → equity monotonic.
  - Engine respects start/end dates and runs expected number of steps.

### Summary

- Modules to implement: `src/strategies/base.py`, `src/execution/paper_broker.py`, `src/backtesting/engine.py`.
- Tests to implement: `tests/test_execution_paper_broker.py`, `tests/test_backtesting_engine.py`.
- Key decisions: use target weights (not raw orders) for simplicity; assume daily close execution; keep strategy, broker, and engine concerns separated for clarity and extensibility.

### Phase 4 Implementation summary

**Status**: Phase 4 complete. Backtest engine, paper broker, and strategy interface implemented with full test coverage.

**What was implemented**:

1. **src/strategies/base.py** (267 lines)
   - `Strategy` protocol defining the interface for all strategies
     - Method: `generate_target_weights(dt, data, portfolio_state) -> dict[str, float]`
     - Receives current timestamp, historical data, and optional portfolio state
     - Returns target weights per symbol (fractions of total equity)
   - `AlwaysCashStrategy`: Trivial strategy holding 100% cash (for testing)
   - `AlwaysLongStrategy`: Simple strategy allocating 100% to a single symbol
   - `BalancedStrategy`: Equal-weight allocation across multiple symbols
   - All with extensive teaching-style docstrings explaining:
     - Why use target weights instead of explicit orders
     - How strategies receive data without time-travel bugs
     - Path-dependency considerations (portfolio_state parameter)

2. **src/execution/paper_broker.py** (383 lines)
   - `Position` dataclass: Tracks symbol, quantity, last_price
   - `PortfolioState` dataclass: Snapshot of portfolio at a timestamp
     - Fields: timestamp, cash, positions, prices, equity, gross_exposure, net_exposure
   - `PaperBroker` class: Simulates broker execution
     - Constructor: `__init__(initial_cash, slippage_bps, fee_per_trade)`
     - `update_prices(prices, dt)`: Update price map without trading
     - `set_target_weights(target_weights)`: Execute trades to achieve targets
       - Converts target weights to dollar values based on current equity
       - Computes required trades (target - current)
       - Applies slippage symmetrically (buys pay more, sells receive less)
       - Charges flat fee per trade
       - Updates cash and positions
     - `get_portfolio_state()`: Return current portfolio snapshot
     - `_execute_trade(symbol, quantity)`: Internal method applying costs
   - Financial assumptions clearly documented:
     - Daily close execution (all trades at last price)
     - Symmetric slippage (basis points on notional)
     - Flat fees per trade
     - Fractional shares allowed (simplifies math)
     - Shorting allowed (negative positions, no margin modeling)
   - Extensive docstrings explaining:
     - Why separate price updates from trading
     - How P&L is computed
     - Target weight semantics and execution logic

3. **src/backtesting/engine.py** (276 lines)
   - `BacktestParams` dataclass: Configuration for backtest runs
     - Fields: start_date, end_date, initial_cash, slippage_bps, fee_per_trade
   - `BacktestResult` dataclass: Output from backtest
     - Fields: equity_curve, portfolio_history, metrics, params
   - `run_backtest(data, strategy, params)`: Main backtest entrypoint
     - Creates PaperBroker from params
     - Determines backtest date range from data and params
     - Iterates forward through time (ascending dates, prevents time travel)
     - For each date:
       - Extracts current closing prices
       - Builds data slice (history up to current date only)
       - Gets current portfolio state
       - Calls strategy for target weights
       - Passes targets to broker
       - Records updated state
     - Post-processing:
       - Builds equity curve from portfolio history
       - Computes metrics using Phase 2 risk_metrics functions:
         - total_return, cagr, sharpe_ratio, sortino_ratio
         - max_drawdown, calmar_ratio
         - final_equity, num_trading_days
     - Returns BacktestResult with all outputs
   - Extensive docstrings explaining:
     - Why iterate forward (simulate real-time, prevent time travel)
     - Data alignment and time-travel prevention mechanisms
     - How equity curve and metrics are computed

4. **tests/test_execution_paper_broker.py** (346 lines, 14 tests)
   - Basic allocation scenarios:
     - Full allocation to single symbol
     - Partial allocation (cash remainder)
     - Multiple symbols
   - Price movement scenarios:
     - Equity increases with price increases
     - Equity decreases with price decreases
   - Rebalancing tests:
     - Switch from one symbol to another
     - Zero allocation liquidates positions
   - Cost model tests:
     - Slippage applied correctly (equity reduced by slippage costs)
     - Fees charged per trade
   - Short selling tests:
     - Negative weights create short positions
     - Exposure metrics computed correctly (gross = sum of abs values)
   - Error handling:
     - No prices before trading
     - Missing price for target symbol
   - All tests use deterministic values with hand-calculated expectations

5. **tests/test_backtesting_engine.py** (396 lines, 9 tests)
   - Helper function `make_simple_price_data()`: Generate synthetic price series
   - Strategy behavior tests:
     - Always-cash strategy maintains constant equity
     - Always-long strategy tracks price movements (up and down)
     - Balanced strategy diversifies across symbols
   - Engine mechanics tests:
     - Date range filtering works correctly
     - Equity curve has expected length
   - Cost integration tests:
     - Slippage and fees reduce returns as expected
     - Daily rebalancing incurs repeated costs
   - Error handling:
     - Empty data raises error
     - Missing required columns raises error
   - Metrics validation:
     - All expected metrics computed and numeric

**Deviations from original design**:

- None. Implementation follows the Phase 4 design spec exactly.
- Added `BalancedStrategy` as an additional test fixture (equal-weight allocation).
- Slippage semantics clarified:
  - Target weights are "ideal" allocations ignoring costs
  - Slippage increases cash spent (can make cash negative)
  - This matches standard backtest behavior (costs create tracking error)
  - Alternative (adjusting quantities for costs) would require iterative solving
- Extended docstrings beyond the original API sketches to fully meet "teaching code" requirement.

**How to use Phase 4 (for new contributors)**:

1. **Define a strategy**:
   ```python
   from src.strategies.base import Strategy
   from src.execution.paper_broker import PortfolioState

   class MyStrategy:
       def generate_target_weights(
           self,
           dt: pd.Timestamp,
           data: dict[str, pd.DataFrame],
           portfolio_state: PortfolioState | None,
       ) -> dict[str, float]:
           # Your logic here
           # Example: 60% QQQ, 40% cash
           return {"QQQ": 0.6}
   ```

2. **Prepare data** (using Phase 3 loaders):
   ```python
   from src.data.loaders import load_qqq_history, load_tqqq_history

   data = {
       "QQQ": load_qqq_history(),
       "TQQQ": load_tqqq_history(),
   }
   ```

3. **Configure backtest**:
   ```python
   from src.backtesting.engine import BacktestParams
   import pandas as pd

   params = BacktestParams(
       start_date=pd.Timestamp("2020-01-01"),
       end_date=pd.Timestamp("2024-01-01"),
       initial_cash=100_000,
       slippage_bps=5.0,  # 5 bps = 0.05%
       fee_per_trade=0.0,  # commission-free
   )
   ```

4. **Run backtest**:
   ```python
   from src.backtesting.engine import run_backtest

   strategy = MyStrategy()
   result = run_backtest(data, strategy, params)

   # Inspect results
   print(result.equity_curve)
   print(result.metrics)

   # Plot equity curve
   result.equity_curve.plot(title="Backtest Equity Curve")
   ```

5. **Understanding broker behavior**:
   - Target weights are fractions of total equity: `1.0 = 100%`, `0.5 = 50%`
   - Weights can be negative for shorts: `-0.3 = 30% short`
   - Broker rebalances at every step (daily by default)
   - Slippage is applied to traded notional (buys pay more, sells receive less)
   - Fees are charged per trade (flat amount)
   - Cash can go negative if costs exceed available cash (no margin calls in Phase 4)

6. **Understanding backtest engine**:
   - Engine iterates forward through time (oldest to newest)
   - Data is filtered to prevent time travel (only data up to current date)
   - Strategy receives full historical data for each symbol
   - Trades execute at daily close prices
   - Equity curve tracks portfolio value over time
   - Metrics are computed using Phase 2 risk_metrics functions

**Key decisions and trade-offs**:

1. **Target weights vs explicit orders**: Chose target weights for simplicity. Strategies think in allocation percentages, broker handles conversion to shares. Explicit orders can be added later without breaking this interface.

2. **Slippage semantics**: Target weights are "ideal" ignoring costs. Slippage increases actual cash spent, creating tracking error. This matches standard backtest behavior and is simpler than iteratively adjusting quantities to stay within cash constraints.

3. **Daily rebalancing**: Strategy is called every day, triggering rebalancing. For low-frequency strategies, this incurs unnecessary transaction costs. Future enhancement: add rebalancing frequency parameter (weekly, monthly, threshold-based).

4. **No margin modeling**: Shorting allowed but no explicit margin requirements, interest charges, or margin calls. Cash can go negative. Sufficient for Phase 4; production would need proper margin accounting.

5. **Fractional shares**: Allowed to simplify rebalancing math. Real brokers may not support this; rounding to whole shares can create small tracking errors.

**Testing Phase 4**:

Run Phase 4 tests:
```bash
pytest tests/test_execution_paper_broker.py tests/test_backtesting_engine.py -v
```

Run all tests (Phase 2 + Phase 3 + Phase 4):
```bash
pytest tests/ -v
```

**Next steps**:

Phase 4 provides a complete backtesting framework. With this foundation, Phase 5 can implement:
- **Strategy 1 (QQQ long/short with TQQQ/SQQQ)**: Use velocity, acceleration, and moving averages from Phase 2 math utilities to generate signals
- **Feature engineering pipelines**: Read raw data (Phase 3), compute features (Phase 2 math), write processed data (Phase 3)
- **Parameter sweeps**: Run multiple backtests with different strategy parameters, compare metrics
- **Results visualization**: Plot equity curves, drawdowns, rolling Sharpe
- **Walk-forward validation**: Train/validate/test splits to avoid overfitting
- All of the above can use the Phase 4 engine and broker without modification

### Phase 2 Implementation summary

**Status**: Phase 2 complete. All core utilities and tests implemented.

**What was implemented**:

1. **src/utils/math.py** (282 lines)
   - Return calculations: `compute_simple_returns()`, `compute_log_returns()`
   - Moving averages: `compute_moving_average_simple()`, `compute_moving_average_exponential()`
   - Volatility measures: `compute_rolling_standard_deviation()`, `compute_annualized_volatility()`
   - Momentum indicators: `compute_velocity()` (via rolling linear regression on log prices), `compute_acceleration()` (first derivative of velocity)
   - All functions include extensive "teaching-style" docstrings explaining conceptual, mathematical, and functional aspects
   - Inline comments at every non-trivial line explaining the implementation

2. **src/analytics/risk_metrics.py** (612 lines)
   - Core performance: `compute_total_return()`, `compute_cagr()`, `compute_annualized_volatility()`, `compute_sharpe_ratio()`, `compute_sortino_ratio()`
   - Drawdown/pain metrics: `compute_drawdown_series()`, `compute_max_drawdown()`, `compute_calmar_ratio()`
   - Relative-to-benchmark: `compute_information_ratio()`, `compute_beta_and_correlation()`
   - Distribution/trade-style: `compute_hit_rate()`, `compute_win_loss_ratio()`
   - Rolling metrics: `compute_rolling_sharpe()`, `compute_rolling_volatility()`
   - Each function includes trading-focused interpretation notes for leveraged QQQ/TQQQ/SQQQ strategies
   - All metrics handle edge cases (zero volatility, no losses, etc.) gracefully

3. **src/utils/time.py** (118 lines)
   - `Clock` protocol defining the abstraction
   - `RealClock` class: returns actual system time via `datetime.now(timezone.utc)`
   - `FrozenClock` class: returns fixed timestamp for deterministic tests/backtests
   - Factory functions: `get_real_clock()`, `get_frozen_clock(fixed_now)`
   - Extensive documentation on why clock abstraction matters for reproducibility and testing

4. **src/analytics/synthetic_data.py** (203 lines)
   - `generate_gbm_paths()`: Geometric Brownian Motion generator for trending markets
   - `generate_ou_paths()`: Ornstein-Uhlenbeck generator for mean-reverting markets
   - Both support reproducible seeds, configurable drift/volatility/mean-reversion parameters
   - Detailed mathematical explanations of stochastic update rules in docstrings and inline comments
   - Returns pandas Series with proper indexing

5. **Test files** (4 files, 484 total lines):
   - `tests/test_utils_math.py`: 17 tests covering returns, averages, volatility, velocity/acceleration on hand-crafted data
   - `tests/test_analytics_risk_metrics.py`: 26 tests covering all risk metrics with synthetic equity curves and return series
   - `tests/test_utils_time.py`: 10 tests verifying RealClock returns current time, FrozenClock returns fixed time, factories work
   - `tests/test_analytics_synthetic_data.py`: 14 tests verifying GBM/OU produce correct lengths, honor seeds, exhibit expected behavior (zero vol, mean reversion, etc.)
   - All tests include comments explaining what is being verified

**Deviations from original design**:

- None. Implementation follows the Phase 2 design spec exactly.
- Added scipy.stats.linregress for velocity calculation (rolling linear regression), which was implied but not explicitly specified in the design.
- Extended docstrings beyond the original API sketches to fully meet the "teaching code" requirement, including edge cases, interpretation notes for leveraged strategies, and usage examples.

**How to use these utilities in later phases**:

For new contributors or when implementing Phase 3+ (backtesting engines, strategies, orchestration):

1. **Math utilities** (`src/utils/math.py`):
   - Use `compute_simple_returns()` or `compute_log_returns()` to convert raw price data into return series for analytics
   - Use moving averages (`compute_moving_average_simple/exponential()`) to build trend indicators in strategies
   - Use `compute_velocity()` and `compute_acceleration()` to detect trend strength and regime changes for QQQ/TQQQ/SQQQ allocation decisions
   - Use `compute_annualized_volatility()` when building position sizing or volatility targeting logic

2. **Risk metrics** (`src/analytics/risk_metrics.py`):
   - Pass backtest equity curves to `compute_sharpe_ratio()`, `compute_sortino_ratio()`, `compute_calmar_ratio()` for risk-adjusted performance measurement
   - Use `compute_drawdown_series()` and `compute_max_drawdown()` to visualize and quantify pain in backtest reports
   - Compare strategy returns to QQQ benchmark using `compute_information_ratio()` and `compute_beta_and_correlation()`
   - Use rolling metrics (`compute_rolling_sharpe()`, `compute_rolling_volatility()`) to detect strategy degradation or regime shifts
   - When evaluating parameter sweeps, use these metrics to rank strategy variants

3. **Clock abstraction** (`src/utils/time.py`):
   - Inject `RealClock()` in production/live orchestration
   - Inject `FrozenClock(date)` in backtests to simulate being at historical timestamps
   - Use `clock.now()` instead of `datetime.now()` everywhere in strategy and engine code
   - This ensures backtests are deterministic and don't accidentally leak future data

4. **Synthetic data** (`src/analytics/synthetic_data.py`):
   - Use `generate_gbm_paths()` to create test price series for validating backtest engines (trending market scenarios)
   - Use `generate_ou_paths()` to stress-test strategies under choppy/mean-reverting conditions
   - Before running strategies on real QQQ data, verify they behave correctly on synthetic paths with known properties
   - Use fixed seeds in tests to ensure reproducibility

**Testing Phase 2**:

Run all Phase 2 tests:
```bash
pytest tests/test_utils_math.py tests/test_analytics_risk_metrics.py tests/test_utils_time.py tests/test_analytics_synthetic_data.py -v
```

Or run all tests:
```bash
pytest tests/ -v
```

**Next steps**:

Phase 2 provides the foundation for Phase 3 (data I/O and canonical schemas). With these utilities in place, we can now:
- Build the backtest engine that uses these metrics to evaluate strategies
- Implement data loading and feature engineering that calls math utilities
- Write the QQQ long/short strategy that uses velocity/acceleration and moving averages
- Generate backtest reports that include all the risk metrics

### Phase 3 Implementation summary

**Status**: Phase 3 complete. All data I/O, schemas, loaders, and tests implemented.

**What was implemented**:

1. **src/data/schemas.py** (271 lines)
   - `SchemaValidationError`: Custom exception for schema violations with actionable error messages
   - `RAW_PRICE_REQUIRED_COLUMNS`: Constant defining required columns for raw OHLCV data (timestamp, open_price, high_price, low_price, closing_price, volume)
   - `PROCESSED_MIN_REQUIRED_COLUMNS`: Constant defining minimum columns for processed data (timestamp, closing_price)
   - `validate_raw_price_schema(df, context)`: Validates raw price DataFrames against schema
     - Checks all required columns are present
     - Validates timestamp column is datetime-parseable
     - Enforces strictly descending timestamp order (newest first, no ties)
     - Raises SchemaValidationError with clear, actionable messages on violations
   - `validate_processed_schema(df, required_features, context)`: Validates processed feature DataFrames
     - Checks base columns (timestamp, closing_price) are present
     - Optionally validates additional required features (e.g., velocity, acceleration)
     - Enforces strictly descending timestamp order
     - Flexible to accommodate different feature sets for different strategies
   - All functions include extensive "teaching-style" docstrings explaining:
     - Conceptual purpose (why schema validation matters)
     - Functional behavior (what it checks, what errors it raises)
     - Why strict descending order and explicit schemas are critical for reproducibility

2. **src/data/io.py** (372 lines)
   - `read_raw_price_csv(path, instrument_name)`: Read raw price CSVs with validation
     - Reads CSV using pandas
     - Parses timestamp column to datetime objects
     - Validates schema (calls validate_raw_price_schema)
     - Returns DataFrame with datetime timestamps in strictly descending order
     - Raises FileNotFoundError if file missing, SchemaValidationError if schema invalid
   - `write_raw_price_csv(df, path)`: Write raw price DataFrames to CSV
     - Validates schema before writing (fail fast)
     - Automatically sorts by timestamp in strictly descending order
     - Converts datetime timestamps to ISO 8601 strings for CSV storage
     - Creates parent directories if they don't exist
     - Writes with stable column order (matching schema)
   - `read_processed_data_csv(path, instrument_name)`: Read processed feature CSVs
     - Similar to read_raw_price_csv but validates processed schema
     - Flexible to handle varying feature sets
   - `write_processed_data_csv(df, path)`: Write processed feature DataFrames
     - Validates base schema (timestamp, closing_price)
     - Auto-sorts by timestamp descending
     - Writes all columns (not just base schema, to preserve features)
   - All functions are the *only* I/O boundary for CSVs in the system (enforces rule: never use pd.read_csv/to_csv directly)
   - Extensive docstrings explaining:
     - Why centralized I/O with validation is essential
     - How timestamp parsing and sort-order enforcement work
     - Error handling and actionable error messages
     - AWS portability benefits (swap local CSV for S3 in one place)

3. **src/data/loaders.py** (223 lines)
   - `load_instrument_history(symbol)`: Generic loader for any instrument by symbol
     - Resolves path under data/raw/{symbol}.csv
     - Delegates to read_raw_price_csv for validation
     - Returns validated DataFrame
   - `load_qqq_history()`: Convenience loader for QQQ (reference instrument for Strategy 1)
   - `load_tqqq_history()`: Convenience loader for TQQQ (3x long QQQ)
   - `load_sqqq_history()`: Convenience loader for SQQQ (3x inverse QQQ)
   - `load_uvxy_history()`: Convenience loader for UVXY (2x leveraged volatility)
   - `load_default_universe()`: Load all default instruments (QQQ, TQQQ, SQQQ, UVXY)
     - Returns dict mapping symbol -> DataFrame
     - Gracefully handles missing instruments (skips FileNotFoundError, omits from dict)
     - Raises schema errors (don't silently skip bad data)
   - All loaders are thin wrappers around io.py functions
   - Docstrings explain:
     - Purpose of each instrument in Strategy 1 context
     - Why convenience loaders improve code readability and maintainability
     - How they decouple strategies from path logic

4. **tests/test_data_io.py** (352 lines, 19 tests)
   - Helper functions:
     - `make_valid_raw_price_df(n_rows)`: Generate test DataFrames matching raw schema
     - `make_valid_processed_df(n_rows)`: Generate test DataFrames matching processed schema
   - Schema validation tests (7 tests):
     - Valid raw and processed DataFrames pass validation
     - Missing required columns raise SchemaValidationError
     - Bad timestamp order (ascending instead of descending) raises error
     - Duplicate timestamps (ties) raise error
     - Required features validation works correctly
   - CSV I/O tests (9 tests):
     - Round-trip read/write for raw and processed data preserves data
     - write_raw_price_csv auto-sorts to descending order
     - Missing files raise FileNotFoundError
     - Missing columns in CSV raise SchemaValidationError with clear messages
     - Bad timestamp formats raise SchemaValidationError
     - Parent directories are created automatically when writing
   - Loader tests (2 tests):
     - load_instrument_history works with temp files (uses monkeypatch to redirect paths)
     - load_default_universe handles missing instruments gracefully
   - Edge case tests (3 tests):
     - Empty DataFrames don't crash validation
     - Single-row DataFrames pass validation (no ordering issues)
   - All tests use tmp_path fixture to avoid polluting real data/ directory
   - Tests include comments explaining what each test verifies

**Deviations from original design**:

- None. Implementation follows the Phase 3 design spec exactly.
- Added automatic parent directory creation in write functions (not explicitly specified but essential for robustness).
- Used monkeypatch in tests to redirect RAW_DATA_DIR to tmp_path (cleaner than creating real files in data/raw/ for tests).
- Extended docstrings beyond the original API sketches to fully meet the "teaching code" requirement, including:
  - Conceptual explanations of why schemas and centralized I/O matter
  - Functional descriptions of inputs, outputs, error handling
  - Teaching notes about production quant system best practices
  - Strategy 1 context for instrument loaders

**How to use Phase 3 data I/O in later phases**:

For new contributors or when implementing Phase 4+ (backtesting engines, strategies, orchestration):

1. **Loading raw price data**:
   ```python
   from src.data.loaders import load_qqq_history, load_tqqq_history

   # Load specific instruments
   qqq = load_qqq_history()  # Returns validated DataFrame
   tqqq = load_tqqq_history()

   # Or load all default instruments at once
   from src.data.loaders import load_default_universe
   universe = load_default_universe()  # Returns dict: {"QQQ": df, "TQQQ": df, ...}
   ```

2. **Reading/writing CSVs directly** (if you need non-standard instruments or custom paths):
   ```python
   from src.data.io import read_raw_price_csv, write_raw_price_csv

   # Read with validation
   df = read_raw_price_csv("data/raw/SPY.csv", instrument_name="SPY")

   # Write with auto-sorting and validation
   write_raw_price_csv(df, "data/raw/SPY.csv")
   ```

3. **Working with processed/feature data**:
   ```python
   from src.data.io import read_processed_data_csv, write_processed_data_csv

   # Read processed features
   qqq_features = read_processed_data_csv("data/processed/QQQ_features.csv")

   # Write features (after feature engineering)
   write_processed_data_csv(qqq_features, "data/processed/QQQ_features.csv")
   ```

4. **Validating DataFrames before use** (if you generate data programmatically):
   ```python
   from src.data.schemas import validate_raw_price_schema, validate_processed_schema

   # Validate before passing to strategies or backtests
   validate_raw_price_schema(df, context="my_data_source")

   # Validate processed data with required features
   validate_processed_schema(
       df,
       required_features=["velocity", "acceleration", "moving_average_50"],
       context="QQQ features"
   )
   ```

**Key rules for Phase 4+ developers**:

1. **Never use pd.read_csv or df.to_csv directly**. Always use src/data/io.py functions to ensure schema enforcement.
2. **All CSVs must have strictly descending timestamps** (newest first). The I/O layer enforces this automatically.
3. **All timestamps are stored as ISO 8601 strings in CSVs** but parsed to datetime objects when read.
4. **Use instrument loaders** (load_qqq_history, etc.) instead of hardcoding paths in strategies or backtests.
5. **Schema validation is non-negotiable**. If a CSV doesn't conform, fix the source data or the schema definition, don't bypass validation.

**Testing Phase 3**:

Run Phase 3 tests:
```bash
pytest tests/test_data_io.py -v
```

Run all tests including Phase 2 + Phase 3:
```bash
pytest tests/ -v
```

**Next steps**:

Phase 3 provides robust data I/O and schema enforcement. With this foundation, Phase 4 can implement:
- Feature engineering pipelines that read raw CSVs, compute features (using Phase 2 math utilities), and write processed CSVs
- Backtest engines that load processed data and evaluate strategies
- Strategy implementations (e.g., Strategy 1: QQQ long/short) that consume feature data and generate signals
- Orchestration workflows that chain data ingestion → feature engineering → backtesting → results export
- All of the above can trust that data conforms to schemas and that I/O is centralized and validated
