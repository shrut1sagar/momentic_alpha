momentic_alpha – Trading System Specification (v1)
1. Project Overview

momentic_alpha is a local Python trading application designed for:

Research & prototyping of systematic trading strategies (initially on QQQ / TQQQ / SQQQ / UVXY).

Robust backtesting using daily bars, with clear data contracts and rich performance metrics.

Future portability to AWS (batch jobs, containers, or serverless), by keeping the core logic stateless and environment-agnostic.

The system is structured as a small library plus a set of CLI actions, with a strong focus on transparency, reproducibility, and debuggability.

2. High-Level Goals

Run reproducible backtests and simulations on local CSV data.

Support multiple strategies, starting with a QQQ long/short trend strategy using TQQQ/SQQQ (and optionally UVXY).

Produce clean outputs: equity curves, trade ledgers, metrics, and plots.

Keep all math and risk logic modular, well-documented, and easy to extend.

Make migration to AWS straightforward by enforcing:

Clear boundaries between I/O, orchestration, and pure logic.

Stateless core functions with explicit state persistence.

3. Design Principles
3.1 Adapters Over Vendors

Define venue-agnostic interfaces for:

Data providers (historical + live).

Broker clients (order submission, fills, positions).

Concrete implementations for Massive, Alpaca, or mocks are plugged in via a registry under src/venues/.

The rest of the system (strategies, backtesting, analytics) depends only on these abstract interfaces.

3.2 Library + Actions Pattern

Core logic lives under src/… as importable modules.

CLI entrypoints live under actions/, one file per high-level action (e.g. run_backtest.py, run_qqq_strategy.py, plot_results.py).

Each action:

Does minimal argument parsing.

Delegates to a single library entrypoint.

Ensures the system can be driven both from the CLI and from notebooks/tests.

3.3 CSV-First Data Contracts

All intermediate artifacts are written to disk in CSV form:

data/raw/ – sourced price data exactly as provided (but normalized to a canonical schema).

data/processed/ – enriched price + feature tables.

data/results/ – backtest outputs (equity, trades, metrics).

CSVs are:

Append-friendly and human-readable.

Easy to inspect manually or with external tools (Excel, pandas, SQL loaders).

Every stage should be reproducible from CSV inputs + config, without hidden state.

3.4 Stateless Core, Explicit State

Core processing functions are pure: input data + parameters → output data / results.

Any persistent or cross-run state is:

Stored in state/ (e.g. state/portfolio.json, run metadata, pointers).

Written intentionally (no hidden caches on disk).

This makes backtesting repeatable and future migration to distributed environments safer.

3.5 Topic-Based Modularity

Each top-level module under src/ owns a clear concern:

analytics/ – feature engineering and risk/metrics calculations.

backtesting/ – engines, runners, and evaluation utilities.

config/ – configuration loading and validation.

data/ – data I/O, caching, and schema enforcement.

execution/ – order normalisation, fills, fees, and broker logic.

strategies/ – strategy logic and signal generation.

utils/ – generic helpers (time, math, logging, types).

venues/ – vendor adapters and venue registries.

orchestration/ – multi-step workflows (e.g. daily pipeline).

3.6 Fail Fast With Clear Remediation

All actions and orchestrators must:

Validate config and schemas up front.

Surface errors with actionable messages (e.g. “missing column closing_price in data/raw/QQQ.csv”).

Failure modes:

Missing environment variables / credentials.

Invalid CSV schemas (dates, columns, sort order).

Misconfigured strategies or parameters.

Errors should point to specific files, keys, and suggested fixes.

4. Directory Layout & Responsibilities

Target repository structure:

.
├─ docs/
│  ├─ SPEC_OVERVIEW.md      # This document
│  ├─ ARCHITECTURE.md       # High-level design (shorter than spec)
│  └─ DEV_NOTES.md          # Running developer log and rationale
├─ config/
│  └─ strategy/             # Strategy parameter files (TOML/YAML/JSON)
├─ data/
│  ├─ raw/                  # Normalized raw OHLCV data
│  ├─ processed/            # Feature-enriched data per instrument
│  └─ results/              # Backtest results and plots
├─ state/                   # Persistent state (portfolio, run metadata)
├─ src/
│  ├─ analytics/
│  ├─ backtesting/
│  ├─ config/
│  ├─ data/
│  ├─ execution/
│  ├─ strategies/
│  ├─ utils/
│  ├─ venues/
│  └─ orchestration/
├─ actions/
└─ tests/

4.1 docs/

SPEC_OVERVIEW.md – master spec (this file).

ARCHITECTURE.md – high-level summary of system architecture and design patterns.

DEV_NOTES.md – chronologically ordered log of:

Decisions made and why.

Trade-offs considered.

Phase descriptions (e.g. “Phase 1 – Scaffolding”, “Phase 2 – Utilities”).

4.2 config/

Holds configuration files (e.g. TOML/YAML/JSON).

config/strategy/ – named strategy configs (e.g. qqq_long_short.toml):

Universe selection.

Parameter sets (MA windows, thresholds, slippage assumptions, etc.).

Configuration will be loaded via strongly-typed settings objects under src/config/.

4.3 data/

data/raw/:

Canonical OHLCV CSVs for instruments: QQQ.csv, TQQQ.csv, SQQQ.csv, UVXY.csv, etc.

data/processed/:

Feature tables aligned on dates.

Derived columns (returns, moving averages, volatility estimates, signals, etc.).

data/results/:

equity_<strategy>_<run_id>.csv

trades_<strategy>_<run_id>.csv

metrics_<strategy>_<run_id>.json

plot_<strategy>_<run_id>.png (equity + Sharpe plots).

4.4 state/

JSON / small files storing:

Last successful run.

Persistent portfolio snapshots (for live transition later).

Misc metadata (e.g. parameter sweeps history).

4.5 src/analytics/

Purpose: feature engineering and risk analytics.

feature_engineering.py:

Compute returns (simple, log).

Momentum, spreads, technical indicators (RSI, moving averages, etc.).

risk_metrics.py:

Portfolio and strategy evaluation metrics (Sharpe, Sortino, ROI, CAGR, etc.).

synthetic_data.py:

Monte Carlo / stochastic generators for synthetic price paths.

Additional modules as project grows (e.g. cross-section analytics).

4.6 src/backtesting/

Purpose: backtest engines and evaluation utilities.

engine.py:

Daily-bar vectorized engine.

Hooks for pluggable strategies and brokers.

runner.py or scenario_runner.py:

Run backtests over parameter grids.

reports.py:

Build structured reports (equity curves, metrics, attribution summaries).

All engines write their outputs via src/data/ utilities.

4.7 src/config/

Purpose: configuration loading and validation.

settings.py:

Top-level Settings dataclass for paths, logging, environment.

strategy_config.py:

Strategy-specific settings models (e.g. QQQLongShortConfig).

Validates:

File paths and existence.

Param types and ranges.

Environment variables (e.g. credentials when venues are added later).

4.8 src/data/

Purpose: data I/O and canonical schema enforcement.

io.py:

Read/write canonical CSVs.

Enforce date ordering and schema.

sources.py:

Adapters to load local CSVs, synthetic data, or (later) vendor APIs.

schemas.py:

Definitions and validation for CSV column expectations.

4.9 src/execution/

Purpose: order handling and portfolio simulation.

paper_broker.py:

Paper trading engine (positions, cash, fills).

Slippage and fee models.

order_models.py:

Standardized order representation (market orders, target weights, etc.).

Future: live execution connectors via venues/.

4.10 src/strategies/

Purpose: strategy logic and signal generation.

base.py:

Shared interfaces / ABCs (e.g. Strategy, SignalGenerator).

qqq_long_short.py:

Strategy 1 implementation (see §8).

Additional modules for future strategies.

4.11 src/utils/

Purpose: generic helper functions and types.

math.py:

Clean, well-documented implementations of standard math functions:

Standard deviation, moving averages, exponential moving averages.

Velocity and acceleration of price or returns.

Annualization helpers.

time.py:

Clock abstraction (real vs frozen time).

logging.py:

Logging setup and standard log format helpers.

types.py, errors.py (optional):

Typed aliases and custom exception classes.

4.12 src/venues/

Purpose: venue abstraction & adapters.

base.py:

ABCs for DataProvider and BrokerClient.

registry.py:

Mapping from venue name → concrete implementation.

massive_adapter.py, alpaca_adapter.py (future):

Implementations that conform to base interfaces.

4.13 src/orchestration/

Purpose: multi-step pipelines.

daily_pipeline.py:

Load config.

Ingest/refresh data.

Build features.

Run strategies and backtests.

Persist results.

clock.py:

Higher-level orchestration around the clock abstraction (batch scheduling hooks).

4.14 actions/

One script per action, e.g.:

actions/fetch_history.py

actions/run_backtest.py

actions/run_qqq_strategy.py

actions/plot_results.py

Each script:

Parses CLI arguments.

Calls a single entrypoint in src/.

4.15 tests/

Mirrors src/ structure where useful:

tests/test_utils_math.py

tests/test_analytics_risk_metrics.py

tests/test_data_io.py

tests/test_backtesting_engine.py

tests/test_strategies_qqq_long_short.py, etc.

Focus:

Correctness of math.

Adherence to CSV contracts.

Basic backtest sanity (e.g. always-cash strategy behaves as expected).

5. Canonical CSV Schemas
5.1 General Rules

Date column:

Name: timestamp.

Type: ISO 8601 date-time string (e.g. 2024-01-15T00:00:00).

Sort order:

Strictly decreasing by timestamp (newest row at the top).

Column names:

Human-readable, descriptive, and snake_case.

Example: closing_price, moving_average_50, rolling_sharpe_60d.

5.2 Raw Price Data (data/raw)

Suggested schema for each instrument (e.g. QQQ.csv):

timestamp – ISO date-time, strictly decreasing.

open_price

high_price

low_price

closing_price

volume

5.3 Processed/Feature Data (data/processed)

Processed CSVs share the same primary key (timestamp), plus features. Example columns:

timestamp

closing_price

daily_return

moving_average_10

moving_average_50

moving_average_100

moving_average_250

volatility_20d

velocity

acceleration

signal_long_probability (if applicable)

position_target_weight_TQQQ, position_target_weight_SQQQ, position_target_weight_cash etc.

5.4 Results Data (data/results)

Equity curve CSV:

timestamp

equity_value

Optional: drawdown, rolling_sharpe_60d

Trades CSV:

timestamp

instrument

side (buy/sell/short_cover/short_open)

quantity

price

gross_value

fees

slippage_cost

Metrics JSON (one per run):

Overall metrics:

sharpe_ratio

sortino_ratio

roi

cagr

volatility_annualized

max_drawdown

calmar_ratio

information_ratio

hit_rate

win_loss_ratio

Rolling metrics (stored as summary stats or as paths in results CSVs).

6. Core Utilities
6.1 src/utils/math.py

Centralized math and statistical helpers:

Moving averages:

compute_simple_moving_average(prices, window)

compute_exponential_moving_average(prices, span)

Volatility / dispersion:

compute_standard_deviation(returns)

compute_annualized_volatility(returns, periods_per_year=252)

Returns:

compute_simple_returns(prices)

compute_log_returns(prices)

Momentum / dynamics:

compute_velocity(series, window) – e.g. slope of recent returns or prices.

compute_acceleration(series, window) – change in velocity over time.

Misc:

Helper functions for rolling operations, normalisation, scaling to [0,1] if needed.

All functions:

Use type hints.

Are documented with docstrings and inline comments.

Are deterministic and side-effect free.

6.2 src/analytics/risk_metrics.py

Implements portfolio performance metrics, given equity curves or return series:

Point metrics:

compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

compute_roi(equity_curve)

compute_cagr(equity_curve, periods_per_year=252)

compute_max_drawdown(equity_curve)

compute_calmar_ratio(equity_curve, periods_per_year=252)

compute_information_ratio(strategy_returns, benchmark_returns, periods_per_year=252)

compute_hit_rate(returns) (fraction positive)

compute_win_loss_ratio(returns) (avg win / avg loss)

Rolling metrics:

compute_rolling_sharpe(returns, window, risk_free_rate=0.0, periods_per_year=252)

Similar functions for rolling Sortino, etc.

Each function should operate on a pd.Series or pd.DataFrame and return simple scalars/series.

6.3 src/utils/time.py – “Freeze the Clock”

Provide a simple abstraction to control “now”:

Real mode:

get_now() → datetime.utcnow() (or system time).

Test/backtest mode:

Accept an injected or configured frozen_now timestamp.

get_now() returns the fixed timestamp.

Integration:

Backtests should use this clock abstraction for date comparisons and logs, so we can simulate being “in the past” and ignore future data.

6.4 src/analytics/synthetic_data.py – Synthetic Market Scenarios

Functions to generate synthetic price paths:

generate_gbm_paths(...) – Geometric Brownian Motion.

generate_mean_reverting_paths(...) – e.g. Ornstein–Uhlenbeck process.

Optional: multi-asset paths with correlation.

Use cases:

Stress-testing strategies.

Validating backtest engine and metrics in controlled environments.

7. Backtesting & Simulation
7.1 Engine (src/backtesting/engine.py)

Daily-bar engine that:

Accepts:

A price/feature DataFrame (indexed or keyed by timestamp).

A strategy function or object.

A broker instance (paper broker).

Backtest parameters (start/end dates, initial capital).

Iterates over dates in time order (internal processing may convert from descending to ascending as needed).

At each date:

Builds a “context” (features and state).

Calls the strategy to get target allocations or orders.

Sends actions to the broker and records fills.

Produces:

Equity curve time series.

Trades ledger.

Metrics using risk_metrics.py.

7.2 Paper Broker (src/execution/paper_broker.py)

Responsibilities:

Maintain positions and cash.

Apply:

Simple slippage models (e.g. basis point spread).

Commission/fee models per trade or per notional.

Accept either:

Target weights per instrument.

Or explicit buy/sell orders (first versions can use target weights).

Outputs:

Trade logs (one row per fill).

Per-date portfolio valuation (needed for equity curve).

7.3 Outputs & Evaluation

For each backtest run:

Equity curve CSV.

Trades CSV.

Metrics JSON.

Evaluation plan:

Parameter sweeps:

Loop over strategy parameter grids (e.g. MA lengths, thresholds).

Walk-forward splits:

Train/validate/test windows.

Scenario comparisons:

Compare metrics across parameter sets and seeds.

8. Strategy Framework & Strategy 1 (QQQ Long/Short)
8.1 Strategy Framework

Abstract interface (in src/strategies/base.py), e.g.:

class Strategy(Protocol):
    def generate_signals(self, data: pd.DataFrame, state: dict | None = None) -> pd.DataFrame:
        ...


Strategies return per-date signals or target weights, e.g.:

target_weight_TQQQ

target_weight_SQQQ

target_weight_cash

(optionally) target_weight_UVXY

8.2 Strategy 1 – QQQ Trend Long/Short with TQQQ / SQQQ

Universe:

QQQ (underlying “reference” instrument).

TQQQ (levered long).

SQQQ (levered short).

UVXY (optional volatility hedge).

Core Idea:

Trend direction based on QQQ:

Use velocity and acceleration to determine trend regime.

Moving average tiers:

Use moving averages (e.g. 50d, 100d, 250d) and their spreads to confirm trend strength and persistence.

Regime logic:

Uptrend → long via TQQQ (levered).

Downtrend → short via SQQQ (or long SQQQ).

Mixed / unclear → cash (and possibly UVXY overlays later).

Inputs (per date):

QQQ closing price and features:

moving_average_50

moving_average_100

moving_average_250

velocity

acceleration

Optionally:

Short-term volatility measures.

UVXY features (for future model versions).

Example Decision Logic (v1 sketch):

Compute:

trend_spread_50_100 = moving_average_50 - moving_average_100

trend_spread_50_250 = moving_average_50 - moving_average_250

Interpret:

Strong uptrend:

trend_spread_50_100 > 0 and trend_spread_50_250 > 0

velocity > 0

acceleration > 0 (trend gaining speed)

Tiring uptrend:

Spreads > 0 but acceleration < 0 (trend still up but slowing)

Downtrend:

trend_spread_50_100 < 0 and trend_spread_50_250 < 0

velocity < 0

acceleration <= 0

Mixed/neutral:

Signs disagree or metrics close to zero thresholds.

Allocation examples:

Strong uptrend:

High positive allocation to TQQQ (subject to volatility constraints).

Tiring uptrend:

Reduced TQQQ allocation.

Strong downtrend:

Allocation to SQQQ (short bias).

Mixed / neutral:

Move to cash (and optionally small UVXY positioning later).

Backtesting objective for Strategy 1:

Generate multiple parameterized variants:

MA windows (e.g. 20/50/100, 50/100/250).

Thresholds for velocity/acceleration magnitudes.

Caps on leverage/position size.

Backtest each variant.

Compare outcomes by:

ROI

Sharpe

Drawdown

Calmar, etc.

Identify scenario(s) with best trade-off between Sharpe and drawdown.

9. Documentation & Communication Standards
9.1 Code Documentation

Every function:

Has a clear docstring describing:

Purpose and intuition.

Parameters and types.

Returns and units.

Inline comments:

Required for any non-obvious logic.

Prefer readability over cleverness:

Straightforward, explicit code is better than compact idioms.

9.2 Developer Notes & Rationale

docs/DEV_NOTES.md maintains:

Phase-by-phase sections (e.g. “Phase 1 – Scaffolding”, “Phase 2 – Utilities”).

For each phase:

What was implemented.

Why design choices were made.

Any trade-offs or later TODOs.

9.3 “Blog-Ready” Documentation

For each major component (e.g. config, math utilities, risk metrics, backtesting engine, Strategy 1), we will eventually produce:

A short README or section.

A narrative explanation that can be turned into a Substack “chapter” (e.g. “Building a Backtesting Engine from Scratch”, “Designing Risk Metrics for Retail Quant Strategies”).

Style:

Educational.

Explains algorithms and architecture in approachable terms for an interested technical reader.