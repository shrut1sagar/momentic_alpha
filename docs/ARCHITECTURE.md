# Architecture

## Overview
momentic_alpha is a local-first Python trading toolkit built as a small library plus CLI actions. Core logic stays stateless and importable under `src/`, while thin scripts in `actions/` provide human-friendly entrypoints for backtests, pipelines, and plotting. Data flows through CSV contracts (`data/raw` → `data/processed` → `data/results`), with any persistent pointers or metadata stored explicitly under `state/`. This separation keeps the system reproducible on a laptop today and portable to AWS batch/containers later.

## Design Principles (short form)
- Adapters over vendors: depend on abstract data/broker interfaces; plug concrete venues under `src/venues/`.
- Library + actions: reusable modules under `src/`; one-file CLI actions that delegate to a single library entrypoint.
- CSV-first contracts: every stage reads/writes canonical CSVs for inspectability and reproducibility.
- Stateless core, explicit state: pure functions for computation; any persistence lives in `state/` on purpose.
- Topic-based modularity: each top-level `src/` package owns one concern with clear boundaries.
- Fail fast, clear remediation: upfront validation and actionable errors pointing to files/keys/columns.

## Module Responsibilities
- `src/analytics/`: Feature engineering, indicators, and risk/metrics computations (Sharpe, drawdown, rolling stats).
- `src/backtesting/`: Engines, runners, and reporting utilities that orchestrate strategies, brokers, and data.
- `src/config/`: Strongly typed settings and strategy config loaders/validators for paths, params, and env.
- `src/data/`: Data I/O, schema enforcement, and sources/loaders for raw/processed/results CSVs.
- `src/execution/`: Order models and paper broker logic (fills, fees, slippage) for simulations; future live bridges.
- `src/strategies/`: Strategy interfaces and implementations that emit signals/target weights.
- `src/utils/`: Generic helpers (time/clock, math, logging, types, errors) shared across modules.
- `src/venues/`: Venue abstractions plus concrete adapters/registries for data providers and brokers.
- `src/orchestration/`: Multi-step pipelines (e.g., daily pipeline) that chain data refresh, feature builds, and runs.
- `actions/`: One script per CLI action; minimal arg parsing, then delegate to a single `src/` entrypoint.

## Data & State Layout
- `data/raw/`: Canonical, normalized OHLCV CSVs from sources.
- `data/processed/`: Feature-enriched tables aligned on timestamps.
- `data/results/`: Backtest outputs (equity curves, trades, metrics, plots).
- `state/`: Intentional run metadata and snapshots (no hidden caches).

## Entry Points & Extensibility
- CLI actions invoke library entrypoints to keep notebooks/tests and scripts consistent.
- Adapters/registries under `src/venues/` let new providers slot in without touching strategies or backtests.
- Stateless core functions plus explicit state files make local runs reproducible and cloud migration safer.
