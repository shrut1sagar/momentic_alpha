# Project: momentic_alpha

## Purpose
You are helping build a modular trading / data analysis project.
Focus on clear architecture, testability, and maintainability.

## Tech stack
- Python 3.11 (or current Python in this venv)
- Layout:
  - `src/momentic_alpha/` – main package
  - `tests/` – unit tests

## Architecture (initial)
- `src/momentic_alpha/config/` – settings, constants
- `src/momentic_alpha/data/` – data loading and caching
- `src/momentic_alpha/strategies/` – trading/signal logic
- `src/momentic_alpha/backtesting/` – backtest engine + metrics
- `src/momentic_alpha/utils/` – small shared helpers

## Coding style
- Use type hints.
- Prefer small, focused functions over large classes.
- Docstrings for public functions.
- Add tests in `tests/` for non-trivial logic.

## AI collaboration rules
- Codex (in Cursor): architect + reviewer (design, plans, reviews).
- Claude Code (in WSL): implementer + refactorer (write code, apply reviews).
- Treat this file as the source of truth for style and structure.
