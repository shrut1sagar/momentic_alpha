# momentic_alpha

Experimental quant/trading project.

This repo is built collaboratively with two AIs:
- Codex (in Cursor) for planning, architecture, and review
- Claude Code (in WSL) for implementation and refactoring

## Configuration

- Use a `.env` file in the repo root for secrets/config; see `.env.example` for placeholders.
- API keys must never be committed. Set `FINNHUB_API_KEY` for Finnhub access; optional overrides:
  - `FINNHUB_BASE_URL` (default `https://finnhub.io/api/v1`)
  - `FINNHUB_MIN_SLEEP_SECONDS` (polite delay between requests, default `0.2`)
  - `FINNHUB_TIMEOUT_SECONDS` (default `30`)
- Optional: `DATA_SOURCE_NAME` to choose a default provider (e.g., `finnhub`, `massive`); actions may also take a `--source` flag.
