#!/usr/bin/env python3
"""
Fetch historical price data from Finnhub and save to data/raw/.

**Conceptual**: This script fetches daily OHLCV bars from Finnhub for multiple
tickers and writes them to data/raw/{TICKER}.csv in canonical format. It supports
incremental updates (fill gaps only) and force mode (overwrite date range).

**Usage**:
    # Fetch default universe with default date range
    python actions/fetch_price_history_finnhub.py

    # Fetch specific tickers
    python actions/fetch_price_history_finnhub.py --tickers QQQ,SPY,TQQQ

    # Fetch from specific start date
    python actions/fetch_price_history_finnhub.py --start 2020-01-01

    # Force overwrite existing data in date range
    python actions/fetch_price_history_finnhub.py --tickers QQQ --force

**Default universe**:
    QQQ, SQQQ, TQQQ, SPY, VIX, UVXY, VIXM, VIXY, VIXX, GLD, OIL, TLT

**Date handling**:
    - Default start: 1970-01-01 (fetch all available history)
    - Default end: today
    - Finnhub free tier typically has data from 1970s onwards

**Incremental vs Force mode**:
    - Incremental (default): Fill gaps in existing data without overwriting
    - Force (--force): Overwrite date range [start, end] with fresh data

**Rate limiting**:
    - Respects settings.min_sleep_seconds between API requests
    - Free tier: 60 API calls/minute (default: 0.2s = 5 calls/sec)

**Error handling**:
    - Per-ticker errors don't crash the entire script
    - Continues to next ticker after logging error
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to Python path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import get_settings
from src.venues.finnhub_data_provider import FinnhubDataProvider, FinnhubAccessError
from src.data.io import read_raw_price_csv, write_raw_price_csv


# Default ticker universe
DEFAULT_TICKERS = [
    "QQQ", "SQQQ", "TQQQ", "SPY",
    "VIX", "UVXY", "VIXM", "VIXY", "VIXX",
    "GLD", "OIL", "TLT"
]


def merge_price_history(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
    force: bool = False,
) -> pd.DataFrame:
    """
    Merge existing and new price data with incremental or force mode.

    **Conceptual**: When fetching data, we often want to update existing CSVs
    rather than overwriting them completely. This function implements two modes:

    **Incremental mode (force=False)**:
      - Keeps all existing data
      - Adds new bars for dates not in existing data
      - Useful for filling gaps or extending time range

    **Force mode (force=True)**:
      - Removes existing data in [start_date, end_date] range
      - Replaces with new data for that range
      - Keeps existing data outside the range
      - Useful for refreshing stale data or fixing errors

    Args:
        existing: Existing DataFrame from CSV (may be empty).
        new: Newly fetched DataFrame from API.
        start_date: Start date of fetch request.
        end_date: End date of fetch request.
        force: If True, replace existing data in [start_date, end_date].
               If False, only add new bars (incremental).

    Returns:
        Merged DataFrame with canonical columns, sorted descending by timestamp.

    Example:
        >>> # Incremental: add 2024 data to existing 2023 data
        >>> existing = pd.DataFrame(...)  # 2023 data
        >>> new = pd.DataFrame(...)  # 2024 data
        >>> merged = merge_price_history(existing, new, dt.date(2024,1,1), dt.date(2024,12,31), force=False)
        >>> # Result: 2023 + 2024 data

        >>> # Force: refresh 2023 data with newly fetched version
        >>> existing = pd.DataFrame(...)  # 2023 + 2024 data (stale 2023)
        >>> new = pd.DataFrame(...)  # fresh 2023 data
        >>> merged = merge_price_history(existing, new, dt.date(2023,1,1), dt.date(2023,12,31), force=True)
        >>> # Result: fresh 2023 + existing 2024 data
    """
    # If no existing data, just return new data
    if existing.empty:
        return new.copy()

    # If no new data, return existing data
    if new.empty:
        return existing.copy()

    # Ensure timestamp column is datetime for both DataFrames
    if not pd.api.types.is_datetime64_any_dtype(existing['timestamp']):
        existing = existing.copy()
        existing['timestamp'] = pd.to_datetime(existing['timestamp'], format='ISO8601')

    if not pd.api.types.is_datetime64_any_dtype(new['timestamp']):
        new = new.copy()
        new['timestamp'] = pd.to_datetime(new['timestamp'], format='ISO8601')

    if force:
        # Force mode: Remove existing data in [start_date, end_date] range
        start_ts = pd.Timestamp(start_date, tz='UTC')
        end_ts = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        # Keep existing data outside the range
        outside_range = existing[
            (existing['timestamp'] < start_ts) | (existing['timestamp'] > end_ts)
        ]

        # Concatenate with new data
        merged = pd.concat([outside_range, new], ignore_index=True)
    else:
        # Incremental mode: Add new bars that don't exist in existing data
        # Use timestamp as the key to determine uniqueness
        existing_timestamps = set(existing['timestamp'].values)
        new_bars = new[~new['timestamp'].isin(existing_timestamps)]

        # Concatenate existing + new bars
        merged = pd.concat([existing, new_bars], ignore_index=True)

    # Sort by timestamp descending (newest first)
    merged = merged.sort_values('timestamp', ascending=False).reset_index(drop=True)

    # Remove duplicate timestamps (keep first = newest in descending sort)
    merged = merged.drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)

    return merged


def fetch_and_save_ticker(
    provider: FinnhubDataProvider,
    ticker: str,
    start_date: dt.date,
    end_date: dt.date,
    output_dir: Path,
    force: bool = False,
) -> bool:
    """
    Fetch price data for a single ticker and save to CSV.

    **Conceptual**: This function handles the full workflow for one ticker:
      1. Check if existing CSV exists
      2. Fetch new data from Finnhub
      3. Merge with existing data (incremental or force)
      4. Write to CSV

    Args:
        provider: FinnhubDataProvider instance.
        ticker: Ticker symbol (e.g., "QQQ").
        start_date: Start date for fetch (inclusive).
        end_date: End date for fetch (inclusive).
        output_dir: Directory where CSV should be saved (e.g., data/raw).
        force: If True, overwrite existing data in date range.

    Returns:
        True if successful, False if error occurred.

    Side effects:
        - Writes or updates data/raw/{TICKER}.csv
        - Prints progress messages to stdout
    """
    output_file = output_dir / f"{ticker}.csv"

    print(f"\n[{ticker}] Fetching data from {start_date} to {end_date}...")

    try:
        # Fetch new data from Finnhub
        new_bars = provider.get_daily_bars(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        print(f"[{ticker}] Fetched {len(new_bars)} bars from Finnhub.")

        # If no new data, skip
        if new_bars.empty:
            print(f"[{ticker}] No data returned from Finnhub for date range. Skipping.")
            return True

        # Check if existing file exists
        if output_file.exists():
            print(f"[{ticker}] Found existing file: {output_file}")
            try:
                existing_bars = read_raw_price_csv(output_file, instrument_name=ticker)
                print(f"[{ticker}] Loaded {len(existing_bars)} existing bars from CSV.")

                # Merge with existing data
                mode_str = "force" if force else "incremental"
                print(f"[{ticker}] Merging in {mode_str} mode...")
                merged_bars = merge_price_history(
                    existing=existing_bars,
                    new=new_bars,
                    start_date=start_date,
                    end_date=end_date,
                    force=force,
                )
                print(f"[{ticker}] After merge: {len(merged_bars)} bars total.")

                final_bars = merged_bars
            except Exception as e:
                print(f"[{ticker}] ERROR: Failed to read/merge existing file: {e}")
                print(f"[{ticker}] Will overwrite with new data.")
                final_bars = new_bars
        else:
            print(f"[{ticker}] No existing file. Creating new CSV.")
            final_bars = new_bars

        # Write to CSV
        write_raw_price_csv(final_bars, output_file)
        print(f"[{ticker}] ✓ Wrote {len(final_bars)} bars to {output_file}")

        return True

    except Exception as e:
        print(f"[{ticker}] ERROR: {e}")
        return False


def main():
    """
    Main entry point for the Finnhub data fetch script.

    **Workflow**:
      1. Parse command-line arguments
      2. Load Finnhub settings from environment
      3. Initialize FinnhubDataProvider
      4. For each ticker:
         a. Fetch data
         b. Merge with existing (if present)
         c. Write to CSV
         d. Sleep for rate limiting
      5. Print summary

    **Exit codes**:
      - 0: Success (all tickers fetched)
      - 1: Partial failure (some tickers failed)
      - 2: Total failure (settings error or all tickers failed)
    """
    parser = argparse.ArgumentParser(
        description="Fetch historical price data from Finnhub and save to data/raw/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--start",
        type=str,
        default="1970-01-01",
        help="Start date (YYYY-MM-DD). Default: 1970-01-01 (all available history).",
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today.",
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help=f"Comma-separated list of tickers. Default: {','.join(DEFAULT_TICKERS)}",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force mode: overwrite existing data in [start, end] range. Default: incremental (fill gaps only).",
    )

    args = parser.parse_args()

    # Parse start date
    try:
        start_date = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    except ValueError:
        print(f"ERROR: Invalid start date format: {args.start}. Expected YYYY-MM-DD.")
        sys.exit(2)

    # Parse end date (default to today)
    if args.end is None:
        end_date = dt.date.today()
    else:
        try:
            end_date = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
        except ValueError:
            print(f"ERROR: Invalid end date format: {args.end}. Expected YYYY-MM-DD.")
            sys.exit(2)

    # Validate date range
    if start_date > end_date:
        print(f"ERROR: start_date ({start_date}) must be <= end_date ({end_date}).")
        sys.exit(2)

    # Parse tickers
    if args.tickers is None:
        tickers = DEFAULT_TICKERS
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]

    if not tickers:
        print("ERROR: No tickers specified.")
        sys.exit(2)

    # Load settings
    try:
        settings = get_settings(require_finnhub=True)
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Please set FINNHUB_API_KEY in your .env file.")
        print("Get a free API key at https://finnhub.io/register")
        sys.exit(2)

    # Initialize provider
    provider = FinnhubDataProvider(settings.finnhub)

    # Output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("Finnhub Price Data Fetch")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Mode: {'FORCE (overwrite)' if args.force else 'INCREMENTAL (fill gaps)'}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Rate limit delay: {settings.finnhub.min_sleep_seconds}s between requests")
    print("=" * 60)

    # Fetch each ticker
    success_count = 0
    failure_count = 0

    for i, ticker in enumerate(tickers):
        # Rate limiting: sleep before each request (except first); see DEV_NOTES for Finnhub rate limit guidance.
        if i > 0 and settings.finnhub.min_sleep_seconds > 0:
            print(f"\n[Rate limit] Sleeping {settings.finnhub.min_sleep_seconds}s...")
            time.sleep(settings.finnhub.min_sleep_seconds)

        # Fetch and save
        try:
            success = fetch_and_save_ticker(
                provider=provider,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                force=args.force,
            )
        except FinnhubAccessError:
            print(f"[FINNHUB] Ticker={ticker} → access denied by Finnhub (plan does not include stock/candle for this symbol). Skipping.")
            # TODO: If FinnhubAccessError is raised, consider falling back to another provider via registry once wired in.
            success = False
        except Exception as e:
            print(f"[FINNHUB] Ticker={ticker} → unexpected error: {e}. Skipping.")
            success = False

        if success:
            success_count += 1
        else:
            failure_count += 1

    # Close provider
    provider.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Fetch complete")
    print("=" * 60)
    print(f"Success: {success_count}/{len(tickers)} tickers")
    print(f"Failures: {failure_count}/{len(tickers)} tickers")
    print("=" * 60)

    # Exit with appropriate code
    if failure_count == 0:
        sys.exit(0)  # Success
    elif success_count == 0:
        sys.exit(2)  # Total failure
    else:
        sys.exit(1)  # Partial failure


if __name__ == "__main__":
    main()
