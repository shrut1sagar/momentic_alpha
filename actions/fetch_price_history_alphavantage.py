#!/usr/bin/env python3
"""
Fetch historical price data from Alpha Vantage and save to data/raw/.

**Conceptual**: This script fetches daily OHLCV bars from Alpha Vantage for multiple
tickers and writes them to data/raw/{TICKER}.csv in canonical format. It supports
incremental updates (fill gaps only) and force mode (overwrite date range).

**Usage**:
    # Fetch default universe with default date range
    python actions/fetch_price_history_alphavantage.py

    # Fetch specific tickers
    python actions/fetch_price_history_alphavantage.py --tickers QQQ,SPY,TQQQ

    # Fetch from specific start date
    python actions/fetch_price_history_alphavantage.py --start 2020-01-01

    # Force overwrite existing data in date range
    python actions/fetch_price_history_alphavantage.py --tickers QQQ --force

**Default universe**:
    QQQ, SQQQ, TQQQ, SPY, VIX, UVXY, VIXM, VIXY, VIXX, GLD, OIL, TLT
    (12 tickers - uses nearly half the 25/day free tier quota!)

**Date handling**:
    - Default start: 1970-01-01 (fetch all available history)
    - Default end: today
    - Alpha Vantage typically has data from 1999 onwards for major ETFs

**Incremental vs Force mode**:
    - Incremental (default): Fill gaps in existing data without overwriting
    - Force (--force): Overwrite date range [start, end] with fresh data

**Rate limiting - CRITICAL**:
    - Alpha Vantage free tier: 25 API calls per day (!), 5 calls per minute
    - This script adds 12-second delay between requests (5 calls/min = 12s apart)
    - BE VERY CONSERVATIVE: fetch few tickers, use incremental mode, cache locally
    - Consider using multiple providers to spread load
    - Paid tiers have higher limits

**Error handling**:
    - Per-ticker errors don't crash the entire script
    - Continues to next ticker after logging error
    - Rate limit errors are caught and reported clearly
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
from src.venues.alphavantage_data_provider import (
    AlphaVantageDataProvider,
    AlphaVantageRateLimitError,
    AlphaVantageApiError,
)
from src.data.io import read_raw_price_csv, write_raw_price_csv


# Default ticker universe
# WARNING: 12 tickers = 12 API calls (nearly half the 25/day free tier quota!)
# Consider fetching in batches or using incremental mode to conserve quota
DEFAULT_TICKERS = [
    "QQQ",    # Nasdaq 100 ETF
    "SQQQ",   # 3x inverse Nasdaq 100 ETF
    "TQQQ",   # 3x leveraged Nasdaq 100 ETF
    "SPY",    # S&P 500 ETF
    "VIX",    # VIX volatility index
    "UVXY",   # 2x VIX short-term futures ETF
    "VIXM",   # VIX mid-term futures ETF
    "VIXY",   # VIX short-term futures ETF
    "VIXX",   # VIX futures ETF (if available)
    "GLD",    # Gold ETF
    "OIL",    # Oil ETF (e.g., USO)
    "TLT",    # 20+ Year Treasury Bond ETF
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
    provider: AlphaVantageDataProvider,
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
      2. Fetch new data from Alpha Vantage
      3. Merge with existing data (incremental or force)
      4. Write to CSV

    Args:
        provider: AlphaVantageDataProvider instance.
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
        # Fetch new data from Alpha Vantage
        new_bars = provider.get_daily_bars(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        print(f"[{ticker}] Fetched {len(new_bars)} bars from Alpha Vantage.")

        # If no new data, skip
        if new_bars.empty:
            print(f"[{ticker}] No data returned from Alpha Vantage for date range. Skipping.")
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

    except AlphaVantageRateLimitError as e:
        print(f"[{ticker}] ERROR: Rate limit exceeded - {e}")
        print(f"[{ticker}] Alpha Vantage free tier: 25 calls/day, 5 calls/minute")
        print(f"[{ticker}] Consider waiting or using incremental mode to reduce API calls.")
        # Propagate to stop further tickers in this run to conserve quota
        raise

    except AlphaVantageApiError as e:
        print(f"[{ticker}] ERROR: Alpha Vantage API error - {e}")
        return False

    except Exception as e:
        print(f"[{ticker}] ERROR: Unexpected error - {e}")
        return False


def main():
    """
    Main entry point for the Alpha Vantage data fetch script.

    **Workflow**:
      1. Parse command-line arguments
      2. Load Alpha Vantage settings from environment
      3. Initialize AlphaVantageDataProvider
      4. For each ticker:
         a. Fetch data
         b. Merge with existing (if present)
         c. Write to CSV
         d. Sleep for rate limiting (12 seconds = 5 calls/min)
      5. Print summary

    **Exit codes**:
      - 0: Success (all tickers fetched)
      - 1: Partial failure (some tickers failed)
      - 2: Total failure (settings error or all tickers failed)
    """
    parser = argparse.ArgumentParser(
        description="Fetch historical price data from Alpha Vantage and save to data/raw/",
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
        settings = get_settings(require_alphavantage=True)
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Please set ALPHAVANTAGE_API_KEY in your .env file.")
        print("Get a free API key at https://www.alphavantage.co/support/#api-key")
        sys.exit(2)

    # Initialize provider
    provider = AlphaVantageDataProvider(settings.alphavantage)

    # Output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("Alpha Vantage Price Data Fetch")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Mode: {'FORCE (overwrite)' if args.force else 'INCREMENTAL (fill gaps)'}")
    print(f"Output directory: {output_dir.absolute()}")

    # Determine rate limit delay (use settings if configured, else default to 12s for 5 calls/min)
    rate_limit_delay = settings.alphavantage.min_sleep_seconds
    if rate_limit_delay <= 0:
        rate_limit_delay = 12.0  # Default: 5 calls/min = 12s between calls
    print(f"Rate limit delay: {rate_limit_delay}s between requests")

    print("=" * 60)
    print("\nWARNING: Alpha Vantage free tier has strict limits:")
    print("  - 25 API calls per day")
    print("  - 5 API calls per minute")
    print(f"  - You are about to make {len(tickers)} API call(s)")
    print("=" * 60)

    # Fetch each ticker
    success_count = 0
    failure_count = 0

    for i, ticker in enumerate(tickers):
        # Rate limiting: sleep before each request (except first)
        # Alpha Vantage free tier: 5 calls/minute (default 12s between calls)
        if i > 0 and rate_limit_delay > 0:
            print(f"\n[Rate limit] Sleeping {rate_limit_delay}s...")
            time.sleep(rate_limit_delay)

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
        except AlphaVantageRateLimitError:
            print(f"[{ticker}] Rate limit reached. Stopping further requests to conserve quota.")
            failure_count += 1
            break

        if success:
            success_count += 1
        else:
            failure_count += 1

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
