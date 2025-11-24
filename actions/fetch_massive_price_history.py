#!/usr/bin/env python3
"""
Fetch historical price data from Massive.com and save to data/raw/.

**Purpose**: This script demonstrates end-to-end usage of the Massive data
provider infrastructure. It fetches daily OHLCV bars from Massive API and
saves them to CSV files in the project's data/raw/ directory.

**Usage**:
    python actions/fetch_massive_price_history.py QQQ --start 2024-01-01
    python actions/fetch_massive_price_history.py QQQ --start 2024-01-01 --end 2024-12-31
    python actions/fetch_massive_price_history.py SPY TQQQ SQQQ --start 2023-01-01

**What this script does**:
  1. Parse command line arguments (symbols, date range)
  2. Load Massive API settings from environment (.env file)
  3. Create MassiveDataProvider
  4. For each symbol:
     a. Fetch daily bars from Massive API
     b. Validate data (check for gaps, anomalies)
     c. Save to data/raw/{symbol}_daily.csv
  5. Print summary (rows fetched, date range, file location)

**Why this script is needed**:
  - Populate data/raw/ with fresh data from Massive
  - One-time setup for new symbols
  - Periodic updates (run weekly/monthly to get latest data)
  - Demonstrates correct usage of MassiveDataProvider

**Teaching note**: This is a "runnable example" - it shows how all the pieces
(settings, client, provider, I/O) fit together. Study this script to understand
the full data ingestion workflow.

**Requirements**:
  - MASSIVE_API_KEY set in .env file
  - Network access to Massive API
  - Write permissions to data/raw/ directory

**Example output**:
    $ python actions/fetch_massive_price_history.py QQQ 2024-01-01 2024-12-31
    Loading Massive settings from environment...
    Fetching QQQ daily bars from 2024-01-01 to 2024-12-31...
    ✓ Fetched 252 bars for QQQ
    ✓ Date range: 2024-01-01 to 2024-12-31
    ✓ Saved to data/raw/QQQ_daily.csv
    Done!
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to Python path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import get_settings
from src.venues.massive_data_provider import MassiveDataProvider
from src.venues.massive_client import (
    MassiveAuthenticationError,
    MassiveSymbolNotFoundError,
    MassiveRateLimitError,
    MassiveServerError,
)
from src.data.io import read_raw_price_csv, write_raw_price_csv


def parse_args():
    """
    Parse command line arguments.

    **Conceptual**: argparse is Python's standard library for CLI argument
    parsing. It provides automatic help text, type validation, and error messages.

    **Why use argparse over manual parsing?**
      - Automatic --help generation
      - Type validation (dates, required args, etc.)
      - Clear error messages for invalid inputs
      - Standard Unix-style CLI conventions

    Returns:
        Namespace with attributes: symbols (list), start (str), end (str)

    Example:
        >>> args = parse_args()  # python script.py QQQ SPY --start 2024-01-01 --end 2024-12-31
        >>> print(args.symbols)  # ['QQQ', 'SPY']
        >>> print(args.start)  # '2024-01-01'
    """
    parser = argparse.ArgumentParser(
        description="Fetch historical price data from Massive.com",
        epilog="""
Examples:
  # Fetch QQQ from 2024-01-01 to today (end date defaults to today)
  python actions/fetch_massive_price_history.py QQQ --start 2024-01-01

  # Fetch QQQ for specific date range
  python actions/fetch_massive_price_history.py QQQ --start 2024-01-01 --end 2024-12-31

  # Fetch multiple symbols to today
  python actions/fetch_massive_price_history.py QQQ SPY TQQQ --start 2024-01-01

  # Force mode: replace existing data in date range
  python actions/fetch_massive_price_history.py QQQ --start 2024-06-01 --end 2024-06-30 --force
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "symbols",
        nargs="+",
        help="One or more ticker symbols to fetch (e.g., QQQ SPY TQQQ)",
    )

    parser.add_argument(
        "--start",
        type=str,
        help="Start date in YYYY-MM-DD format (default: 2020-01-01)",
        default="2020-01-01",
    )

    parser.add_argument(
        "--end",
        type=str,
        help="End date in YYYY-MM-DD format (default: today)",
        default=None,  # Will be set to today if not provided
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for CSV files (default: data/raw/)",
        default="data/raw",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force mode: replace existing data in the specified date range (default: incremental)",
    )

    args = parser.parse_args()

    # If end date not provided, use today
    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    return args


def validate_date(date_str: str) -> pd.Timestamp:
    """
    Validate and parse date string to timezone-aware timestamp.

    **Conceptual**: Convert user input (string) to validated pd.Timestamp.
    We require UTC timezone for consistency.

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns:
        Timezone-aware pd.Timestamp (UTC).

    Raises:
        ValueError: If date string is invalid.

    Example:
        >>> ts = validate_date("2024-01-01")
        >>> print(ts)  # Timestamp('2024-01-01 00:00:00+0000', tz='UTC')
    """
    try:
        # Parse date string and add UTC timezone
        ts = pd.Timestamp(date_str, tz="UTC")
        return ts
    except Exception as e:
        raise ValueError(
            f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD. Error: {e}"
        ) from e


def merge_price_history(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    force: bool = False,
) -> pd.DataFrame:
    """
    Merge existing price history with newly fetched data.

    **Conceptual**: This function implements incremental and force update modes:
      - Incremental mode (force=False): Preserves existing data, only adds new dates.
      - Force mode (force=True): Replaces data in the specified date range with new data.

    **Why incremental by default?**
      - Avoids overwriting existing data unless explicitly requested.
      - Faster (only fetch missing dates if API supports it).
      - Safer (preserves historical data from past fetches).

    **When to use force mode?**
      - Backfill corrections (vendor fixed bad data).
      - Re-fetch after data quality issues.
      - Update split-adjusted prices.

    **Merge logic**:
      Incremental mode (force=False):
        1. Keep all existing rows outside [start, end].
        2. Keep existing rows inside [start, end] that are NOT in new_df.
        3. Add all rows from new_df (only new dates).
        4. Result: Existing data preserved, new dates added.

      Force mode (force=True):
        1. Keep all existing rows outside [start, end].
        2. Discard existing rows inside [start, end].
        3. Add all rows from new_df.
        4. Result: Data in [start, end] replaced with new_df.

    **Teaching note**: Incremental updates are common in data pipelines. The key
    decision is: do we trust existing data (incremental) or re-fetch everything
    (force)? Incremental is safer and faster; force is useful for corrections.

    Args:
        existing_df: Existing price history (any sort order).
        new_df: Newly fetched data (any sort order).
        start: Start of requested date range (inclusive).
        end: End of requested date range (inclusive).
        force: If True, replace data in [start, end]; if False, only add new dates.

    Returns:
        Merged DataFrame (unsorted - caller should sort before writing).

    Example (incremental mode):
        >>> existing = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02']),
        ...     'closing_price': [100.0, 101.0],
        ... })
        >>> new = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2024-01-02', '2024-01-03']),
        ...     'closing_price': [101.5, 102.0],
        ... })
        >>> merged = merge_price_history(existing, new, start, end, force=False)
        >>> # Result has [2024-01-01, 2024-01-02 (from existing), 2024-01-03 (from new)]
        >>> # 2024-01-02 from existing is preserved

    Example (force mode):
        >>> merged = merge_price_history(existing, new, start, end, force=True)
        >>> # Result has [2024-01-01 (outside range), 2024-01-02 (from new), 2024-01-03 (from new)]
        >>> # 2024-01-02 from existing is replaced with version from new
    """
    # Ensure timestamp columns are datetime
    existing_df = existing_df.copy()
    new_df = new_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(existing_df['timestamp']):
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
    if not pd.api.types.is_datetime64_any_dtype(new_df['timestamp']):
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

    # Make timestamps timezone-aware if they aren't already
    if existing_df['timestamp'].dt.tz is None:
        existing_df['timestamp'] = existing_df['timestamp'].dt.tz_localize('UTC')
    if new_df['timestamp'].dt.tz is None:
        new_df['timestamp'] = new_df['timestamp'].dt.tz_localize('UTC')

    # Normalize start/end to be timezone-aware
    if start.tz is None:
        start = start.tz_localize('UTC')
    if end.tz is None:
        end = end.tz_localize('UTC')

    if force:
        # Force mode: Replace data in [start, end] with new_df
        # Keep existing data outside [start, end]
        outside_range = existing_df[
            (existing_df['timestamp'] < start) | (existing_df['timestamp'] > end)
        ]
        # Combine with new_df
        merged = pd.concat([outside_range, new_df], ignore_index=True)
    else:
        # Incremental mode: Only add dates from new_df that aren't in existing_df
        # Keep all existing data (inside and outside range)
        # Add only new dates from new_df
        existing_dates = set(existing_df['timestamp'])
        new_rows = new_df[~new_df['timestamp'].isin(existing_dates)]
        merged = pd.concat([existing_df, new_rows], ignore_index=True)

    # Remove duplicates (shouldn't happen, but defensive)
    merged = merged.drop_duplicates(subset=['timestamp'], keep='last')

    return merged


def fetch_and_save_symbol(
    provider: MassiveDataProvider,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    force: bool = False,
) -> None:
    """
    Fetch daily bars for a symbol and save to CSV with incremental/force mode.

    **Conceptual**: This function encapsulates the fetch-validate-merge-save workflow
    for a single symbol. It supports both incremental updates (default) and force
    mode (--force flag).

    **Workflow**:
      1. Check if CSV file already exists
      2. Fetch new data from Massive API
      3. If file exists:
         - Load existing data
         - Merge with new data (incremental or force mode)
      4. Save merged data to CSV (via write_raw_price_csv for schema enforcement)
      5. Print summary

    **Incremental vs force mode**:
      - Incremental (default): Only add new dates, preserve existing data
      - Force (--force): Replace data in [start, end] with newly fetched data

    Args:
        provider: MassiveDataProvider instance.
        symbol: Ticker symbol to fetch.
        start: Start date (inclusive).
        end: End date (inclusive).
        output_dir: Directory to save CSV files.
        force: If True, replace existing data in range; if False, incremental.

    Raises:
        Various exceptions from provider (auth, not found, timeout, etc.)
    """
    mode_str = "force" if force else "incremental"
    print(f"Fetching {symbol} daily bars from {start.date()} to {end.date()} (mode: {mode_str})...")

    # Construct output file path
    output_file = output_dir / f"{symbol}_daily.csv"

    # Check if file already exists
    file_exists = output_file.exists()

    # Fetch new data from Massive
    new_bars = provider.fetch_daily_bars(symbol=symbol, start=start, end=end)

    # Validate: check that we got some data
    if len(new_bars) == 0:
        print(f"  ⚠ No data returned for {symbol} in range {start.date()} to {end.date()}")
        print(f"  (This may be normal if symbol didn't exist yet or no trading days in range)")
        return

    print(f"  ✓ Fetched {len(new_bars)} new bars from API")

    # Merge with existing data if file exists
    if file_exists:
        try:
            existing_bars = read_raw_price_csv(output_file, instrument_name=symbol)
            existing_count = len(existing_bars)
            print(f"  ✓ Loaded {existing_count} existing bars from {output_file}")

            # Merge existing with new
            merged_bars = merge_price_history(existing_bars, new_bars, start, end, force=force)
            merged_count = len(merged_bars)

            # Calculate stats
            added_count = merged_count - existing_count
            print(f"  ✓ Merged: {existing_count} existing + {len(new_bars)} new = {merged_count} total ({added_count:+d} net)")

            final_bars = merged_bars
        except Exception as e:
            print(f"  ⚠ Warning: Could not load existing file: {e}")
            print(f"  ⚠ Writing new data only (existing data may be lost)")
            final_bars = new_bars
    else:
        print(f"  ✓ No existing file found, writing {len(new_bars)} new bars")
        final_bars = new_bars

    # Check actual date range in final data
    # Note: Data can be in any order at this point, so sort for display
    timestamps = final_bars['timestamp'].sort_values()
    oldest_date = timestamps.iloc[0].date()
    newest_date = timestamps.iloc[-1].date()
    print(f"  ✓ Final date range: {oldest_date} to {newest_date}")

    # Save to CSV using write_raw_price_csv (enforces schema and descending order)
    write_raw_price_csv(final_bars, output_file)
    print(f"  ✓ Saved to {output_file}")


def main():
    """
    Main entry point for the script.

    **Conceptual**: The main() function orchestrates the entire workflow:
    load settings → create provider → fetch each symbol → save results.

    **Error handling strategy**:
      - Catch auth errors early (before fetching any symbols)
      - Catch per-symbol errors (rate limit, not found) and continue to next symbol
      - Let unexpected errors bubble up (with stack trace for debugging)

    **Exit codes**:
      - 0: Success (all symbols fetched)
      - 1: Configuration error (missing API key, invalid dates)
      - 2: Fatal error (network down, server error)
    """
    try:
        # Parse command line arguments
        args = parse_args()

        # Validate dates
        try:
            start = validate_date(args.start)
            end = validate_date(args.end)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if start > end:
            print(f"Error: start date ({start.date()}) must be <= end date ({end.date()})", file=sys.stderr)
            sys.exit(1)

        # Ensure output directory exists
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load Massive settings from environment
        print("Loading Massive settings from environment...")
        try:
            settings = get_settings(require_massive=True)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("", file=sys.stderr)
            print("Please set MASSIVE_API_KEY in your .env file:", file=sys.stderr)
            print("  MASSIVE_API_KEY=your_key_here", file=sys.stderr)
            print("  MASSIVE_BASE_URL=https://api.massive.com  # optional", file=sys.stderr)
            sys.exit(1)

        # Create data provider
        provider = MassiveDataProvider(settings.massive)

        print(f"Fetching {len(args.symbols)} symbol(s) from {start.date()} to {end.date()}")
        print("")

        # Fetch each symbol
        success_count = 0
        error_count = 0

        for symbol in args.symbols:
            try:
                fetch_and_save_symbol(provider, symbol, start, end, output_dir, force=args.force)
                success_count += 1
                print("")  # Blank line between symbols

            except MassiveAuthenticationError as e:
                # Auth error - this is fatal, don't continue
                print(f"Error: Authentication failed: {e}", file=sys.stderr)
                print("Check your MASSIVE_API_KEY in .env file", file=sys.stderr)
                sys.exit(2)

            except MassiveSymbolNotFoundError as e:
                # Symbol not found - log and continue to next symbol
                print(f"  ✗ Symbol '{symbol}' not found: {e}")
                print("")
                error_count += 1
                continue

            except MassiveRateLimitError as e:
                # Rate limit - warn and exit (don't hammer API)
                print(f"Error: Rate limit exceeded: {e}", file=sys.stderr)
                print("Slow down requests or wait before retrying.", file=sys.stderr)
                sys.exit(2)

            except MassiveServerError as e:
                # Server error - log and continue (may be temporary)
                print(f"  ✗ Server error for '{symbol}': {e}")
                print("")
                error_count += 1
                continue

            except Exception as e:
                # Unexpected error - log and continue
                print(f"  ✗ Unexpected error for '{symbol}': {e}")
                print("")
                error_count += 1
                continue

        # Print final summary
        print("=" * 60)
        print(f"Done! Successfully fetched {success_count}/{len(args.symbols)} symbols.")
        if error_count > 0:
            print(f"Failed: {error_count} symbols (see errors above)")

        # Exit with error code if any failures
        sys.exit(0 if error_count == 0 else 1)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(130)  # Standard Unix exit code for Ctrl+C

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
