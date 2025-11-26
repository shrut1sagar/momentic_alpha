#!/usr/bin/env python3
"""
Normalize timestamps in all existing CSV files to canonical format.

**Purpose**: This migration script converts legacy "YYYY-MM-DDTHH:MM:SS" timestamps
to the new canonical format "YYYY-MM-DD HH:MM:SS" across all project CSV files.

**What it does**:
  1. Scans data/raw/, data/processed/, and data/results/ for CSV files
  2. For each file with a 'timestamp' column:
     - Reads the file into a DataFrame
     - Normalizes the timestamp column (format + descending order)
     - Writes back to the same file path
  3. Prints a summary of changes (row count, date range)

**When to run**: Run this once after deploying the timestamp normalization changes
to convert all existing data files to the new format.

**Usage**:
    From project root:
    ```bash
    python actions/normalise_all_timestamps.py
    ```

**Safety**:
  - Overwrites files in place (make a backup first if needed)
  - Only touches files with a 'timestamp' column
  - Preserves all other columns unchanged
  - Enforces descending sort order (newest first)

**Teaching note**: Migration scripts are common when changing data formats.
Good practices:
  - Make it idempotent (safe to run multiple times)
  - Print clear output showing what changed
  - Test on a copy of data first
  - Consider adding a --dry-run flag for safety
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is on path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.io import normalize_timestamp_column


def normalize_csv_file(file_path: Path, dry_run: bool = False) -> dict | None:
    """
    Normalize timestamps in a single CSV file.

    **Process**:
      1. Read CSV into DataFrame
      2. Check if 'timestamp' column exists
      3. If yes: normalize timestamps and write back
      4. Return summary dict with file info

    Args:
        file_path: Path to CSV file to normalize.
        dry_run: If True, only report what would be done without writing.

    Returns:
        Dict with keys 'file', 'rows', 'min_date', 'max_date' if normalized.
        None if file has no timestamp column or is empty.
    """
    try:
        # Read CSV
        df = pd.read_csv(file_path)

        # Skip if no timestamp column
        if 'timestamp' not in df.columns:
            return None

        # Skip if empty
        if df.empty:
            return None

        # Normalize timestamps
        df_normalized = normalize_timestamp_column(df, col='timestamp')

        # Extract summary info before writing
        min_timestamp = df_normalized['timestamp'].min()
        max_timestamp = df_normalized['timestamp'].max()
        row_count = len(df_normalized)

        # Write back if not dry run
        if not dry_run:
            # Convert timestamp to canonical string format
            df_normalized['timestamp'] = df_normalized['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_normalized.to_csv(file_path, index=False)

        return {
            'file': str(file_path.relative_to(repo_root)),
            'rows': row_count,
            'min_date': min_timestamp.strftime('%Y-%m-%d'),
            'max_date': max_timestamp.strftime('%Y-%m-%d'),
        }

    except Exception as e:
        print(f"  ✗ Error processing {file_path.relative_to(repo_root)}: {e}")
        return None


def main():
    """
    Main entrypoint for timestamp normalization migration.

    Steps:
      1. Define directories and patterns to scan
      2. Find all matching CSV files
      3. Normalize each file with a timestamp column
      4. Print summary of changes
    """
    print("=" * 80)
    print("Timestamp Normalization Migration")
    print("=" * 80)
    print()
    print("This script normalizes timestamps in all CSV files to the canonical format:")
    print("  - Format: YYYY-MM-DD HH:MM:SS (space, not 'T')")
    print("  - Timezone: UTC")
    print("  - Sort order: Descending (newest first)")
    print()

    # Define directories and patterns to scan
    # These are the locations where time-series CSV files live
    directories_to_scan = [
        repo_root / "data" / "raw",
        repo_root / "data" / "processed",
        repo_root / "data" / "results",
    ]

    # Collect all CSV files
    csv_files = []
    for directory in directories_to_scan:
        if directory.exists():
            csv_files.extend(directory.glob("*.csv"))
            # Also scan subdirectories
            csv_files.extend(directory.glob("**/*.csv"))

    # Remove duplicates (from nested globs)
    csv_files = sorted(set(csv_files))

    print(f"Found {len(csv_files)} CSV files to scan")
    print()

    # Normalize each file
    print("Processing files...")
    print("-" * 80)

    results = []
    skipped_count = 0

    for csv_file in csv_files:
        result = normalize_csv_file(csv_file, dry_run=False)

        if result is not None:
            results.append(result)
            print(f"  ✓ {result['file']:50s} {result['rows']:6d} rows  "
                  f"{result['min_date']} to {result['max_date']}")
        else:
            skipped_count += 1

    print("-" * 80)
    print()

    # Print summary
    print("=" * 80)
    print("Migration Complete!")
    print("=" * 80)
    print()
    print(f"Files processed:    {len(results)}")
    print(f"Files skipped:      {skipped_count} (no timestamp column or empty)")
    print(f"Total rows updated: {sum(r['rows'] for r in results):,}")
    print()
    print("All timestamps are now in canonical format: YYYY-MM-DD HH:MM:SS")
    print()

    # Show example verification command
    if results:
        example_file = results[0]['file']
        print("Verify changes with:")
        print(f"  head data/{example_file.split('/')[-1]}")
        print()


if __name__ == "__main__":
    main()
