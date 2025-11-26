"""
CSV readers and writers with schema enforcement.

**Conceptual**: This module is the *only* I/O boundary for CSV data in the system.
All raw price data, processed features, and backtest results must pass through
these functions to ensure schema compliance. This centralization provides:
  - Consistent timestamp handling (ISO 8601 strings <-> datetime objects).
  - Automatic validation (via schemas.py) at every read/write.
  - Guaranteed sort order (strictly descending by timestamp, newest first).
  - Clear error messages when data doesn't conform.

**Rule**: Never use pd.read_csv or df.to_csv directly in strategies, engines,
or orchestration code. Always import and use these functions instead. This
keeps data contracts enforceable and makes AWS migration easier (we can swap
out local CSV I/O for S3 reads/writes in one place).

**Teaching note**: In production quant systems, centralizing I/O behind a
validation layer is essential. Ad-hoc reads/writes lead to schema drift,
silent bugs, and non-reproducible results. By enforcing schemas here, we
ensure that every downstream consumer (strategies, backtests, metrics) can
trust the data format.
"""

import pandas as pd
from pathlib import Path

from src.data.schemas import (
    validate_raw_price_schema,
    validate_processed_schema,
    SchemaValidationError,
)


def normalize_timestamp_column(
    df: pd.DataFrame,
    col: str = "timestamp",
    ensure_date_column: bool = False,
) -> pd.DataFrame:
    """
    Normalize a timestamp column to the project's canonical format.

    **Canonical format**:
    - Column name: "timestamp"
    - String format on disk: "YYYY-MM-DD HH:MM:SS" (space, not 'T')
    - Timezone: Treated as UTC (no explicit timezone stored)
    - Sort order: Strictly descending by timestamp (newest first)

    **Robustness**:
    - Handles timestamps that are already datetime64 dtype (no re-parsing needed)
    - Handles timestamps that are strings in either legacy or new format
    - Returns empty DataFrames unchanged (no crash on empty data)
    - Preserves all other columns unchanged

    **Functionally**:
    - Parses the timestamp column using pd.to_datetime if not already datetime
      (handles both "YYYY-MM-DDTHH:MM:SS" and "YYYY-MM-DD HH:MM:SS" formats)
    - Sorts rows in strictly descending order by timestamp (newest first)
    - Optionally adds a human-readable "date" column (YYYY-MM-DD) for inspection
    - Leaves timestamp as datetime64 dtype for in-memory use

    **Why this function exists**:
    - Centralizes timestamp normalization logic for consistency
    - Makes it easy to standardize CSVs written by actions/scripts
    - Ensures all CSV outputs follow the same canonical format
    - Provides a single place to update if format changes

    **Teaching note**: Timestamp handling is a common source of bugs in quant systems.
    By centralizing normalization, we ensure that all CSVs (raw data, features,
    backtest results) follow the same conventions. This makes data inspection,
    debugging, and validation much easier.

    Args:
        df: DataFrame containing a timestamp column.
        col: Name of the timestamp column (default: "timestamp").
        ensure_date_column: If True, adds a "date" column (YYYY-MM-DD) for
                           human-readable inspection. Defaults to False.

    Returns:
        DataFrame with normalized timestamp column, sorted descending by timestamp.
        The timestamp column remains as datetime64 dtype (not converted to string).

    Raises:
        KeyError: If the specified timestamp column doesn't exist.
        ValueError: If the timestamp column can't be parsed as datetime.

    Example:
        >>> # Works with string timestamps
        >>> df = pd.DataFrame({
        ...     'timestamp': ['2024-01-15T00:00:00', '2024-01-14 12:00:00'],
        ...     'value': [100, 200]
        ... })
        >>> normalized = normalize_timestamp_column(df)
        >>> normalized['timestamp'].dtype
        dtype('<M8[ns]')
        >>> normalized['timestamp'].iloc[0] > normalized['timestamp'].iloc[1]
        True  # Descending order confirmed

        >>> # Works with already-parsed datetimes
        >>> df2 = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2024-01-15', '2024-01-14']),
        ...     'value': [100, 200]
        ... })
        >>> normalized2 = normalize_timestamp_column(df2)
        >>> normalized2['timestamp'].dtype
        dtype('<M8[ns]')
    """
    # Handle empty DataFrame gracefully - return unchanged
    if df.empty:
        return df.copy()

    # Make a copy to avoid modifying the input
    df_normalized = df.copy()

    # Check if column exists
    if col not in df_normalized.columns:
        raise KeyError(
            f"Timestamp column '{col}' not found in DataFrame. "
            f"Available columns: {list(df_normalized.columns)}"
        )

    # Parse timestamp column to datetime if it's not already datetime type
    # This handles both string timestamps and already-parsed datetime objects
    if not pd.api.types.is_datetime64_any_dtype(df_normalized[col]):
        # Column is not datetime - parse it
        try:
            # pd.to_datetime with format='ISO8601' handles:
            # - "YYYY-MM-DDTHH:MM:SS" (legacy format with 'T')
            # - "YYYY-MM-DD HH:MM:SS" (new canonical format with space)
            # - Mixed formats in the same column
            df_normalized[col] = pd.to_datetime(df_normalized[col], format='ISO8601')
        except Exception as e:
            raise ValueError(
                f"Failed to parse '{col}' column as datetime. "
                f"Expected ISO 8601 format (e.g., '2024-01-15 00:00:00' or '2024-01-15T00:00:00'). "
                f"Error: {e}"
            )
    # else: Column is already datetime64 dtype - no parsing needed

    # Sort by timestamp in strictly descending order (newest first)
    # This is the canonical sort order for all time-series data in this project
    df_normalized = df_normalized.sort_values(col, ascending=False).reset_index(drop=True)

    # Optionally add a "date" column for human inspection
    # This provides a human-readable date without the time component
    if ensure_date_column and 'date' not in df_normalized.columns:
        df_normalized['date'] = df_normalized[col].dt.strftime('%Y-%m-%d')

    return df_normalized


def write_normalized_csv(
    df: pd.DataFrame,
    path: Path | str,
    timestamp_col: str = "timestamp",
    ensure_date_column: bool = False,
) -> None:
    """
    Write a DataFrame to CSV with normalized timestamps and canonical format.

    **Canonical format on disk**:
    - Timestamps written as "YYYY-MM-DD HH:MM:SS" (space, not 'T')
    - Rows sorted strictly descending by timestamp (newest first)
    - Optional "date" column for human inspection
    - All other columns preserved unchanged

    **Robustness**:
    - Handles DataFrames where timestamp is already datetime64 or still string
    - Handles empty DataFrames (writes empty CSV with headers)
    - Preserves all columns except timestamp (which gets normalized)
    - Creates parent directory if it doesn't exist

    **Functionally**:
    - Calls normalize_timestamp_column() to parse and sort timestamps
    - Converts timestamp datetime to canonical string format for CSV storage
    - Writes CSV with index=False (no row numbers)

    **Teaching note**: This is a convenience wrapper for writing CSVs from
    actions/scripts. It handles all the normalization boilerplate so scripts
    don't have to repeat the same logic. Use this for backtest results,
    analysis outputs, and any CSV that contains a time series.

    Args:
        df: DataFrame to write (must contain timestamp column).
        path: Path where CSV should be written.
        timestamp_col: Name of timestamp column (default: "timestamp").
        ensure_date_column: If True, adds "date" column (YYYY-MM-DD) for
                           human inspection (default: False).

    Raises:
        KeyError: If timestamp column doesn't exist.
        ValueError: If timestamp column can't be parsed as datetime.
        OSError: If file can't be written (permissions, disk full, etc.).

    Example:
        >>> # Works with datetime timestamps
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2024-01-15', '2024-01-14']),
        ...     'value': [100, 200]
        ... })
        >>> write_normalized_csv(df, "output.csv")
        # output.csv will have timestamps as "2024-01-15 00:00:00", sorted descending

        >>> # Also works with string timestamps
        >>> df2 = pd.DataFrame({
        ...     'timestamp': ['2024-01-15T00:00:00', '2024-01-14 12:00:00'],
        ...     'value': [100, 200]
        ... })
        >>> write_normalized_csv(df2, "output.csv")
        # Same result - both formats are normalized to "YYYY-MM-DD HH:MM:SS"
    """
    # Convert path to Path object for consistent handling
    path = Path(path)

    # Ensure parent directory exists (create if necessary)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize timestamp column (parse if string, sort descending, add date column if requested)
    df_to_write = normalize_timestamp_column(df, col=timestamp_col, ensure_date_column=ensure_date_column)

    # Convert timestamp from datetime64 to canonical string format: "YYYY-MM-DD HH:MM:SS"
    # This is the on-disk representation that readers will parse back to datetime
    # Skip formatting if DataFrame is empty (no .dt accessor on empty Series)
    if not df_to_write.empty:
        df_to_write[timestamp_col] = df_to_write[timestamp_col].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Write to CSV with index=False (no row numbers column)
    try:
        df_to_write.to_csv(path, index=False)
    except Exception as e:
        raise OSError(
            f"Failed to write CSV to {path}. Error: {e}"
        )


def read_raw_price_csv(
    path: Path | str,
    instrument_name: str | None = None,
) -> pd.DataFrame:
    """
    Read a raw price CSV file with schema validation.

    **Conceptual**: Loads OHLCV data from disk (e.g., data/raw/QQQ.csv) and
    ensures it conforms to the raw price schema (timestamp, OHLC, volume).
    This function is the gateway for all raw market data entering the system.

    **Functionally**:
      - Reads CSV using pandas.
      - Parses `timestamp` column to datetime objects.
      - Validates schema (required columns, timestamp format, descending order).
      - Returns a DataFrame ready for feature engineering or direct use.

    **Timestamp format handling**:
      - Accepts both legacy "YYYY-MM-DDTHH:MM:SS" and new "YYYY-MM-DD HH:MM:SS" formats
      - Always returns timestamp as datetime64 dtype (not string)
      - pd.to_datetime handles both formats automatically via format='ISO8601'

    **Why parse timestamps on read?**
      - Datetime objects are easier to work with than strings (filtering, sorting,
        diffing, timezone handling).
      - Parsing once at the I/O boundary avoids repeated parsing downstream.
      - Schema validation can check for malformed timestamps immediately.

    **Why validate on read?**
      - Fail fast: detect schema violations before data enters pipelines.
      - Clear errors: user knows exactly which file/column/row is broken.
      - Trust downstream: strategies and engines can assume valid schemas.

    **Teaching note**: Reading data is the first line of defense against bugs.
    By validating here, we prevent bad data from propagating through complex
    pipelines where it would be much harder to diagnose. The cost is a bit of
    upfront validation time; the benefit is correctness and debuggability.

    Args:
        path: Path to the raw price CSV file (e.g., "data/raw/QQQ.csv").
              Can be string or pathlib.Path.
        instrument_name: Optional name of the instrument (e.g., "QQQ") for
                        error message context. If not provided, uses the path.

    Returns:
        DataFrame with columns matching the raw price schema:
          - timestamp (datetime64 dtype, strictly descending)
          - open_price, high_price, low_price, closing_price (float)
          - volume (int or float depending on source)

    Raises:
        FileNotFoundError: If the file doesn't exist.
        SchemaValidationError: If the CSV doesn't conform to raw price schema.
        pd.errors.ParserError: If CSV is malformed (pandas can't parse it).

    Example:
        >>> df = read_raw_price_csv("data/raw/QQQ.csv", instrument_name="QQQ")
        >>> df.head(3)
                  timestamp  open_price  high_price  low_price  closing_price  volume
        0 2024-01-15 00:00:00       400.0       405.0      398.0          403.0  1000000
        1 2024-01-14 00:00:00       398.0       401.0      397.0          400.0  950000
        2 2024-01-13 00:00:00       395.0       399.0      394.0          398.0  900000
    """
    # Convert path to Path object for consistent handling
    path = Path(path)

    # Build context string for error messages
    context = instrument_name or str(path)

    # Check if file exists before attempting to read
    if not path.exists():
        raise FileNotFoundError(
            f"Raw price CSV not found: {path}. "
            f"Ensure the file exists and the path is correct."
        )

    # Read CSV into DataFrame
    # Note: We don't parse dates yet; we'll do it explicitly below for better error handling
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SchemaValidationError(
            f"{context}: Failed to read CSV. Error: {e}"
        )

    # Parse timestamp column to datetime
    # This must happen before validation so validate_raw_price_schema can check datetime dtype
    # Accepts both legacy "YYYY-MM-DDTHH:MM:SS" and new "YYYY-MM-DD HH:MM:SS" formats
    if 'timestamp' in df.columns:
        try:
            # pd.to_datetime with format='ISO8601' handles both formats automatically
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        except Exception as e:
            raise SchemaValidationError(
                f"{context}: Failed to parse 'timestamp' column as datetime. "
                f"Expected ISO 8601 format (e.g., '2024-01-15 00:00:00' or '2024-01-15T00:00:00'). "
                f"Error: {e}"
            )
    else:
        # Validation will catch this, but we raise here for clarity
        raise SchemaValidationError(
            f"{context}: 'timestamp' column missing. "
            f"Found columns: {list(df.columns)}."
        )

    # Validate schema
    validate_raw_price_schema(df, context=context)

    # Return the validated DataFrame
    return df


def write_raw_price_csv(
    df: pd.DataFrame,
    path: Path | str,
) -> None:
    """
    Write a raw price DataFrame to CSV with schema enforcement.

    **Conceptual**: Saves OHLCV data to disk (e.g., data/raw/QQQ.csv) after
    validating schema and enforcing sort order. This ensures that any data
    written to data/raw/ is guaranteed to be in the canonical format.

    **Functionally**:
      - Validates schema before writing (fail fast if invalid).
      - Sorts data by timestamp in strictly descending order (newest first).
      - Converts datetime timestamps to ISO 8601 strings.
      - Writes CSV with index=False and stable column order.

    **Why sort before writing?**
      - Guarantee consistent sort order in all raw CSVs (newest first).
      - Readers can rely on this without defensive sorting.
      - Prevents accidental sort-order bugs in downstream code.

    **Why validate before writing?**
      - Catch schema violations early (before writing bad data to disk).
      - Prevent polluting data/raw/ with malformed files.

    **Teaching note**: Writing data is just as critical as reading. By validating
    and sorting here, we ensure that data/raw/ always contains well-formed CSVs.
    This makes debugging easier (you can manually inspect files and trust the
    format) and prevents downstream surprises.

    Args:
        df: DataFrame with raw price schema (timestamp, OHLC, volume).
        path: Path where CSV should be written (e.g., "data/raw/QQQ.csv").
              Parent directory must exist.

    Raises:
        SchemaValidationError: If DataFrame doesn't conform to raw price schema.
        OSError: If parent directory doesn't exist or file can't be written.

    Returns:
        None (side effect: writes file to disk).

    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2024-01-15', '2024-01-14']),
        ...     'open_price': [400.0, 398.0],
        ...     'high_price': [405.0, 401.0],
        ...     'low_price': [398.0, 397.0],
        ...     'closing_price': [403.0, 400.0],
        ...     'volume': [1000000, 950000],
        ... })
        >>> write_raw_price_csv(df, "data/raw/QQQ.csv")
        # data/raw/QQQ.csv now contains the DataFrame sorted by timestamp (descending)
    """
    # Convert path to Path object
    path = Path(path)
    context = str(path)

    # Make a copy to avoid modifying the input DataFrame
    df_to_write = df.copy()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by timestamp in descending order (newest first)
    # This ensures the output CSV always has consistent ordering
    if 'timestamp' in df_to_write.columns:
        df_to_write = df_to_write.sort_values('timestamp', ascending=False).reset_index(drop=True)

    # Validate schema before writing (fail fast if invalid)
    validate_raw_price_schema(df_to_write, context=context)

    # Convert timestamp to canonical format: "YYYY-MM-DD HH:MM:SS" (space, not 'T')
    df_to_write['timestamp'] = df_to_write['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Write to CSV with stable column order
    # index=False: don't write row numbers
    # columns: explicit order matching schema
    try:
        df_to_write.to_csv(
            path,
            index=False,
            columns=[
                'timestamp',
                'open_price',
                'high_price',
                'low_price',
                'closing_price',
                'volume',
            ],
        )
    except Exception as e:
        raise OSError(
            f"{context}: Failed to write CSV. Error: {e}"
        )


def read_processed_data_csv(
    path: Path | str,
    instrument_name: str | None = None,
) -> pd.DataFrame:
    """
    Read a processed feature CSV file with schema validation.

    **Conceptual**: Loads processed data (e.g., data/processed/QQQ_features.csv)
    that contains price plus engineered features (moving averages, returns,
    velocity, etc.). Validates the base schema (timestamp, closing_price) to
    ensure downstream code can rely on these columns.

    **Functionally**:
      - Reads CSV using pandas.
      - Parses `timestamp` column to datetime objects.
      - Validates base processed schema (timestamp, closing_price, descending order).
      - Returns DataFrame ready for strategy consumption or further processing.

    **Why only validate base schema?**
      - Processed CSVs can have varying feature sets (different strategies need
        different features).
      - We validate the minimum required columns here (timestamp, closing_price).
      - Strategies can optionally pass required_features to validate_processed_schema
        if they need specific features.

    **Teaching note**: Processed data is the output of feature engineering pipelines.
    By validating the base schema here, we ensure that feature engineering didn't
    accidentally drop critical columns or break sort order. Additional feature
    validation can happen at the strategy level if needed.

    Args:
        path: Path to the processed CSV file (e.g., "data/processed/QQQ_features.csv").
        instrument_name: Optional instrument name for error context.

    Returns:
        DataFrame with at minimum:
          - timestamp (datetime64 dtype, strictly descending)
          - closing_price (float)
          - Additional feature columns (if present in the file).

    Raises:
        FileNotFoundError: If file doesn't exist.
        SchemaValidationError: If CSV doesn't conform to processed schema.

    Example:
        >>> df = read_processed_data_csv("data/processed/QQQ_features.csv", instrument_name="QQQ")
        >>> df.columns.tolist()
        ['timestamp', 'closing_price', 'daily_return', 'moving_average_50', 'velocity']
    """
    # Convert path to Path object
    path = Path(path)
    context = instrument_name or str(path)

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data CSV not found: {path}. "
            f"Ensure the file exists and the path is correct."
        )

    # Read CSV
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SchemaValidationError(
            f"{context}: Failed to read CSV. Error: {e}"
        )

    # Parse timestamp column
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        except Exception as e:
            raise SchemaValidationError(
                f"{context}: Failed to parse 'timestamp' column as datetime. "
                f"Expected ISO 8601 format. Error: {e}"
            )
    else:
        raise SchemaValidationError(
            f"{context}: 'timestamp' column missing. "
            f"Found columns: {list(df.columns)}."
        )

    # Validate base processed schema (timestamp, closing_price, descending order)
    validate_processed_schema(df, required_features=None, context=context)

    return df


def write_processed_data_csv(
    df: pd.DataFrame,
    path: Path | str,
) -> None:
    """
    Write a processed feature DataFrame to CSV with schema enforcement.

    **Conceptual**: Saves processed data (price + features) to disk after
    validating schema and enforcing sort order. This is the output of feature
    engineering pipelines and the input to backtesting/strategies.

    **Functionally**:
      - Validates base processed schema (timestamp, closing_price, descending order).
      - Sorts by timestamp in strictly descending order (newest first).
      - Converts timestamps to ISO 8601 strings.
      - Writes CSV with index=False.

    **Why validate before writing?**
      - Ensure feature engineering didn't accidentally break the schema.
      - Catch missing timestamp/closing_price before writing to disk.
      - Guarantee downstream consumers can rely on the base schema.

    **Teaching note**: Processed data is often generated programmatically (e.g.,
    by feature engineering scripts or pipelines). Validating before writing
    catches bugs in those pipelines early. If a feature engineering step
    accidentally drops closing_price or breaks sort order, we want to know
    immediately, not later when a backtest mysteriously fails.

    Args:
        df: DataFrame with processed schema (timestamp, closing_price, + features).
        path: Path where CSV should be written (e.g., "data/processed/QQQ_features.csv").

    Raises:
        SchemaValidationError: If DataFrame doesn't conform to processed schema.
        OSError: If file can't be written.

    Returns:
        None (side effect: writes file to disk).

    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2024-01-15', '2024-01-14']),
        ...     'closing_price': [403.0, 400.0],
        ...     'daily_return': [0.0075, 0.005],
        ...     'velocity': [0.12, 0.10],
        ... })
        >>> write_processed_data_csv(df, "data/processed/QQQ_features.csv")
    """
    # Convert path to Path object
    path = Path(path)
    context = str(path)

    # Make a copy to avoid modifying input
    df_to_write = df.copy()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by timestamp descending
    if 'timestamp' in df_to_write.columns:
        df_to_write = df_to_write.sort_values('timestamp', ascending=False).reset_index(drop=True)

    # Validate schema before writing
    validate_processed_schema(df_to_write, required_features=None, context=context)

    # Convert timestamp to canonical format: "YYYY-MM-DD HH:M:SS" (space, not 'T')
    df_to_write['timestamp'] = df_to_write['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Write to CSV
    # Note: We write all columns (not just a subset) because processed data
    # can have varying feature sets
    try:
        df_to_write.to_csv(path, index=False)
    except Exception as e:
        raise OSError(
            f"{context}: Failed to write CSV. Error: {e}"
        )
