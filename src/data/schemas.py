"""
Canonical CSV schemas and validation for raw and processed data.

**Conceptual**: This module defines the "data contracts" for the entire system,
ensuring that every CSV file (raw OHLCV, processed features, results) follows
a predictable schema. Explicit schemas and early validation are critical for:
  - Reproducible backtests: every run uses data in the same format.
  - Debugging: clear error messages point exactly to schema violations.
  - AWS portability: strict contracts make batch pipelines reliable.

**Schema philosophy**:
  - All CSVs have a `timestamp` column (ISO 8601 date-time strings).
  - All CSVs are sorted in strictly descending order by timestamp (newest first).
  - Column names are snake_case, human-readable, and descriptive.
  - Validation raises SchemaValidationError with actionable messages.

**Teaching note**: In a CSV-first system, enforcing schemas at every I/O boundary
prevents subtle bugs (e.g., missing columns, wrong sort order, bad timestamps)
from propagating into metrics or backtest results. This design trades a bit of
upfront validation cost for massive gains in reliability and debuggability.
"""

import pandas as pd
from pathlib import Path


class SchemaValidationError(Exception):
    """
    Raised when a DataFrame does not conform to the expected schema.

    **Conceptual**: This exception signals schema violations (missing columns,
    bad timestamps, wrong sort order) and should include enough context
    (file path, instrument name, specific issue) for quick remediation.

    **Usage**: Catch this in orchestration/pipeline code to log and halt
    processing, or let it propagate to the user with a clear error message.
    """
    pass


# Raw price schema constants
RAW_PRICE_REQUIRED_COLUMNS = [
    'timestamp',
    'open_price',
    'high_price',
    'low_price',
    'closing_price',
    'volume',
]

# Processed schema minimal required columns
PROCESSED_MIN_REQUIRED_COLUMNS = [
    'timestamp',
    'closing_price',
]


def validate_raw_price_schema(
    df: pd.DataFrame,
    context: str | None = None,
) -> None:
    """
    Validate that a DataFrame conforms to the raw price schema.

    **Conceptual**: Raw price data (e.g., data/raw/QQQ.csv) must have OHLCV
    columns, valid timestamps, and strictly descending order. This function
    acts as a gatekeeper, rejecting malformed data before it enters the system.

    **Functionally**:
      - Checks that all required columns (timestamp, OHLC, volume) are present.
      - Verifies that `timestamp` is parseable as datetime (or already parsed).
      - Enforces strictly descending order by timestamp (no ties, no gaps allowed
        in sort order direction).
      - Raises SchemaValidationError with actionable message on any violation.

    **Why strict descending order?**
      - Newest-first is our convention for all CSVs (aligns with how we think
        about "recent" data first).
      - Strict (no ties) prevents ambiguous orderings that could cause
        non-deterministic behavior in joins or lookups.
      - Clear validation here means backtests and feature engineering can assume
        sorted data without defensive checks everywhere.

    **Teaching note**: In a production quant system, schema validation at data
    boundaries is non-negotiable. A single missing column or unsorted file can
    invalidate months of backtest results. Better to fail fast with a clear
    error than silently produce garbage.

    Args:
        df: DataFrame to validate (should come from pd.read_csv or similar).
        context: Optional string describing the source (e.g., "data/raw/QQQ.csv"
                 or "QQQ instrument"). Included in error messages for clarity.

    Raises:
        SchemaValidationError: If schema validation fails (missing columns,
                              bad timestamps, wrong sort order).

    Returns:
        None (function is side-effect only; raises on error, returns on success).
    """
    # Build context prefix for error messages
    ctx = f"{context}: " if context else ""

    # Check 1: All required columns must be present
    missing_cols = set(RAW_PRICE_REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise SchemaValidationError(
            f"{ctx}Missing required columns: {sorted(missing_cols)}. "
            f"Expected columns: {RAW_PRICE_REQUIRED_COLUMNS}. "
            f"Found columns: {list(df.columns)}."
        )

    # Check 2: timestamp column must be present and parseable as datetime
    # If timestamp is not already datetime dtype, this is a problem
    # (io.py should have already parsed it, but we validate here too)
    if 'timestamp' not in df.columns:
        raise SchemaValidationError(
            f"{ctx}'timestamp' column missing. Cannot validate schema."
        )

    # Try to ensure timestamp is datetime type
    # If it's already datetime, this is a no-op; if it's string, parse it
    timestamp_col = df['timestamp']
    if not pd.api.types.is_datetime64_any_dtype(timestamp_col):
        try:
            # Attempt to parse as datetime
            timestamp_col = pd.to_datetime(timestamp_col, format='ISO8601')
        except Exception as e:
            raise SchemaValidationError(
                f"{ctx}'timestamp' column contains non-parseable values. "
                f"Expected ISO 8601 date-time strings. Error: {e}"
            )

    # Check 3: Timestamps must be in strictly descending order (newest first)
    # Strict means no ties: each timestamp must be strictly less than the previous
    if len(timestamp_col) > 1:
        # Check if strictly descending: ts[i] > ts[i+1] for all i
        # Equivalent to checking that diff is negative and no zeros
        diffs = timestamp_col.diff()  # diffs[i] = ts[i] - ts[i-1]
        # For descending, we expect diffs[1:] to all be negative (ts[i] < ts[i-1])
        # diffs[0] is NaN (no previous value), so we check diffs[1:]
        if len(diffs) > 1:
            # Drop the first NaN
            diffs_valid = diffs.iloc[1:]
            # All diffs should be negative for strictly descending
            if not (diffs_valid < pd.Timedelta(0)).all():
                # Find the first violation
                bad_indices = diffs_valid[diffs_valid >= pd.Timedelta(0)].index.tolist()
                raise SchemaValidationError(
                    f"{ctx}Timestamps are not in strictly descending order. "
                    f"Violations found at row indices: {bad_indices[:5]} (showing first 5). "
                    f"Expected: each timestamp strictly less than the previous. "
                    f"Hint: Sort your CSV by timestamp in descending order (newest first) "
                    f"and ensure no duplicate timestamps."
                )


def validate_processed_schema(
    df: pd.DataFrame,
    required_features: list[str] | None = None,
    context: str | None = None,
) -> None:
    """
    Validate that a DataFrame conforms to the processed feature schema.

    **Conceptual**: Processed data (e.g., data/processed/QQQ_features.csv)
    extends raw price data with additional features (moving averages, returns,
    volatility, velocity, etc.). This function validates the base schema
    (timestamp + closing_price) plus any additional required features.

    **Functionally**:
      - Checks that minimum required columns (timestamp, closing_price) are present.
      - Optionally checks for additional required features (e.g., ['velocity', 'acceleration']).
      - Validates timestamp format and strictly descending order (same as raw schema).
      - Raises SchemaValidationError on any violation.

    **Why allow optional required_features?**
      - Different strategies may require different feature sets.
      - This function can validate that specific features exist before running
        a strategy that depends on them (fail fast if features are missing).
      - Example: Strategy 1 (QQQ long/short) requires velocity, acceleration,
        and several moving averages. We can pass those as required_features
        to validate before backtesting.

    **Teaching note**: Processed data is the output of feature engineering.
    Validating it ensures that feature pipelines ran correctly and that
    strategies receive the data they expect. This prevents runtime errors
    deep in backtest loops when a feature is missing or misnamed.

    Args:
        df: DataFrame to validate (from pd.read_csv or feature engineering pipeline).
        required_features: Optional list of additional column names that must be present
                          beyond the base schema (timestamp, closing_price).
                          Example: ['velocity', 'acceleration', 'moving_average_50'].
        context: Optional string describing the source (e.g., "data/processed/QQQ_features.csv").

    Raises:
        SchemaValidationError: If schema validation fails.

    Returns:
        None (side-effect only; raises on error).
    """
    # Build context prefix for error messages
    ctx = f"{context}: " if context else ""

    # Build full list of required columns
    required_cols = PROCESSED_MIN_REQUIRED_COLUMNS.copy()
    if required_features:
        required_cols.extend(required_features)

    # Check 1: All required columns must be present
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise SchemaValidationError(
            f"{ctx}Missing required columns: {sorted(missing_cols)}. "
            f"Expected at minimum: {PROCESSED_MIN_REQUIRED_COLUMNS}. "
            f"Additional required features: {required_features or []}. "
            f"Found columns: {list(df.columns)}."
        )

    # Check 2 & 3: timestamp validation and descending order
    # Reuse the same timestamp checks as raw schema
    if 'timestamp' not in df.columns:
        raise SchemaValidationError(
            f"{ctx}'timestamp' column missing. Cannot validate schema."
        )

    # Ensure timestamp is datetime type
    timestamp_col = df['timestamp']
    if not pd.api.types.is_datetime64_any_dtype(timestamp_col):
        try:
            timestamp_col = pd.to_datetime(timestamp_col, format='ISO8601')
        except Exception as e:
            raise SchemaValidationError(
                f"{ctx}'timestamp' column contains non-parseable values. "
                f"Expected ISO 8601 date-time strings. Error: {e}"
            )

    # Check strictly descending order
    if len(timestamp_col) > 1:
        diffs = timestamp_col.diff()
        if len(diffs) > 1:
            diffs_valid = diffs.iloc[1:]
            if not (diffs_valid < pd.Timedelta(0)).all():
                bad_indices = diffs_valid[diffs_valid >= pd.Timedelta(0)].index.tolist()
                raise SchemaValidationError(
                    f"{ctx}Timestamps are not in strictly descending order. "
                    f"Violations found at row indices: {bad_indices[:5]} (showing first 5). "
                    f"Expected: each timestamp strictly less than the previous. "
                    f"Hint: Sort your CSV by timestamp in descending order (newest first) "
                    f"and ensure no duplicate timestamps."
                )
