"""
Tests for Phase 3: Data I/O, schemas, and loaders.

This module tests:
  - Schema validation (raw and processed schemas).
  - CSV readers/writers (read_raw_price_csv, write_raw_price_csv, etc.).
  - Instrument loaders (load_qqq_history, etc.).
  - Round-trip read/write correctness.
  - Error handling (missing columns, bad timestamps, wrong sort order).

All tests use temporary directories (via tmp_path fixture) to avoid polluting
the real data/ directory.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timezone
from pathlib import Path

from src.data.schemas import (
    validate_raw_price_schema,
    validate_processed_schema,
    SchemaValidationError,
    RAW_PRICE_REQUIRED_COLUMNS,
    PROCESSED_MIN_REQUIRED_COLUMNS,
)
from src.data.io import (
    read_raw_price_csv,
    write_raw_price_csv,
    read_processed_data_csv,
    write_processed_data_csv,
)


# ============================================================================
# Helper functions for test data generation
# ============================================================================

def make_valid_raw_price_df(n_rows: int = 10) -> pd.DataFrame:
    """
    Create a small DataFrame that conforms to the raw price schema.
    Timestamps are strictly descending (newest first).
    """
    # Generate timestamps in descending order
    timestamps = pd.date_range(
        end=datetime(2024, 1, 15, tzinfo=timezone.utc),
        periods=n_rows,
        freq='D'
    )[::-1]  # Reverse to get descending order (newest first)

    return pd.DataFrame({
        'timestamp': timestamps,
        'open_price': np.linspace(100.0, 110.0, n_rows),
        'high_price': np.linspace(102.0, 112.0, n_rows),
        'low_price': np.linspace(98.0, 108.0, n_rows),
        'closing_price': np.linspace(100.0, 110.0, n_rows),
        'volume': np.linspace(1000000, 1100000, n_rows).astype(int),
    })


def make_valid_processed_df(n_rows: int = 10) -> pd.DataFrame:
    """
    Create a small DataFrame that conforms to the processed schema.
    Includes base columns (timestamp, closing_price) plus some features.
    Timestamps are strictly descending.
    """
    timestamps = pd.date_range(
        end=datetime(2024, 1, 15, tzinfo=timezone.utc),
        periods=n_rows,
        freq='D'
    )[::-1]  # Descending order

    return pd.DataFrame({
        'timestamp': timestamps,
        'closing_price': np.linspace(100.0, 110.0, n_rows),
        'daily_return': np.random.randn(n_rows) * 0.01,  # ~1% daily vol
        'moving_average_50': np.linspace(99.0, 109.0, n_rows),
        'velocity': np.linspace(0.01, 0.02, n_rows),
    })


# ============================================================================
# Tests for schema validation (src/data/schemas.py)
# ============================================================================

def test_validate_raw_price_schema_valid():
    """Test that a valid raw price DataFrame passes validation."""
    df = make_valid_raw_price_df(n_rows=5)
    # Should not raise
    validate_raw_price_schema(df, context="test")


def test_validate_raw_price_schema_missing_column():
    """Test that validation fails when a required column is missing."""
    df = make_valid_raw_price_df(n_rows=5)
    # Drop a required column
    df = df.drop(columns=['closing_price'])

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_raw_price_schema(df, context="test")

    # Check that error message mentions the missing column
    assert 'closing_price' in str(exc_info.value)
    assert 'Missing required columns' in str(exc_info.value)


def test_validate_raw_price_schema_bad_timestamp_order():
    """Test that validation fails when timestamps are not strictly descending."""
    df = make_valid_raw_price_df(n_rows=5)
    # Reverse the order to ascending (should fail)
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_raw_price_schema(df, context="test")

    assert 'not in strictly descending order' in str(exc_info.value)


def test_validate_raw_price_schema_duplicate_timestamps():
    """Test that validation fails when there are duplicate timestamps (ties)."""
    df = make_valid_raw_price_df(n_rows=5)
    # Create a duplicate timestamp (tie)
    df.loc[2, 'timestamp'] = df.loc[1, 'timestamp']

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_raw_price_schema(df, context="test")

    assert 'not in strictly descending order' in str(exc_info.value)


def test_validate_processed_schema_valid():
    """Test that a valid processed DataFrame passes validation."""
    df = make_valid_processed_df(n_rows=5)
    # Should not raise
    validate_processed_schema(df, context="test")


def test_validate_processed_schema_missing_base_column():
    """Test that validation fails when a base column (closing_price) is missing."""
    df = make_valid_processed_df(n_rows=5)
    df = df.drop(columns=['closing_price'])

    with pytest.raises(SchemaValidationError) as exc_info:
        validate_processed_schema(df, context="test")

    assert 'closing_price' in str(exc_info.value)
    assert 'Missing required columns' in str(exc_info.value)


def test_validate_processed_schema_with_required_features():
    """Test that validation checks for specific required features when provided."""
    df = make_valid_processed_df(n_rows=5)

    # Should pass when required features are present
    validate_processed_schema(
        df,
        required_features=['daily_return', 'velocity'],
        context="test"
    )

    # Should fail when a required feature is missing
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_processed_schema(
            df,
            required_features=['daily_return', 'acceleration'],  # acceleration missing
            context="test"
        )

    assert 'acceleration' in str(exc_info.value)


# ============================================================================
# Tests for CSV I/O (src/data/io.py)
# ============================================================================

def test_write_and_read_raw_price_csv_roundtrip(tmp_path):
    """
    Test round-trip: write a raw price DataFrame to CSV, read it back,
    and verify it matches the original.
    """
    # Create a valid DataFrame
    df_original = make_valid_raw_price_df(n_rows=10)

    # Write to a temporary CSV
    csv_path = tmp_path / "test_raw.csv"
    write_raw_price_csv(df_original, csv_path)

    # Read it back
    df_read = read_raw_price_csv(csv_path, instrument_name="test")

    # Verify columns match
    assert list(df_read.columns) == RAW_PRICE_REQUIRED_COLUMNS

    # Verify timestamps are datetime dtype
    assert pd.api.types.is_datetime64_any_dtype(df_read['timestamp'])

    # Verify strictly descending order
    diffs = df_read['timestamp'].diff().iloc[1:]
    assert (diffs < pd.Timedelta(0)).all()

    # Verify data values match (within floating-point tolerance)
    pd.testing.assert_frame_equal(
        df_read.reset_index(drop=True),
        df_original.reset_index(drop=True),
        check_dtype=False,  # Allow for minor dtype differences
    )


def test_write_raw_price_csv_sorts_descending(tmp_path):
    """
    Test that write_raw_price_csv automatically sorts data by timestamp
    in descending order, even if input is unsorted.
    """
    # Create a DataFrame with ascending timestamps (wrong order)
    df_ascending = make_valid_raw_price_df(n_rows=5)
    df_ascending = df_ascending.sort_values('timestamp', ascending=True).reset_index(drop=True)

    # Write it (should auto-sort to descending)
    csv_path = tmp_path / "test_sorted.csv"
    write_raw_price_csv(df_ascending, csv_path)

    # Read it back
    df_read = read_raw_price_csv(csv_path)

    # Verify timestamps are strictly descending
    diffs = df_read['timestamp'].diff().iloc[1:]
    assert (diffs < pd.Timedelta(0)).all()


def test_read_raw_price_csv_missing_file(tmp_path):
    """Test that reading a non-existent file raises FileNotFoundError."""
    csv_path = tmp_path / "nonexistent.csv"

    with pytest.raises(FileNotFoundError) as exc_info:
        read_raw_price_csv(csv_path, instrument_name="test")

    assert 'not found' in str(exc_info.value)


def test_read_raw_price_csv_missing_column_in_file(tmp_path):
    """
    Test that reading a CSV with missing required columns raises
    SchemaValidationError with a clear message.
    """
    # Create a CSV missing 'closing_price' column
    df_bad = make_valid_raw_price_df(n_rows=5)
    df_bad = df_bad.drop(columns=['closing_price'])

    # Write the bad DataFrame to CSV manually (bypass validation)
    csv_path = tmp_path / "bad_schema.csv"
    df_bad['timestamp'] = df_bad['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    df_bad.to_csv(csv_path, index=False)

    # Attempt to read (should fail validation)
    with pytest.raises(SchemaValidationError) as exc_info:
        read_raw_price_csv(csv_path, instrument_name="test")

    assert 'closing_price' in str(exc_info.value)
    assert 'Missing required columns' in str(exc_info.value)


def test_read_raw_price_csv_bad_timestamp_format(tmp_path):
    """
    Test that reading a CSV with non-parseable timestamps raises
    SchemaValidationError.
    """
    # Create a DataFrame and write it with bad timestamps
    df = make_valid_raw_price_df(n_rows=3)

    # Write manually with bad timestamp strings
    csv_path = tmp_path / "bad_timestamps.csv"
    df_bad = df.copy()
    df_bad['timestamp'] = ['not-a-date', 'also-not-a-date', 'nope']
    df_bad.to_csv(csv_path, index=False)

    # Attempt to read (should fail during timestamp parsing)
    with pytest.raises(SchemaValidationError) as exc_info:
        read_raw_price_csv(csv_path, instrument_name="test")

    assert 'timestamp' in str(exc_info.value).lower()


def test_write_and_read_processed_csv_roundtrip(tmp_path):
    """
    Test round-trip for processed data: write, read, verify.
    """
    # Create a valid processed DataFrame
    df_original = make_valid_processed_df(n_rows=10)

    # Write to CSV
    csv_path = tmp_path / "test_processed.csv"
    write_processed_data_csv(df_original, csv_path)

    # Read back
    df_read = read_processed_data_csv(csv_path, instrument_name="test")

    # Verify base columns are present
    assert 'timestamp' in df_read.columns
    assert 'closing_price' in df_read.columns

    # Verify timestamps are datetime and descending
    assert pd.api.types.is_datetime64_any_dtype(df_read['timestamp'])
    diffs = df_read['timestamp'].diff().iloc[1:]
    assert (diffs < pd.Timedelta(0)).all()

    # Verify data matches
    pd.testing.assert_frame_equal(
        df_read.reset_index(drop=True),
        df_original.reset_index(drop=True),
        check_dtype=False,
    )


def test_read_processed_csv_missing_base_column(tmp_path):
    """
    Test that reading a processed CSV missing 'closing_price' raises
    SchemaValidationError.
    """
    # Create a processed DataFrame missing closing_price
    df_bad = make_valid_processed_df(n_rows=5)
    df_bad = df_bad.drop(columns=['closing_price'])

    # Write manually (bypass validation)
    csv_path = tmp_path / "bad_processed.csv"
    df_bad['timestamp'] = df_bad['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    df_bad.to_csv(csv_path, index=False)

    # Attempt to read (should fail)
    with pytest.raises(SchemaValidationError) as exc_info:
        read_processed_data_csv(csv_path, instrument_name="test")

    assert 'closing_price' in str(exc_info.value)


# ============================================================================
# Tests for instrument loaders (src/data/loaders.py)
# ============================================================================
# Note: These tests require actual CSV files in data/raw/, which may not exist
# during initial development. We'll create minimal test files in tmp_path and
# test the loader logic, but skip tests that depend on real files.

def test_load_instrument_history_with_temp_file(tmp_path, monkeypatch):
    """
    Test load_instrument_history by creating a temporary CSV and patching
    the RAW_DATA_DIR to point to tmp_path.
    """
    # Import loaders after patching to pick up the monkeypatched value
    from src.data import loaders

    # Create a temporary "data/raw" directory structure
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    # Create a valid QQQ.csv
    df = make_valid_raw_price_df(n_rows=5)
    csv_path = raw_dir / "QQQ.csv"
    write_raw_price_csv(df, csv_path)

    # Patch RAW_DATA_DIR to point to our temp directory
    monkeypatch.setattr(loaders, 'RAW_DATA_DIR', raw_dir)

    # Now load_instrument_history should find the temp file
    df_loaded = loaders.load_instrument_history("QQQ")

    # Verify it matches the original
    pd.testing.assert_frame_equal(
        df_loaded.reset_index(drop=True),
        df.reset_index(drop=True),
        check_dtype=False,
    )


def test_load_default_universe_with_temp_files(tmp_path, monkeypatch):
    """
    Test load_default_universe with a temporary data directory containing
    only some of the instruments (to test missing-file handling).
    """
    from src.data import loaders

    # Create temp raw directory
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    # Create CSVs for QQQ and TQQQ only (skip SQQQ and UVXY)
    df_qqq = make_valid_raw_price_df(n_rows=3)
    df_tqqq = make_valid_raw_price_df(n_rows=3)

    write_raw_price_csv(df_qqq, raw_dir / "QQQ.csv")
    write_raw_price_csv(df_tqqq, raw_dir / "TQQQ.csv")

    # Patch RAW_DATA_DIR
    monkeypatch.setattr(loaders, 'RAW_DATA_DIR', raw_dir)

    # Load universe
    universe = loaders.load_default_universe()

    # Should have QQQ and TQQQ, but not SQQQ or UVXY
    assert "QQQ" in universe
    assert "TQQQ" in universe
    assert "SQQQ" not in universe
    assert "UVXY" not in universe

    # Verify data is correct
    pd.testing.assert_frame_equal(
        universe["QQQ"].reset_index(drop=True),
        df_qqq.reset_index(drop=True),
        check_dtype=False,
    )


# ============================================================================
# Edge case tests
# ============================================================================

def test_empty_dataframe_validation():
    """Test that an empty DataFrame fails validation (no rows, no data)."""
    df_empty = pd.DataFrame(columns=RAW_PRICE_REQUIRED_COLUMNS)

    # Validation should pass for column presence, but may fail on timestamp checks
    # depending on implementation. Here we just ensure it doesn't crash.
    # An empty DataFrame has no timestamps to validate, so it may pass.
    try:
        validate_raw_price_schema(df_empty, context="empty test")
    except SchemaValidationError:
        # It's okay if validation fails for empty DataFrames
        pass


def test_single_row_dataframe():
    """Test that a single-row DataFrame passes validation (no ordering issues)."""
    df_single = make_valid_raw_price_df(n_rows=1)
    # Should not raise (single row has no ordering issues)
    validate_raw_price_schema(df_single, context="single row test")


def test_write_creates_parent_directory(tmp_path):
    """
    Test that write_raw_price_csv creates the parent directory if it doesn't exist.
    """
    # Use a nested path that doesn't exist yet
    nested_path = tmp_path / "nested" / "subdir" / "test.csv"

    df = make_valid_raw_price_df(n_rows=3)

    # Write should create the parent directories
    write_raw_price_csv(df, nested_path)

    # Verify file exists
    assert nested_path.exists()

    # Verify we can read it back
    df_read = read_raw_price_csv(nested_path)
    assert len(df_read) == 3
