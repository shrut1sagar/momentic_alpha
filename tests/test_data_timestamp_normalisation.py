"""
Tests for timestamp normalization functionality.

**Purpose**: Verify that timestamp normalization handles all edge cases:
  - Legacy format ("YYYY-MM-DDTHH:MM:SS") → new format ("YYYY-MM-DD HH:MM:SS")
  - Already-parsed datetime64 types
  - Empty DataFrames
  - Mixed format strings
  - Descending sort order enforcement
  - Optional date column creation

**Why this matters**: Consistent timestamp format across the project prevents
bugs in date filtering, merging, and backtest alignment. These tests ensure
the normalization logic is robust and won't crash on edge cases.

**Coverage**:
  - normalize_timestamp_column() function
  - write_normalized_csv() / read_raw_price_csv() round-trip
  - On-disk format verification
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from src.data.io import (
    normalize_timestamp_column,
    write_normalized_csv,
    read_raw_price_csv,
)


# ============================================================================
# Tests for normalize_timestamp_column()
# ============================================================================

def test_normalize_timestamp_column_legacy_format():
    """Test normalization of legacy "YYYY-MM-DDTHH:MM:SS" timestamps."""
    df = pd.DataFrame({
        'timestamp': ['2024-01-03T00:00:00', '2024-01-02T00:00:00', '2024-01-01T00:00:00'],
        'value': [30, 20, 10],
    })

    result = normalize_timestamp_column(df)

    # Check timestamp is datetime64
    assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])

    # Check descending order (newest first)
    assert result['timestamp'].iloc[0] == pd.Timestamp('2024-01-03')
    assert result['timestamp'].iloc[1] == pd.Timestamp('2024-01-02')
    assert result['timestamp'].iloc[2] == pd.Timestamp('2024-01-01')

    # Check values are preserved and aligned
    assert result['value'].tolist() == [30, 20, 10]


def test_normalize_timestamp_column_new_format():
    """Test normalization of new "YYYY-MM-DD HH:MM:SS" timestamps."""
    df = pd.DataFrame({
        'timestamp': ['2024-01-03 00:00:00', '2024-01-02 00:00:00', '2024-01-01 00:00:00'],
        'value': [30, 20, 10],
    })

    result = normalize_timestamp_column(df)

    # Check timestamp is datetime64
    assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])

    # Check descending order
    assert result['timestamp'].iloc[0] == pd.Timestamp('2024-01-03')
    assert result['timestamp'].iloc[2] == pd.Timestamp('2024-01-01')


def test_normalize_timestamp_column_mixed_formats():
    """Test normalization handles mixed legacy and new format timestamps."""
    df = pd.DataFrame({
        'timestamp': [
            '2024-01-04 00:00:00',  # New format
            '2024-01-03T00:00:00',  # Legacy format
            '2024-01-02 00:00:00',  # New format
            '2024-01-01T00:00:00',  # Legacy format
        ],
        'value': [40, 30, 20, 10],
    })

    result = normalize_timestamp_column(df)

    # All should parse correctly and be in descending order
    assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
    assert result['timestamp'].iloc[0] == pd.Timestamp('2024-01-04')
    assert result['timestamp'].iloc[1] == pd.Timestamp('2024-01-03')
    assert result['timestamp'].iloc[2] == pd.Timestamp('2024-01-02')
    assert result['timestamp'].iloc[3] == pd.Timestamp('2024-01-01')


def test_normalize_timestamp_column_already_datetime():
    """Test normalization handles timestamps that are already datetime64 dtype."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-03', '2024-01-02', '2024-01-01']),
        'value': [30, 20, 10],
    })

    # Timestamp is already datetime64, should not re-parse
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])

    result = normalize_timestamp_column(df)

    # Should still sort to descending order
    assert result['timestamp'].iloc[0] == pd.Timestamp('2024-01-03')
    assert result['timestamp'].iloc[2] == pd.Timestamp('2024-01-01')


def test_normalize_timestamp_column_empty_dataframe():
    """Test normalization handles empty DataFrames without crashing."""
    df = pd.DataFrame({
        'timestamp': [],
        'value': [],
    })

    result = normalize_timestamp_column(df)

    # Should return empty DataFrame unchanged
    assert result.empty
    assert list(result.columns) == ['timestamp', 'value']


def test_normalize_timestamp_column_enforces_descending_order():
    """Test that normalization always enforces descending order (newest first)."""
    # Create DataFrame in ascending order (oldest first)
    df = pd.DataFrame({
        'timestamp': ['2024-01-01T00:00:00', '2024-01-02T00:00:00', '2024-01-03T00:00:00'],
        'value': [10, 20, 30],
    })

    result = normalize_timestamp_column(df)

    # Should be reordered to descending (newest first)
    assert result['timestamp'].iloc[0] == pd.Timestamp('2024-01-03')
    assert result['timestamp'].iloc[1] == pd.Timestamp('2024-01-02')
    assert result['timestamp'].iloc[2] == pd.Timestamp('2024-01-01')

    # Values should be reordered to match
    assert result['value'].tolist() == [30, 20, 10]


def test_normalize_timestamp_column_preserves_other_columns():
    """Test that normalization preserves all other columns unchanged."""
    df = pd.DataFrame({
        'timestamp': ['2024-01-03T00:00:00', '2024-01-02T00:00:00', '2024-01-01T00:00:00'],
        'closing_price': [100.0, 101.0, 102.0],
        'volume': [1000, 2000, 3000],
        'symbol': ['QQQ', 'QQQ', 'QQQ'],
    })

    result = normalize_timestamp_column(df)

    # All columns should be present
    assert set(result.columns) == {'timestamp', 'closing_price', 'volume', 'symbol'}

    # Values should be preserved (and aligned with sorted timestamps)
    assert result['closing_price'].tolist() == [100.0, 101.0, 102.0]
    assert result['volume'].tolist() == [1000, 2000, 3000]
    assert result['symbol'].tolist() == ['QQQ', 'QQQ', 'QQQ']


def test_normalize_timestamp_column_adds_date_column():
    """Test optional date column creation for human inspection."""
    df = pd.DataFrame({
        'timestamp': ['2024-01-03T00:00:00', '2024-01-02T00:00:00', '2024-01-01T00:00:00'],
        'value': [30, 20, 10],
    })

    result = normalize_timestamp_column(df, ensure_date_column=True)

    # Should have added a 'date' column
    assert 'date' in result.columns

    # Date column should be YYYY-MM-DD string format
    assert result['date'].iloc[0] == '2024-01-03'
    assert result['date'].iloc[1] == '2024-01-02'
    assert result['date'].iloc[2] == '2024-01-01'


def test_normalize_timestamp_column_missing_column_raises_error():
    """Test that normalization raises KeyError if timestamp column is missing."""
    df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'value': [10, 20],
    })

    with pytest.raises(KeyError):
        normalize_timestamp_column(df, col='timestamp')


def test_normalize_timestamp_column_custom_column_name():
    """Test normalization with a custom timestamp column name."""
    df = pd.DataFrame({
        'date': ['2024-01-03T00:00:00', '2024-01-02T00:00:00', '2024-01-01T00:00:00'],
        'value': [30, 20, 10],
    })

    result = normalize_timestamp_column(df, col='date')

    # Should normalize the 'date' column
    assert pd.api.types.is_datetime64_any_dtype(result['date'])
    assert result['date'].iloc[0] == pd.Timestamp('2024-01-03')


# ============================================================================
# Tests for write_normalized_csv() and round-trip
# ============================================================================

def test_write_normalized_csv_creates_canonical_format(tmp_path):
    """Test that write_normalized_csv writes timestamps in canonical format."""
    df = pd.DataFrame({
        'timestamp': ['2024-01-03T00:00:00', '2024-01-02T00:00:00', '2024-01-01T00:00:00'],
        'value': [30, 20, 10],
    })

    csv_path = tmp_path / "test_normalized.csv"
    write_normalized_csv(df, csv_path)

    # Read back as raw text to verify on-disk format
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Header
    assert lines[0].strip() == 'timestamp,value'

    # Data rows should use "YYYY-MM-DD HH:MM:SS" format (space, not 'T')
    assert lines[1].strip() == '2024-01-03 00:00:00,30'
    assert lines[2].strip() == '2024-01-02 00:00:00,20'
    assert lines[3].strip() == '2024-01-01 00:00:00,10'


def test_write_normalized_csv_enforces_descending_order(tmp_path):
    """Test that write_normalized_csv enforces descending sort order."""
    # Create DataFrame in ascending order
    df = pd.DataFrame({
        'timestamp': ['2024-01-01T00:00:00', '2024-01-02T00:00:00', '2024-01-03T00:00:00'],
        'value': [10, 20, 30],
    })

    csv_path = tmp_path / "test_sorted.csv"
    write_normalized_csv(df, csv_path)

    # Read back
    df_read = pd.read_csv(csv_path)

    # Should be in descending order (newest first)
    assert df_read['timestamp'].iloc[0] == '2024-01-03 00:00:00'
    assert df_read['timestamp'].iloc[1] == '2024-01-02 00:00:00'
    assert df_read['timestamp'].iloc[2] == '2024-01-01 00:00:00'

    # Values should be reordered to match
    assert df_read['value'].tolist() == [30, 20, 10]


def test_write_normalized_csv_round_trip_with_read_raw_price_csv(tmp_path):
    """Test write_normalized_csv → read_raw_price_csv round-trip."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-03', '2024-01-02', '2024-01-01']),
        'open_price': [100.0, 101.0, 102.0],
        'high_price': [100.5, 101.5, 102.5],
        'low_price': [99.5, 100.5, 101.5],
        'closing_price': [100.2, 101.2, 102.2],
        'volume': [1000, 2000, 3000],
    })

    csv_path = tmp_path / "test_roundtrip.csv"
    write_normalized_csv(df, csv_path)

    # Read back using read_raw_price_csv
    df_read = read_raw_price_csv(csv_path, instrument_name='TEST')

    # Should parse timestamps correctly
    assert pd.api.types.is_datetime64_any_dtype(df_read['timestamp'])

    # Should be in descending order
    assert df_read['timestamp'].iloc[0] == pd.Timestamp('2024-01-03')
    assert df_read['timestamp'].iloc[2] == pd.Timestamp('2024-01-01')

    # Values should match
    assert df_read['closing_price'].iloc[0] == pytest.approx(100.2)
    assert df_read['closing_price'].iloc[1] == pytest.approx(101.2)
    assert df_read['closing_price'].iloc[2] == pytest.approx(102.2)


def test_write_normalized_csv_handles_empty_dataframe(tmp_path):
    """Test write_normalized_csv handles empty DataFrames without crashing."""
    df = pd.DataFrame({
        'timestamp': [],
        'value': [],
    })

    csv_path = tmp_path / "test_empty.csv"
    write_normalized_csv(df, csv_path)

    # Should create file with header only
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    assert len(lines) == 1  # Header only
    assert lines[0].strip() == 'timestamp,value'


def test_write_normalized_csv_creates_parent_directories(tmp_path):
    """Test write_normalized_csv creates parent directories if they don't exist."""
    df = pd.DataFrame({
        'timestamp': ['2024-01-01T00:00:00'],
        'value': [10],
    })

    # Path with nested directories that don't exist
    csv_path = tmp_path / "nested" / "dir" / "test.csv"

    write_normalized_csv(df, csv_path)

    # Should have created the file and all parent dirs
    assert csv_path.exists()
    assert csv_path.parent.exists()


def test_write_normalized_csv_with_date_column(tmp_path):
    """Test write_normalized_csv with optional date column."""
    df = pd.DataFrame({
        'timestamp': ['2024-01-03T00:00:00', '2024-01-02T00:00:00', '2024-01-01T00:00:00'],
        'value': [30, 20, 10],
    })

    csv_path = tmp_path / "test_with_date.csv"
    write_normalized_csv(df, csv_path, ensure_date_column=True)

    # Read back
    df_read = pd.read_csv(csv_path)

    # Should have added date column
    assert 'date' in df_read.columns
    assert df_read['date'].iloc[0] == '2024-01-03'
    assert df_read['date'].iloc[1] == '2024-01-02'
    assert df_read['date'].iloc[2] == '2024-01-01'


# ============================================================================
# Integration test: legacy format compatibility
# ============================================================================

def test_read_raw_price_csv_accepts_both_formats(tmp_path):
    """Test that read_raw_price_csv accepts both legacy and new timestamp formats."""
    # Create CSV with legacy format timestamps
    legacy_csv = tmp_path / "legacy.csv"
    with open(legacy_csv, 'w') as f:
        f.write("timestamp,open_price,high_price,low_price,closing_price,volume\n")
        f.write("2024-01-03T00:00:00,100.0,100.5,99.5,100.2,1000\n")
        f.write("2024-01-02T00:00:00,101.0,101.5,100.5,101.2,2000\n")
        f.write("2024-01-01T00:00:00,102.0,102.5,101.5,102.2,3000\n")

    # Create CSV with new format timestamps
    new_csv = tmp_path / "new.csv"
    with open(new_csv, 'w') as f:
        f.write("timestamp,open_price,high_price,low_price,closing_price,volume\n")
        f.write("2024-01-03 00:00:00,100.0,100.5,99.5,100.2,1000\n")
        f.write("2024-01-02 00:00:00,101.0,101.5,100.5,101.2,2000\n")
        f.write("2024-01-01 00:00:00,102.0,102.5,101.5,102.2,3000\n")

    # Both should parse successfully
    df_legacy = read_raw_price_csv(legacy_csv, instrument_name='TEST')
    df_new = read_raw_price_csv(new_csv, instrument_name='TEST')

    # Both should have datetime64 timestamps
    assert pd.api.types.is_datetime64_any_dtype(df_legacy['timestamp'])
    assert pd.api.types.is_datetime64_any_dtype(df_new['timestamp'])

    # Both should have same values
    pd.testing.assert_frame_equal(df_legacy, df_new)


# ============================================================================
# Edge case: non-midnight timestamps
# ============================================================================

def test_normalize_timestamp_column_with_non_midnight_times():
    """Test normalization handles timestamps with non-midnight times."""
    df = pd.DataFrame({
        'timestamp': [
            '2024-01-03T14:30:00',  # 2:30 PM
            '2024-01-02T09:15:00',  # 9:15 AM
            '2024-01-01T16:45:00',  # 4:45 PM
        ],
        'value': [30, 20, 10],
    })

    result = normalize_timestamp_column(df)

    # Should preserve time components
    assert result['timestamp'].iloc[0] == pd.Timestamp('2024-01-03 14:30:00')
    assert result['timestamp'].iloc[1] == pd.Timestamp('2024-01-02 09:15:00')
    assert result['timestamp'].iloc[2] == pd.Timestamp('2024-01-01 16:45:00')


def test_write_normalized_csv_preserves_time_components(tmp_path):
    """Test that write_normalized_csv preserves non-midnight time components."""
    df = pd.DataFrame({
        'timestamp': ['2024-01-03T14:30:00', '2024-01-02T09:15:00', '2024-01-01T16:45:00'],
        'value': [30, 20, 10],
    })

    csv_path = tmp_path / "test_times.csv"
    write_normalized_csv(df, csv_path)

    # Read back as text to verify format
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Should preserve time components
    assert '2024-01-03 14:30:00' in lines[1]
    assert '2024-01-02 09:15:00' in lines[2]
    assert '2024-01-01 16:45:00' in lines[3]
