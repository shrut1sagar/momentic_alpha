"""
Tests for Massive fetch action merge logic.

**Purpose**: Verify that the merge_price_history function correctly handles
incremental and force mode updates.

**Testing philosophy**: Test the merge logic in isolation without hitting
the API. Use synthetic test data with known timestamps and values.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import actions module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the merge function directly from the actions script
from actions.fetch_massive_price_history import merge_price_history


def make_test_df(dates_and_prices):
    """
    Create a test DataFrame with timestamps and prices.

    Args:
        dates_and_prices: List of (date_str, price) tuples.

    Returns:
        DataFrame with timestamp, closing_price, and other required columns.
    """
    timestamps = [pd.Timestamp(date, tz='UTC') for date, _ in dates_and_prices]
    prices = [price for _, price in dates_and_prices]

    return pd.DataFrame({
        'timestamp': timestamps,
        'open_price': prices,
        'high_price': [p * 1.01 for p in prices],
        'low_price': [p * 0.99 for p in prices],
        'closing_price': prices,
        'volume': [1000000] * len(prices),
    })


def test_merge_price_history_incremental_no_overlap():
    """Test incremental mode with no overlapping dates."""
    # Existing: 2024-01-01, 2024-01-02
    existing = make_test_df([
        ('2024-01-01', 100.0),
        ('2024-01-02', 101.0),
    ])

    # New: 2024-01-03, 2024-01-04
    new = make_test_df([
        ('2024-01-03', 102.0),
        ('2024-01-04', 103.0),
    ])

    start = pd.Timestamp('2024-01-03', tz='UTC')
    end = pd.Timestamp('2024-01-04', tz='UTC')

    # Incremental mode: should add all new dates
    merged = merge_price_history(existing, new, start, end, force=False)

    # Should have 4 rows (all dates)
    assert len(merged) == 4

    # Check that all dates are present
    merged_dates = set(merged['timestamp'])
    expected_dates = {
        pd.Timestamp('2024-01-01', tz='UTC'),
        pd.Timestamp('2024-01-02', tz='UTC'),
        pd.Timestamp('2024-01-03', tz='UTC'),
        pd.Timestamp('2024-01-04', tz='UTC'),
    }
    assert merged_dates == expected_dates


def test_merge_price_history_incremental_with_overlap():
    """Test incremental mode with overlapping dates (preserves existing)."""
    # Existing: 2024-01-01, 2024-01-02
    existing = make_test_df([
        ('2024-01-01', 100.0),
        ('2024-01-02', 101.0),
    ])

    # New: 2024-01-02 (different price), 2024-01-03
    new = make_test_df([
        ('2024-01-02', 999.0),  # Different price, should be ignored in incremental mode
        ('2024-01-03', 102.0),
    ])

    start = pd.Timestamp('2024-01-02', tz='UTC')
    end = pd.Timestamp('2024-01-03', tz='UTC')

    # Incremental mode: should preserve existing 2024-01-02, add 2024-01-03
    merged = merge_price_history(existing, new, start, end, force=False)

    # Should have 3 rows
    assert len(merged) == 3

    # Check that 2024-01-02 has ORIGINAL price (101.0), not new price (999.0)
    row_jan_02 = merged[merged['timestamp'] == pd.Timestamp('2024-01-02', tz='UTC')]
    assert len(row_jan_02) == 1
    assert row_jan_02.iloc[0]['closing_price'] == 101.0  # Original, not 999.0

    # Check that 2024-01-03 was added
    row_jan_03 = merged[merged['timestamp'] == pd.Timestamp('2024-01-03', tz='UTC')]
    assert len(row_jan_03) == 1
    assert row_jan_03.iloc[0]['closing_price'] == 102.0


def test_merge_price_history_force_mode_replaces_data():
    """Test force mode replaces data in specified range."""
    # Existing: 2024-01-01, 2024-01-02, 2024-01-03
    existing = make_test_df([
        ('2024-01-01', 100.0),
        ('2024-01-02', 101.0),
        ('2024-01-03', 102.0),
    ])

    # New: 2024-01-02 (corrected price), 2024-01-03 (corrected price)
    new = make_test_df([
        ('2024-01-02', 201.0),  # Corrected price
        ('2024-01-03', 202.0),  # Corrected price
    ])

    start = pd.Timestamp('2024-01-02', tz='UTC')
    end = pd.Timestamp('2024-01-03', tz='UTC')

    # Force mode: should replace 2024-01-02 and 2024-01-03 with new values
    merged = merge_price_history(existing, new, start, end, force=True)

    # Should have 3 rows
    assert len(merged) == 3

    # Check that 2024-01-01 is preserved (outside range)
    row_jan_01 = merged[merged['timestamp'] == pd.Timestamp('2024-01-01', tz='UTC')]
    assert len(row_jan_01) == 1
    assert row_jan_01.iloc[0]['closing_price'] == 100.0

    # Check that 2024-01-02 has NEW price (201.0)
    row_jan_02 = merged[merged['timestamp'] == pd.Timestamp('2024-01-02', tz='UTC')]
    assert len(row_jan_02) == 1
    assert row_jan_02.iloc[0]['closing_price'] == 201.0  # New, not 101.0

    # Check that 2024-01-03 has NEW price (202.0)
    row_jan_03 = merged[merged['timestamp'] == pd.Timestamp('2024-01-03', tz='UTC')]
    assert len(row_jan_03) == 1
    assert row_jan_03.iloc[0]['closing_price'] == 202.0  # New, not 102.0


def test_merge_price_history_force_mode_preserves_outside_range():
    """Test force mode preserves data outside specified range."""
    # Existing: 2024-01-01, 2024-01-02, 2024-01-03, 2024-01-04, 2024-01-05
    existing = make_test_df([
        ('2024-01-01', 100.0),
        ('2024-01-02', 101.0),
        ('2024-01-03', 102.0),
        ('2024-01-04', 103.0),
        ('2024-01-05', 104.0),
    ])

    # New: 2024-01-03 only (force update for this one day)
    new = make_test_df([
        ('2024-01-03', 999.0),
    ])

    start = pd.Timestamp('2024-01-03', tz='UTC')
    end = pd.Timestamp('2024-01-03', tz='UTC')

    # Force mode: should replace only 2024-01-03
    merged = merge_price_history(existing, new, start, end, force=True)

    # Should have 5 rows
    assert len(merged) == 5

    # Check that dates outside range are preserved
    row_jan_01 = merged[merged['timestamp'] == pd.Timestamp('2024-01-01', tz='UTC')]
    assert row_jan_01.iloc[0]['closing_price'] == 100.0

    row_jan_02 = merged[merged['timestamp'] == pd.Timestamp('2024-01-02', tz='UTC')]
    assert row_jan_02.iloc[0]['closing_price'] == 101.0

    row_jan_04 = merged[merged['timestamp'] == pd.Timestamp('2024-01-04', tz='UTC')]
    assert row_jan_04.iloc[0]['closing_price'] == 103.0

    row_jan_05 = merged[merged['timestamp'] == pd.Timestamp('2024-01-05', tz='UTC')]
    assert row_jan_05.iloc[0]['closing_price'] == 104.0

    # Check that 2024-01-03 was replaced
    row_jan_03 = merged[merged['timestamp'] == pd.Timestamp('2024-01-03', tz='UTC')]
    assert row_jan_03.iloc[0]['closing_price'] == 999.0


def test_merge_price_history_no_duplicates():
    """Test that merge removes duplicate timestamps."""
    # Existing: 2024-01-01, 2024-01-02
    existing = make_test_df([
        ('2024-01-01', 100.0),
        ('2024-01-02', 101.0),
    ])

    # New: 2024-01-02 (duplicate), 2024-01-03
    new = make_test_df([
        ('2024-01-02', 102.0),
        ('2024-01-03', 103.0),
    ])

    start = pd.Timestamp('2024-01-02', tz='UTC')
    end = pd.Timestamp('2024-01-03', tz='UTC')

    # Incremental mode
    merged = merge_price_history(existing, new, start, end, force=False)

    # Check no duplicates
    assert len(merged) == len(merged['timestamp'].unique())


def test_merge_price_history_handles_timezone_naive():
    """Test that merge handles timezone-naive timestamps."""
    # Create timezone-naive timestamps
    existing = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'open_price': [100.0, 101.0],
        'high_price': [101.0, 102.0],
        'low_price': [99.0, 100.0],
        'closing_price': [100.0, 101.0],
        'volume': [1000000, 1000000],
    })

    new = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-03']),
        'open_price': [102.0],
        'high_price': [103.0],
        'low_price': [101.0],
        'closing_price': [102.0],
        'volume': [1000000],
    })

    start = pd.Timestamp('2024-01-03')  # Also naive
    end = pd.Timestamp('2024-01-03')

    # Should not raise error, should localize to UTC
    merged = merge_price_history(existing, new, start, end, force=False)

    # Should have 3 rows
    assert len(merged) == 3

    # Timestamps should now be timezone-aware (UTC)
    assert merged['timestamp'].dt.tz is not None


def test_merge_price_history_empty_new_data():
    """Test merge with empty new data."""
    # Existing: 2024-01-01, 2024-01-02
    existing = make_test_df([
        ('2024-01-01', 100.0),
        ('2024-01-02', 101.0),
    ])

    # Empty new data
    new = make_test_df([])

    start = pd.Timestamp('2024-01-03', tz='UTC')
    end = pd.Timestamp('2024-01-04', tz='UTC')

    # Should return existing data unchanged
    merged = merge_price_history(existing, new, start, end, force=False)

    assert len(merged) == 2
    assert set(merged['timestamp']) == set(existing['timestamp'])


def test_merge_price_history_incremental_adds_only_new_dates():
    """Test that incremental mode truly only adds dates not in existing."""
    # Existing: 2024-01-01, 2024-01-03, 2024-01-05 (gaps)
    existing = make_test_df([
        ('2024-01-01', 100.0),
        ('2024-01-03', 102.0),
        ('2024-01-05', 104.0),
    ])

    # New: 2024-01-02, 2024-01-03 (duplicate), 2024-01-04
    new = make_test_df([
        ('2024-01-02', 201.0),
        ('2024-01-03', 202.0),  # Already exists, should be ignored
        ('2024-01-04', 203.0),
    ])

    start = pd.Timestamp('2024-01-02', tz='UTC')
    end = pd.Timestamp('2024-01-04', tz='UTC')

    # Incremental mode: should add 2024-01-02 and 2024-01-04, preserve existing 2024-01-03
    merged = merge_price_history(existing, new, start, end, force=False)

    # Should have 5 rows
    assert len(merged) == 5

    # Check 2024-01-03 has ORIGINAL price
    row_jan_03 = merged[merged['timestamp'] == pd.Timestamp('2024-01-03', tz='UTC')]
    assert row_jan_03.iloc[0]['closing_price'] == 102.0  # Original, not 202.0

    # Check 2024-01-02 and 2024-01-04 were added
    row_jan_02 = merged[merged['timestamp'] == pd.Timestamp('2024-01-02', tz='UTC')]
    assert row_jan_02.iloc[0]['closing_price'] == 201.0

    row_jan_04 = merged[merged['timestamp'] == pd.Timestamp('2024-01-04', tz='UTC')]
    assert row_jan_04.iloc[0]['closing_price'] == 203.0
