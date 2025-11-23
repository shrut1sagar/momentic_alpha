"""
Tests for src/utils/time.py

These tests verify the clock abstraction works correctly for both real and frozen time.
"""

from datetime import datetime, timezone, timedelta
import time

from src.utils.time import (
    Clock,
    RealClock,
    FrozenClock,
    get_real_clock,
    get_frozen_clock,
)


def test_real_clock_returns_current_time():
    """Test that RealClock returns a time close to actual current time."""
    clock = RealClock()

    # Get current UTC time before and after calling clock.now()
    before = datetime.now(timezone.utc)
    clock_time = clock.now()
    after = datetime.now(timezone.utc)

    # clock_time should be between before and after (within a small window)
    assert before <= clock_time <= after

    # Verify timezone is UTC
    assert clock_time.tzinfo == timezone.utc


def test_real_clock_advances():
    """Test that RealClock returns different times on successive calls."""
    clock = RealClock()

    time1 = clock.now()
    # Small delay to ensure time advances
    time.sleep(0.01)
    time2 = clock.now()

    # time2 should be after time1
    assert time2 > time1


def test_frozen_clock_returns_fixed_time():
    """Test that FrozenClock always returns the configured timestamp."""
    fixed_time = datetime(2015, 1, 5, 12, 30, 45, tzinfo=timezone.utc)
    clock = FrozenClock(fixed_time)

    # Call now() multiple times; should always return fixed_time
    assert clock.now() == fixed_time
    assert clock.now() == fixed_time
    assert clock.now() == fixed_time


def test_frozen_clock_different_instances():
    """Test that different FrozenClock instances can have different times."""
    time1 = datetime(2015, 1, 5, tzinfo=timezone.utc)
    time2 = datetime(2020, 6, 15, tzinfo=timezone.utc)

    clock1 = FrozenClock(time1)
    clock2 = FrozenClock(time2)

    # Each clock returns its own fixed time
    assert clock1.now() == time1
    assert clock2.now() == time2
    assert clock1.now() != clock2.now()


def test_get_real_clock_factory():
    """Test that get_real_clock() returns a working RealClock."""
    clock = get_real_clock()

    # Should return a Clock (duck typing or Protocol check)
    assert hasattr(clock, 'now')
    assert callable(clock.now)

    # Should return current time
    now = clock.now()
    assert isinstance(now, datetime)
    assert now.tzinfo == timezone.utc


def test_get_frozen_clock_factory():
    """Test that get_frozen_clock() returns a working FrozenClock."""
    fixed_time = datetime(2018, 3, 10, 8, 15, 0, tzinfo=timezone.utc)
    clock = get_frozen_clock(fixed_time)

    # Should return a Clock
    assert hasattr(clock, 'now')
    assert callable(clock.now)

    # Should return the fixed time
    assert clock.now() == fixed_time


def test_frozen_clock_simulates_backtest():
    """Test that FrozenClock can simulate stepping through time in a backtest."""
    # Simulate a backtest loop where we step through dates
    dates = [
        datetime(2015, 1, 5, tzinfo=timezone.utc),
        datetime(2015, 1, 6, tzinfo=timezone.utc),
        datetime(2015, 1, 7, tzinfo=timezone.utc),
    ]

    clocks = [FrozenClock(date) for date in dates]

    # Verify each clock returns its respective date
    for clock, expected_date in zip(clocks, dates):
        assert clock.now() == expected_date


def test_frozen_clock_timezone_aware():
    """Test that FrozenClock works with timezone-aware datetimes."""
    # Create a timezone-aware datetime
    fixed_time = datetime(2020, 7, 4, 10, 30, tzinfo=timezone.utc)
    clock = FrozenClock(fixed_time)

    # Should preserve timezone
    assert clock.now().tzinfo == timezone.utc
    assert clock.now() == fixed_time


def test_frozen_clock_immutable():
    """Test that calling now() doesn't change the FrozenClock's internal state."""
    fixed_time = datetime(2021, 12, 25, tzinfo=timezone.utc)
    clock = FrozenClock(fixed_time)

    # Call now() multiple times
    for _ in range(10):
        result = clock.now()
        # Each call should return the exact same object/value
        assert result == fixed_time
        # Modifying result shouldn't affect the clock
        # (datetime is immutable, but verify clock still returns original)

    # Clock should still return the original fixed time
    assert clock.now() == fixed_time
