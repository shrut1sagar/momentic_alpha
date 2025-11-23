"""
Time and clock abstractions for deterministic testing and backtesting.

This module provides a simple, testable way to obtain "now" via a clock object
rather than calling datetime.now() directly. This enables deterministic tests
and backtests by "freezing" time to a specific timestamp.

The key insight: depending on a Clock abstraction instead of system time makes
your code testable, reproducible, and backtest-friendly. You can simulate being
"in the past" to test strategy logic without touching real wall-clock time.
"""

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    """
    Abstract time source protocol.

    **Conceptual**: A Clock is any object that can answer the question "what time
    is it right now?" By depending on this abstraction instead of directly calling
    datetime.now() or datetime.utcnow(), code becomes testable and deterministic.
    This is critical for backtesting, where we need to simulate being at a specific
    historical timestamp and ensure no future data leaks into decisions.

    **Usage**: Consumers should accept a Clock instance (injected via constructor
    or function parameter) and call clock.now() whenever they need the current time.
    In production, pass a RealClock; in tests and backtests, pass a FrozenClock.

    **Example**:
        # In a strategy or backtest engine:
        def run(data, clock: Clock):
            current_time = clock.now()
            # Use current_time for logic, logging, filenames, etc.

        # In production:
        run(data, RealClock())

        # In tests or backtest:
        run(data, FrozenClock(datetime(2015, 1, 5, tzinfo=timezone.utc)))
    """

    def now(self) -> datetime:
        """
        Return the current time according to this clock.

        Returns:
            datetime object representing "now" (timezone-aware, UTC preferred).
        """
        ...


class RealClock:
    """
    Clock that returns the actual current system time (UTC).

    **Conceptual**: Use this in production or live environments where you need
    real wall-clock time. RealClock delegates to datetime.now(timezone.utc),
    ensuring you always get the current UTC timestamp.

    **Usage**:
        clock = RealClock()
        current_time = clock.now()  # Returns current UTC time
    """

    def now(self) -> datetime:
        """
        Return the current UTC time from the system clock.

        Returns:
            datetime object with current time in UTC timezone.
        """
        # Use timezone.utc to ensure timezone-aware datetime
        return datetime.now(timezone.utc)


class FrozenClock:
    """
    Clock that always returns a fixed timestamp (for deterministic tests/backtests).

    **Conceptual**: Use this in tests and backtests to simulate being at a specific
    point in time. FrozenClock "freezes" time to a configured timestamp, making
    your code deterministic and reproducible. This is essential for:
      - Unit tests: verify time-dependent logic without flaky system clock issues.
      - Backtests: simulate being "in the past" to ensure no future data leakage.
      - Debugging: reproduce exact conditions by fixing timestamps.

    **Usage**:
        # Freeze time to January 5, 2015 at midnight UTC
        clock = FrozenClock(datetime(2015, 1, 5, tzinfo=timezone.utc))
        current_time = clock.now()  # Always returns 2015-01-05T00:00:00+00:00

        # In a backtest loop, you can update the frozen clock as you step through dates:
        for date in date_range:
            clock = FrozenClock(date)
            # Run strategy with this clock
    """

    def __init__(self, fixed_now: datetime):
        """
        Initialize a FrozenClock with a fixed timestamp.

        Args:
            fixed_now: The datetime to return on every call to now().
                       Should be timezone-aware (UTC recommended).
        """
        self._fixed_now = fixed_now

    def now(self) -> datetime:
        """
        Return the configured fixed timestamp.

        Returns:
            The datetime set at initialization (immutable).
        """
        return self._fixed_now


def get_real_clock() -> Clock:
    """
    Factory function to create a RealClock instance.

    **Conceptual**: Convenience factory for obtaining a real-time clock.
    Useful for dependency injection or config-driven clock selection.

    **Usage**:
        clock = get_real_clock()
        current_time = clock.now()

    Returns:
        RealClock instance.
    """
    return RealClock()


def get_frozen_clock(fixed_now: datetime) -> Clock:
    """
    Factory function to create a FrozenClock with a given timestamp.

    **Conceptual**: Convenience factory for creating a frozen clock. This is
    particularly useful in test setup or backtest initialization, where you
    want to cleanly specify the simulated "now" without directly constructing
    the FrozenClock class.

    **Usage**:
        # In a test or backtest setup:
        clock = get_frozen_clock(datetime(2015, 1, 5, tzinfo=timezone.utc))
        # Pass clock to functions/classes that need deterministic time

    Args:
        fixed_now: The datetime to freeze at (timezone-aware recommended).

    Returns:
        FrozenClock instance configured with fixed_now.
    """
    return FrozenClock(fixed_now)
