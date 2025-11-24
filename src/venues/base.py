"""
Base abstractions for data providers (venues).

**Conceptual**: This module defines the DataProvider protocol, which is the
abstract interface that all data sources must implement. By defining a common
protocol, we decouple our backtesting and analysis code from specific vendors
(Massive, Alpaca, Polygon, etc.).

**Why protocols over inheritance?**
  - Protocols are structural typing (duck typing) - no need to inherit from base class.
  - More flexible - any class that implements the required methods is a DataProvider.
  - Better for testing - easy to create mock providers without complex inheritance.
  - Pythonic - follows "if it walks like a duck and quacks like a duck, it's a duck".

**Teaching note**: In traditional OOP, you'd create an abstract base class with
abstract methods. Python's Protocol (PEP 544) is more flexible - it only checks
that the required methods exist with the right signatures, not that the class
explicitly inherits from DataProvider. This is called "structural subtyping".

**Data consistency guarantees**:
All DataProvider implementations MUST ensure:
  1. Canonical column names (timestamp, open_price, high_price, low_price,
     closing_price, volume).
  2. Timezone-aware timestamps (UTC recommended for consistency).
  3. Strictly decreasing timestamp order (newest first) - matches Phase 3 spec.
  4. No duplicate timestamps for the same symbol.
  5. No missing values in required columns (open, high, low, close, volume).

**Why these guarantees matter**:
  - Downstream code (strategies, backtester) can assume clean, consistent data.
  - No need for defensive checks at every data access point.
  - Bugs are caught early (in data provider) rather than silently propagating.
  - Different data sources produce identical DataFrame structure (vendor-agnostic code).
"""

from typing import Protocol
import pandas as pd


class DataProvider(Protocol):
    """
    Protocol for fetching historical market data from any data source.

    **Conceptual**: This protocol defines the contract that all data providers
    must fulfill. Any class that implements fetch_daily_bars() with the correct
    signature is considered a DataProvider, regardless of how it's implemented
    (HTTP API, database, CSV files, etc.).

    **Design philosophy**:
      - Minimal surface area (one method to start - easy to implement, hard to break).
      - Focus on daily bars (most common use case for backtesting).
      - Can be extended with fetch_intraday_bars(), fetch_options_chain(), etc. later.

    **Why start with daily bars only?**
      - Simplifies initial implementation (most strategies use daily data).
      - Easier to test (less data volume, simpler edge cases).
      - Can add intraday/options methods later without breaking existing code.

    **Implementation requirements**:
    Any class implementing this protocol MUST:
      1. Return DataFrame with exactly these columns:
         - timestamp: pd.Timestamp (timezone-aware, preferably UTC)
         - open_price: float
         - high_price: float
         - low_price: float
         - closing_price: float
         - volume: int or float
      2. Sort rows by timestamp in DESCENDING order (newest first).
      3. Handle missing data gracefully (raise clear error, don't return partial data).
      4. Validate date range (raise error if start > end).
      5. Raise descriptive errors (not generic "request failed").

    **Example usage**:
        >>> from src.venues.massive_data_provider import MassiveDataProvider
        >>> from src.config.settings import get_settings
        >>>
        >>> settings = get_settings(require_massive=True)
        >>> provider = MassiveDataProvider(settings.massive)
        >>>
        >>> # Fetch QQQ daily bars for 2024
        >>> bars = provider.fetch_daily_bars(
        ...     symbol="QQQ",
        ...     start=pd.Timestamp("2024-01-01", tz="UTC"),
        ...     end=pd.Timestamp("2024-12-31", tz="UTC"),
        ... )
        >>>
        >>> # Data is always in descending order (newest first)
        >>> assert bars.iloc[0]["timestamp"] > bars.iloc[-1]["timestamp"]
        >>> print(bars.head())
        #     timestamp              open_price  high_price  low_price  closing_price  volume
        # 0   2024-12-31 00:00:00+00:00  450.23      452.11     449.80  451.50        5000000
        # 1   2024-12-30 00:00:00+00:00  448.90      450.50     448.20  450.10        4800000
        # ...

    **Testing strategy**:
    When testing code that uses DataProvider, create a simple mock:
        >>> class MockDataProvider:
        ...     def fetch_daily_bars(self, symbol, start, end):
        ...         # Return synthetic test data
        ...         dates = pd.date_range(end, start, freq='D', tz='UTC')[::-1]
        ...         return pd.DataFrame({
        ...             'timestamp': dates,
        ...             'open_price': [100.0] * len(dates),
        ...             'high_price': [101.0] * len(dates),
        ...             'low_price': [99.0] * len(dates),
        ...             'closing_price': [100.5] * len(dates),
        ...             'volume': [1000000] * len(dates),
        ...         })
        >>>
        >>> # Use in tests without hitting real API
        >>> mock_provider = MockDataProvider()
        >>> bars = mock_provider.fetch_daily_bars("TEST", start, end)
    """

    def fetch_daily_bars(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for a symbol over a date range.

        **Conceptual**: This method retrieves historical daily price data from
        the underlying data source (API, database, CSV files, etc.). It's the
        primary interface for loading market data into the system.

        **Why daily bars?**
          - Most backtests use daily data (end-of-day signals).
          - Simpler to work with than intraday (no time-of-day complexity).
          - Less data volume (faster downloads, less storage).
          - Can always add intraday methods later if needed.

        **OHLCV explanation**:
          - Open: first trade price of the day
          - High: highest trade price of the day
          - Low: lowest trade price of the day
          - Close: last trade price of the day
          - Volume: total shares/contracts traded

        **Date range semantics**:
          - `start` and `end` are INCLUSIVE (data includes both boundary dates).
          - If start > end, raise ValueError (invalid range).
          - If no data exists in range, return empty DataFrame (with correct columns).
          - Timezone-aware timestamps are required (prevents DST/timezone bugs).

        **Return value guarantees**:
        The returned DataFrame MUST have:
          1. Columns: timestamp, open_price, high_price, low_price, closing_price, volume
          2. Rows sorted by timestamp DESCENDING (newest first, matches Phase 3 spec)
          3. Timezone-aware timestamps (preferably UTC)
          4. No NaN values in price/volume columns
          5. No duplicate timestamps

        **Why descending order?**
          - Phase 3 spec mandates strictly decreasing timestamps.
          - Most recent data appears first (often what you want to see).
          - Consistent with how databases often return time series (ORDER BY timestamp DESC).
          - Downstream code (strategies, features) can rely on this invariant.

        **Error handling**:
        Implementations should raise descriptive errors:
          - ValueError: Invalid parameters (start > end, empty symbol, etc.)
          - ConnectionError: Network/API failures
          - KeyError: Symbol not found
          - TimeoutError: Request took too long

        Args:
            symbol: Ticker symbol (e.g., "QQQ", "SPY", "AAPL").
                   Should be exchange-normalized (provider handles mapping).
            start: Start date (inclusive), timezone-aware.
                  Earliest date to fetch data for.
            end: End date (inclusive), timezone-aware.
                Latest date to fetch data for.

        Returns:
            DataFrame with columns:
              - timestamp (pd.Timestamp, tz-aware): Date of the bar
              - open_price (float): Opening price
              - high_price (float): Highest price
              - low_price (float): Lowest price
              - closing_price (float): Closing price
              - volume (float): Trading volume

            Sorted by timestamp in DESCENDING order (newest first).
            Empty DataFrame (with columns) if no data in range.

        Raises:
            ValueError: If start > end or symbol is empty/invalid.
            ConnectionError: If network/API request fails.
            KeyError: If symbol is not found in data source.
            TimeoutError: If request exceeds timeout threshold.

        Example:
            >>> provider = SomeDataProvider()
            >>> bars = provider.fetch_daily_bars(
            ...     symbol="QQQ",
            ...     start=pd.Timestamp("2024-01-01", tz="UTC"),
            ...     end=pd.Timestamp("2024-01-31", tz="UTC"),
            ... )
            >>> print(bars.shape)  # (22, 6) - 22 trading days in January 2024
            >>> print(bars.columns.tolist())
            ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
            >>> # Most recent date is first
            >>> print(bars.iloc[0]['timestamp'])
            Timestamp('2024-01-31 00:00:00+0000', tz='UTC')
        """
        ...
