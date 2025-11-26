"""
Finnhub data provider implementation.

**Conceptual**: This module provides a data adapter for the Finnhub API.
It bridges between FinnhubClient (which returns raw JSON) and the application
(which expects pandas DataFrames with canonical column names).

**Layered architecture**:
  1. FinnhubClient: HTTP layer - handles requests/responses, returns JSON
  2. FinnhubDataProvider (this file): Adapter layer - converts JSON to DataFrame
  3. Strategies/Backtester: Business logic - uses DataFrames

Each layer has a single responsibility, making the system testable and maintainable.

**Why separate client from provider?**
  - Client knows about HTTP, provider knows about data transformation
  - Can test provider logic without making HTTP requests (mock client)
  - Can reuse client for different data types (daily, intraday, quote)
  - Clear separation of concerns (HTTP vs business logic)

**Usage note**: Designed for research/free-tier usage; respect Finnhub rate limits
documented in DEV_NOTES (default pacing via min_sleep_seconds) and avoid embedding
secrets in code. The webhook secret applies only to webhook endpoints, not pulls.
"""

import datetime as dt
import pandas as pd
from typing import Optional
from src.config.settings import FinnhubSettings
from src.venues.finnhub_client import FinnhubClient


class FinnhubDataProviderError(Exception):
    """
    Base exception for FinnhubDataProvider errors.

    **Conceptual**: Separate from FinnhubClientError to distinguish between
    HTTP-layer errors (client) and data-layer errors (provider).

    Client errors: Network issues, auth failures, 404s, rate limits
    Provider errors: Malformed responses, missing columns, validation failures
    """
    pass

class FinnhubAccessError(RuntimeError):
    """
    Raised when Finnhub reports that the client has no access to the requested
    resource (plan/permissions issue on stock/candle).
    """
    pass


class FinnhubDataProvider:
    """
    Data provider for Finnhub API.

    **Conceptual**: This class fetches historical daily bars from Finnhub
    and converts them to DataFrames with canonical column names.

    **Data transformation pipeline**:
      1. Validate input parameters (ticker, date range)
      2. Convert dt.date to Unix timestamps (Finnhub uses epochs)
      3. Use FinnhubClient to fetch raw JSON from API
      4. Parse JSON arrays (c, h, l, o, t, v) into records
      5. Convert to pandas DataFrame
      6. Convert Unix timestamps to pd.Timestamp with UTC timezone
      7. Filter to [start_date, end_date] inclusive window
      8. Validate output (no NaNs, correct columns, etc.)
      9. Return DataFrame (unsorted, unformatted - IO layer handles that)

    **Example usage**:
        >>> from src.config.settings import get_settings
        >>> settings = get_settings(require_finnhub=True)
        >>> provider = FinnhubDataProvider(settings.finnhub)
        >>>
        >>> # Fetch AAPL daily bars
        >>> bars = provider.get_daily_bars(
        ...     ticker="AAPL",
        ...     start_date=dt.date(2024, 1, 1),
        ...     end_date=dt.date(2024, 1, 31),
        ... )
        >>> print(bars.shape)  # (22, 6) - 22 trading days in January
        >>> print(bars.columns.tolist())
        ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
    """

    def __init__(self, settings: FinnhubSettings, client: Optional[FinnhubClient] = None):
        """
        Initialize Finnhub data provider.

        **Conceptual**: The provider needs FinnhubSettings to configure the
        HTTP client. You can optionally inject a pre-configured client (useful
        for testing with mocks).

        **Dependency injection pattern**:
        By allowing optional client injection, we can test this class without
        making real HTTP requests:
            >>> mock_client = MockFinnhubClient()
            >>> provider = FinnhubDataProvider(settings, client=mock_client)

        If no client is provided, we create a real one:
            >>> provider = FinnhubDataProvider(settings)
            >>> # provider.client is a real FinnhubClient

        Args:
            settings: Finnhub API configuration.
            client: Optional pre-configured FinnhubClient (for testing/DI).
                   If None, creates a new client from settings.

        Example:
            >>> from src.config.settings import get_settings
            >>> settings = get_settings(require_finnhub=True)
            >>> provider = FinnhubDataProvider(settings.finnhub)
        """
        self.settings = settings
        self.client = client if client is not None else FinnhubClient(settings)

    def get_daily_bars(
        self,
        ticker: str,
        start_date: dt.date,
        end_date: dt.date,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars from Finnhub.

        **Conceptual**: This method fetches historical daily bars and returns
        a DataFrame with canonical columns. It does NOT sort or format the
        timestamp - that's the IO layer's responsibility.

        **Implementation steps**:
          1. Validate inputs (ticker not empty, start_date <= end_date)
          2. Convert dt.date to Unix timestamps for API (seconds since epoch)
          3. Call FinnhubClient.get_candles() to fetch JSON
          4. Parse JSON response (arrays: c, h, l, o, t, v, s)
          5. Convert arrays to list of records (one per bar)
          6. Convert to DataFrame
          7. Convert Unix timestamps to pd.Timestamp with UTC timezone
          8. Filter to [start_date, end_date] inclusive window
          9. Validate output (no missing values, correct columns)
          10. Return DataFrame (unsorted, unformatted)

        **Finnhub API response format**:
        Finnhub returns JSON with parallel arrays:
            {
                "c": [451.50, 452.10, ...],   # Closing prices
                "h": [452.11, 453.20, ...],   # High prices
                "l": [449.80, 450.30, ...],   # Low prices
                "o": [450.23, 451.50, ...],   # Opening prices
                "t": [1706745600, 1706832000, ...],  # Unix timestamps (seconds)
                "v": [5000000, 5100000, ...], # Volumes
                "s": "ok"                     # Status
            }

        **Column names**:
        Finnhub fields → DataFrame columns:
          - t → timestamp (converted from Unix timestamp to pd.Timestamp UTC)
          - o → open_price
          - h → high_price
          - l → low_price
          - c → closing_price
          - v → volume

        **Timezone handling**:
        Finnhub returns Unix timestamps (seconds since epoch), which are inherently
        UTC. We convert these to pd.Timestamp with explicit UTC timezone.

        **Why Unix timestamps?**
          - No timezone ambiguity (always UTC)
          - Easy to work with for date range filtering
          - Compact representation (single integer)

        **Error handling**:
        This method may raise:
          - ValueError: Invalid inputs (empty ticker, start_date > end_date, etc.)
          - FinnhubAuthenticationError: Bad API key (from client)
          - FinnhubSymbolNotFoundError: Symbol not found or no data (from client)
          - FinnhubRateLimitError: Rate limit exceeded (from client)
          - FinnhubDataProviderError: Malformed response, validation failures
          - FinnhubAccessError: Finnhub reports no access to stock/candle for this key/symbol.
          - requests.Timeout: Request timeout (from client)

        Args:
            ticker: Ticker symbol (e.g., "AAPL", "QQQ", "SPY").
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            DataFrame with columns:
              - timestamp: pd.Timestamp (UTC timezone)
              - open_price: float
              - high_price: float
              - low_price: float
              - closing_price: float
              - volume: float

            Unsorted, with timestamp as pd.Timestamp (not formatted string).
            Empty DataFrame (with correct columns) if no data in range.

        Raises:
            ValueError: If inputs are invalid.
            FinnhubDataProviderError: If response is malformed or validation fails.
            FinnhubAuthenticationError: If API authentication fails.
            FinnhubSymbolNotFoundError: If symbol not found.
            FinnhubRateLimitError: If rate limit exceeded.
            requests.Timeout: If request times out.

        Example:
            >>> provider = FinnhubDataProvider(settings)
            >>> bars = provider.get_daily_bars(
            ...     ticker="AAPL",
            ...     start_date=dt.date(2024, 1, 1),
            ...     end_date=dt.date(2024, 1, 31),
            ... )
            >>> print(bars.shape)  # (22, 6)
            >>> print(bars.columns.tolist())
            ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
        """
        # Step 1: Validate inputs
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date must be provided")

        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be <= end_date ({end_date})"
            )

        # Step 2: Convert dates to Unix timestamps (seconds since epoch)
        # Finnhub API expects integer Unix timestamps
        # Use datetime.combine to convert date to datetime, then to timestamp
        start_dt = dt.datetime.combine(start_date, dt.time.min)
        end_dt = dt.datetime.combine(end_date, dt.time.max)

        from_timestamp = int(start_dt.replace(tzinfo=dt.timezone.utc).timestamp())
        to_timestamp = int(end_dt.replace(tzinfo=dt.timezone.utc).timestamp())

        # Step 3: Fetch candles from Finnhub API via client
        try:
            response = self.client.get_candles(
                symbol=ticker.strip().upper(),
                resolution="D",  # Daily candles
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
            )
        except Exception:
            # Re-raise client errors (auth, not found, rate limit, timeout, etc.)
            # We don't wrap them because they're already descriptive
            raise

        # Step 4: Parse JSON response
        # Finnhub returns parallel arrays: {"c": [...], "h": [...], "l": [...], ...}
        # We need to transpose these into a list of records

        # Finnhub may return an error field even on HTTP 200 if the plan lacks access
        if isinstance(response, dict) and "error" in response:
            err_msg = str(response.get("error", "")).lower()
            if "don't have access" in err_msg or "no access" in err_msg or "access" in err_msg:
                raise FinnhubAccessError(
                    f"Finnhub reports access denied for ticker={ticker.strip().upper()} resolution=D "
                    f"on stock/candle (plan/permissions issue)."
                )

        # Extract arrays from response
        closes = response.get('c', [])
        highs = response.get('h', [])
        lows = response.get('l', [])
        opens = response.get('o', [])
        timestamps = response.get('t', [])
        volumes = response.get('v', [])

        # Check that all arrays have the same length
        lengths = [len(closes), len(highs), len(lows), len(opens), len(timestamps), len(volumes)]
        if len(set(lengths)) > 1:
            raise FinnhubDataProviderError(
                f"Finnhub response has inconsistent array lengths: {dict(zip(['c', 'h', 'l', 'o', 't', 'v'], lengths))}"
            )

        # If no data, return empty DataFrame with canonical columns
        if len(timestamps) == 0:
            return pd.DataFrame(columns=[
                'timestamp', 'open_price', 'high_price',
                'low_price', 'closing_price', 'volume'
            ])

        # Step 5: Convert arrays to DataFrame
        # Build records from parallel arrays
        records = []
        for i in range(len(timestamps)):
            records.append({
                'timestamp': timestamps[i],
                'open_price': opens[i],
                'high_price': highs[i],
                'low_price': lows[i],
                'closing_price': closes[i],
                'volume': volumes[i],
            })

        df = pd.DataFrame(records)

        # Step 6: Convert Unix timestamps to pd.Timestamp with UTC timezone
        # Finnhub timestamps are Unix timestamps (seconds since epoch)
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        except Exception as e:
            raise FinnhubDataProviderError(
                f"Failed to parse timestamps from Unix format: {e}. "
                f"Sample timestamp: {timestamps[0] if timestamps else None}"
            ) from e

        # Step 7: Filter to [start_date, end_date] inclusive window
        # Convert start_date and end_date to pd.Timestamp for comparison
        start_ts = pd.Timestamp(start_date, tz='UTC')
        end_ts = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].reset_index(drop=True)

        # Step 8: Validate output
        # Check for missing values in critical columns
        critical_cols = ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
        for col in critical_cols:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                raise FinnhubDataProviderError(
                    f"Column '{col}' has {nan_count} NaN values. "
                    f"Data quality issue in Finnhub response."
                )

        # Check that prices are positive
        price_cols = ['open_price', 'high_price', 'low_price', 'closing_price']
        for col in price_cols:
            if (df[col] <= 0).any():
                raise FinnhubDataProviderError(
                    f"Column '{col}' has non-positive values. "
                    f"Data quality issue."
                )

        # Check that high >= low
        if (df['high_price'] < df['low_price']).any():
            raise FinnhubDataProviderError(
                "Found bars where high_price < low_price. Data quality issue."
            )

        # Step 9: Select only canonical columns (drop any extras)
        df = df[critical_cols]

        # Step 10: Return DataFrame (unsorted, unformatted - IO layer handles that)
        return df

    def close(self):
        """
        Close the underlying HTTP client.

        **Conceptual**: Release HTTP connection pool resources and reset rate limit state.
        Call this when done using the provider, especially in long-running scripts.

        **Better pattern**: Use context manager:
            >>> with FinnhubDataProvider(settings) as provider:
            ...     bars = provider.get_daily_bars("AAPL", start_date, end_date)
            >>> # Client automatically closed

        Example:
            >>> provider = FinnhubDataProvider(settings)
            >>> try:
            ...     bars = provider.get_daily_bars("AAPL", start_date, end_date)
            ... finally:
            ...     provider.close()
        """
        self.client.close()

    def __enter__(self):
        """Enable context manager support (with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up client when exiting context manager."""
        self.close()
        return False  # Don't suppress exceptions
