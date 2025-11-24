"""
Massive.com data provider implementation.

**Conceptual**: This module implements the DataProvider protocol for Massive.com.
It bridges between MassiveClient (which returns raw JSON) and the application
(which expects pandas DataFrames with canonical column names and ordering).

**Layered architecture**:
  1. MassiveClient: HTTP layer - handles requests/responses, returns JSON
  2. MassiveDataProvider (this file): Adapter layer - converts JSON to DataFrame
  3. Strategies/Backtester: Business logic - uses DataFrames

Each layer has a single responsibility, making the system testable and maintainable.

**Why separate client from provider?**
  - Client knows about HTTP, provider knows about data transformation
  - Can test provider logic without making HTTP requests (mock client)
  - Can reuse client for different data types (daily, intraday, options)
  - Clear separation of concerns (HTTP vs business logic)

**Teaching note**: This is the "Adapter Pattern" - we adapt Massive's API
response format to our application's expected format (DataProvider protocol).
Different vendors have different API formats, but all providers return the
same DataFrame structure.
"""

import pandas as pd
from typing import Optional
from src.config.settings import MassiveSettings
from src.venues.massive_client import MassiveClient
from src.venues.base import DataProvider


class MassiveDataProviderError(Exception):
    """
    Base exception for MassiveDataProvider errors.

    **Conceptual**: Separate from MassiveClientError to distinguish between
    HTTP-layer errors (client) and data-layer errors (provider).

    Client errors: Network issues, auth failures, 404s
    Provider errors: Malformed responses, missing columns, validation failures
    """
    pass


class MassiveDataProvider:
    """
    DataProvider implementation for Massive.com API.

    **Conceptual**: This class implements the DataProvider protocol, fetching
    historical daily bars from Massive.com and converting them to standardized
    DataFrames that conform to our application's data contract.

    **Data transformation pipeline**:
      1. Validate input parameters (symbol, date range)
      2. Use MassiveClient to fetch raw JSON from API
      3. Parse JSON into list of bar records
      4. Convert to pandas DataFrame
      5. Rename columns to canonical names (closing_price, not close)
      6. Add timezone info to timestamps (UTC)
      7. Sort by timestamp ASCENDING (oldest first, matching Massive sort=asc)
      8. Validate output (no NaNs, correct columns, etc.)
      9. Return DataFrame

    **Why this class doesn't inherit from DataProvider**:
    Python Protocols use structural typing (duck typing). As long as this class
    has a fetch_daily_bars() method with the right signature, it satisfies the
    DataProvider protocol. No need for explicit inheritance.

    **Example usage**:
        >>> from src.config.settings import get_settings
        >>> settings = get_settings(require_massive=True)
        >>> provider = MassiveDataProvider(settings.massive)
        >>>
        >>> # Fetch QQQ daily bars
        >>> bars = provider.fetch_daily_bars(
        ...     symbol="QQQ",
        ...     start=pd.Timestamp("2024-01-01", tz="UTC"),
        ...     end=pd.Timestamp("2024-01-31", tz="UTC"),
        ... )
        >>> print(bars.shape)  # (22, 6) - 22 trading days in January
        >>> print(bars.columns.tolist())
        ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
        >>> # Oldest date first (ascending order)
        >>> print(bars.iloc[0]['timestamp'])  # 2024-01-01
        >>> print(bars.iloc[-1]['timestamp'])  # 2024-01-31
    """

    def __init__(self, settings: MassiveSettings, client: Optional[MassiveClient] = None):
        """
        Initialize Massive data provider.

        **Conceptual**: The provider needs MassiveSettings to configure the
        HTTP client. You can optionally inject a pre-configured client (useful
        for testing with mocks).

        **Dependency injection pattern**:
        By allowing optional client injection, we can test this class without
        making real HTTP requests:
            >>> mock_client = MockMassiveClient()
            >>> provider = MassiveDataProvider(settings, client=mock_client)

        If no client is provided, we create a real one:
            >>> provider = MassiveDataProvider(settings)
            >>> # provider.client is a real MassiveClient

        Args:
            settings: Massive API configuration.
            client: Optional pre-configured MassiveClient (for testing/DI).
                   If None, creates a new client from settings.

        Example:
            >>> from src.config.settings import get_settings
            >>> settings = get_settings(require_massive=True)
            >>> provider = MassiveDataProvider(settings.massive)
        """
        self.settings = settings
        self.client = client if client is not None else MassiveClient(settings)

    def fetch_daily_bars(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars from Massive.com.

        **Conceptual**: This method implements the DataProvider protocol,
        fetching historical daily bars and returning a DataFrame with canonical
        columns and ordering.

        **Implementation steps**:
          1. Validate inputs (symbol not empty, start <= end, timestamps tz-aware)
          2. Convert pd.Timestamp to ISO date strings for API (YYYY-MM-DD)
          3. Call MassiveClient.get_daily_bars() to fetch JSON
          4. Parse JSON response (extract bars list)
          5. Convert to DataFrame
          6. Rename columns to canonical names
          7. Parse timestamp strings to pd.Timestamp with timezone
          8. Sort by timestamp DESCENDING (newest first)
          9. Validate output (no missing values, correct columns)
          10. Return DataFrame

        **Massive API response format** (hypothetical):
        We assume Massive returns JSON like:
            {
                "symbol": "QQQ",
                "bars": [
                    {
                        "date": "2024-01-31",
                        "open": 450.23,
                        "high": 452.11,
                        "low": 449.80,
                        "close": 451.50,
                        "volume": 5000000
                    },
                    ...
                ],
                "count": 22
            }

        The actual format may differ - adjust parsing logic based on real API docs.

        **Column mapping**:
        Massive API → Canonical names:
          - date → timestamp (parsed to pd.Timestamp with UTC timezone)
          - open → open_price
          - high → high_price
          - low → low_price
          - close → closing_price
          - volume → volume (unchanged)

        **Timezone handling**:
        All timestamps are converted to UTC timezone for consistency. Most
        market data APIs return dates without time (just YYYY-MM-DD), which we
        interpret as market close in UTC.

        **Why UTC?**
          - Avoids DST bugs (UTC has no DST)
          - Consistent across all symbols (even international markets)
          - Standard for financial data
          - Easy to convert to local timezone if needed

        **Error handling**:
        This method may raise:
          - ValueError: Invalid inputs (empty symbol, start > end, etc.)
          - MassiveAuthenticationError: Bad API key (from client)
          - MassiveSymbolNotFoundError: Symbol not found (from client)
          - MassiveDataProviderError: Malformed response, validation failures
          - requests.Timeout: Request timeout (from client)

        Args:
            symbol: Ticker symbol (e.g., "QQQ", "SPY").
            start: Start date (inclusive), must be timezone-aware.
            end: End date (inclusive), must be timezone-aware.

        Returns:
            DataFrame with columns:
              - timestamp: pd.Timestamp (UTC timezone)
              - open_price: float
              - high_price: float
              - low_price: float
              - closing_price: float
              - volume: float

            Sorted by timestamp DESCENDING (newest first).
            Empty DataFrame (with correct columns) if no data in range.

        Raises:
            ValueError: If inputs are invalid.
            MassiveDataProviderError: If response is malformed or validation fails.
            MassiveAuthenticationError: If API authentication fails.
            MassiveSymbolNotFoundError: If symbol not found.
            requests.Timeout: If request times out.

        Example:
            >>> provider = MassiveDataProvider(settings)
            >>> bars = provider.fetch_daily_bars(
            ...     symbol="QQQ",
            ...     start=pd.Timestamp("2024-01-01", tz="UTC"),
            ...     end=pd.Timestamp("2024-01-31", tz="UTC"),
            ... )
            >>> print(bars.shape)  # (22, 6)
            >>> # Check ordering: newest first
            >>> assert bars.iloc[0]['timestamp'] > bars.iloc[-1]['timestamp']
        """
        # Step 1: Validate inputs
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")

        if start is None or end is None:
            raise ValueError("start and end must be provided")

        if start > end:
            raise ValueError(
                f"start ({start}) must be <= end ({end})"
            )

        # Check that timestamps are timezone-aware
        if start.tz is None or end.tz is None:
            raise ValueError(
                "start and end must be timezone-aware. "
                "Use pd.Timestamp('2024-01-01', tz='UTC')"
            )

        # Step 2: Convert timestamps to ISO date strings for API
        # Massive API expects YYYY-MM-DD format
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        # Step 3: Fetch DataFrame from Massive API via client
        try:
            df_raw = self.client.get_daily_bars(
                symbol=symbol.strip().upper(),
                start_date=start_str,
                end_date=end_str,
            )
        except Exception as e:
            # Re-raise client errors (auth, not found, timeout, etc.)
            # We don't wrap them because they're already descriptive
            raise

        # If no data in range, return empty DataFrame with canonical columns
        if df_raw is None or len(df_raw) == 0:
            return pd.DataFrame(columns=[
                'timestamp', 'open_price', 'high_price',
                'low_price', 'closing_price', 'volume'
            ])

        df = df_raw.copy()

        # Step 4: Rename columns from Massive short/clear names to canonical names
        column_mapping = {
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'closing_price',
        }
        # Required columns from client
        required_client_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_client_columns if col not in df.columns]
        if missing_cols:
            raise MassiveDataProviderError(
                f"Response missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.rename(columns=column_mapping)

        # Step 5: Ensure timestamp is datetime with UTC (client already parses)
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            except Exception as e:
                raise MassiveDataProviderError(
                    f"Failed to parse timestamps: {e}. "
                    f"Sample timestamp: {df['timestamp'].iloc[0] if len(df) > 0 else None}"
                ) from e

        # Step 6: Sort by timestamp ASCENDING (Massive returns asc when requested)
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

        # Step 9: Validate output
        # Check for missing values in critical columns
        critical_cols = ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
        for col in critical_cols:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                raise MassiveDataProviderError(
                    f"Column '{col}' has {nan_count} NaN values. "
                    f"Data quality issue in Massive response."
                )

        # Check that prices are positive
        price_cols = ['open_price', 'high_price', 'low_price', 'closing_price']
        for col in price_cols:
            if (df[col] <= 0).any():
                raise MassiveDataProviderError(
                    f"Column '{col}' has non-positive values. "
                    f"Data quality issue."
                )

        # Check that high >= low
        if (df['high_price'] < df['low_price']).any():
            raise MassiveDataProviderError(
                "Found bars where high_price < low_price. Data quality issue."
            )

        # Step 10: Select only canonical columns (drop any extras from API)
        df = df[critical_cols]

        return df

    def close(self):
        """
        Close the underlying HTTP client.

        **Conceptual**: Release HTTP connection pool resources.
        Call this when done using the provider, especially in long-running scripts.

        **Better pattern**: Use context manager:
            >>> with MassiveDataProvider(settings) as provider:
            ...     bars = provider.fetch_daily_bars("QQQ", start, end)
            >>> # Client automatically closed

        Example:
            >>> provider = MassiveDataProvider(settings)
            >>> try:
            ...     bars = provider.fetch_daily_bars("QQQ", start, end)
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
