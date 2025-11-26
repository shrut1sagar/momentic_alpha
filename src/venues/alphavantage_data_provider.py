"""
Alpha Vantage data provider implementation.

**Conceptual**: This module provides a data adapter for the Alpha Vantage API.
It fetches historical daily price data and converts it to pandas DataFrames
with canonical column names that match our system's schema.

**Why Alpha Vantage?**
  - Free tier available (25 API calls per day, 5 calls per minute)
  - Comprehensive historical data for US stocks, ETFs, forex, crypto
  - No credit card required for basic access
  - Good for small-scale research and development

**API Structure**:
  - Base URL: https://www.alphavantage.co/query
  - Authentication: API key passed as query parameter
  - Function: TIME_SERIES_DAILY for daily OHLCV data
  - Response format: JSON with nested time series data

**Rate Limits (Free Tier)**:
  - 25 API calls per day (very restrictive!)
  - 5 API calls per minute
  - Be very conservative: use incremental mode, cache results locally

**Teaching note**: Alpha Vantage's free tier is quite limited compared to
other providers. In production, you'd likely use a paid tier or a different
provider (Polygon, Alpaca, etc.). However, it's useful for learning and
small-scale backtesting where you don't need frequent data updates.

**Usage pattern**:
  - Fetch data once and cache it locally in data/raw/
  - Use incremental mode to fill gaps without wasting API calls
  - Consider using multiple providers to spread load
"""

import datetime as dt
import pandas as pd
import requests
from typing import Optional

from src.config.settings import AlphaVantageSettings


class AlphaVantageApiError(RuntimeError):
    """
    Generic error returned by Alpha Vantage API.

    **Conceptual**: Raised when Alpha Vantage returns an error message
    in the JSON response (e.g., invalid symbol, malformed request).

    **Recovery**: Check error message for details. Common issues:
      - Invalid ticker symbol
      - Malformed API request
      - API maintenance/outage
    """
    pass


class AlphaVantageRateLimitError(AlphaVantageApiError):
    """
    Raised when Alpha Vantage reports rate limit or quota exceeded.

    **Conceptual**: Alpha Vantage free tier has very strict limits:
      - 25 API calls per day
      - 5 API calls per minute

    When you hit these limits, the API returns a "Note" field with a
    message about API rate limits.

    **Recovery**:
      - Wait until next day (daily limit) or next minute (minute limit)
      - Use incremental mode to minimize API calls
      - Consider upgrading to paid tier for higher limits
      - Spread load across multiple providers
    """
    pass


class AlphaVantageDataProvider:
    """
    Data provider for Alpha Vantage API.

    **Conceptual**: This class fetches historical daily OHLCV bars from
    Alpha Vantage and converts them to DataFrames with canonical column names.

    **Data transformation pipeline**:
      1. Validate input parameters (ticker, date range)
      2. Build HTTP GET request to Alpha Vantage API
      3. Fetch JSON response with time series data
      4. Check for error messages or rate limit warnings
      5. Parse nested JSON structure (time series dictionary)
      6. Convert to pandas DataFrame
      7. Rename columns to canonical names (open_price, etc.)
      8. Parse timestamps and convert to UTC
      9. Filter to [start_date, end_date] inclusive window
      10. Validate output (no NaNs, correct columns, etc.)
      11. Return DataFrame (unsorted - IO layer handles that)

    **Alpha Vantage API response format**:
    ```json
    {
      "Meta Data": {
        "1. Information": "Daily Prices (open, high, low, close) and Volumes",
        "2. Symbol": "QQQ",
        "3. Last Refreshed": "2025-11-26",
        "4. Output Size": "Full size",
        "5. Time Zone": "US/Eastern"
      },
      "Time Series (Daily)": {
        "2025-11-26": {
          "1. open": "450.23",
          "2. high": "452.11",
          "3. low": "449.80",
          "4. close": "451.50",
          "5. volume": "5000000"
        },
        "2025-11-25": {
          ...
        }
      }
    }
    ```

    **Column mapping**:
    Alpha Vantage fields → DataFrame columns:
      - Date (dictionary key) → timestamp (converted to pd.Timestamp UTC)
      - "1. open" → open_price
      - "2. high" → high_price
      - "3. low" → low_price
      - "4. close" → closing_price
      - "5. volume" → volume

    **Timezone handling**:
    Alpha Vantage returns dates as "YYYY-MM-DD" strings without time component.
    We convert these to pd.Timestamp with UTC timezone at midnight (00:00:00).
    This matches our canonical format.

    **Example usage**:
        >>> from src.config.settings import get_settings
        >>> settings = get_settings(require_alphavantage=True)
        >>> provider = AlphaVantageDataProvider(settings.alphavantage)
        >>>
        >>> # Fetch QQQ daily bars
        >>> bars = provider.get_daily_bars(
        ...     ticker="QQQ",
        ...     start_date=dt.date(2024, 1, 1),
        ...     end_date=dt.date(2024, 1, 31),
        ... )
        >>> print(bars.shape)  # (21, 6) - 21 trading days in January
        >>> print(bars.columns.tolist())
        ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
    """

    def __init__(self, settings: AlphaVantageSettings):
        """
        Initialize Alpha Vantage data provider.

        **Conceptual**: The provider needs AlphaVantageSettings to configure
        API requests (API key, base URL, function, outputsize).

        Args:
            settings: Alpha Vantage API configuration.

        Raises:
            ValueError: If settings are invalid (caught by settings validation).

        Example:
            >>> from src.config.settings import get_settings
            >>> settings = get_settings(require_alphavantage=True)
            >>> provider = AlphaVantageDataProvider(settings.alphavantage)
        """
        self.settings = settings

        # Validate API key at initialization
        if not self.settings.api_key:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY is not set; cannot use AlphaVantageDataProvider. "
                "Get a free API key at https://www.alphavantage.co/support/#api-key"
            )

    def get_daily_bars(
        self,
        ticker: str,
        start_date: dt.date,
        end_date: dt.date,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars from Alpha Vantage.

        **Conceptual**: This method fetches historical daily bars and returns
        a DataFrame with canonical columns. It does NOT sort or format the
        timestamp - that's the IO layer's responsibility (write_raw_price_csv).

        **Implementation steps**:
          1. Validate inputs (ticker not empty, start_date <= end_date)
          2. Build HTTP GET request to Alpha Vantage API
          3. Make request with timeout (default 30 seconds)
          4. Parse JSON response
          5. Check for error messages or rate limit warnings
          6. Extract "Time Series (Daily)" dictionary
          7. Convert nested dict to list of records
          8. Convert to DataFrame
          9. Rename columns to canonical names
          10. Parse timestamps (date strings → pd.Timestamp UTC)
          11. Filter to [start_date, end_date] inclusive window
          12. Validate output (no NaNs, correct columns, positive prices)
          13. Return DataFrame (unsorted, unformatted)

        **Alpha Vantage quirks**:
          - Returns ALL available data (outputsize=full), ignoring date range
          - We must filter client-side to [start_date, end_date]
          - Dates are in US/Eastern timezone (we convert to UTC)
          - Returns nested dict (not parallel arrays like Finnhub)
          - Error messages appear in "Error Message" field
          - Rate limit warnings appear in "Note" field

        **Error handling**:
        This method may raise:
          - ValueError: Invalid inputs (empty ticker, start_date > end_date)
          - AlphaVantageRateLimitError: Rate limit or quota exceeded
          - AlphaVantageApiError: Invalid symbol, malformed request, API error
          - requests.Timeout: Request timeout
          - requests.RequestException: Network errors

        Args:
            ticker: Ticker symbol (e.g., "QQQ", "SPY", "AAPL").
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
            AlphaVantageRateLimitError: If rate limit exceeded.
            AlphaVantageApiError: If API returns error message.
            requests.Timeout: If request times out.
            requests.RequestException: If network error occurs.

        Example:
            >>> provider = AlphaVantageDataProvider(settings)
            >>> bars = provider.get_daily_bars(
            ...     ticker="QQQ",
            ...     start_date=dt.date(2024, 1, 1),
            ...     end_date=dt.date(2024, 1, 31),
            ... )
            >>> print(bars.shape)  # (21, 6)
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

        # Step 2: Build HTTP GET request
        url = self.settings.base_url
        params = {
            "function": self.settings.function,
            "symbol": ticker.strip().upper(),
            "outputsize": self.settings.outputsize,
            "apikey": self.settings.api_key,
        }

        # Step 3: Make HTTP request with timeout
        try:
            response = requests.get(url, params=params, timeout=self.settings.timeout_seconds)
            response.raise_for_status()  # Raise exception for 4xx/5xx status codes
        except requests.exceptions.Timeout:
            raise AlphaVantageApiError(
                f"Request to Alpha Vantage API timed out after {self.settings.timeout_seconds} seconds. "
                f"URL: {url}"
            )
        except requests.exceptions.RequestException as e:
            raise AlphaVantageApiError(
                f"Network error when connecting to Alpha Vantage API: {e}"
            )

        # Step 4: Parse JSON response
        try:
            data = response.json()
        except ValueError as e:
            raise AlphaVantageApiError(
                f"Failed to parse JSON response from Alpha Vantage: {e}. "
                f"Response text: {response.text[:500]}"
            )

        # Lightweight debug for diagnosing empty responses on key tickers
        if ticker.strip().upper() == "QQQ":
            print(f"[ALPHAVANTAGE DEBUG] Top-level keys: {list(data.keys())}")
            if "Note" in data:
                print(f"[ALPHAVANTAGE DEBUG] Note: {data.get('Note')}")
            if "Error Message" in data:
                print(f"[ALPHAVANTAGE DEBUG] Error Message: {data.get('Error Message')}")

        # Step 5: Check for error messages or rate limit warnings
        # Rate limit: Alpha Vantage returns "Note" field with message
        if "Note" in data:
            note_msg = str(data["Note"])
            if "rate limit" in note_msg.lower() or "api call" in note_msg.lower():
                raise AlphaVantageRateLimitError(
                    f"Alpha Vantage rate limit exceeded. Free tier: 25 calls/day, 5 calls/minute. "
                    f"Message: {note_msg}"
                )

        # API error: Alpha Vantage returns "Error Message" field
        if "Error Message" in data:
            error_msg = str(data["Error Message"])
            raise AlphaVantageApiError(
                f"Alpha Vantage API error for ticker '{ticker}': {error_msg}"
            )

        # Step 6: Extract "Time Series (Daily)" dictionary
        # Some responses may vary the exact key name; look for any key containing "Time Series"
        time_series_key = None
        for k in data.keys():
            if "time series" in k.lower():
                time_series_key = k
                break

        if time_series_key is None or time_series_key not in data:
            raise AlphaVantageApiError(
                f"Time series data missing for ticker '{ticker}'. Keys present: {list(data.keys())}"
            )

        time_series = data[time_series_key]

        if ticker.strip().upper() == "QQQ" and isinstance(time_series, dict):
            keys_sorted = sorted(time_series.keys())
            print(f"[ALPHAVANTAGE DEBUG] Time series entries: {len(time_series)}")
            if keys_sorted:
                print(f"[ALPHAVANTAGE DEBUG] Oldest date: {keys_sorted[0]}, Newest date: {keys_sorted[-1]}")

        # If time series is empty, return empty DataFrame
        if not time_series:
            return pd.DataFrame(columns=[
                'timestamp', 'open_price', 'high_price',
                'low_price', 'closing_price', 'volume'
            ])

        # Step 7: Convert nested dict to list of records
        # Alpha Vantage format: {"2025-11-26": {"1. open": "450.23", ...}, ...}
        # We want: [{"date": "2025-11-26", "open": "450.23", ...}, ...]
        records = []
        for date_str, daily_data in time_series.items():
            records.append({
                'date': date_str,
                'open': daily_data.get('1. open'),
                'high': daily_data.get('2. high'),
                'low': daily_data.get('3. low'),
                'close': daily_data.get('4. close'),
                'volume': daily_data.get('5. volume'),
            })

        # Step 8: Convert to DataFrame
        df = pd.DataFrame(records)

        # Step 9: Rename columns to canonical names
        df = df.rename(columns={
            'date': 'timestamp',
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'closing_price',
            # 'volume' stays as 'volume'
        })

        # Step 10: Parse timestamps (date strings → pd.Timestamp UTC)
        # Alpha Vantage returns dates as "YYYY-MM-DD" (no time component)
        # We convert to pd.Timestamp with UTC timezone at midnight
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d', utc=True)
        except Exception as e:
            raise AlphaVantageApiError(
                f"Failed to parse timestamps from Alpha Vantage response: {e}. "
                f"Sample timestamp: {df['timestamp'].iloc[0] if len(df) > 0 else None}"
            )

        # Convert price and volume columns to numeric (they come as strings)
        numeric_cols = ['open_price', 'high_price', 'low_price', 'closing_price', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Step 11: Filter to [start_date, end_date] inclusive window using date component
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        df = df.loc[mask].reset_index(drop=True)

        # If no data in range after filtering, return empty DataFrame
        if df.empty:
            return pd.DataFrame(columns=[
                'timestamp', 'open_price', 'high_price',
                'low_price', 'closing_price', 'volume'
            ])

        # Step 12: Validate output
        # Check for missing values in critical columns
        critical_cols = ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
        for col in critical_cols:
            if col not in df.columns:
                raise AlphaVantageApiError(
                    f"Column '{col}' missing from Alpha Vantage response. "
                    f"Available columns: {list(df.columns)}"
                )
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                raise AlphaVantageApiError(
                    f"Column '{col}' has {nan_count} NaN values. "
                    f"Data quality issue in Alpha Vantage response."
                )

        # Check that prices are positive
        price_cols = ['open_price', 'high_price', 'low_price', 'closing_price']
        for col in price_cols:
            if (df[col] <= 0).any():
                raise AlphaVantageApiError(
                    f"Column '{col}' has non-positive values. "
                    f"Data quality issue."
                )

        # Check that high >= low
        if (df['high_price'] < df['low_price']).any():
            raise AlphaVantageApiError(
                "Found bars where high_price < low_price. Data quality issue."
            )

        # Step 13: Select only canonical columns (drop any extras)
        df = df[critical_cols]

        # Step 14: Return DataFrame (unsorted, unformatted - IO layer handles that)
        return df
