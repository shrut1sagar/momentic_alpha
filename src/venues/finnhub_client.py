"""
HTTP client for Finnhub API.

**Conceptual**: This module provides a thin wrapper around HTTP requests to
Finnhub's REST API. It handles authentication, request construction, error
handling, and response parsing. It does NOT handle business logic like
converting JSON to DataFrames - that's FinnhubDataProvider's job.

**Why separate HTTP client from DataProvider?**
  - Single Responsibility Principle: HTTP client knows about HTTP, DataProvider knows about data.
  - Testability: Can mock HTTP responses without testing DataFrame logic.
  - Reusability: Same client can be used by different data providers (daily, intraday, quote).
  - Debugging: Easy to log raw HTTP requests/responses without DataFrame noise.

**Design pattern**: This is a "thin client" - minimal logic, just HTTP mechanics.
All business logic (parsing, validation, DataFrame construction) lives in
FinnhubDataProvider. This keeps the client simple and focused.

**Finnhub API structure**:
  - Authentication: Query parameter `token=YOUR_API_KEY`
  - Endpoint for candles: /stock/candle
  - Parameters: symbol, resolution, from (unix timestamp), to (unix timestamp)
  - Response format: {"c": [closes], "h": [highs], "l": [lows], "o": [opens],
                      "t": [timestamps], "v": [volumes], "s": "ok"|"no_data"}

**Teaching note**: In production systems, you often have multiple layers:
  1. HTTP client (this file) - handles requests/responses
  2. Data adapter (FinnhubDataProvider) - converts API responses to domain objects
  3. Business logic (strategies) - uses domain objects

This separation makes each layer testable in isolation.
"""

import requests
import time
from typing import Dict, Any
from src.config.settings import FinnhubSettings


class FinnhubClientError(Exception):
    """
    Base exception for Finnhub API client errors.

    **Conceptual**: Custom exceptions make error handling more precise.
    Caller can catch FinnhubClientError to handle all Finnhub-related errors,
    or catch specific subclasses for fine-grained handling.

    **Why custom exceptions?**
      - More informative than generic Exception.
      - Can attach context (status code, response body, etc.).
      - Caller can distinguish between different error types.
      - Makes error handling explicit in code.
    """
    pass


class FinnhubAuthenticationError(FinnhubClientError):
    """
    Raised when API key is invalid or missing.

    **Conceptual**: 401 Unauthorized or 403 Forbidden from API.
    Usually means API key is wrong, expired, or missing.

    **Recovery**: Check FINNHUB_API_KEY in .env, verify it's still valid.
    Get a free API key at https://finnhub.io/register
    """
    pass


class FinnhubSymbolNotFoundError(FinnhubClientError):
    """
    Raised when requested symbol is not found or has no data.

    **Conceptual**: Finnhub returns status="no_data" when symbol is invalid
    or has no candle data for the requested time range.

    **Recovery**: Verify symbol spelling, check if it's a valid ticker,
    ensure the date range has trading days.
    """
    pass


class FinnhubRateLimitError(FinnhubClientError):
    """
    Raised when API rate limit is exceeded.

    **Conceptual**: 429 Too Many Requests.
    Free tier: 60 API calls/minute. Paid tiers: higher limits.

    **Recovery**: Add delay between requests (min_sleep_seconds in settings),
    implement exponential backoff, or upgrade to paid tier.
    """
    pass


class FinnhubServerError(FinnhubClientError):
    """
    Raised when Finnhub API returns 5xx server error.

    **Conceptual**: API server is having issues (not your fault).

    **Recovery**: Retry with exponential backoff, contact Finnhub support if persistent.
    """
    pass


class FinnhubClient:
    """
    Thin HTTP client for Finnhub API.

    **Conceptual**: This class wraps the Finnhub API, handling authentication,
    request construction, and error handling. It returns raw JSON responses
    without any interpretation or transformation.

    **Responsibilities**:
      - Construct API request URLs with authentication
      - Add rate limiting (min_sleep_seconds delay)
      - Make HTTP requests with timeout
      - Handle HTTP errors (401, 403, 429, 500, timeout)
      - Parse JSON responses
      - Raise descriptive exceptions

    **NOT responsible for**:
      - Converting JSON to DataFrames (that's FinnhubDataProvider's job)
      - Data validation (also FinnhubDataProvider)
      - Business logic (strategies)

    **Finnhub API documentation**: https://finnhub.io/docs/api/

    **Example usage**:
        >>> from src.config.settings import FinnhubSettings
        >>> settings = FinnhubSettings.from_env()
        >>> client = FinnhubClient(settings)
        >>>
        >>> # Fetch daily candles for AAPL
        >>> response = client.get_candles(
        ...     symbol="AAPL",
        ...     resolution="D",  # Daily candles
        ...     from_timestamp=1609459200,  # 2021-01-01 00:00:00 UTC
        ...     to_timestamp=1640995200,    # 2022-01-01 00:00:00 UTC
        ... )
        >>> print(type(response))  # dict
        >>> print(response.keys())  # ['c', 'h', 'l', 'o', 't', 'v', 's']
    """

    def __init__(self, settings: FinnhubSettings):
        """
        Initialize Finnhub HTTP client with settings.

        **Conceptual**: The client needs API credentials and configuration
        (base URL, timeout, rate limit delay) to make requests. All settings
        are provided via FinnhubSettings (loaded from environment).

        **Why pass settings object instead of individual params?**
          - Future-proof: Easy to add new settings without changing signature.
          - Type safety: Settings is a validated dataclass.
          - Testability: Easy to create fake settings for tests.

        Args:
            settings: Finnhub configuration (API key, base URL, timeout, etc.).

        Raises:
            ValueError: If settings are invalid (caught by FinnhubSettings validation).
        """
        self.settings = settings
        self.session = requests.Session()
        self._last_request_time: float = 0.0

    def _wait_for_rate_limit(self) -> None:
        """
        Enforce minimum delay between API requests.

        **Conceptual**: Finnhub free tier has rate limits (60 calls/minute).
        To avoid hitting limits, we enforce a minimum delay between requests.

        **Why sleep here instead of after request?**
          - Prevents back-to-back requests if client makes multiple calls quickly.
          - More predictable timing (no race conditions).

        **Implementation**: Track time of last request, sleep if needed to enforce
        min_sleep_seconds delay.
        """
        if self.settings.min_sleep_seconds > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.settings.min_sleep_seconds:
                time.sleep(self.settings.min_sleep_seconds - elapsed)

        self._last_request_time = time.time()

    def get_candles(
        self,
        symbol: str,
        resolution: str,
        from_timestamp: int,
        to_timestamp: int,
    ) -> Dict[str, Any]:
        """
        Fetch candle/OHLCV data for a symbol.

        **Conceptual**: This method calls Finnhub's /stock/candle endpoint to
        fetch historical OHLCV (open/high/low/close/volume) bars for a stock.

        **Finnhub candle API**:
          - Endpoint: GET /stock/candle
          - Parameters:
            - symbol: Stock ticker (e.g., "AAPL", "TSLA")
            - resolution: Candle resolution ("1", "5", "15", "30", "60", "D", "W", "M")
                         For daily data, use "D"
            - from: Start time (Unix timestamp in seconds)
            - to: End time (Unix timestamp in seconds)
            - token: API key (added automatically from settings)
          - Response:
            {
              "c": [close_prices],  # Closing prices
              "h": [high_prices],   # High prices
              "l": [low_prices],    # Low prices
              "o": [open_prices],   # Opening prices
              "t": [timestamps],    # Unix timestamps
              "v": [volumes],       # Volumes
              "s": "ok"|"no_data"   # Status
            }

        **Why Unix timestamps?**
          - Finnhub API uses Unix timestamps (seconds since 1970-01-01 00:00:00 UTC).
          - No timezone ambiguity (always UTC).
          - Easy to compute (use datetime.timestamp()).

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "QQQ").
            resolution: Candle resolution. Use "D" for daily candles.
            from_timestamp: Start time as Unix timestamp (seconds since epoch).
            to_timestamp: End time as Unix timestamp (seconds since epoch).

        Returns:
            Dict containing candle data with keys: c, h, l, o, t, v, s.

        Raises:
            FinnhubAuthenticationError: If API key is invalid.
            FinnhubSymbolNotFoundError: If symbol is not found or has no data.
            FinnhubRateLimitError: If rate limit exceeded.
            FinnhubServerError: If API returns 5xx error.
            FinnhubClientError: For other API errors.

        Example:
            >>> import pandas as pd
            >>> client = FinnhubClient(settings)
            >>> # Fetch AAPL daily data for 2024
            >>> start = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp())
            >>> end = int(pd.Timestamp("2024-12-31", tz="UTC").timestamp())
            >>> candles = client.get_candles("AAPL", "D", start, end)
            >>> print(candles['s'])  # "ok" if successful
            >>> print(len(candles['c']))  # Number of candles
        """
        # Enforce rate limit delay
        self._wait_for_rate_limit()

        # Construct request URL
        url = f"{self.settings.base_url}/stock/candle"

        # Build query parameters
        params = {
            "symbol": symbol.upper(),  # Normalize to uppercase
            "resolution": resolution,
            "from": from_timestamp,
            "to": to_timestamp,
            "token": self.settings.api_key,  # Finnhub uses query param for auth
        }

        # Make HTTP request
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.settings.timeout_seconds,
            )
        except requests.exceptions.Timeout:
            raise FinnhubClientError(
                f"Request to Finnhub API timed out after {self.settings.timeout_seconds}s. "
                f"URL: {url}"
            )
        except requests.exceptions.ConnectionError as e:
            raise FinnhubClientError(
                f"Connection error when connecting to Finnhub API: {e}"
            )
        except requests.exceptions.RequestException as e:
            raise FinnhubClientError(
                f"Unexpected error during Finnhub API request: {e}"
            )

        # Handle HTTP error status codes
        if response.status_code == 401 or response.status_code == 403:
            raise FinnhubAuthenticationError(
                f"Authentication failed (status {response.status_code}). "
                f"Check that FINNHUB_API_KEY is set correctly. "
                f"Get a free API key at https://finnhub.io/register"
            )
        elif response.status_code == 429:
            raise FinnhubRateLimitError(
                f"Rate limit exceeded (status 429). "
                f"Free tier: 60 API calls/minute. "
                f"Consider increasing min_sleep_seconds or upgrading to paid tier."
            )
        elif response.status_code >= 500:
            raise FinnhubServerError(
                f"Finnhub server error (status {response.status_code}). "
                f"This is not your fault. Retry later or contact support."
            )
        elif response.status_code >= 400:
            raise FinnhubClientError(
                f"Finnhub API error (status {response.status_code}). "
                f"Response: {response.text}"
            )

        # Parse JSON response
        try:
            data = response.json()
        except ValueError as e:
            raise FinnhubClientError(
                f"Failed to parse JSON response from Finnhub: {e}. "
                f"Response text: {response.text}"
            )

        # Check status field in response
        status = data.get('s', '')
        if status == 'no_data':
            raise FinnhubSymbolNotFoundError(
                f"No data available for symbol '{symbol}' in requested date range. "
                f"Verify symbol spelling and date range."
            )
        elif status != 'ok':
            raise FinnhubClientError(
                f"Unexpected status from Finnhub API: '{status}'. "
                f"Response: {data}"
            )

        return data

    def close(self) -> None:
        """
        Close the HTTP session.

        **Conceptual**: Cleanup method to close the underlying requests.Session.
        Good practice to call this when done with the client to free resources.

        **When to call**:
          - After batch fetching multiple symbols.
          - In __del__ for automatic cleanup.
          - When used as context manager (__exit__).
        """
        self.session.close()

    def __enter__(self):
        """Support for context manager (with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager (with statement)."""
        self.close()
        return False  # Don't suppress exceptions
