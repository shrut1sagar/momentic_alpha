"""
HTTP client for Massive.com API.

**Conceptual**: This module provides a thin wrapper around HTTP requests to
Massive.com's API. It handles authentication, request construction, error
handling, and response parsing. It does NOT handle business logic like
converting JSON to DataFrames - that's MassiveDataProvider's job.

**Why separate HTTP client from DataProvider?**
  - Single Responsibility Principle: HTTP client knows about HTTP, DataProvider knows about data.
  - Testability: Can mock HTTP responses without testing DataFrame logic.
  - Reusability: Same client can be used by different data providers (daily, intraday, options).
  - Debugging: Easy to log raw HTTP requests/responses without DataFrame noise.

**Design pattern**: This is a "thin client" - minimal logic, just HTTP mechanics.
All business logic (parsing, validation, DataFrame construction) lives in
MassiveDataProvider. This keeps the client simple and focused.

**Teaching note**: In production systems, you often have multiple layers:
  1. HTTP client (this file) - handles requests/responses
  2. Data adapter (MassiveDataProvider) - converts API responses to domain objects
  3. Business logic (strategies) - uses domain objects

This separation makes each layer testable in isolation.
"""

import requests
from typing import Dict, Any, Optional
import pandas as pd
from src.config.settings import MassiveSettings


class MassiveClientError(Exception):
    """
    Base exception for Massive API client errors.

    **Conceptual**: Custom exceptions make error handling more precise.
    Caller can catch MassiveClientError to handle all Massive-related errors,
    or catch specific subclasses for fine-grained handling.

    **Why custom exceptions?**
      - More informative than generic Exception.
      - Can attach context (status code, response body, etc.).
      - Caller can distinguish between different error types.
      - Makes error handling explicit in code.
    """
    pass


class MassiveAuthenticationError(MassiveClientError):
    """
    Raised when API key is invalid or missing.

    **Conceptual**: 401 Unauthorized or 403 Forbidden from API.
    Usually means API key is wrong, expired, or missing.

    **Recovery**: Check MASSIVE_API_KEY in .env, verify it's still valid.
    """
    pass


class MassiveSymbolNotFoundError(MassiveClientError):
    """
    Raised when requested symbol is not found in Massive's database.

    **Conceptual**: 404 Not Found for a specific symbol.
    Symbol may be delisted, invalid ticker, or not supported by Massive.

    **Recovery**: Verify symbol spelling, check if it's a valid ticker.
    """
    pass


class MassiveRateLimitError(MassiveClientError):
    """
    Raised when API rate limit is exceeded.

    **Conceptual**: 429 Too Many Requests.
    You've made too many API calls in a short time window.

    **Recovery**: Add delay between requests, implement backoff/retry logic.
    """
    pass


class MassiveServerError(MassiveClientError):
    """
    Raised when Massive API returns 5xx server error.

    **Conceptual**: API server is having issues (not your fault).

    **Recovery**: Retry with exponential backoff, contact Massive support if persistent.
    """
    pass


class MassiveClient:
    """
    Thin HTTP client for Massive.com API.

    **Conceptual**: This class wraps the Massive API, handling authentication,
    request construction, and error handling. It returns raw JSON responses
    without any interpretation or transformation.

    **Responsibilities**:
      - Construct API request URLs
      - Add authentication headers
      - Make HTTP requests with timeout
      - Handle HTTP errors (401, 404, 429, 500, timeout)
      - Parse JSON responses
      - Raise descriptive exceptions

    **NOT responsible for**:
      - Converting JSON to DataFrames (that's MassiveDataProvider's job)
      - Data validation (also MassiveDataProvider)
      - Business logic (strategies)

    **API endpoint structure** (example):
    Massive.com API is hypothetical for this project. In reality, each vendor
    has their own endpoint structure. Common patterns:
      - REST: GET /v1/bars/daily?symbol=QQQ&start=2024-01-01&end=2024-12-31
      - GraphQL: POST /graphql with query in body
      - WebSocket: Real-time streaming (not used here)

    For this implementation, we'll assume a REST-style API.

    **Example usage**:
        >>> from src.config.settings import MassiveSettings
        >>> settings = MassiveSettings.from_env()
        >>> client = MassiveClient(settings)
        >>>
        >>> # Fetch daily bars for QQQ
        >>> response = client.get_daily_bars(
        ...     symbol="QQQ",
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ... )
        >>> print(type(response))  # dict
        >>> print(response.keys())  # ['bars', 'symbol', 'count', ...]
    """

    def __init__(self, settings: MassiveSettings):
        """
        Initialize Massive HTTP client with settings.

        **Conceptual**: The client needs API credentials and configuration
        (base URL, timeout) to make requests. All settings are provided via
        MassiveSettings (loaded from environment).

        **Why pass settings object instead of individual params?**
          - Cleaner interface (one object vs many params)
          - Easy to extend (add new settings without changing signature)
          - Type-safe (settings validated at construction)

        Args:
            settings: Massive API configuration (base_url, api_key, timeout_seconds).

        Example:
            >>> from src.config.settings import get_settings
            >>> settings = get_settings(require_massive=True)
            >>> client = MassiveClient(settings.massive)
        """
        self.settings = settings
        self.session = requests.Session()

        # Set default headers for all requests
        # Authorization: most APIs use Bearer token or API key in header
        self.session.headers.update({
            "Authorization": f"Bearer {self.settings.api_key}",
            "Accept": "application/json",
            "User-Agent": "momentic_alpha/1.0",
        })

    def get_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars from Massive API.

        **Conceptual**: This method makes an HTTP GET request to Massive's
        daily bars endpoint and returns the raw JSON response. It does NOT
        parse or validate the data - that's the caller's responsibility.

        **HTTP request details**:
          - Method: GET
          - URL: {base_url}/v1/bars/daily
          - Query params: symbol, start, end
          - Headers: Authorization (from session), Accept: application/json
          - Timeout: From settings (default 30 seconds)

        **API response format** (hypothetical):
        We assume Massive returns JSON like:
            {
                "symbol": "QQQ",
                "bars": [
                    {"date": "2024-01-31", "open": 450.0, "high": 452.0, ...},
                    {"date": "2024-01-30", "open": 448.0, "high": 450.0, ...},
                    ...
                ],
                "count": 22
            }

        The exact format varies by vendor. This implementation is a placeholder
        that will need adjustment based on actual Massive API documentation.

        **Error handling**:
        Raises specific exceptions for different error types:
          - MassiveAuthenticationError: 401/403 (bad API key)
          - MassiveSymbolNotFoundError: 404 (symbol not found)
          - MassiveRateLimitError: 429 (rate limit exceeded)
          - MassiveServerError: 5xx (server error)
          - requests.Timeout: Request took too long
          - MassiveClientError: Other errors

        Args:
            symbol: Ticker symbol (e.g., "QQQ", "SPY").
            start_date: Start date in ISO format (YYYY-MM-DD).
            end_date: End date in ISO format (YYYY-MM-DD).

        Returns:
            Raw JSON response as dict. Structure depends on Massive API format.

        Raises:
            MassiveAuthenticationError: If API key is invalid.
            MassiveSymbolNotFoundError: If symbol not found.
            MassiveRateLimitError: If rate limit exceeded.
            MassiveServerError: If server returns 5xx error.
            requests.Timeout: If request exceeds timeout.
            MassiveClientError: For other API errors.
            ValueError: If parameters are invalid (empty symbol, etc.).

        Example:
            >>> client = MassiveClient(settings)
            >>> try:
            ...     response = client.get_daily_bars("QQQ", "2024-01-01", "2024-01-31")
            ...     print(f"Got {response['count']} bars for {response['symbol']}")
            ... except MassiveAuthenticationError:
            ...     print("Check your API key!")
            ... except MassiveSymbolNotFoundError:
            ...     print("Symbol not found")
        """
        # Validate inputs
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")

        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required")

        # Construct request URL and params (Massive v2 aggs endpoint)
        url = (
            f"{self.settings.base_url}"
            f"/v2/aggs/ticker/{symbol.strip().upper()}"
            f"/range/1/day/{start_date}/{end_date}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",  # oldest -> newest
            "limit": 50000,
        }

        try:
            # Make HTTP request with timeout
            response = self.session.get(
                url,
                params=params,
                timeout=self.settings.timeout_seconds,
            )

            # Handle HTTP errors
            if response.status_code == 401 or response.status_code == 403:
                raise MassiveAuthenticationError(
                    f"Authentication failed (status {response.status_code}). "
                    f"Check your MASSIVE_API_KEY. Response: {response.text}"
                )

            if response.status_code == 404:
                raise MassiveSymbolNotFoundError(
                    f"Symbol '{symbol}' not found. Response: {response.text}"
                )

            if response.status_code == 429:
                raise MassiveRateLimitError(
                    f"Rate limit exceeded. Slow down requests. Response: {response.text}"
                )

            if response.status_code >= 500:
                raise MassiveServerError(
                    f"Massive API server error (status {response.status_code}). "
                    f"Response: {response.text}"
                )

            # For other 4xx errors
            if 400 <= response.status_code < 500:
                raise MassiveClientError(
                    f"Client error (status {response.status_code}). "
                    f"Request may be malformed. Response: {response.text}"
                )

            # Raise for any other non-2xx status
            response.raise_for_status()

            # Parse JSON response
            try:
                data = response.json()
            except ValueError as e:
                raise MassiveClientError(
                    f"Failed to parse JSON response: {e}. Response: {response.text}"
                )

            results = data.get("results")
            if results is None:
                raise MassiveClientError(
                    f"Response missing 'results' field. Keys: {list(data.keys())}"
                )

            if not isinstance(results, list):
                raise MassiveClientError(
                    f"Expected 'results' to be a list, got {type(results)}"
                )

            if len(results) == 0:
                return pd.DataFrame(
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "vwap",
                        "trade_count",
                    ]
                )

            try:
                df = pd.DataFrame(results)
            except Exception as e:
                raise MassiveClientError(
                    f"Failed to convert results to DataFrame: {e}"
                )

            required_fields = ["t", "o", "h", "l", "c", "v"]
            missing = [f for f in required_fields if f not in df.columns]
            if missing:
                raise MassiveClientError(
                    f"Missing required fields in response: {missing}. "
                    f"Available: {list(df.columns)}"
                )

            # Map Massive short fields to clearer names
            rename_map = {
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "trade_count",
            }
            df = df.rename(columns=rename_map)

            # Convert timestamp from ms to UTC datetime
            try:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], unit="ms", utc=True
                )
            except Exception as e:
                raise MassiveClientError(
                    f"Failed to parse timestamps: {e}. "
                    f"Sample: {df['timestamp'].iloc[0] if len(df) else None}"
                )

            # Sort ascending by timestamp (Massive returns asc when sort=asc)
            df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)

            # Ensure column order
            cols = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            if "vwap" in df.columns:
                cols.append("vwap")
            if "trade_count" in df.columns:
                cols.append("trade_count")
            df = df[cols]

            return df

        except requests.Timeout as e:
            raise requests.Timeout(
                f"Request to Massive API timed out after {self.settings.timeout_seconds}s. "
                f"Check network connection or increase timeout."
            ) from e

        except requests.ConnectionError as e:
            raise MassiveClientError(
                f"Failed to connect to Massive API at {self.settings.base_url}. "
                f"Check network connection and base URL."
            ) from e

        except requests.RequestException as e:
            # Catch-all for other requests errors
            raise MassiveClientError(
                f"HTTP request failed: {e}"
            ) from e

    def close(self):
        """
        Close the HTTP session and release resources.

        **Conceptual**: HTTP sessions maintain connection pools for performance.
        When done using the client, close the session to free resources.

        **When to call this**:
          - At end of long-running script
          - In finally block or context manager
          - Not needed for short scripts (resources auto-released on exit)

        **Better pattern**: Use context manager (with statement):
            >>> with MassiveClient(settings) as client:
            ...     bars = client.get_daily_bars("QQQ", "2024-01-01", "2024-12-31")
            >>> # Session automatically closed after 'with' block

        Example:
            >>> client = MassiveClient(settings)
            >>> try:
            ...     bars = client.get_daily_bars("QQQ", "2024-01-01", "2024-12-31")
            ... finally:
            ...     client.close()
        """
        self.session.close()

    def __enter__(self):
        """Enable context manager support (with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up session when exiting context manager."""
        self.close()
        return False  # Don't suppress exceptions
