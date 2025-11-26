"""
Configuration settings for the trading system.

**Conceptual**: This module provides strongly-typed configuration objects that
load from environment variables (via .env files). All settings are validated
at startup, ensuring fail-fast behavior if configuration is missing or invalid.

**Why centralized config?**
  - Single source of truth for all settings (paths, API keys, etc.).
  - Easy to test (inject fake settings instead of reading from environment).
  - Fail-fast validation (missing API key → clear error at startup, not mid-run).
  - Secrets management (API keys loaded from .env, never hardcoded).

**Teaching note**: In production systems, configuration is critical infrastructure:
  - Dev/staging/prod environments have different configs (different API URLs, keys).
  - Secrets should never be committed to git (.env in .gitignore).
  - Config errors should be caught early (at startup, not when API call fails).
  - Strongly-typed config (dataclasses) prevents typos and provides IDE autocomplete.

This module uses python-dotenv to load .env files and dataclasses for type safety.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Try to load .env file if present (dev/local environments)
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed; assume environment variables are set externally
    pass


@dataclass(frozen=True)
class MassiveSettings:
    """
    Configuration for Massive.com data provider.

    **Conceptual**: Massive.com is a data vendor providing historical price data
    for stocks, ETFs, etc. This settings object stores the API credentials and
    base URL needed to authenticate and fetch data.

    **Why separate settings class?**
      - Clear separation of concerns (Massive settings vs other vendors).
      - Easy to add other data providers (Alpaca, Polygon, etc.) without mixing configs.
      - Testable (inject fake settings in tests, don't hit real API).

    **Security note**: API keys are secrets and should:
      - Be loaded from environment variables (MASSIVE_API_KEY).
      - Never be hardcoded in source code.
      - Never be committed to git (use .env file in .gitignore).
      - Be rotated regularly in production.

    Attributes:
        base_url: Base URL for Massive API (e.g., "https://api.massive.com").
                 Defaults to standard production URL if not set.
        api_key: Massive API key for authentication.
                REQUIRED - raises ValueError if not provided.
        timeout_seconds: HTTP request timeout in seconds (default 30).
                        Prevents hanging on slow/failed API calls.
    """
    base_url: str
    api_key: str
    timeout_seconds: int = 30

    def __post_init__(self):
        """Validate settings after initialization."""
        if not self.api_key:
            raise ValueError(
                "MASSIVE_API_KEY is required but not set. "
                "Please set it in your .env file or environment variables."
            )
        if not self.base_url:
            raise ValueError(
                "MASSIVE_BASE_URL is required but not set. "
                "Please set it in your .env file or environment variables."
            )

    @classmethod
    def from_env(cls) -> "MassiveSettings":
        """
        Load Massive settings from environment variables.

        **Conceptual**: This factory method reads environment variables and
        constructs a validated MassiveSettings object. It's the standard way
        to initialize settings in production and development.

        **Environment variables**:
          - MASSIVE_API_KEY (required): Your Massive API key.
          - MASSIVE_BASE_URL (optional): Base URL for Massive API.
            Defaults to "https://api.massive.com" if not set.
          - MASSIVE_TIMEOUT_SECONDS (optional): HTTP timeout in seconds.
            Defaults to 30 if not set.

        Returns:
            MassiveSettings object with values loaded from environment.

        Raises:
            ValueError: If MASSIVE_API_KEY is missing or empty.

        Usage example:
            >>> # In .env file:
            >>> # MASSIVE_API_KEY=your_key_here
            >>> # MASSIVE_BASE_URL=https://api.massive.com
            >>>
            >>> settings = MassiveSettings.from_env()
            >>> print(settings.api_key)  # "your_key_here"
        """
        api_key = os.getenv("MASSIVE_API_KEY", "")
        base_url = os.getenv("MASSIVE_BASE_URL", "https://api.massive.com")
        timeout_str = os.getenv("MASSIVE_TIMEOUT_SECONDS", "30")

        try:
            timeout_seconds = int(timeout_str)
        except ValueError:
            raise ValueError(
                f"MASSIVE_TIMEOUT_SECONDS must be an integer, got: {timeout_str}"
            )

        return cls(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class FinnhubSettings:
    """
    Configuration for Finnhub data provider.

    **Conceptual**: Finnhub is a real-time financial data provider offering stock
    prices, forex, crypto, and economic data via their REST API. This settings
    object stores the API key and configuration needed to fetch historical candle data.

    **Why separate Finnhub settings?**
      - Allows using multiple data providers simultaneously (Massive for some symbols, Finnhub for others).
      - Each provider has different API structure, rate limits, and requirements.
      - Easy to test with mock settings.

    **Security note**: API keys are secrets and should:
      - Be loaded from environment variables (FINNHUB_API_KEY).
      - Never be hardcoded in source code.
      - Never be committed to git (use .env file in .gitignore).

    **Rate limiting**: Finnhub free tier has limits (typically 60 API calls/minute).
    The min_sleep_seconds parameter adds a small delay between requests to avoid
    hitting rate limits. For paid tiers, this can be set to 0.

    Attributes:
        api_key: Finnhub API token for authentication.
                REQUIRED - raises ValueError if not provided.
        base_url: Base URL for Finnhub API (default: https://finnhub.io/api/v1).
        min_sleep_seconds: Minimum delay between API requests to avoid rate limits.
                          Default 0.2 seconds (3 requests/second, well under free tier limit).
        timeout_seconds: HTTP request timeout in seconds (default 30).
    """
    api_key: str
    base_url: str = "https://finnhub.io/api/v1"
    min_sleep_seconds: float = 0.2
    timeout_seconds: int = 30

    def __post_init__(self):
        """Validate settings after initialization."""
        if not self.api_key:
            raise ValueError(
                "FINNHUB_API_KEY is required but not set. "
                "Please set it in your .env file or environment variables. "
                "Get a free API key at https://finnhub.io/register"
            )
        if self.min_sleep_seconds < 0:
            raise ValueError(
                f"min_sleep_seconds must be non-negative, got: {self.min_sleep_seconds}"
            )

    @classmethod
    def from_env(cls) -> "FinnhubSettings":
        """
        Load Finnhub settings from environment variables.

        **Conceptual**: This factory method reads environment variables and
        constructs a validated FinnhubSettings object. It's the standard way
        to initialize settings in production and development.

        **Environment variables**:
          - FINNHUB_API_KEY (required): Your Finnhub API token.
            Get a free key at https://finnhub.io/register
          - FINNHUB_BASE_URL (optional): Base URL for Finnhub API.
            Defaults to "https://finnhub.io/api/v1" if not set.
          - FINNHUB_MIN_SLEEP_SECONDS (optional): Delay between requests in seconds.
            Defaults to 0.2 (3 requests/second) if not set.
          - FINNHUB_TIMEOUT_SECONDS (optional): HTTP timeout in seconds.
            Defaults to 30 if not set.

        Returns:
            FinnhubSettings object with values loaded from environment.

        Raises:
            ValueError: If FINNHUB_API_KEY is missing or empty.

        Usage example:
            >>> # In .env file:
            >>> # FINNHUB_API_KEY=your_token_here
            >>>
            >>> settings = FinnhubSettings.from_env()
            >>> print(settings.base_url)  # "https://finnhub.io/api/v1"
        """
        api_key = os.getenv("FINNHUB_API_KEY", "")
        base_url = os.getenv("FINNHUB_BASE_URL", "https://finnhub.io/api/v1")
        min_sleep_str = os.getenv("FINNHUB_MIN_SLEEP_SECONDS", "0.2")
        timeout_str = os.getenv("FINNHUB_TIMEOUT_SECONDS", "30")

        try:
            min_sleep_seconds = float(min_sleep_str)
        except ValueError:
            raise ValueError(
                f"FINNHUB_MIN_SLEEP_SECONDS must be a number, got: {min_sleep_str}"
            )

        try:
            timeout_seconds = int(timeout_str)
        except ValueError:
            raise ValueError(
                f"FINNHUB_TIMEOUT_SECONDS must be an integer, got: {timeout_str}"
            )

        return cls(
            api_key=api_key,
            base_url=base_url,
            min_sleep_seconds=min_sleep_seconds,
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class AlphaVantageSettings:
    """
    Configuration for Alpha Vantage data provider.

    **Conceptual**: Alpha Vantage is a free/freemium financial data provider offering
    stock prices, forex, crypto, and technical indicators via their REST API. This
    settings object stores the API key and configuration needed to fetch historical
    daily candle data.

    **Why separate Alpha Vantage settings?**
      - Allows using multiple data providers simultaneously (Massive for some symbols, Alpha Vantage for others).
      - Each provider has different API structure, rate limits, and requirements.
      - Easy to test with mock settings.

    **Security note**: API keys are secrets and should:
      - Be loaded from environment variables (ALPHAVANTAGE_API_KEY).
      - Never be hardcoded in source code.
      - Never be committed to git (use .env file in .gitignore).

    **Rate limiting**: Alpha Vantage free tier has strict limits:
      - 25 API calls per day (!)
      - 5 API calls per minute
      Be very conservative with fetching data. Consider using incremental mode
      and caching results to avoid hitting limits.

    Attributes:
        api_key: Alpha Vantage API token for authentication.
                REQUIRED - raises ValueError if not provided.
        base_url: Base URL for Alpha Vantage API (default: https://www.alphavantage.co/query).
        function: API function to call (default: TIME_SERIES_DAILY).
        outputsize: Output size - "compact" (last 100 days) or "full" (all available).
                   Default: "full" for maximum historical data.
        min_sleep_seconds: Minimum delay between API requests to avoid rate limits.
                          Default 0.0 seconds (no enforced delay - script can add delays).
        timeout_seconds: HTTP request timeout in seconds (default 30).
    """
    api_key: str
    base_url: str = "https://www.alphavantage.co/query"
    function: str = "TIME_SERIES_DAILY"
    outputsize: str = "full"
    min_sleep_seconds: float = 0.0
    timeout_seconds: int = 30

    def __post_init__(self):
        """Validate settings after initialization."""
        if not self.api_key:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY is required but not set. "
                "Please set it in your .env file or environment variables. "
                "Get a free API key at https://www.alphavantage.co/support/#api-key"
            )

    @classmethod
    def from_env(cls) -> "AlphaVantageSettings":
        """
        Load Alpha Vantage settings from environment variables.

        **Conceptual**: This factory method reads environment variables and
        constructs a validated AlphaVantageSettings object. It's the standard way
        to initialize settings in production and development.

        **Environment variables**:
          - ALPHAVANTAGE_API_KEY (required): Your Alpha Vantage API token.
            Get a free key at https://www.alphavantage.co/support/#api-key
          - ALPHAVANTAGE_BASE_URL (optional): Base URL for Alpha Vantage API.
            Defaults to "https://www.alphavantage.co/query" if not set.
          - ALPHAVANTAGE_FUNCTION (optional): API function to call.
            Defaults to "TIME_SERIES_DAILY" if not set.
          - ALPHAVANTAGE_OUTPUTSIZE (optional): "compact" or "full".
            Defaults to "full" if not set.
          - ALPHAVANTAGE_MIN_SLEEP_SECONDS (optional): Delay between requests in seconds.
            Defaults to 0.0 if not set.
          - ALPHAVANTAGE_TIMEOUT_SECONDS (optional): HTTP timeout in seconds.
            Defaults to 30 if not set.

        Returns:
            AlphaVantageSettings object with values loaded from environment.

        Raises:
            ValueError: If ALPHAVANTAGE_API_KEY is missing or empty.

        Usage example:
            >>> # In .env file:
            >>> # ALPHAVANTAGE_API_KEY=your_token_here
            >>>
            >>> settings = AlphaVantageSettings.from_env()
            >>> print(settings.base_url)  # "https://www.alphavantage.co/query"
        """
        api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
        base_url = os.getenv("ALPHAVANTAGE_BASE_URL", "https://www.alphavantage.co/query")
        function = os.getenv("ALPHAVANTAGE_FUNCTION", "TIME_SERIES_DAILY")
        outputsize = os.getenv("ALPHAVANTAGE_OUTPUTSIZE", "full")
        min_sleep_str = os.getenv("ALPHAVANTAGE_MIN_SLEEP_SECONDS", "0.0")
        timeout_str = os.getenv("ALPHAVANTAGE_TIMEOUT_SECONDS", "30")

        try:
            min_sleep_seconds = float(min_sleep_str)
        except ValueError:
            raise ValueError(
                f"ALPHAVANTAGE_MIN_SLEEP_SECONDS must be a number, got: {min_sleep_str}"
            )

        try:
            timeout_seconds = int(timeout_str)
        except ValueError:
            raise ValueError(
                f"ALPHAVANTAGE_TIMEOUT_SECONDS must be an integer, got: {timeout_str}"
            )

        return cls(
            api_key=api_key,
            base_url=base_url,
            function=function,
            outputsize=outputsize,
            min_sleep_seconds=min_sleep_seconds,
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class YFinanceSettings:
    """
    Configuration for YFinance data provider.

    **Conceptual**: Yahoo Finance via yfinance is a free data source that doesn't
    require an API key. This settings object stores configuration for how to fetch
    and process data from Yahoo Finance.

    **Why separate YFinance settings?**
      - Allows customizing fetch behavior (adjustments, intervals, threading)
      - Easy to test with different configurations
      - Consistent with other provider patterns

    **No API Key**: Unlike Finnhub or Alpha Vantage, yfinance doesn't require
    authentication. It scrapes Yahoo Finance web pages.

    Attributes:
        interval: Data interval (default: "1d" for daily bars).
                 Other options: "1m", "5m", "15m", "30m", "60m", "1h", "1wk", "1mo"
        auto_adjust: Automatically adjust OHLC for splits/dividends (default: False).
                    If True, returns adjusted prices without separate Adj Close column.
        back_adjust: Back-adjust prices (default: False).
                    Only used if auto_adjust=False.
        prepost: Include pre/post market data (default: False).
        threads: Enable multi-threading for faster downloads (default: True).
        proxy: Optional HTTP proxy URL for requests (default: None).
               Format: "http://user:pass@host:port" or "http://host:port"
    """
    interval: str = "1d"
    auto_adjust: bool = False
    back_adjust: bool = False
    prepost: bool = False
    threads: bool = True
    proxy: Optional[str] = None

    @classmethod
    def from_env(cls) -> "YFinanceSettings":
        """
        Load YFinance settings from environment variables.

        **Conceptual**: This factory method reads environment variables and
        constructs a YFinanceSettings object. Since yfinance doesn't require
        an API key, this mainly loads optional configuration.

        **Environment variables**:
          - YFINANCE_INTERVAL (optional): Data interval (default: "1d").
          - YFINANCE_AUTO_ADJUST (optional): Auto-adjust prices (default: "false").
          - YFINANCE_BACK_ADJUST (optional): Back-adjust prices (default: "false").
          - YFINANCE_PREPOST (optional): Include pre/post market (default: "false").
          - YFINANCE_THREADS (optional): Enable threading (default: "true").
          - YFINANCE_PROXY (optional): HTTP proxy URL (default: None).

        Returns:
            YFinanceSettings object with values loaded from environment.

        Usage example:
            >>> # In .env file (all optional):
            >>> # YFINANCE_INTERVAL=1d
            >>> # YFINANCE_AUTO_ADJUST=false
            >>> # YFINANCE_PROXY=http://proxy.example.com:8080
            >>>
            >>> settings = YFinanceSettings.from_env()
            >>> print(settings.interval)  # "1d"
        """
        interval = os.getenv("YFINANCE_INTERVAL", "1d")
        auto_adjust = os.getenv("YFINANCE_AUTO_ADJUST", "false").lower() in ("true", "1", "yes")
        back_adjust = os.getenv("YFINANCE_BACK_ADJUST", "false").lower() in ("true", "1", "yes")
        prepost = os.getenv("YFINANCE_PREPOST", "false").lower() in ("true", "1", "yes")
        threads = os.getenv("YFINANCE_THREADS", "true").lower() in ("true", "1", "yes")
        proxy = os.getenv("YFINANCE_PROXY", None)  # None if not set

        return cls(
            interval=interval,
            auto_adjust=auto_adjust,
            back_adjust=back_adjust,
            prepost=prepost,
            threads=threads,
            proxy=proxy,
        )


@dataclass(frozen=True)
class Settings:
    """
    Global settings for the trading system.

    **Conceptual**: This is the top-level settings object that aggregates all
    subsystem settings (data providers, paths, etc.). It provides a single
    entrypoint for accessing configuration throughout the application.

    **Why top-level settings object?**
      - Single import point for all config (from src.config import settings).
      - Easy to extend with new subsystems (add broker settings, logging config, etc.).
      - Testable (inject fake Settings object in tests).

    **Usage pattern**:
      ```python
      from src.config.settings import Settings

      # Load settings from environment
      settings = Settings.from_env()

      # Access subsystem settings
      massive_settings = settings.massive
      ```

    Attributes:
        massive: Massive.com data provider settings.
                Can be None if Massive is not configured (optional dependency).
        finnhub: Finnhub data provider settings.
                Can be None if Finnhub is not configured (optional dependency).
        alphavantage: Alpha Vantage data provider settings.
                     Can be None if Alpha Vantage is not configured (optional dependency).
        yfinance: Yahoo Finance data provider settings.
                 Always available (no API key required).
    """
    massive: Optional[MassiveSettings] = None
    finnhub: Optional[FinnhubSettings] = None
    alphavantage: Optional[AlphaVantageSettings] = None
    yfinance: Optional[YFinanceSettings] = None

    @classmethod
    def from_env(cls, require_massive: bool = False, require_finnhub: bool = False, require_alphavantage: bool = False) -> "Settings":
        """
        Load global settings from environment variables.

        **Conceptual**: This factory method loads all subsystem settings from
        the environment. It's the standard way to initialize settings at
        application startup.

        **Design decision**: Data provider settings are optional by default. This
        allows the system to work without any providers configured (e.g., using
        local CSV files only). Set require_massive=True, require_finnhub=True,
        or require_alphavantage=True to enforce that specific providers are
        configured (useful for data ingestion scripts).

        Args:
            require_massive: If True, raise error if Massive settings are missing.
                           If False (default), Massive settings are optional.
            require_finnhub: If True, raise error if Finnhub settings are missing.
                           If False (default), Finnhub settings are optional.
            require_alphavantage: If True, raise error if Alpha Vantage settings are missing.
                                If False (default), Alpha Vantage settings are optional.

        Returns:
            Settings object with all subsystem settings loaded from environment.

        Raises:
            ValueError: If require_massive=True and MASSIVE_API_KEY is missing,
                       or if require_finnhub=True and FINNHUB_API_KEY is missing,
                       or if require_alphavantage=True and ALPHAVANTAGE_API_KEY is missing.

        Usage example:
            >>> # Load settings (all providers optional)
            >>> settings = Settings.from_env()
            >>>
            >>> # Load settings (Massive required)
            >>> settings = Settings.from_env(require_massive=True)
            >>>
            >>> # Load settings (Finnhub required)
            >>> settings = Settings.from_env(require_finnhub=True)
            >>>
            >>> # Load settings (Alpha Vantage required)
            >>> settings = Settings.from_env(require_alphavantage=True)
        """
        # Try to load Massive settings
        massive_settings = None
        try:
            massive_settings = MassiveSettings.from_env()
        except ValueError as e:
            if require_massive:
                raise ValueError(
                    f"Massive settings are required but could not be loaded: {e}"
                )
            # Otherwise, Massive is optional - continue without it

        # Try to load Finnhub settings
        finnhub_settings = None
        try:
            finnhub_settings = FinnhubSettings.from_env()
        except ValueError as e:
            if require_finnhub:
                raise ValueError(
                    f"Finnhub settings are required but could not be loaded: {e}"
                )
            # Otherwise, Finnhub is optional - continue without it

        # Try to load Alpha Vantage settings
        alphavantage_settings = None
        try:
            alphavantage_settings = AlphaVantageSettings.from_env()
        except ValueError as e:
            if require_alphavantage:
                raise ValueError(
                    f"Alpha Vantage settings are required but could not be loaded: {e}"
                )
            # Otherwise, Alpha Vantage is optional - continue without it

        # Load YFinance settings (always succeeds - no API key required)
        yfinance_settings = YFinanceSettings.from_env()

        return cls(
            massive=massive_settings,
            finnhub=finnhub_settings,
            alphavantage=alphavantage_settings,
            yfinance=yfinance_settings,
        )


# Convenience singleton for accessing settings throughout the application
# Note: This loads settings at module import time. For testing, inject custom settings.
# Example: tests can create Settings(massive=fake_massive_settings) instead of using this.
_default_settings: Optional[Settings] = None


def get_settings(require_massive: bool = False, require_finnhub: bool = False, require_alphavantage: bool = False) -> Settings:
    """
    Get the global settings singleton.

    **Conceptual**: This function provides lazy initialization of settings.
    Settings are loaded from environment on first call, then cached for reuse.

    **Why lazy initialization?**
      - Settings only loaded when actually needed (not at module import).
      - Tests can bypass this by creating their own Settings objects.
      - Clear error messages if settings are accessed before configuration.

    **Teaching note**: Singleton pattern is useful for config, but be careful:
      - Makes testing harder (global state).
      - Can hide dependencies (function uses config without declaring it).
      - Alternative: dependency injection (pass settings as argument).
    Here we use singleton for convenience, but allow tests to inject custom settings.

    Args:
        require_massive: If True, raise error if Massive settings are missing.
        require_finnhub: If True, raise error if Finnhub settings are missing.
        require_alphavantage: If True, raise error if Alpha Vantage settings are missing.

    Returns:
        Global Settings singleton.

    Raises:
        ValueError: If require_massive=True and Massive not configured,
                   or if require_finnhub=True and Finnhub not configured,
                   or if require_alphavantage=True and Alpha Vantage not configured.

    Usage example:
        >>> from src.config.settings import get_settings
        >>>
        >>> # Get settings (all providers optional)
        >>> settings = get_settings()
        >>> if settings.massive:
        >>>     print("Massive configured")
        >>> if settings.finnhub:
        >>>     print("Finnhub configured")
        >>> if settings.alphavantage:
        >>>     print("Alpha Vantage configured")
        >>>
        >>> # Get settings (Massive required)
        >>> settings = get_settings(require_massive=True)
        >>> print(settings.massive.api_key)
        >>>
        >>> # Get settings (Finnhub required)
        >>> settings = get_settings(require_finnhub=True)
        >>> print(settings.finnhub.api_key)
        >>>
        >>> # Get settings (Alpha Vantage required)
        >>> settings = get_settings(require_alphavantage=True)
        >>> print(settings.alphavantage.api_key)
    """
    global _default_settings

    if _default_settings is None:
        _default_settings = Settings.from_env(
            require_massive=require_massive,
            require_finnhub=require_finnhub,
            require_alphavantage=require_alphavantage,
        )

    # If already loaded but require_massive is True, check that Massive is present
    if require_massive and _default_settings.massive is None:
        raise ValueError(
            "Massive settings are required but not configured. "
            "Please set MASSIVE_API_KEY in your .env file."
        )

    # If already loaded but require_finnhub is True, check that Finnhub is present
    if require_finnhub and _default_settings.finnhub is None:
        raise ValueError(
            "Finnhub settings are required but not configured. "
            "Please set FINNHUB_API_KEY in your .env file. "
            "Get a free API key at https://finnhub.io/register"
        )

    # If already loaded but require_alphavantage is True, check that Alpha Vantage is present
    if require_alphavantage and _default_settings.alphavantage is None:
        raise ValueError(
            "Alpha Vantage settings are required but not configured. "
            "Please set ALPHAVANTAGE_API_KEY in your .env file. "
            "Get a free API key at https://www.alphavantage.co/support/#api-key"
        )

    return _default_settings


def reset_settings():
    """
    Reset the global settings singleton (for testing).

    **Conceptual**: This function clears the cached settings singleton, forcing
    it to be reloaded on next access. Used in tests to ensure each test gets
    fresh settings from environment.

    **Testing pattern**:
      ```python
      def test_something():
          # Reset settings before test
          reset_settings()

          # Set environment variables for this test
          os.environ["MASSIVE_API_KEY"] = "test_key"

          # Get fresh settings
          settings = get_settings()
          assert settings.massive.api_key == "test_key"
      ```

    Returns:
        None (side effect: clears global settings cache).
    """
    global _default_settings
    _default_settings = None
