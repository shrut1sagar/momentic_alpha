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
    """
    massive: Optional[MassiveSettings] = None

    @classmethod
    def from_env(cls, require_massive: bool = False) -> "Settings":
        """
        Load global settings from environment variables.

        **Conceptual**: This factory method loads all subsystem settings from
        the environment. It's the standard way to initialize settings at
        application startup.

        **Design decision**: Massive settings are optional by default. This allows
        the system to work without Massive configured (e.g., using local CSV files).
        Set require_massive=True to enforce that Massive is configured (useful for
        data ingestion scripts).

        Args:
            require_massive: If True, raise error if Massive settings are missing.
                           If False (default), Massive settings are optional.

        Returns:
            Settings object with all subsystem settings loaded from environment.

        Raises:
            ValueError: If require_massive=True and MASSIVE_API_KEY is missing.

        Usage example:
            >>> # Load settings (Massive optional)
            >>> settings = Settings.from_env()
            >>>
            >>> # Load settings (Massive required)
            >>> settings = Settings.from_env(require_massive=True)
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

        return cls(massive=massive_settings)


# Convenience singleton for accessing settings throughout the application
# Note: This loads settings at module import time. For testing, inject custom settings.
# Example: tests can create Settings(massive=fake_massive_settings) instead of using this.
_default_settings: Optional[Settings] = None


def get_settings(require_massive: bool = False) -> Settings:
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

    Returns:
        Global Settings singleton.

    Raises:
        ValueError: If require_massive=True and Massive not configured.

    Usage example:
        >>> from src.config.settings import get_settings
        >>>
        >>> # Get settings (Massive optional)
        >>> settings = get_settings()
        >>> if settings.massive:
        >>>     print("Massive configured")
        >>>
        >>> # Get settings (Massive required)
        >>> settings = get_settings(require_massive=True)
        >>> print(settings.massive.api_key)
    """
    global _default_settings

    if _default_settings is None:
        _default_settings = Settings.from_env(require_massive=require_massive)

    # If already loaded but require_massive is True, check that Massive is present
    if require_massive and _default_settings.massive is None:
        raise ValueError(
            "Massive settings are required but not configured. "
            "Please set MASSIVE_API_KEY in your .env file."
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
