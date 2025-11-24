"""
Tests for MassiveDataProvider.

**Purpose**: Verify that MassiveDataProvider correctly:
  1. Converts JSON responses to DataFrames
  2. Maps column names to canonical format
  3. Sorts data in descending order
  4. Validates data quality (no NaNs, positive prices, etc.)
  5. Handles edge cases (empty results, malformed responses)

**Testing philosophy**: Use mocked MassiveClient (no real HTTP or API calls).
  - Fast and deterministic
  - Can test error conditions easily
  - Focus on DataFrame transformation logic
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.config.settings import MassiveSettings
from src.venues.massive_data_provider import MassiveDataProvider, MassiveDataProviderError
from src.venues.massive_client import MassiveSymbolNotFoundError


@pytest.fixture
def massive_settings():
    """Create test settings for MassiveDataProvider."""
    return MassiveSettings(
        base_url="https://api.test-massive.com",
        api_key="test_api_key_123",
        timeout_seconds=30,
    )


@pytest.fixture
def mock_client():
    """
    Create a mock MassiveClient for testing.

    **Conceptual**: Instead of mocking HTTP requests, we mock the entire
    MassiveClient. This lets us test MassiveDataProvider's logic without
    worrying about HTTP details.

    Returns:
        Mock object that can be configured to return test responses.
    """
    return Mock()


def make_test_bars_df(bars_data):
    """
    Create a test DataFrame in the format returned by MassiveClient.

    Args:
        bars_data: List of dicts with keys: timestamp, open, high, low, close, volume

    Returns:
        DataFrame with the structure returned by MassiveClient.get_daily_bars()
    """
    if not bars_data:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(bars_data)
    # Convert timestamps to datetime if they're strings
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def test_fetch_daily_bars_success(massive_settings, mock_client):
    """
    Test successful fetch and DataFrame conversion.

    **Conceptual**: This is the happy path - client returns valid DataFrame,
    provider renames columns to canonical format.
    """
    # Configure mock client to return test DataFrame (client now returns DF directly)
    mock_client.get_daily_bars.return_value = make_test_bars_df([
        {"timestamp": "2024-01-31", "open": 450.0, "high": 452.0, "low": 449.0, "close": 451.0, "volume": 5000000},
        {"timestamp": "2024-01-30", "open": 448.0, "high": 450.0, "low": 447.0, "close": 449.0, "volume": 4800000},
        {"timestamp": "2024-01-29", "open": 446.0, "high": 448.0, "low": 445.0, "close": 447.0, "volume": 4600000},
    ])

    # Create provider with mocked client
    provider = MassiveDataProvider(massive_settings, client=mock_client)

    # Fetch data
    start = pd.Timestamp("2024-01-29", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")
    bars = provider.fetch_daily_bars("QQQ", start, end)

    # Verify client was called with correct params
    mock_client.get_daily_bars.assert_called_once_with(
        symbol="QQQ",
        start_date="2024-01-29",
        end_date="2024-01-31",
    )

    # Verify DataFrame structure
    assert isinstance(bars, pd.DataFrame)
    assert len(bars) == 3

    # Verify canonical columns
    expected_cols = ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
    assert list(bars.columns) == expected_cols

    # Verify data is sorted ASCENDING (oldest first, as returned by client)
    # Write layer will enforce descending order when writing to CSV
    assert bars.iloc[0]['timestamp'] == pd.Timestamp("2024-01-29", tz="UTC")
    assert bars.iloc[1]['timestamp'] == pd.Timestamp("2024-01-30", tz="UTC")
    assert bars.iloc[2]['timestamp'] == pd.Timestamp("2024-01-31", tz="UTC")

    # Verify timestamp is timezone-aware
    assert bars.iloc[0]['timestamp'].tz is not None

    # Verify price values for last row (newest)
    assert bars.iloc[2]['closing_price'] == 451.0
    assert bars.iloc[2]['open_price'] == 450.0
    assert bars.iloc[2]['high_price'] == 452.0
    assert bars.iloc[2]['low_price'] == 449.0
    assert bars.iloc[2]['volume'] == 5000000


def test_fetch_daily_bars_empty_result(massive_settings, mock_client):
    """Test that empty response returns empty DataFrame with correct columns."""
    # Mock client returns empty DataFrame
    mock_client.get_daily_bars.return_value = make_test_bars_df([])

    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")
    bars = provider.fetch_daily_bars("QQQ", start, end)

    # Should return empty DataFrame with correct columns
    assert len(bars) == 0
    expected_cols = ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
    assert list(bars.columns) == expected_cols


def test_fetch_daily_bars_normalizes_symbol(massive_settings, mock_client):
    """Test that symbol is normalized to uppercase."""
    mock_client.get_daily_bars.return_value = make_test_bars_df([])

    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")

    # Pass lowercase symbol with whitespace
    provider.fetch_daily_bars("  qqq  ", start, end)

    # Verify client was called with normalized symbol
    mock_client.get_daily_bars.assert_called_once_with(
        symbol="QQQ",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )


def test_fetch_daily_bars_empty_symbol(massive_settings, mock_client):
    """Test that empty symbol raises ValueError."""
    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")

    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        provider.fetch_daily_bars("", start, end)

    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        provider.fetch_daily_bars("   ", start, end)


def test_fetch_daily_bars_missing_dates(massive_settings, mock_client):
    """Test that None dates raise ValueError."""
    provider = MassiveDataProvider(massive_settings, client=mock_client)

    with pytest.raises(ValueError, match="start and end must be provided"):
        provider.fetch_daily_bars("QQQ", None, pd.Timestamp("2024-01-31", tz="UTC"))

    with pytest.raises(ValueError, match="start and end must be provided"):
        provider.fetch_daily_bars("QQQ", pd.Timestamp("2024-01-01", tz="UTC"), None)


def test_fetch_daily_bars_start_after_end(massive_settings, mock_client):
    """Test that start > end raises ValueError."""
    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-12-31", tz="UTC")
    end = pd.Timestamp("2024-01-01", tz="UTC")

    with pytest.raises(ValueError, match="start .* must be <= end"):
        provider.fetch_daily_bars("QQQ", start, end)


def test_fetch_daily_bars_naive_timestamps(massive_settings, mock_client):
    """Test that timezone-naive timestamps raise ValueError."""
    provider = MassiveDataProvider(massive_settings, client=mock_client)

    # Create naive timestamps (no timezone)
    start = pd.Timestamp("2024-01-01")  # No tz argument
    end = pd.Timestamp("2024-01-31")

    with pytest.raises(ValueError, match="must be timezone-aware"):
        provider.fetch_daily_bars("QQQ", start, end)


# JSON parsing tests removed - client now returns DataFrames directly


def test_fetch_daily_bars_missing_required_columns(massive_settings, mock_client):
    """Test that missing required columns raises error."""
    # Mock client returns DataFrame missing 'close' column
    df = pd.DataFrame([
        {"timestamp": "2024-01-31", "open": 450.0, "high": 452.0, "low": 449.0, "volume": 5000000},
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    mock_client.get_daily_bars.return_value = df

    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")

    with pytest.raises(MassiveDataProviderError, match="missing required columns"):
        provider.fetch_daily_bars("QQQ", start, end)


def test_fetch_daily_bars_nan_values(massive_settings, mock_client):
    """Test that NaN values in data raise error."""
    # Mock client returns DataFrame with NaN
    df = make_test_bars_df([
        {"timestamp": "2024-01-31", "open": 450.0, "high": None, "low": 449.0, "close": 451.0, "volume": 5000000},
    ])
    mock_client.get_daily_bars.return_value = df

    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")

    with pytest.raises(MassiveDataProviderError, match="NaN values"):
        provider.fetch_daily_bars("QQQ", start, end)


def test_fetch_daily_bars_negative_prices(massive_settings, mock_client):
    """Test that negative prices raise error."""
    # Mock client returns DataFrame with negative price
    mock_client.get_daily_bars.return_value = make_test_bars_df([
        {"timestamp": "2024-01-31", "open": 450.0, "high": 452.0, "low": 449.0, "close": -10.0, "volume": 5000000},
    ])

    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")

    with pytest.raises(MassiveDataProviderError, match="non-positive values"):
        provider.fetch_daily_bars("QQQ", start, end)


def test_fetch_daily_bars_zero_prices(massive_settings, mock_client):
    """Test that zero prices raise error."""
    # Mock client returns DataFrame with zero price
    mock_client.get_daily_bars.return_value = make_test_bars_df([
        {"timestamp": "2024-01-31", "open": 0.0, "high": 452.0, "low": 449.0, "close": 451.0, "volume": 5000000},
    ])

    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")

    with pytest.raises(MassiveDataProviderError, match="non-positive values"):
        provider.fetch_daily_bars("QQQ", start, end)


def test_fetch_daily_bars_high_less_than_low(massive_settings, mock_client):
    """Test that high < low raises error."""
    # Mock client returns DataFrame with high < low (invalid)
    mock_client.get_daily_bars.return_value = make_test_bars_df([
        {"timestamp": "2024-01-31", "open": 450.0, "high": 448.0, "low": 452.0, "close": 451.0, "volume": 5000000},
    ])

    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")

    with pytest.raises(MassiveDataProviderError, match="high_price < low_price"):
        provider.fetch_daily_bars("QQQ", start, end)


def test_fetch_daily_bars_client_error_propagates(massive_settings, mock_client):
    """Test that client errors (auth, not found, etc.) propagate correctly."""
    # Mock client raises MassiveSymbolNotFoundError
    mock_client.get_daily_bars.side_effect = MassiveSymbolNotFoundError("Symbol 'INVALID' not found")

    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")

    # Error should propagate unchanged
    with pytest.raises(MassiveSymbolNotFoundError, match="INVALID"):
        provider.fetch_daily_bars("INVALID", start, end)


def test_fetch_daily_bars_sorts_ascending(massive_settings, mock_client):
    """
    Test that data is sorted in ascending order as per updated spec.

    **Conceptual**: Updated design has client return ascending data,
    which provider preserves (write layer enforces descending on disk).
    """
    # Mock client returns data in ASCENDING order (oldest first)
    mock_client.get_daily_bars.return_value = make_test_bars_df([
        {"timestamp": "2024-01-29", "open": 446.0, "high": 448.0, "low": 445.0, "close": 447.0, "volume": 4600000},
        {"timestamp": "2024-01-30", "open": 448.0, "high": 450.0, "low": 447.0, "close": 449.0, "volume": 4800000},
        {"timestamp": "2024-01-31", "open": 450.0, "high": 452.0, "low": 449.0, "close": 451.0, "volume": 5000000},
    ])

    provider = MassiveDataProvider(massive_settings, client=mock_client)
    start = pd.Timestamp("2024-01-29", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")
    bars = provider.fetch_daily_bars("QQQ", start, end)

    # Verify data is sorted ASCENDING (oldest first, as returned by client)
    # Write layer will enforce descending order when writing to CSV
    assert bars.iloc[0]['timestamp'] == pd.Timestamp("2024-01-29", tz="UTC")
    assert bars.iloc[1]['timestamp'] == pd.Timestamp("2024-01-30", tz="UTC")
    assert bars.iloc[2]['timestamp'] == pd.Timestamp("2024-01-31", tz="UTC")


def test_provider_context_manager(massive_settings, mock_client):
    """Test that provider can be used as context manager."""
    with MassiveDataProvider(massive_settings, client=mock_client) as provider:
        assert isinstance(provider, MassiveDataProvider)

    # Client's close method should have been called
    mock_client.close.assert_called_once()


def test_provider_close(massive_settings, mock_client):
    """Test that close() closes the underlying client."""
    provider = MassiveDataProvider(massive_settings, client=mock_client)
    provider.close()

    # Verify client.close() was called
    mock_client.close.assert_called_once()


def test_provider_creates_client_if_not_provided(massive_settings):
    """Test that provider creates its own client if none is injected."""
    with patch('src.venues.massive_data_provider.MassiveClient') as MockClientClass:
        # Create provider without injecting client
        provider = MassiveDataProvider(massive_settings)

        # Verify MassiveClient was instantiated with settings
        MockClientClass.assert_called_once_with(massive_settings)
        assert provider.client == MockClientClass.return_value
