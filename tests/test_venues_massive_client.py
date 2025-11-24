"""
Tests for MassiveClient HTTP wrapper.

**Purpose**: Verify that MassiveClient correctly constructs requests, handles
responses, and raises appropriate exceptions for different HTTP error codes.

**Testing philosophy**: Use mocked HTTP responses (no real API calls).
  - Fast (no network I/O)
  - Deterministic (no flaky tests due to network issues)
  - No API key required
  - Can test error conditions (rate limits, server errors) easily

**Teaching note**: This demonstrates the "test pyramid" principle:
  - Many unit tests (fast, isolated, mocked)
  - Fewer integration tests (slower, real API)
  - Fewest end-to-end tests (slowest, full system)

We test MassiveClient at the unit level with mocks. Real API integration
can be tested manually or in a separate integration test suite.
"""

import pytest
import pandas as pd
import requests
from unittest.mock import Mock, patch, MagicMock
from src.config.settings import MassiveSettings
from src.venues.massive_client import (
    MassiveClient,
    MassiveAuthenticationError,
    MassiveSymbolNotFoundError,
    MassiveRateLimitError,
    MassiveServerError,
    MassiveClientError,
)


@pytest.fixture
def massive_settings():
    """
    Create test settings for MassiveClient.

    **Conceptual**: Pytest fixtures are reusable test components.
    This fixture provides MassiveSettings with fake credentials for testing.

    **Why fixture instead of creating settings in each test?**
      - DRY (Don't Repeat Yourself) - settings defined once
      - Consistent across tests
      - Easy to modify (change in one place)

    Returns:
        MassiveSettings with test credentials.
    """
    return MassiveSettings(
        base_url="https://api.test-massive.com",
        api_key="test_api_key_123",
        timeout_seconds=30,
    )


def test_massive_client_initialization(massive_settings):
    """Test that client initializes with correct headers."""
    client = MassiveClient(massive_settings)

    # Check that session headers include auth token
    assert "Authorization" in client.session.headers
    assert client.session.headers["Authorization"] == "Bearer test_api_key_123"
    assert client.session.headers["Accept"] == "application/json"


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_success(mock_get, massive_settings):
    """
    Test successful fetch of daily bars.

    **Conceptual**: We mock requests.Session.get to return a fake response
    without making a real HTTP request. This tests the happy path.

    **Mock response structure**:
    We simulate Massive API returning JSON with bars list.
    """
    # Create mock response (Polygon/Massive API format)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "ticker": "QQQ",
        "results": [
            {"t": 1706745600000, "o": 450.0, "h": 452.0, "l": 449.0, "c": 451.0, "v": 5000000},
            {"t": 1706659200000, "o": 448.0, "h": 450.0, "l": 447.0, "c": 449.0, "v": 4800000},
        ],
        "resultsCount": 2,
    }
    mock_get.return_value = mock_response

    # Create client and fetch data
    client = MassiveClient(massive_settings)
    result = client.get_daily_bars(
        symbol="QQQ",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    # Verify request was made with correct params
    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert call_args.kwargs["params"]["adjusted"] == "true"
    assert call_args.kwargs["params"]["sort"] == "asc"
    assert call_args.kwargs["timeout"] == 30

    # Verify response is a DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'timestamp' in result.columns
    assert 'close' in result.columns


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_authentication_error_401(mock_get, massive_settings):
    """Test that 401 Unauthorized raises MassiveAuthenticationError."""
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.text = "Invalid API key"
    mock_get.return_value = mock_response

    client = MassiveClient(massive_settings)

    with pytest.raises(MassiveAuthenticationError, match="Authentication failed"):
        client.get_daily_bars("QQQ", "2024-01-01", "2024-01-31")


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_authentication_error_403(mock_get, massive_settings):
    """Test that 403 Forbidden raises MassiveAuthenticationError."""
    mock_response = Mock()
    mock_response.status_code = 403
    mock_response.text = "API key expired"
    mock_get.return_value = mock_response

    client = MassiveClient(massive_settings)

    with pytest.raises(MassiveAuthenticationError, match="Authentication failed"):
        client.get_daily_bars("QQQ", "2024-01-01", "2024-01-31")


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_symbol_not_found(mock_get, massive_settings):
    """Test that 404 Not Found raises MassiveSymbolNotFoundError."""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = "Symbol 'INVALID' not found"
    mock_get.return_value = mock_response

    client = MassiveClient(massive_settings)

    with pytest.raises(MassiveSymbolNotFoundError, match="Symbol 'INVALID' not found"):
        client.get_daily_bars("INVALID", "2024-01-01", "2024-01-31")


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_rate_limit_error(mock_get, massive_settings):
    """Test that 429 Too Many Requests raises MassiveRateLimitError."""
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.text = "Rate limit exceeded. Try again in 60 seconds."
    mock_get.return_value = mock_response

    client = MassiveClient(massive_settings)

    with pytest.raises(MassiveRateLimitError, match="Rate limit exceeded"):
        client.get_daily_bars("QQQ", "2024-01-01", "2024-01-31")


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_server_error_500(mock_get, massive_settings):
    """Test that 500 Internal Server Error raises MassiveServerError."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"
    mock_get.return_value = mock_response

    client = MassiveClient(massive_settings)

    with pytest.raises(MassiveServerError, match="server error"):
        client.get_daily_bars("QQQ", "2024-01-01", "2024-01-31")


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_server_error_503(mock_get, massive_settings):
    """Test that 503 Service Unavailable raises MassiveServerError."""
    mock_response = Mock()
    mock_response.status_code = 503
    mock_response.text = "Service temporarily unavailable"
    mock_get.return_value = mock_response

    client = MassiveClient(massive_settings)

    with pytest.raises(MassiveServerError, match="server error"):
        client.get_daily_bars("QQQ", "2024-01-01", "2024-01-31")


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_client_error_400(mock_get, massive_settings):
    """Test that 400 Bad Request raises MassiveClientError."""
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Bad request: invalid date format"
    mock_get.return_value = mock_response

    client = MassiveClient(massive_settings)

    with pytest.raises(MassiveClientError, match="Client error"):
        client.get_daily_bars("QQQ", "invalid-date", "2024-01-31")


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_timeout(mock_get, massive_settings):
    """Test that request timeout raises requests.Timeout."""
    # Mock get to raise Timeout exception
    mock_get.side_effect = requests.Timeout("Connection timed out")

    client = MassiveClient(massive_settings)

    with pytest.raises(requests.Timeout, match="timed out"):
        client.get_daily_bars("QQQ", "2024-01-01", "2024-01-31")


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_connection_error(mock_get, massive_settings):
    """Test that connection error raises MassiveClientError."""
    # Mock get to raise ConnectionError
    mock_get.side_effect = requests.ConnectionError("Failed to connect")

    client = MassiveClient(massive_settings)

    with pytest.raises(MassiveClientError, match="Failed to connect"):
        client.get_daily_bars("QQQ", "2024-01-01", "2024-01-31")


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_malformed_json(mock_get, massive_settings):
    """Test that malformed JSON response raises MassiveClientError."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "This is not JSON"
    mock_response.json.side_effect = ValueError("No JSON object could be decoded")
    mock_get.return_value = mock_response

    client = MassiveClient(massive_settings)

    with pytest.raises(MassiveClientError, match="Failed to parse JSON"):
        client.get_daily_bars("QQQ", "2024-01-01", "2024-01-31")


def test_get_daily_bars_empty_symbol(massive_settings):
    """Test that empty symbol raises ValueError."""
    client = MassiveClient(massive_settings)

    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        client.get_daily_bars("", "2024-01-01", "2024-01-31")

    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        client.get_daily_bars("   ", "2024-01-01", "2024-01-31")


def test_get_daily_bars_missing_dates(massive_settings):
    """Test that missing dates raise ValueError."""
    client = MassiveClient(massive_settings)

    with pytest.raises(ValueError, match="start_date and end_date are required"):
        client.get_daily_bars("QQQ", "", "2024-01-31")

    with pytest.raises(ValueError, match="start_date and end_date are required"):
        client.get_daily_bars("QQQ", "2024-01-01", "")


@patch("src.venues.massive_client.requests.Session.get")
def test_get_daily_bars_normalizes_symbol(mock_get, massive_settings):
    """Test that symbol is normalized to uppercase and trimmed."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ticker": "QQQ", "results": [], "resultsCount": 0}
    mock_get.return_value = mock_response

    client = MassiveClient(massive_settings)
    result = client.get_daily_bars("  qqq  ", "2024-01-01", "2024-01-31")

    # Verify symbol was normalized in the URL path
    call_args = mock_get.call_args
    # The symbol should be in the URL path, not in params
    assert "QQQ" in call_args.args[0]

    # Verify response is a DataFrame (even if empty)
    assert isinstance(result, pd.DataFrame)


def test_client_context_manager(massive_settings):
    """Test that client can be used as context manager."""
    with MassiveClient(massive_settings) as client:
        assert isinstance(client, MassiveClient)
        assert client.session is not None

    # Session should be closed after exiting context
    # (We can't easily test this without accessing internals, but context manager works)


def test_client_close(massive_settings):
    """Test that close() closes the session."""
    client = MassiveClient(massive_settings)
    session = client.session

    # Spy on session.close
    with patch.object(session, 'close') as mock_close:
        client.close()
        mock_close.assert_called_once()
