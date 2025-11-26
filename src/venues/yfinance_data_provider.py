"""
YFinance data provider implementation.

**Conceptual**: This module provides a data adapter for Yahoo Finance via the
yfinance library. Unlike API-based providers (Finnhub, Alpha Vantage), yfinance
scrapes Yahoo Finance web pages, making it free to use with no API key required.

**Why YFinance?**
  - Free tier with no API key required
  - Comprehensive historical data for global stocks, ETFs, indices, forex, crypto
  - Easy to use via the yfinance Python package
  - No rate limits for reasonable usage
  - Good for research, backtesting, and small-scale applications

**Data Source**:
  - yfinance library (https://github.com/ranaroussi/yfinance)
  - Scrapes Yahoo Finance website
  - Returns pandas DataFrames directly
  - Supports daily, intraday, and other intervals

**Limitations**:
  - Web scraping can be unreliable (Yahoo may change their website)
  - No official API support or guarantees
  - Data quality may vary
  - Some corporate actions may not be perfectly adjusted

**Teaching note**: yfinance is excellent for learning and prototyping, but for
production systems you'd typically use a paid data provider with SLAs and
official API support. However, many quant researchers use yfinance successfully
for backtesting and research.

**Usage pattern**:
  - No API key needed - just install yfinance and fetch data
  - Fast for single tickers, slower for bulk downloads
  - Can fetch adjusted or unadjusted data
  - Supports multi-threading for faster bulk downloads
"""

import datetime as dt
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError(
        "yfinance library not installed. Install it with: pip install yfinance"
    )

# Import YFinanceSettings from config module (centralized settings)
from src.config.settings import YFinanceSettings


class YFinanceError(RuntimeError):
    """
    Generic error when fetching data from yfinance.

    **Conceptual**: Raised when yfinance encounters issues fetching or parsing
    data from Yahoo Finance. This could be due to network errors, invalid tickers,
    or changes to Yahoo Finance's website structure.

    **Recovery**:
      - Check ticker symbol spelling
      - Verify date range is valid
      - Check internet connection
      - Try again later (Yahoo Finance may be temporarily down)
      - Consider using a different data provider as backup
    """
    pass


class YFinanceEmptyDataError(YFinanceError):
    """
    Raised when yfinance returns no data for a given ticker/date range.

    **Conceptual**: This can happen for several reasons:
      - Invalid ticker symbol
      - Date range is outside available data
      - Ticker was delisted or suspended
      - Weekend/holiday (no trading)

    **Recovery**:
      - Verify ticker symbol is correct
      - Check if ticker exists on Yahoo Finance website
      - Adjust date range to include trading days
      - Use a different ticker or data source
    """
    pass


class YFinanceDataProvider:
    """
    Data provider for Yahoo Finance via yfinance library.

    **Conceptual**: This class fetches historical daily OHLCV bars from Yahoo
    Finance and converts them to DataFrames with canonical column names.

    **Data transformation pipeline**:
      1. Validate input parameters (ticker, date range)
      2. Call yf.download() to fetch data from Yahoo Finance
      3. Check if result is empty
      4. Reset index to convert DatetimeIndex to timestamp column
      5. Rename columns to canonical names (open_price, etc.)
      6. Convert timestamps to UTC timezone
      7. Filter to [start_date, end_date] inclusive window
      8. Validate output (no NaNs, correct columns, etc.)
      9. Return DataFrame (unsorted - IO layer handles that)

    **YFinance response format**:
    yfinance returns a DataFrame with:
      - Index: DatetimeIndex with dates
      - Columns: Open, High, Low, Close, Adj Close, Volume
      - All prices as floats
      - Volume as int64

    **Column mapping**:
    YFinance columns → DataFrame columns:
      - Date (index) → timestamp (converted to UTC)
      - Open → open_price
      - High → high_price
      - Low → low_price
      - Close → closing_price
      - Volume → volume
      - Adj Close → dropped (not part of canonical schema)

    **Timezone handling**:
    yfinance returns dates as DatetimeIndex (usually timezone-naive).
    We convert to UTC timezone for consistency with other providers.

    **Example usage**:
        >>> from src.config.settings import get_settings
        >>> settings = get_settings()
        >>> provider = YFinanceDataProvider(settings.yfinance)
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

    def __init__(self, settings: YFinanceSettings):
        """
        Initialize YFinance data provider.

        **Conceptual**: The provider needs YFinanceSettings to configure
        how data is fetched from Yahoo Finance (interval, adjustments, threading).

        Args:
            settings: YFinance configuration.

        Example:
            >>> from src.config.settings import YFinanceSettings
            >>> settings = YFinanceSettings(interval="1d", auto_adjust=False)
            >>> provider = YFinanceDataProvider(settings)
        """
        self.settings = settings

    def get_daily_bars(
        self,
        ticker: str,
        start_date: dt.date,
        end_date: dt.date,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars from Yahoo Finance via yfinance.

        **Conceptual**: This method fetches historical daily bars and returns
        a DataFrame with canonical columns. It does NOT sort or format the
        timestamp - that's the IO layer's responsibility (write_raw_price_csv).

        **Implementation steps**:
          1. Validate inputs (ticker not empty, start_date <= end_date)
          2. Call yf.download() with date range and settings
          3. Check if result is empty (raise YFinanceEmptyDataError)
          4. Handle MultiIndex columns (flatten for single ticker)
          5. Reset index to convert DatetimeIndex to timestamp column
          6. Rename columns to canonical names
          7. Drop Adj Close column (not part of canonical schema)
          8. Normalize timestamps to tz-naive UTC (canonical format)
          9. Filter to [start_date, end_date] inclusive window
          10. Validate output (no NaNs, correct columns, positive prices)
          11. Return DataFrame (unsorted, unformatted)

        **YFinance quirks**:
          - Returns DataFrame with DatetimeIndex (not a column)
          - Column names are capitalized (Open, High, Low, Close, Volume)
          - Includes "Adj Close" column for adjusted closing price
          - May return empty DataFrame for invalid tickers
          - Date range is inclusive [start, end]
          - Timezone is usually naive (no explicit timezone)

        **Error handling**:
        This method may raise:
          - ValueError: Invalid inputs (empty ticker, start_date > end_date)
          - YFinanceEmptyDataError: No data returned for ticker/date range
          - YFinanceError: Other errors during data fetch or processing

        Args:
            ticker: Ticker symbol (e.g., "QQQ", "SPY", "AAPL").
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            DataFrame with columns:
              - timestamp: datetime64[ns] (tz-naive, interpreted as UTC)
              - open_price: float
              - high_price: float
              - low_price: float
              - closing_price: float
              - volume: float

            Unsorted, with timestamp as tz-naive datetime (not formatted string).
            Never returns empty DataFrame - raises YFinanceEmptyDataError instead.

        Raises:
            ValueError: If inputs are invalid.
            YFinanceEmptyDataError: If no data returned for ticker/date range.
            YFinanceError: If error occurs during data fetch or processing.

        Example:
            >>> provider = YFinanceDataProvider(settings)
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

        # Step 2: Call yf.download() to fetch data
        # Note: yfinance expects string dates in YYYY-MM-DD format
        # Add 1 day to end_date because yfinance end date is exclusive
        try:
            df = yf.download(
                ticker.strip().upper(),
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=self.settings.interval,
                auto_adjust=self.settings.auto_adjust,
                back_adjust=self.settings.back_adjust,
                prepost=self.settings.prepost,
                progress=False,  # Disable progress bar
                threads=self.settings.threads,
                proxy=self.settings.proxy,
            )
        except Exception as e:
            raise YFinanceError(
                f"Error fetching data from yfinance for ticker '{ticker}': {e}"
            )

        # Step 3: Check if result is empty
        if df is None or df.empty:
            raise YFinanceEmptyDataError(
                f"No data returned from yfinance for ticker '{ticker}' "
                f"in date range [{start_date}, {end_date}]. "
                f"Ticker may be invalid or delisted."
            )

        # Step 4: Handle MultiIndex columns (yfinance sometimes returns multi-level columns)
        # For single ticker, flatten the columns to just the price column names
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-index columns by taking the first level (price column names)
            df.columns = df.columns.get_level_values(0)

        # Step 5: Reset index to convert DatetimeIndex to timestamp column
        # yfinance returns data with dates as the index
        df = df.reset_index()

        # Step 6: Rename columns to canonical names
        # yfinance columns: Date, Open, High, Low, Close, Adj Close, Volume
        # canonical columns: timestamp, open_price, high_price, low_price, closing_price, volume
        column_mapping = {
            "Date": "timestamp",
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "closing_price",
            "Volume": "volume",
        }
        df = df.rename(columns=column_mapping)

        # Step 7: Drop Adj Close column if present (not part of canonical schema)
        if "Adj Close" in df.columns:
            df = df.drop(columns=["Adj Close"])

        # Step 8: Normalize timestamps to tz-naive UTC (canonical format)
        # **Canonical contract**: All timestamps in DataFrames are tz-naive datetime64[ns],
        # interpreted as UTC. This matches the IO layer's contract and allows seamless
        # merging with existing CSVs.

        # First, parse to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Ensure timestamps are in UTC, then strip timezone info to make tz-naive
        if df['timestamp'].dt.tz is None:
            # Already tz-naive - assume UTC (yfinance default)
            pass
        else:
            # Convert to UTC, then strip timezone info to make tz-naive
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

        # Step 9: Filter to [start_date, end_date] inclusive window
        # Use tz-naive timestamps for filtering to match the canonical contract
        start_ts = pd.Timestamp(start_date)  # tz-naive
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].reset_index(drop=True)

        # Check if filtering removed all data
        if df.empty:
            raise YFinanceEmptyDataError(
                f"No data for ticker '{ticker}' in date range [{start_date}, {end_date}] "
                f"after filtering. Data may exist outside this range."
            )

        # Step 10: Validate output
        # Check for required columns
        required_cols = ['timestamp', 'open_price', 'high_price', 'low_price', 'closing_price', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise YFinanceError(
                    f"Column '{col}' missing from yfinance response. "
                    f"Available columns: {list(df.columns)}"
                )

        # Check for missing values in critical columns
        for col in required_cols:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                raise YFinanceError(
                    f"Column '{col}' has {nan_count} NaN values. "
                    f"Data quality issue in yfinance response."
                )

        # Check that prices are positive
        price_cols = ['open_price', 'high_price', 'low_price', 'closing_price']
        for col in price_cols:
            if (df[col] <= 0).any():
                raise YFinanceError(
                    f"Column '{col}' has non-positive values. "
                    f"Data quality issue."
                )

        # Check that high >= low
        if (df['high_price'] < df['low_price']).any():
            raise YFinanceError(
                "Found bars where high_price < low_price. Data quality issue."
            )

        # Step 11: Select only canonical columns and return
        df = df[required_cols]

        return df
