"""
Instrument-specific convenience loaders for raw price data.

**Conceptual**: This module provides simple, named functions to load specific
instruments (QQQ, TQQQ, SQQQ, UVXY) without hardcoding paths in strategy or
backtest code. These are thin wrappers around io.py that:
  - Resolve standard paths under data/raw/.
  - Provide clear, self-documenting names (load_qqq_history vs magic paths).
  - Centralize path logic so strategies stay portable.

**Why convenience loaders?**
  - Readability: `load_qqq_history()` is clearer than `read_raw_price_csv("data/raw/QQQ.csv")`.
  - Decoupling: Strategies don't hardcode paths; if we change directory structure,
    only loaders.py needs updates.
  - Discoverability: New contributors can see which instruments are available by
    reading this file.
  - Testing: Easy to mock or replace loaders in unit tests.

**Teaching note**: In a trading system, data loading is a cross-cutting concern.
By centralizing it here, we make strategy code cleaner (strategies just say
"give me QQQ data" without worrying about where it lives). This also helps
with future cloud migration: we can swap local paths for S3 keys without
touching strategy logic.
"""

import pandas as pd
from pathlib import Path

from src.data.io import read_raw_price_csv


# Project root is 2 levels up from src/data/loaders.py
# This allows us to resolve data/raw/ relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Standard data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_instrument_history(symbol: str) -> pd.DataFrame:
    """
    Load raw price history for a given instrument symbol.

    **Conceptual**: Generic loader that resolves the standard path for an
    instrument's raw price CSV (data/raw/<SYMBOL>.csv) and reads it with
    full schema validation.

    **Functionally**:
      - Builds path: data/raw/{symbol}.csv (e.g., data/raw/QQQ.csv).
      - Delegates to read_raw_price_csv for schema validation.
      - Returns DataFrame with validated raw price schema.

    **Why generic + specific loaders?**
      - Generic (this function): useful for dynamic loading (e.g., loop over symbols).
      - Specific (load_qqq_history, etc.): self-documenting, easier to discover.

    **Teaching note**: This function demonstrates the "generic with specific
    conveniences" pattern. It's flexible enough for dynamic use cases but
    also supports explicit wrappers for common instruments.

    Args:
        symbol: Instrument symbol (e.g., "QQQ", "TQQQ", "SQQQ", "UVXY").
                Symbol is case-sensitive and must match the CSV filename.

    Returns:
        DataFrame with raw price schema (timestamp, OHLC, volume).

    Raises:
        FileNotFoundError: If data/raw/{symbol}.csv doesn't exist.
        SchemaValidationError: If CSV doesn't conform to raw price schema.

    Example:
        >>> qqq = load_instrument_history("QQQ")
        >>> tqqq = load_instrument_history("TQQQ")
        >>> qqq.head(2)
                  timestamp  open_price  high_price  low_price  closing_price  volume
        0 2024-01-15 00:00:00       400.0       405.0      398.0          403.0  1000000
        1 2024-01-14 00:00:00       398.0       401.0      397.0          400.0  950000
    """
    # Build path to raw CSV
    csv_path = RAW_DATA_DIR / f"{symbol}.csv"

    # Load via read_raw_price_csv for schema validation
    return read_raw_price_csv(csv_path, instrument_name=symbol)


def load_qqq_history() -> pd.DataFrame:
    """
    Load raw price history for QQQ (Invesco QQQ Trust).

    **Conceptual**: QQQ is the primary "reference instrument" for Strategy 1
    (QQQ-based long/short trend-following). This is the underlying index ETF
    whose price, velocity, and acceleration determine allocation decisions.

    **Functionally**:
      - Loads data/raw/QQQ.csv with schema validation.
      - Returns DataFrame with raw price schema.

    **Strategy 1 context**: QQQ price data is used to compute features (moving
    averages, velocity, acceleration) that drive trend detection. Execution
    happens via TQQQ (long) or SQQQ (short), but QQQ is the signal source.

    **Teaching note**: Explicit loaders for key instruments serve as documentation.
    When a new contributor asks "which instruments does Strategy 1 use?", they
    can grep for load_qqq_history, load_tqqq_history, etc., and immediately see
    the universe.

    Returns:
        DataFrame with QQQ raw price data (timestamp, OHLC, volume).

    Raises:
        FileNotFoundError: If data/raw/QQQ.csv doesn't exist.
        SchemaValidationError: If QQQ.csv doesn't conform to schema.

    Example:
        >>> qqq = load_qqq_history()
        >>> qqq['closing_price'].mean()
        398.5
    """
    return load_instrument_history("QQQ")


def load_tqqq_history() -> pd.DataFrame:
    """
    Load raw price history for TQQQ (ProShares UltraPro QQQ).

    **Conceptual**: TQQQ is a 3x leveraged long ETF tracking QQQ. Strategy 1
    uses TQQQ for long exposure when trend detection indicates an uptrend.

    **Functionally**:
      - Loads data/raw/TQQQ.csv with schema validation.
      - Returns DataFrame with raw price schema.

    **Strategy 1 context**: When QQQ shows strong uptrend signals (positive
    velocity, acceleration, favorable moving average spreads), Strategy 1
    allocates to TQQQ for amplified upside exposure.

    **Teaching note**: TQQQ is leveraged, so it exhibits higher volatility and
    drawdown than QQQ. Backtests using TQQQ must account for leverage costs,
    tracking error, and volatility decay over time.

    Returns:
        DataFrame with TQQQ raw price data.

    Raises:
        FileNotFoundError: If data/raw/TQQQ.csv doesn't exist.
        SchemaValidationError: If TQQQ.csv doesn't conform to schema.
    """
    return load_instrument_history("TQQQ")


def load_sqqq_history() -> pd.DataFrame:
    """
    Load raw price history for SQQQ (ProShares UltraPro Short QQQ).

    **Conceptual**: SQQQ is a 3x leveraged inverse ETF tracking QQQ. Strategy 1
    uses SQQQ for short exposure when trend detection indicates a downtrend.

    **Functionally**:
      - Loads data/raw/SQQQ.csv with schema validation.
      - Returns DataFrame with raw price schema.

    **Strategy 1 context**: When QQQ shows strong downtrend signals (negative
    velocity, acceleration, unfavorable moving average spreads), Strategy 1
    allocates to SQQQ for amplified downside protection / profit.

    **Teaching note**: SQQQ moves inversely to QQQ with 3x leverage. This means
    when QQQ falls 1%, SQQQ rises ~3%. However, SQQQ also suffers from volatility
    decay and tracking error, so holding it long-term (without rebalancing) can
    be costly. Strategy 1 aims to hold SQQQ only during detected downtrends.

    Returns:
        DataFrame with SQQQ raw price data.

    Raises:
        FileNotFoundError: If data/raw/SQQQ.csv doesn't exist.
        SchemaValidationError: If SQQQ.csv doesn't conform to schema.
    """
    return load_instrument_history("SQQQ")


def load_uvxy_history() -> pd.DataFrame:
    """
    Load raw price history for UVXY (ProShares Ultra VIX Short-Term Futures ETF).

    **Conceptual**: UVXY is a 2x leveraged volatility ETF. Strategy 1 optionally
    uses UVXY as a volatility hedge during mixed/neutral regimes or as a
    diversification overlay.

    **Functionally**:
      - Loads data/raw/UVXY.csv with schema validation.
      - Returns DataFrame with raw price schema.

    **Strategy 1 context**: UVXY can be used to profit from or hedge against
    volatility spikes (e.g., during market panics when QQQ trends are unclear).
    Initial Strategy 1 versions may not use UVXY; it's included here for future
    enhancement.

    **Teaching note**: UVXY is a complex instrument (2x leveraged, futures-based,
    suffers from severe contango decay). It's not intended for buy-and-hold.
    Strategy 1 would only allocate to UVXY tactically during specific regimes.

    Returns:
        DataFrame with UVXY raw price data.

    Raises:
        FileNotFoundError: If data/raw/UVXY.csv doesn't exist.
        SchemaValidationError: If UVXY.csv doesn't conform to schema.
    """
    return load_instrument_history("UVXY")


def load_default_universe() -> dict[str, pd.DataFrame]:
    """
    Load raw price histories for the default instrument universe.

    **Conceptual**: Strategy 1 (QQQ long/short with TQQQ/SQQQ/UVXY) uses a
    specific set of instruments. This function loads all of them in one call,
    returning a dictionary keyed by symbol.

    **Functionally**:
      - Attempts to load QQQ, TQQQ, SQQQ, UVXY from data/raw/.
      - Skips any missing instruments (logs/warns but doesn't fail).
      - Returns dict mapping symbol -> DataFrame.

    **Why allow missing instruments?**
      - UVXY may not be available in all datasets (it's optional for Strategy 1).
      - In testing, we may have only QQQ available.
      - Failing hard on missing instruments would make the system brittle.
      - Strategies can check which instruments are available and adapt.

    **Teaching note**: In production, you might want stricter enforcement (fail
    if required instruments are missing). Here we prioritize flexibility for
    development/testing. Strategies should validate that their required
    instruments are present before running.

    Returns:
        Dictionary mapping symbol (str) to DataFrame (raw price data).
        Example: {"QQQ": df_qqq, "TQQQ": df_tqqq, "SQQQ": df_sqqq, "UVXY": df_uvxy}.
        Missing instruments are omitted from the dict (no entry vs empty DataFrame).

    Example:
        >>> universe = load_default_universe()
        >>> universe.keys()
        dict_keys(['QQQ', 'TQQQ', 'SQQQ', 'UVXY'])
        >>> qqq_df = universe["QQQ"]
        >>> qqq_df.head(2)
                  timestamp  open_price  ...
    """
    universe = {}
    symbols = ["QQQ", "TQQQ", "SQQQ", "UVXY"]

    for symbol in symbols:
        try:
            universe[symbol] = load_instrument_history(symbol)
        except FileNotFoundError:
            # Silently skip missing instruments
            # In production, you might want to log a warning here
            continue
        except Exception as e:
            # Re-raise schema validation errors or other unexpected issues
            # (we only want to skip file-not-found, not schema problems)
            raise

    return universe
