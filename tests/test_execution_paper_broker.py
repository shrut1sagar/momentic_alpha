"""
Tests for the paper broker and portfolio model.

This module tests the PaperBroker's ability to:
  - Accept target weights and execute trades.
  - Track cash and positions correctly.
  - Apply slippage and fees.
  - Handle price movements and rebalancing.

All tests use simple, deterministic scenarios with hand-calculated expected values.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone

from src.execution.paper_broker import PaperBroker, Position, PortfolioState


def test_paper_broker_initial_state():
    """Test that broker initializes with correct cash and no positions."""
    broker = PaperBroker(initial_cash=100_000)
    state = broker.get_portfolio_state()

    # Should have all cash, no positions
    assert state.cash == 100_000
    assert state.positions == {}
    assert state.equity == 100_000
    assert state.gross_exposure == 0.0
    assert state.net_exposure == 0.0


def test_paper_broker_full_allocation_single_symbol():
    """
    Test allocating 100% to a single symbol.

    Scenario:
      - Start with $100,000 cash
      - Symbol TEST trades at $100
      - Allocate 100% to TEST
      - No slippage or fees

    Expected:
      - Buy 1,000 shares of TEST ($100,000 / $100)
      - Cash = $0
      - Position = 1,000 shares
      - Equity = $100,000
    """
    # Setup
    broker = PaperBroker(initial_cash=100_000, slippage_bps=0.0, fee_per_trade=0.0)
    dt = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))

    # Update price
    broker.update_prices({"TEST": 100.0}, dt)

    # Allocate 100% to TEST
    broker.set_target_weights({"TEST": 1.0})

    # Check state
    state = broker.get_portfolio_state()

    assert state.cash == pytest.approx(0.0, abs=1e-6)
    assert "TEST" in state.positions
    assert state.positions["TEST"] == pytest.approx(1000.0, abs=1e-6)
    assert state.equity == pytest.approx(100_000.0, abs=1e-6)
    assert state.gross_exposure == pytest.approx(100_000.0, abs=1e-6)
    assert state.net_exposure == pytest.approx(100_000.0, abs=1e-6)


def test_paper_broker_partial_allocation():
    """
    Test allocating 50% to a symbol (50% cash remainder).

    Scenario:
      - Start with $100,000 cash
      - Symbol TEST trades at $100
      - Allocate 50% to TEST

    Expected:
      - Buy 500 shares ($50,000 / $100)
      - Cash = $50,000
      - Equity = $100,000
    """
    broker = PaperBroker(initial_cash=100_000)
    dt = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))

    broker.update_prices({"TEST": 100.0}, dt)
    broker.set_target_weights({"TEST": 0.5})

    state = broker.get_portfolio_state()

    assert state.cash == pytest.approx(50_000.0, abs=1e-6)
    assert state.positions["TEST"] == pytest.approx(500.0, abs=1e-6)
    assert state.equity == pytest.approx(100_000.0, abs=1e-6)


def test_paper_broker_price_increase():
    """
    Test that equity increases when prices go up.

    Scenario:
      - Start with $100,000, allocate 100% to TEST at $100
      - Price increases to $110
      - Update prices (no rebalancing)

    Expected:
      - Position value increases to $110,000
      - Equity increases to $110,000
      - Cash remains $0
    """
    broker = PaperBroker(initial_cash=100_000)
    dt1 = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))
    dt2 = pd.Timestamp(datetime(2024, 1, 16, tzinfo=timezone.utc))

    # Initial allocation at $100
    broker.update_prices({"TEST": 100.0}, dt1)
    broker.set_target_weights({"TEST": 1.0})

    # Price increases to $110
    broker.update_prices({"TEST": 110.0}, dt2)

    state = broker.get_portfolio_state()

    # Position unchanged (1000 shares)
    assert state.positions["TEST"] == pytest.approx(1000.0, abs=1e-6)
    # Equity increases
    assert state.equity == pytest.approx(110_000.0, abs=1e-6)
    # Cash still zero
    assert state.cash == pytest.approx(0.0, abs=1e-6)


def test_paper_broker_price_decrease():
    """
    Test that equity decreases when prices go down.

    Scenario:
      - Start with $100,000, allocate 100% to TEST at $100
      - Price decreases to $90

    Expected:
      - Equity decreases to $90,000
    """
    broker = PaperBroker(initial_cash=100_000)
    dt1 = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))
    dt2 = pd.Timestamp(datetime(2024, 1, 16, tzinfo=timezone.utc))

    broker.update_prices({"TEST": 100.0}, dt1)
    broker.set_target_weights({"TEST": 1.0})

    broker.update_prices({"TEST": 90.0}, dt2)

    state = broker.get_portfolio_state()

    assert state.equity == pytest.approx(90_000.0, abs=1e-6)


def test_paper_broker_rebalancing():
    """
    Test rebalancing from one symbol to another.

    Scenario:
      - Allocate 100% to TEST_A at $100
      - Rebalance to 100% TEST_B at $200

    Expected:
      - Liquidate TEST_A (sell 1000 shares)
      - Buy TEST_B (500 shares at $200)
      - Equity unchanged (ignoring costs)
    """
    broker = PaperBroker(initial_cash=100_000, slippage_bps=0.0, fee_per_trade=0.0)
    dt = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))

    # Initial: 100% TEST_A
    broker.update_prices({"TEST_A": 100.0, "TEST_B": 200.0}, dt)
    broker.set_target_weights({"TEST_A": 1.0})

    state1 = broker.get_portfolio_state()
    assert state1.positions["TEST_A"] == pytest.approx(1000.0, abs=1e-6)
    assert "TEST_B" not in state1.positions

    # Rebalance: 100% TEST_B
    broker.set_target_weights({"TEST_B": 1.0})

    state2 = broker.get_portfolio_state()
    assert "TEST_A" not in state2.positions or state2.positions.get("TEST_A", 0.0) == pytest.approx(0.0, abs=1e-6)
    assert state2.positions["TEST_B"] == pytest.approx(500.0, abs=1e-6)
    assert state2.equity == pytest.approx(100_000.0, abs=1e-6)


def test_paper_broker_slippage():
    """
    Test that slippage is applied correctly.

    Scenario:
      - Start with $100,000
      - Slippage = 10 bps (0.1%)
      - Buy TEST at $100 with 100% allocation

    Expected:
      - Target is 100% of equity = $100,000 worth
      - Trade quantity = $100,000 / $100 = 1,000 shares
      - Effective buy price = $100 * (1 + 0.001) = $100.10
      - Cash spent = 1,000 * $100.10 = $100,100
      - Cash balance = $100,000 - $100,100 = -$100 (negative due to slippage)
      - Equity = -$100 + (1000 * $100) = $99,900 (slippage cost reduces equity)
    """
    broker = PaperBroker(initial_cash=100_000, slippage_bps=10.0, fee_per_trade=0.0)
    dt = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))

    broker.update_prices({"TEST": 100.0}, dt)
    broker.set_target_weights({"TEST": 1.0})

    state = broker.get_portfolio_state()

    # Trade targets $100k worth at $100 = 1000 shares
    assert state.positions["TEST"] == pytest.approx(1000.0, abs=1e-6)

    # But we pay slippage: 1000 * $100.10 = $100,100
    # Cash = $100,000 - $100,100 = -$100
    assert state.cash == pytest.approx(-100.0, abs=1e-2)

    # Equity = cash + position_value = -100 + (1000 * 100) = $99,900
    assert state.equity == pytest.approx(99_900.0, abs=1.0)


def test_paper_broker_fees():
    """
    Test that per-trade fees are applied.

    Scenario:
      - Start with $100,000
      - Fee = $10 per trade
      - Allocate 100% to TEST at $100

    Expected:
      - One trade (buy TEST)
      - Fee = $10
      - Net cash spent = $100,000 + $10 = $100,010
      - Equity after = $100,000 - $10 = $99,990
    """
    broker = PaperBroker(initial_cash=100_000, slippage_bps=0.0, fee_per_trade=10.0)
    dt = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))

    broker.update_prices({"TEST": 100.0}, dt)
    broker.set_target_weights({"TEST": 1.0})

    state = broker.get_portfolio_state()

    # Cash should be negative by the fee amount
    # Initial cash - (position value + fee) = 100000 - (100000 + 10) = -10
    assert state.cash == pytest.approx(-10.0, abs=1e-6)

    # Position value is still $100,000 (1000 shares * $100)
    # But total equity = cash + position value = -10 + 100000 = 99990
    assert state.equity == pytest.approx(99_990.0, abs=1e-6)


def test_paper_broker_multiple_symbols():
    """
    Test allocating across multiple symbols.

    Scenario:
      - $100,000 initial cash
      - Allocate: 50% to TEST_A at $100, 30% to TEST_B at $200, 20% cash

    Expected:
      - TEST_A position: $50,000 / $100 = 500 shares
      - TEST_B position: $30,000 / $200 = 150 shares
      - Cash: $20,000
      - Equity: $100,000
    """
    broker = PaperBroker(initial_cash=100_000, slippage_bps=0.0, fee_per_trade=0.0)
    dt = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))

    broker.update_prices({"TEST_A": 100.0, "TEST_B": 200.0}, dt)
    broker.set_target_weights({"TEST_A": 0.5, "TEST_B": 0.3})

    state = broker.get_portfolio_state()

    assert state.positions["TEST_A"] == pytest.approx(500.0, abs=1e-6)
    assert state.positions["TEST_B"] == pytest.approx(150.0, abs=1e-6)
    assert state.cash == pytest.approx(20_000.0, abs=1e-6)
    assert state.equity == pytest.approx(100_000.0, abs=1e-6)


def test_paper_broker_negative_weights_shorting():
    """
    Test that negative weights create short positions.

    Scenario:
      - $100,000 initial cash
      - Allocate: -0.5 to TEST (short 50% of equity)

    Expected:
      - Short position: -500 shares (negative quantity)
      - Cash increases by sale proceeds: $100,000 + $50,000 = $150,000
      - Equity: $100,000 (cash - short liability)
    """
    broker = PaperBroker(initial_cash=100_000, slippage_bps=0.0, fee_per_trade=0.0)
    dt = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))

    broker.update_prices({"TEST": 100.0}, dt)
    broker.set_target_weights({"TEST": -0.5})

    state = broker.get_portfolio_state()

    # Short position (negative quantity)
    assert state.positions["TEST"] == pytest.approx(-500.0, abs=1e-6)

    # Cash increases from short sale proceeds
    assert state.cash == pytest.approx(150_000.0, abs=1e-6)

    # Equity = cash + position_value
    # position_value for short = -500 * 100 = -50000
    # equity = 150000 + (-50000) = 100000
    assert state.equity == pytest.approx(100_000.0, abs=1e-6)


def test_paper_broker_error_no_prices():
    """Test that setting target weights before updating prices raises an error."""
    broker = PaperBroker(initial_cash=100_000)

    with pytest.raises(ValueError, match="Cannot set target weights before updating prices"):
        broker.set_target_weights({"TEST": 1.0})


def test_paper_broker_error_missing_price():
    """Test that setting target for a symbol with no price raises an error."""
    broker = PaperBroker(initial_cash=100_000)
    dt = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))

    # Update price for TEST_A only
    broker.update_prices({"TEST_A": 100.0}, dt)

    # Try to allocate to TEST_B (no price available)
    with pytest.raises(ValueError, match="No price available for symbol 'TEST_B'"):
        broker.set_target_weights({"TEST_B": 0.5})


def test_paper_broker_zero_allocation():
    """
    Test that zero target weight liquidates positions.

    Scenario:
      - Allocate 100% to TEST
      - Then set target weight to 0.0 (liquidate)

    Expected:
      - Position liquidated (no TEST position)
      - Cash = $100,000 (back to initial)
    """
    broker = PaperBroker(initial_cash=100_000, slippage_bps=0.0, fee_per_trade=0.0)
    dt = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))

    # Initial allocation
    broker.update_prices({"TEST": 100.0}, dt)
    broker.set_target_weights({"TEST": 1.0})

    # Liquidate
    broker.set_target_weights({"TEST": 0.0})

    state = broker.get_portfolio_state()

    assert "TEST" not in state.positions or state.positions.get("TEST", 0.0) == pytest.approx(0.0, abs=1e-9)
    assert state.cash == pytest.approx(100_000.0, abs=1e-6)
    assert state.equity == pytest.approx(100_000.0, abs=1e-6)


def test_portfolio_state_exposure_metrics():
    """
    Test that exposure metrics (gross, net) are calculated correctly.

    Scenario:
      - Long $60k in TEST_A, short $30k in TEST_B

    Expected:
      - Gross exposure = $60k + $30k = $90k (sum of abs values)
      - Net exposure = $60k - $30k = $30k (long - short)
    """
    broker = PaperBroker(initial_cash=100_000, slippage_bps=0.0, fee_per_trade=0.0)
    dt = pd.Timestamp(datetime(2024, 1, 15, tzinfo=timezone.utc))

    broker.update_prices({"TEST_A": 100.0, "TEST_B": 100.0}, dt)
    broker.set_target_weights({"TEST_A": 0.6, "TEST_B": -0.3})

    state = broker.get_portfolio_state()

    # TEST_A: 600 shares * $100 = $60,000 (long)
    # TEST_B: -300 shares * $100 = -$30,000 (short)
    # Gross = $60k + $30k = $90k
    # Net = $60k - $30k = $30k

    assert state.gross_exposure == pytest.approx(90_000.0, abs=1e-6)
    assert state.net_exposure == pytest.approx(30_000.0, abs=1e-6)
