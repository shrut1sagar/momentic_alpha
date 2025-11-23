"""
Tests for src/analytics/synthetic_data.py

These tests verify that synthetic data generators (GBM, OU) produce series of
expected length, honor reproducibility via seeds, and exhibit expected behavior
in trivial cases (zero volatility, strong mean reversion, etc.).
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.synthetic_data import (
    generate_gbm_paths,
    generate_ou_paths,
)


def test_generate_gbm_paths_correct_length():
    """Test that GBM generates a series of correct length."""
    n_steps = 100
    prices = generate_gbm_paths(
        initial_price=100.0,
        drift=0.10,
        volatility=0.20,
        n_steps=n_steps,
        seed=42,
    )

    # Should have n_steps + 1 values (including initial price)
    assert len(prices) == n_steps + 1


def test_generate_gbm_paths_initial_price():
    """Test that GBM starts at the specified initial price."""
    initial_price = 150.0
    prices = generate_gbm_paths(
        initial_price=initial_price,
        drift=0.05,
        volatility=0.15,
        n_steps=50,
        seed=42,
    )

    # First price should be initial_price
    assert prices.iloc[0] == initial_price


def test_generate_gbm_paths_zero_volatility():
    """Test that GBM with zero volatility produces deterministic exponential growth."""
    # With volatility = 0, path should be deterministic: S_t = S_0 * exp(μ * t)
    initial_price = 100.0
    drift = 0.10
    volatility = 0.0
    n_steps = 252
    dt = 1 / 252

    prices = generate_gbm_paths(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        n_steps=n_steps,
        dt=dt,
        seed=42,
    )

    # Expected final price: 100 * exp(0.10 * 1) ≈ 110.52 (after 1 year = 252 steps)
    # Using discrete formula: S_t = S_0 * exp((μ - 0.5*σ^2)*dt*n) = S_0 * exp(μ * 1) when σ=0
    expected_final = initial_price * np.exp(drift * n_steps * dt)
    assert np.isclose(prices.iloc[-1], expected_final, rtol=1e-6)


def test_generate_gbm_paths_seed_reproducibility():
    """Test that GBM with same seed produces identical paths."""
    params = {
        'initial_price': 100.0,
        'drift': 0.10,
        'volatility': 0.20,
        'n_steps': 100,
        'dt': 1 / 252,
    }

    prices1 = generate_gbm_paths(**params, seed=42)
    prices2 = generate_gbm_paths(**params, seed=42)

    # Should be identical
    pd.testing.assert_series_equal(prices1, prices2)


def test_generate_gbm_paths_different_seeds():
    """Test that GBM with different seeds produces different paths."""
    params = {
        'initial_price': 100.0,
        'drift': 0.10,
        'volatility': 0.20,
        'n_steps': 100,
        'dt': 1 / 252,
    }

    prices1 = generate_gbm_paths(**params, seed=42)
    prices2 = generate_gbm_paths(**params, seed=99)

    # Should be different (with high probability)
    assert not prices1.equals(prices2)


def test_generate_gbm_paths_positive_drift_trend():
    """Test that GBM with positive drift tends upward on average."""
    # Generate many paths and check average final price > initial
    initial_price = 100.0
    drift = 0.20  # 20% drift
    n_steps = 252
    dt = 1 / 252

    final_prices = []
    for seed in range(20):
        prices = generate_gbm_paths(
            initial_price=initial_price,
            drift=drift,
            volatility=0.15,
            n_steps=n_steps,
            dt=dt,
            seed=seed,
        )
        final_prices.append(prices.iloc[-1])

    # Average final price should be > initial price (positive drift)
    avg_final = np.mean(final_prices)
    assert avg_final > initial_price


def test_generate_ou_paths_correct_length():
    """Test that OU generates a series of correct length."""
    n_steps = 100
    prices = generate_ou_paths(
        initial_price=100.0,
        mean_reversion_speed=2.0,
        long_term_mean=100.0,
        volatility=0.10,
        n_steps=n_steps,
        seed=42,
    )

    # Should have n_steps + 1 values
    assert len(prices) == n_steps + 1


def test_generate_ou_paths_initial_price():
    """Test that OU starts at the specified initial price."""
    initial_price = 120.0
    prices = generate_ou_paths(
        initial_price=initial_price,
        mean_reversion_speed=1.5,
        long_term_mean=100.0,
        volatility=0.05,
        n_steps=50,
        seed=42,
    )

    # First price should be initial_price
    assert prices.iloc[0] == initial_price


def test_generate_ou_paths_zero_volatility_converges_to_mean():
    """Test that OU with zero volatility converges deterministically to long-term mean."""
    # With σ = 0, OU is deterministic: X_t → θ exponentially
    initial_price = 150.0
    long_term_mean = 100.0
    mean_reversion_speed = 5.0  # Fast reversion
    volatility = 0.0
    n_steps = 500
    dt = 1 / 252

    prices = generate_ou_paths(
        initial_price=initial_price,
        mean_reversion_speed=mean_reversion_speed,
        long_term_mean=long_term_mean,
        volatility=volatility,
        n_steps=n_steps,
        dt=dt,
        seed=42,
    )

    # After many steps with strong mean reversion, should be very close to long_term_mean
    # Deterministic solution: X_t = θ + (X_0 - θ) * exp(-κ*t)
    # After t = n_steps * dt, X_t ≈ θ (if κ is large and t is large)
    final_price = prices.iloc[-1]
    # Use slightly relaxed tolerance to account for Euler discretization error
    assert np.isclose(final_price, long_term_mean, atol=1e-2)


def test_generate_ou_paths_strong_mean_reversion_stays_near_mean():
    """Test that OU with strong mean reversion stays close to long-term mean."""
    # High κ → tight oscillation around θ
    long_term_mean = 100.0
    prices = generate_ou_paths(
        initial_price=100.0,
        mean_reversion_speed=10.0,  # Very strong reversion
        long_term_mean=long_term_mean,
        volatility=0.05,  # Low noise
        n_steps=252,
        dt=1 / 252,
        seed=42,
    )

    # Most prices should be within a small range of long_term_mean
    # (not a strict test, but qualitative check)
    mean_price = prices.mean()
    assert np.abs(mean_price - long_term_mean) < 10.0  # Within 10 units on average


def test_generate_ou_paths_seed_reproducibility():
    """Test that OU with same seed produces identical paths."""
    params = {
        'initial_price': 100.0,
        'mean_reversion_speed': 2.0,
        'long_term_mean': 100.0,
        'volatility': 0.15,
        'n_steps': 100,
        'dt': 1 / 252,
    }

    prices1 = generate_ou_paths(**params, seed=42)
    prices2 = generate_ou_paths(**params, seed=42)

    # Should be identical
    pd.testing.assert_series_equal(prices1, prices2)


def test_generate_ou_paths_different_seeds():
    """Test that OU with different seeds produces different paths."""
    params = {
        'initial_price': 100.0,
        'mean_reversion_speed': 2.0,
        'long_term_mean': 100.0,
        'volatility': 0.15,
        'n_steps': 100,
        'dt': 1 / 252,
    }

    prices1 = generate_ou_paths(**params, seed=42)
    prices2 = generate_ou_paths(**params, seed=99)

    # Should be different (with high probability)
    assert not prices1.equals(prices2)


def test_generate_ou_paths_mean_reversion_from_above():
    """Test that OU reverts downward when starting above long-term mean."""
    # Start above mean, should drift downward on average
    initial_price = 150.0
    long_term_mean = 100.0
    prices = generate_ou_paths(
        initial_price=initial_price,
        mean_reversion_speed=3.0,
        long_term_mean=long_term_mean,
        volatility=0.05,
        n_steps=252,
        dt=1 / 252,
        seed=42,
    )

    # Final price should be closer to long_term_mean than initial_price
    final_price = prices.iloc[-1]
    initial_distance = abs(initial_price - long_term_mean)
    final_distance = abs(final_price - long_term_mean)
    assert final_distance < initial_distance


def test_generate_ou_paths_mean_reversion_from_below():
    """Test that OU reverts upward when starting below long-term mean."""
    # Start below mean, should drift upward on average
    initial_price = 50.0
    long_term_mean = 100.0
    prices = generate_ou_paths(
        initial_price=initial_price,
        mean_reversion_speed=3.0,
        long_term_mean=long_term_mean,
        volatility=0.05,
        n_steps=252,
        dt=1 / 252,
        seed=42,
    )

    # Final price should be closer to long_term_mean than initial_price
    final_price = prices.iloc[-1]
    initial_distance = abs(initial_price - long_term_mean)
    final_distance = abs(final_price - long_term_mean)
    assert final_distance < initial_distance
