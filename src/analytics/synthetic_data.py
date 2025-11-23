"""
Synthetic market data generators for testing and validation.

This module provides functions to generate synthetic price paths using
stochastic processes commonly used in quantitative finance:
  - Geometric Brownian Motion (GBM): for trending, compounding equity-like behavior
  - Ornstein-Uhlenbeck (OU): for mean-reverting, choppy/sideways behavior

These generators are invaluable for:
  - Validating backtest engine and risk metrics under controlled conditions
  - Stress-testing strategies against known regimes (pure trend vs pure mean-reversion)
  - Teaching: understanding how different processes behave and affect metrics
"""

import numpy as np
import pandas as pd


def generate_gbm_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    n_steps: int,
    dt: float = 1 / 252,
    seed: int | None = None,
) -> pd.Series:
    """
    Generate a price path using Geometric Brownian Motion (GBM).

    **Conceptual**: GBM is the classic model for equity prices, exhibiting
    trending and compounding behavior. It assumes log returns are normally
    distributed and independent, with constant drift (μ) and volatility (σ).
    This is the foundation of Black-Scholes option pricing and is a reasonable
    first-order approximation for stock/ETF behavior. For traders, GBM simulates
    "normal" market conditions with a steady upward (or downward) drift plus noise.

    **Mathematical**: The continuous-time GBM process is:
        dS = μ * S * dt + σ * S * dW
    where μ is drift, σ is volatility, and dW is a Wiener process (Brownian motion).

    The discrete update (Euler-Maruyama scheme) for each time step is:
        S_{t+1} = S_t * exp((μ - 0.5 * σ^2) * dt + σ * sqrt(dt) * Z_t)
    where Z_t ~ N(0, 1) is a standard normal random variable.

    The (μ - 0.5 * σ^2) term is the drift correction (Itô's lemma adjustment)
    ensuring the expected log price grows at rate μ.

    **Functionally**:
    - Input:
        - initial_price: Starting price (e.g., 100.0).
        - drift: Annualized drift rate μ (e.g., 0.10 for 10% expected return/year).
        - volatility: Annualized volatility σ (e.g., 0.20 for 20% annual vol).
        - n_steps: Number of time steps to generate.
        - dt: Time increment per step (default 1/252 for daily steps).
        - seed: Random seed for reproducibility (None for random).
    - Output: pandas Series of length (n_steps + 1) including initial price,
      indexed by step number (0, 1, 2, ..., n_steps).
    - Use case: Generate test data for returns/volatility metrics and path-dependent
      logic under controlled drift/vol regimes. Compare strategies on trending data.

    **Interpretation**:
    - drift > 0: upward trending market (bullish).
    - drift < 0: downward trending market (bearish).
    - volatility controls "wiggliness" of the path.
    - Setting volatility = 0 yields a deterministic exponential path.

    **Edge cases**:
    - n_steps = 0 → returns just [initial_price].
    - Very high volatility → extreme price swings, potential for near-zero prices.
    - Negative initial_price is undefined (prices should be positive).

    Args:
        initial_price: Starting price of the asset (must be positive).
        drift: Annualized drift rate (e.g., 0.10 for 10%).
        volatility: Annualized volatility (e.g., 0.20 for 20%).
        n_steps: Number of steps to simulate.
        dt: Time increment per step (1/252 for daily).
        seed: Random seed for reproducibility.

    Returns:
        pandas Series of prices indexed by step number.
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Initialize price array: first element is initial price
    prices = np.zeros(n_steps + 1)
    prices[0] = initial_price

    # Generate n_steps of standard normal random variables for the Wiener process
    # Z_t ~ N(0, 1)
    Z = np.random.standard_normal(n_steps)

    # Compute the GBM update for each step
    # S_{t+1} = S_t * exp((μ - 0.5*σ^2)*dt + σ*sqrt(dt)*Z_t)
    # The drift term (μ - 0.5*σ^2)*dt accounts for Itô correction
    drift_term = (drift - 0.5 * volatility**2) * dt
    diffusion_term = volatility * np.sqrt(dt) * Z

    # Loop through each step and update price
    for t in range(n_steps):
        # Compute the multiplicative factor: exp(drift_term + diffusion_term)
        factor = np.exp(drift_term + diffusion_term[t])
        # Update price: S_{t+1} = S_t * factor
        prices[t + 1] = prices[t] * factor

    # Return as pandas Series indexed by step number
    return pd.Series(prices, index=range(n_steps + 1), name='price')


def generate_ou_paths(
    initial_price: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
    n_steps: int,
    dt: float = 1 / 252,
    seed: int | None = None,
) -> pd.Series:
    """
    Generate a price path using an Ornstein-Uhlenbeck (OU) mean-reverting process.

    **Conceptual**: The OU process models mean reversion, where prices are pulled
    back toward a long-term average. This mimics choppy, sideways, or range-bound
    markets rather than trending markets. It's useful for modeling commodities,
    interest rates, volatility indices, or any asset that tends to revert to a
    mean rather than drift indefinitely. For traders, OU simulates challenging
    conditions for trend-following strategies (whipsaws, false breakouts).

    **Mathematical**: The continuous-time OU process is:
        dX = κ * (θ - X) * dt + σ * dW
    where:
      - κ (kappa) is mean reversion speed: higher κ → faster reversion.
      - θ (theta) is the long-term mean: price is pulled toward θ.
      - σ (sigma) is volatility: controls noise around the mean.
      - dW is a Wiener process.

    The discrete update (Euler-Maruyama scheme) is:
        X_{t+1} = X_t + κ * (θ - X_t) * dt + σ * sqrt(dt) * Z_t
    where Z_t ~ N(0, 1).

    **Functionally**:
    - Input:
        - initial_price: Starting price (e.g., 100.0).
        - mean_reversion_speed (κ): How quickly price reverts to mean (e.g., 2.0).
        - long_term_mean (θ): The equilibrium price level (e.g., 100.0).
        - volatility (σ): Annualized volatility controlling noise (e.g., 0.15).
        - n_steps: Number of time steps.
        - dt: Time increment per step (default 1/252 for daily).
        - seed: Random seed for reproducibility.
    - Output: pandas Series of length (n_steps + 1) indexed by step number.
    - Use case: Stress-test mean-reversion vs trend metrics and verify drawdown/vol
      calculations in non-trending, choppy conditions. Validate that trend strategies
      correctly reduce exposure or move to cash in sideways markets.

    **Interpretation**:
    - High κ: strong mean reversion (tight range around θ).
    - Low κ: weak mean reversion (more drift-like).
    - volatility controls how much the price oscillates around θ.
    - Setting volatility = 0 → deterministic exponential decay/growth toward θ.

    **Edge cases**:
    - n_steps = 0 → returns just [initial_price].
    - Very high κ → price snaps quickly to long_term_mean.
    - Very low κ → process behaves more like random walk.

    Args:
        initial_price: Starting price of the asset.
        mean_reversion_speed: Speed of reversion to long-term mean (κ, kappa).
        long_term_mean: Equilibrium price level (θ, theta).
        volatility: Annualized volatility (σ, sigma).
        n_steps: Number of steps to simulate.
        dt: Time increment per step (1/252 for daily).
        seed: Random seed for reproducibility.

    Returns:
        pandas Series of prices indexed by step number.
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Initialize price array: first element is initial price
    prices = np.zeros(n_steps + 1)
    prices[0] = initial_price

    # Generate n_steps of standard normal random variables
    # Z_t ~ N(0, 1)
    Z = np.random.standard_normal(n_steps)

    # Compute the OU update for each step
    # X_{t+1} = X_t + κ*(θ - X_t)*dt + σ*sqrt(dt)*Z_t
    for t in range(n_steps):
        # Mean reversion term: κ*(θ - X_t)*dt pulls price toward long_term_mean
        mean_reversion_term = mean_reversion_speed * (long_term_mean - prices[t]) * dt

        # Diffusion term: σ*sqrt(dt)*Z_t adds noise
        diffusion_term = volatility * np.sqrt(dt) * Z[t]

        # Update price: X_{t+1} = X_t + mean_reversion_term + diffusion_term
        prices[t + 1] = prices[t] + mean_reversion_term + diffusion_term

    # Return as pandas Series indexed by step number
    return pd.Series(prices, index=range(n_steps + 1), name='price')
