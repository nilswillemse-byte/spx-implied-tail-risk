import numpy as np
from scipy.stats import norm

from bs import bs_price, implied_vol


def test_implied_vol_roundtrip():
    """Price an option with a known vol, then recover that vol."""
    F, K, T, df, sigma = 5000, 4800, 0.25, 0.99, 0.20
    for kind in ("call", "put"):
        price = bs_price(F, K, T, sigma, df, kind)
        assert abs(implied_vol(price, F, K, T, df, kind) - sigma) < 1e-6


def test_crash_probability_recovered():
    """Put-spread formula must reproduce the known Black-Scholes probability."""
    F, T, df, sigma = 5000, 0.25, 0.99, 0.20
    K, h = 4500, 1.0
    p_hi = bs_price(F, K + h, T, sigma, df, "put")
    p_lo = bs_price(F, K - h, T, sigma, df, "put")
    estimate = (p_hi - p_lo) / (2 * h) / df  # e^{rT} = 1/df
    d2 = (np.log(F / K) - 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    assert abs(estimate - norm.cdf(-d2)) < 1e-4
