"""Black-Scholes price and implied volatility, written in terms of the
forward price F and discount factor df (both come from put-call parity on Day 2).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(F, K, T, sigma, df, kind):
    """European option price. kind is 'call' or 'put'."""
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, df, kind):
    """Find the volatility that makes the formula match a market price."""
    f = lambda s: bs_price(F, K, T, s, df, kind) - price
    try:
        return brentq(f, 1e-4, 5.0)
    except ValueError:  # price outside the possible range (bad quote)
        return np.nan
