"""Day 3: from the smile to crash probabilities.

For each expiry:
  1. Fit a smooth curve to the implied-vol smile (dropping outlier quotes).
  2. Reprice puts on a fine strike grid from the smooth smile.
  3. Risk-neutral CDF:  P(S_T < K) = (1/df) * dPut/dK   (the put-spread idea).
  4. Compare with (a) a flat-vol Black-Scholes world and (b) how often the
     S&P 500 really fell that much over the same horizon since 1993.
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from bs import bs_price

DROPS = (0.05, 0.10, 0.20)

smiles = pd.read_csv("smiles.csv", parse_dates=["expiry"])
chain_path = sorted(glob.glob("chain_SPX_*.csv"))[-1]
spot = pd.read_csv(chain_path)["spot"].iloc[0]

px = yf.download("^GSPC", start="1993-01-01", auto_adjust=True, progress=False)["Close"].squeeze()


def fit_smile(x, iv, deg=4):
    """Polynomial fit in log-moneyness, repeatedly dropping outlier quotes."""
    keep = np.ones(len(x), dtype=bool)
    for _ in range(3):
        coef = np.polyfit(x[keep], iv[keep], deg)
        resid = iv - np.polyval(coef, x)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep])))
        keep = np.abs(resid) < 4 * 1.4826 * mad + 1e-9
    return coef


def flat_vol_prob(K, F, T, sigma):
    """P(S_T < K) if volatility were the same at every strike (bell-curve world)."""
    d2 = (np.log(F / K) - 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return norm.cdf(-d2)


rows = []
fig, ax = plt.subplots(figsize=(8, 5))
d_grid = np.linspace(0.02, 0.30, 120)

for (expiry, g), color in zip(smiles.groupby("expiry"), ["tab:blue", "tab:orange", "tab:green"]):
    T, F, df = g["T"].iloc[0], g["F"].iloc[0], g["df"].iloc[0]
    x = np.log(g["strike"].to_numpy() / F)
    iv = g["iv"].to_numpy()
    coef = fit_smile(x, iv)
    atm = np.polyval(coef, 0.0)

    # Reprice puts on a fine grid from the smooth smile, then differentiate
    xs = np.linspace(x.min(), x.max(), 2000)
    K = F * np.exp(xs)
    put = bs_price(F, K, T, np.polyval(coef, xs), df, "put")
    cdf = np.gradient(put, K) / df
    bad = int((np.diff(cdf) < -1e-6).sum())

    horizon = int(round(T * 252))
    ret = (px.shift(-horizon) / px - 1).dropna()

    print(f"\n{expiry.date()}  ({horizon} trading days, ATM vol {atm:.1%}, "
          f"non-monotone CDF points: {bad})")
    for d in DROPS:
        k = (1 - d) * spot
        rn = float(np.interp(k, K, cdf))
        flat = float(flat_vol_prob(k, F, T, atm))
        hist = float((ret <= -d).mean())
        print(f"  drop >= {d:.0%}:  market-implied {rn:.2%} | flat-vol {flat:.2%} | historical {hist:.2%}")
        rows.append(dict(expiry=expiry, horizon_days=horizon, drop=d,
                         implied=rn, flat_vol=flat, historical=hist))

    rn_curve = np.interp((1 - d_grid) * spot, K, cdf)
    ax.plot(d_grid * 100, rn_curve * 100, color=color, label=f"{expiry.date()} market-implied")
    ax.plot(d_grid * 100, flat_vol_prob((1 - d_grid) * spot, F, T, atm) * 100,
            color=color, linestyle="--", alpha=0.6)

ax.set_xlabel("Index drop (%)")
ax.set_ylabel("Probability of falling at least this much (%)")
ax.set_title("Market-implied crash probability (solid) vs flat-vol bell curve (dashed)")
ax.set_yscale("log")
ax.legend()
fig.savefig("tail_probability.png", dpi=150)
pd.DataFrame(rows).to_csv("crash_probs.csv", index=False)
print("\nSaved tail_probability.png and crash_probs.csv")
plt.show()
