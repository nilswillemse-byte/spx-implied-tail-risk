"""Day 2: from the saved option chain to the implied-volatility smile.

For each expiry:
  1. Regress (call - put) on strike. Put-call parity says
        C - P = df * (F - K)   ->   slope = -df,  intercept = df * F
     so one regression gives the discount factor df and the forward F.
  2. Use out-of-the-money options only (puts below F, calls above F).
  3. Solve for each option's implied volatility and plot the smile.
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bs import implied_vol

path = sorted(glob.glob("chain_SPX_*.csv"))[-1]
print("Using", path)
chain = pd.read_csv(path, parse_dates=["expiry"])
spot = chain["spot"].iloc[0]

all_smiles = []
fig, ax = plt.subplots(figsize=(8, 5))

for expiry, g in chain.groupby("expiry"):
    T = g["T"].iloc[0]
    calls = g[g["type"] == "call"].drop_duplicates("strike").set_index("strike")["mid"]
    puts = g[g["type"] == "put"].drop_duplicates("strike").set_index("strike")["mid"]

    # Step 1: parity regression on strikes near the money
    both = calls.index.intersection(puts.index)
    near = both[(both > 0.95 * spot) & (both < 1.05 * spot)]
    diff = calls[near] - puts[near]
    slope, intercept = np.polyfit(near.to_numpy(), diff.to_numpy(), 1)
    df = -slope
    F = intercept / df
    rate = -np.log(df) / T

    # Step 2 and 3: out-of-the-money options -> implied vols
    rows = []
    for K, price in puts.items():
        if K < F:
            rows.append((K, implied_vol(price, F, K, T, df, "put")))
    for K, price in calls.items():
        if K >= F:
            rows.append((K, implied_vol(price, F, K, T, df, "call")))

    smile = pd.DataFrame(rows, columns=["strike", "iv"]).dropna().sort_values("strike")
    smile = smile[(smile["strike"] > 0.7 * spot) & (smile["strike"] < 1.2 * spot) & (smile["iv"] < 1.5)]

    atm_iv = np.interp(F, smile["strike"], smile["iv"])
    print(f"{expiry.date()}  T={T:.3f}y  forward={F:,.1f}  df={df:.4f}  "
          f"implied rate={rate:.2%}  ATM vol={atm_iv:.1%}  strikes used={len(smile)}")

    smile = smile.assign(expiry=expiry, T=T, F=F, df=df)
    all_smiles.append(smile)
    ax.plot(smile["strike"] / F, smile["iv"] * 100, ".", label=f"{expiry.date()}")

ax.set_xlabel("Strike / forward")
ax.set_ylabel("Implied volatility (%)")
ax.set_title("SPX implied volatility smile")
ax.legend()
fig.savefig("smile.png", dpi=150)

pd.concat(all_smiles).to_csv("smiles.csv", index=False)
print("Saved smile.png and smiles.csv")
plt.show()
