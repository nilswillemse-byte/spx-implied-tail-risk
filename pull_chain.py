"""Day 1: pull option chains, clean them, save a CSV snapshot.

Run during US market hours (bids are often 0 after close).
Setup: pip install yfinance pandas
"""
import pandas as pd
import yfinance as yf

TICKER = "^SPX"            # European-style. Fallback: "SPY" (American-style)
TARGET_DAYS = [30, 60, 90]  # roughly 1, 2, 3 months out

tk = yf.Ticker(TICKER)
spot = float(tk.history(period="5d")["Close"].iloc[-1])
today = pd.Timestamp.now().normalize()

# Pick the listed expiry closest to each target horizon
expiries = pd.to_datetime(tk.options)
chosen = []
for d in TARGET_DAYS:
    target = today + pd.Timedelta(days=d)
    e = expiries[abs(expiries - target).argmin()]
    if e not in chosen:
        chosen.append(e)

frames = []
for e in chosen:
    ch = tk.option_chain(e.strftime("%Y-%m-%d"))
    for kind, df in (("call", ch.calls), ("put", ch.puts)):
        df = df[["strike", "bid", "ask", "volume", "openInterest"]].copy()
        df["type"] = kind
        df["expiry"] = e
        frames.append(df)

chain = pd.concat(frames, ignore_index=True)
chain["T"] = (chain["expiry"] - today).dt.days / 365
chain["mid"] = (chain["bid"] + chain["ask"]) / 2
chain["spot"] = spot

# Cleaning: drop zero-bid and crossed quotes
chain = chain[(chain["bid"] > 0) & (chain["ask"] >= chain["bid"])]

out = f"chain_{TICKER.strip('^')}_{today:%Y%m%d}.csv"
chain.to_csv(out, index=False)

print(f"Spot: {spot:.2f}")
print(chain.groupby(["expiry", "type"]).size())
print(f"Saved {out}")

