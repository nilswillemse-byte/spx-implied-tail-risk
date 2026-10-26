import yfinance as yf
#trading days p month#
HORIZON = 21
px = yf.download("^GSPC", start="1993-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
ret = (px.shift(-HORIZON) / px - 1).dropna()
print(f"Windows tested: {len(ret)}")
for drop in (-0.05, -0.10, -0.20):
    freq = (ret <= drop).mean()
    print(f"P(fall of {abs(drop):.0%} or more in a month): {freq:.2%}")
