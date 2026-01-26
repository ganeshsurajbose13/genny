import yfinance as yf

df = yf.download("RELIANCE.NS", start="2023-01-01", end="2024-01-01", interval="1d")
print(df.head())
