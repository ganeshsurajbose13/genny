"""
Train LSTM models for all symbols. Run once, or when you want to refresh.
Usage:
    python train_lstm.py              # train all symbols
    python train_lstm.py RELIANCE.NS  # train one symbol
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# reuse the fetch function and symbol list from main.py
from main import fetch_yahoo_df, SYMBOLS, MODEL_DIR


def train_one(symbol, epochs=20, seq_len=60):
    print(f"[{symbol}] fetching data...")
    df = fetch_yahoo_df(symbol, interval="1d", range_period="2y")
    if df is None or len(df) < seq_len + 20:
        print(f"[{symbol}] SKIP — not enough data")
        return False

    close = df["Close"].dropna()
    if len(close) < 120:
        print(f"[{symbol}] SKIP — only {len(close)} bars")
        return False

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.values.reshape(-1, 1))

    X, y = [], []
    # Predict NEXT bar (i+1) from current window
    for i in range(seq_len, len(scaled) - 1):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i + 1, 0])

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)

    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(seq_len, 1)),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    out_path = f"{MODEL_DIR}/{symbol.replace('.', '_')}.keras"
    model.save(out_path)
    print(f"[{symbol}] OK — saved {out_path}  (samples={len(X)})")
    return True


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = SYMBOLS

    ok, fail = 0, 0
    for sym in targets:
        try:
            if train_one(sym):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"[{sym}] ERROR — {type(e).__name__}: {e}")
            fail += 1

    print(f"\nDONE — trained {ok}, skipped/errored {fail}")
