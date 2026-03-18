from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

SYMBOLS = ["RELIANCE.NS","HDFCBANK.NS","TCS.NS","INFY.NS","ICICIBANK.NS"]


# ==============================
# INDICATORS
# ==============================
def indicators(df):
    df["EMA5"] = df["Close"].ewm(span=5).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift())
        )
    )
    df["ATR"] = df["TR"].rolling(14).mean()

    df["Vol_Avg"] = df["Volume"].rolling(20).mean()

    return df


# ==============================
# TREND
# ==============================
def dow_trend(df):
    if df is None or len(df) < 10:
        return "SIDEWAYS"

    highs = df["High"].rolling(5).max()
    lows = df["Low"].rolling(5).min()

    if highs.iloc[-1] > highs.iloc[-5] and lows.iloc[-1] > lows.iloc[-5]:
        return "UPTREND"
    elif highs.iloc[-1] < highs.iloc[-5] and lows.iloc[-1] < lows.iloc[-5]:
        return "DOWNTREND"
    return "SIDEWAYS"


# ==============================
# RSI DIVERGENCE
# ==============================
def rsi_divergence(df):
    if len(df) < 10:
        return "NONE"

    if df["Close"].iloc[-1] > df["Close"].iloc[-5] and df["RSI"].iloc[-1] < df["RSI"].iloc[-5]:
        return "BEARISH"
    if df["Close"].iloc[-1] < df["Close"].iloc[-5] and df["RSI"].iloc[-1] > df["RSI"].iloc[-5]:
        return "BULLISH"
    return "NONE"


# ==============================
# SUPPORT / RESISTANCE
# ==============================
def support_resistance(df):
    return float(df["Low"].rolling(20).min().iloc[-1]), float(df["High"].rolling(20).max().iloc[-1])


# ==============================
# BREAKOUT
# ==============================
def breakout(df):
    support, resistance = support_resistance(df)
    price = df["Close"].iloc[-1]

    if price > resistance:
        return "BREAKOUT_UP"
    elif price < support:
        return "BREAKOUT_DOWN"
    return "NO_BREAKOUT"


# ==============================
# GMMA
# ==============================
def gmma(df):
    short = df["Close"].ewm(span=5).mean()
    long = df["Close"].ewm(span=30).mean()
    return "BULLISH" if short.iloc[-1] > long.iloc[-1] else "BEARISH"


# ==============================
# ML MODEL
# ==============================
def train_model(df):
    df = df.dropna().copy()

    if len(df) < 50:
        return 50.0

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    X = df[["RSI","EMA5","EMA20","ATR"]]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42)
    model.fit(X[:-1], y[:-1])

    prob = model.predict_proba(X.iloc[-1:])[0][1]
    return float(prob * 100)


# ==============================
# BACKTEST
# ==============================
def backtest(df):
    wins, total = 0, 0
    for i in range(20, len(df)-1):
        if df["RSI"].iloc[i] < 30:
            total += 1
            if df["Close"].iloc[i+1] > df["Close"].iloc[i]:
                wins += 1

    return {
        "win_rate": round((wins/total)*100,2) if total else 0,
        "trades": total
    }


# ==============================
# MULTI TIMEFRAME
# ==============================
def get_tf(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return indicators(df)


# ==============================
# ENTRY QUALITY
# ==============================
def entry_quality(tf):
    up = tf.count("UPTREND")
    down = tf.count("DOWNTREND")

    if up >= 4:
        return "HIGH_PROB_BUY"
    elif down >= 4:
        return "HIGH_PROB_SELL"
    elif up >= 3:
        return "MEDIUM_BUY"
    elif down >= 3:
        return "MEDIUM_SELL"
    return "LOW_QUALITY"


# ==============================
# ANALYZE
# ==============================
@app.get("/analyze")
def analyze(symbol: str):

    df_1h = get_tf(symbol, "1h", "1mo")
    df_4h = get_tf(symbol, "1h", "3mo")
    df_d = get_tf(symbol, "1d", "6mo")
    df_w = get_tf(symbol, "1wk", "1y")
    df_m = get_tf(symbol, "1mo", "5y")

    if df_d is None:
        return {"error": "No Data"}

    latest = df_d.iloc[-1]

    price = float(latest["Close"])
    rsi = float(latest["RSI"])
    atr = float(latest["ATR"])

    trend_1h = dow_trend(df_1h)
    trend_4h = dow_trend(df_4h)
    trend_d = dow_trend(df_d)
    trend_w = dow_trend(df_w)
    trend_m = dow_trend(df_m)

    tf_trends = [trend_1h, trend_4h, trend_d, trend_w, trend_m]

    div = rsi_divergence(df_d)
    gmma_signal = gmma(df_d)
    prob = train_model(df_d)
    breakout_signal = breakout(df_d)

    volume = latest["Volume"] > latest["Vol_Avg"]
    entry = entry_quality(tf_trends)

    # ======================
    # SCORING
    # ======================
    score = 0

    score += (tf_trends.count("UPTREND") - tf_trends.count("DOWNTREND")) * 12

    score += 10 if gmma_signal == "BULLISH" else -10

    if breakout_signal == "BREAKOUT_UP":
        score += 15
    elif breakout_signal == "BREAKOUT_DOWN":
        score -= 15

    if div == "BULLISH": score += 15
    if div == "BEARISH": score -= 15

    if rsi < 30: score += 10
    elif rsi > 70: score -= 10

    if volume: score += 5

    if prob > 65: score += 8
    elif prob < 35: score -= 8

    # ======================
    # FINAL SIGNAL
    # ======================
    if score > 45:
        signal = "STRONG BUY"
    elif score > 20:
        signal = "BUY"
    elif score < -45:
        signal = "STRONG SELL"
    elif score < -20:
        signal = "SELL"
    else:
        signal = "HOLD"

    # ======================
    # TRADE FILTER
    # ======================
    if entry == "LOW_QUALITY":
        signal = "HOLD"

    # ======================
    # FIXED STOP LOSS
    # ======================
    support, resistance = support_resistance(df_d)

    if signal in ["BUY", "STRONG BUY"]:
        stop_loss = max(price - (1.2 * atr), support)
        target = price + (price - stop_loss) * 2

    elif signal in ["SELL", "STRONG SELL"]:
        stop_loss = min(price + (1.2 * atr), resistance)
        target = price - (stop_loss - price) * 2

    else:
        stop_loss = price - atr
        target = price + atr

    stop_loss = float(round(stop_loss, 2))
    target = float(round(target, 2))

    return {
        "symbol": symbol,
        "signal": signal,
        "entry_quality": entry,
        "score": score,
        "timeframes": {
            "1H": trend_1h,
            "4H": trend_4h,
            "DAILY": trend_d,
            "WEEKLY": trend_w,
            "MONTHLY": trend_m
        },
        "breakout": breakout_signal,
        "divergence": div,
        "price": price,
        "rsi": rsi,
        "ml_probability": prob,
        "volume_strong": bool(volume),
        "risk": {
            "stop_loss": stop_loss,
            "target": target
        },
        "backtest": backtest(df_d)
    }


# ==============================
# SCANNER
# ==============================
@app.get("/scanner")
def scanner():
    results = []

    for sym in SYMBOLS:
        try:
            df = yf.download(sym, period="3mo")

            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = indicators(df)

            rsi = float(df["RSI"].iloc[-1])

            signal = "BUY" if rsi < 35 else "SELL" if rsi > 65 else "HOLD"

            results.append({
                "symbol": sym,
                "signal": signal,
                "score": float(100 - abs(50 - rsi))
            })

        except:
            continue

    return {
        "TOP_BUY": sorted(results, key=lambda x: -x["score"])[:5],
        "TOP_SELL": sorted(results, key=lambda x: x["score"])[:5]
    }
