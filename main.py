from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

# ==============================
# STOCK LIST (EXPANDED)
# ==============================
SYMBOLS = sorted(list(set([
    "RELIANCE.NS","HDFCBANK.NS","BHARTIARTL.NS","TCS.NS","ICICIBANK.NS",
    "SBIN.NS","INFY.NS","BAJFINANCE.NS","ITC.NS","HINDUNILVR.NS",
    "LT.NS","MARUTI.NS","AXISBANK.NS","TECHM.NS","SUNPHARMA.NS",
    "NESTLEIND.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","POWERGRID.NS",
    "HCLTECH.NS","ONGC.NS","TATACONSUM.NS","ADANIENT.NS","BPCL.NS",
    "COALINDIA.NS","EICHERMOT.NS","DIVISLAB.NS","JSWSTEEL.NS","NTPC.NS",
    "GRASIM.NS","BRITANNIA.NS","CIPLA.NS","SHREECEM.NS","HDFCLIFE.NS",
    "ICICIPRULI.NS","UPL.NS","SBILIFE.NS","ICICIGI.NS","BAJAJFINSV.NS",
    "ADANIGREEN.NS","INDUSINDBK.NS","PIDILITIND.NS","ADANIPORTS.NS",
    "GAIL.NS","M&M.NS","ZEEL.NS","DLF.NS","TATAPOWER.NS","VEDL.NS",
    "COFORGE.NS","BEL.NS","TATAELXSI.NS","COLPAL.NS","HINDALCO.NS",
    "CERA.NS","AUROPHARMA.NS","MUTHOOTFIN.NS","TORNTPOWER.NS",
    "BANDHANBNK.NS","LUPIN.NS","SRF.NS","NMDC.NS","AMBUJACEM.NS",
    "HAL.NS","SIEMENS.NS","INDIGO.NS","M&MFIN.NS","BOSCHLTD.NS",
    "PAGEIND.NS","BHEL.NS","TATASTEEL.NS"
])))

# ==============================
# SAFE DATA FETCH
# ==============================
def get_tf(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        if df.empty:
            return None

        return indicators(df)

    except Exception as e:
        print(f"Error {symbol}: {e}")
        return None


# ==============================
# INDICATORS
# ==============================
def indicators(df):
    df["EMA5"] = df["Close"].ewm(span=5).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift())
        )
    )
    df["ATR"] = df["TR"].rolling(14).mean()

    # MACD
    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

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
# MACD SIGNAL
# ==============================
def macd_signal(df):
    if df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]:
        return "BULLISH"
    return "BEARISH"


# ==============================
# SUPPORT / RESISTANCE
# ==============================
def support_resistance(df):
    support = df["Low"].rolling(50).min().iloc[-1]
    resistance = df["High"].rolling(50).max().iloc[-1]
    return float(support), float(resistance)


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
# ML MODEL
# ==============================
def train_model(df):
    df = df.dropna().copy()

    if len(df) < 50:
        return 50.0

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    X = df[["RSI","EMA5","EMA20","ATR","MACD"]]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42)
    model.fit(X[:-1], y[:-1])

    prob = model.predict_proba(X.iloc[-1:])[0][1]
    return float(prob * 100)


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

    macd_sig = macd_signal(df_d)
    prob = train_model(df_d)
    breakout_signal = breakout(df_d)

    volume = latest["Volume"] > latest["Vol_Avg"]

    # ======================
    # SCORING (UPGRADED)
    # ======================
    score = 0

    score += (tf_trends.count("UPTREND") - tf_trends.count("DOWNTREND")) * 12
    score += 12 if macd_sig == "BULLISH" else -12

    if breakout_signal == "BREAKOUT_UP":
        score += 15
    elif breakout_signal == "BREAKOUT_DOWN":
        score -= 15

    if rsi < 30: score += 10
    elif rsi > 70: score -= 10

    if volume: score += 6

    if prob > 65: score += 10
    elif prob < 35: score -= 10

    # ======================
    # SIGNAL
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
    # SMART SL / TARGET
    # ======================
    support, resistance = support_resistance(df_d)

    if signal in ["BUY","STRONG BUY"]:
        stop_loss = support
        target = resistance

    elif signal in ["SELL","STRONG SELL"]:
        stop_loss = resistance
        target = support

    else:
        stop_loss = price - atr
        target = price + atr

    return {
        "symbol": symbol,
        "signal": signal,
        "score": score,
        "price": price,
        "rsi": rsi,
        "ml_probability": prob,
        "timeframes": {
            "1H": trend_1h,
            "4H": trend_4h,
            "DAILY": trend_d,
            "WEEKLY": trend_w,
            "MONTHLY": trend_m
        },
        "breakout": breakout_signal,
        "macd": macd_sig,
        "volume_strong": bool(volume),
        "risk": {
            "stop_loss": round(stop_loss,2),
            "target": round(target,2)
        }
    }


# ==============================
# SCANNER (FAST)
# ==============================
@app.get("/scanner")
def scanner():
    results = []

    for sym in SYMBOLS:
        try:
            df = yf.download(sym, period="3mo", progress=False)

            if df.empty:
                continue

            df = indicators(df)
            rsi = float(df["RSI"].iloc[-1])

            signal = "BUY" if rsi < 35 else "SELL" if rsi > 65 else "HOLD"

            results.append({
                "symbol": sym,
                "signal": signal,
                "score": float(100 - abs(50 - rsi))
            })

            time.sleep(0.2)

        except:
            continue

    return {
        "TOP_BUY": sorted(results, key=lambda x: -x["score"])[:5],
        "TOP_SELL": sorted(results, key=lambda x: x["score"])[:5]
    }
