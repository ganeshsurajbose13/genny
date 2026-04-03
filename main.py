from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import time

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")


# ==============================
# STOCK LIST
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
# FETCH DATA
# ==============================
def get_tf(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)

        if df is None or df.empty or len(df) < 50:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()
        return indicators(df)

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


# ==============================
# INDICATORS
# ==============================
def indicators(df):

    df["EMA5"] = df["Close"].ewm(span=5).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(abs(df["High"] - df["Close"].shift()),
                   abs(df["Low"] - df["Close"].shift()))
    )
    df["ATR"] = df["TR"].rolling(14).mean()

    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

    if "Volume" in df.columns:
        df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)
        df["Vol_Avg"] = df["Volume"].rolling(20).mean()
    else:
        df["VWAP"] = df["Close"]
        df["Vol_Avg"] = 0

    return df


# ==============================
# TREND
# ==============================
def trend_score(trend):
    return 1 if trend == "UPTREND" else -1 if trend == "DOWNTREND" else 0


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
# HELPERS
# ==============================
def macd_signal(df):
    return "BULLISH" if df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1] else "BEARISH"


def vwap_signal(df):
    return "BULLISH" if df["Close"].iloc[-1] > df["VWAP"].iloc[-1] else "BEARISH"


def breakout(df):
    high = df["High"].rolling(20).max().iloc[-2]
    low = df["Low"].rolling(20).min().iloc[-2]
    price = df["Close"].iloc[-1]

    if price > high:
        return "BREAKOUT_UP"
    elif price < low:
        return "BREAKOUT_DOWN"
    return "NO_BREAKOUT"


# ==============================
# 🔥 ML MODEL (IMPROVED)
# ==============================
def train_model(df):
    df = df.dropna().copy()

    if len(df) < 100:
        return 50.0

    df["RETURN"] = df["Close"].pct_change()
    df["EMA_DIFF"] = df["EMA5"] - df["EMA20"]
    df["PRICE_VWAP_DIFF"] = df["Close"] - df["VWAP"]
    df["ATR_PCT"] = df["ATR"] / df["Close"]
    df["VOL_SPIKE"] = df["Volume"] / (df["Vol_Avg"] + 1e-9)

    df["target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)

    features = ["RSI","EMA_DIFF","ATR_PCT","MACD","PRICE_VWAP_DIFF","VOL_SPIKE"]

    df = df.dropna()
    if len(df) < 60:
        return 50.0

    X = df[features]
    y = df["target"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        random_state=42
    )

    model.fit(X_scaled[:-1], y[:-1])

    prob = model.predict_proba([X_scaled[-1]])[0][1] * 100

    prob = (prob * 0.75) + 10
    return round(prob, 2)


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

    t1 = dow_trend(df_1h)
    t4 = dow_trend(df_4h)
    td = dow_trend(df_d)
    tw = dow_trend(df_w)
    tm = dow_trend(df_m)

    score = (
        trend_score(t1)*5 +
        trend_score(t4)*8 +
        trend_score(td)*15 +
        trend_score(tw)*20 +
        trend_score(tm)*25
    )

    macd_sig = macd_signal(df_d)
    vwap_sig = vwap_signal(df_d)
    prob = train_model(df_d)
    brk = breakout(df_d)

    volume = latest["Volume"] > latest["Vol_Avg"]

    score += 12 if macd_sig == "BULLISH" else -12
    score += 8 if vwap_sig == "BULLISH" else -8
    score += 6 if volume else 0

    if brk == "BREAKOUT_UP":
        score += 20
    elif brk == "BREAKOUT_DOWN":
        score -= 20

    if rsi < 30: score += 10
    elif rsi > 70: score -= 10

    if prob > 70:
        score += 15
    elif prob > 60:
        score += 8
    elif prob < 30:
        score -= 15
    elif prob < 40:
        score -= 8

    if score > 50:
        signal = "STRONG BUY"
    elif score > 20:
        signal = "BUY"
    elif score < -50:
        signal = "STRONG SELL"
    elif score < -20:
        signal = "SELL"
    else:
        signal = "HOLD"

    risk = {}

    if signal in ["BUY","STRONG BUY","HOLD"]:
        risk["stop_loss_small"] = round(price - 1 * atr,2)
        risk["stop_loss_big"]   = round(price - 2 * atr,2)
        risk["target_small"]    = round(price + 2 * atr,2)
        risk["target_big"]      = round(price + 4 * atr,2)
    else:
        risk["stop_loss_small"] = round(price + 1 * atr,2)
        risk["stop_loss_big"]   = round(price + 2 * atr,2)
        risk["target_small"]    = round(price - 2 * atr,2)
        risk["target_big"]      = round(price - 4 * atr,2)

    return {
        "symbol": symbol,
        "signal": signal,
        "score": round(score,2),
        "price": price,
        "rsi": round(rsi,2),
        "ml_probability": prob,
        "timeframes": {
            "1H": t1,
            "4H": t4,
            "DAILY": td,
            "WEEKLY": tw,
            "MONTHLY": tm
        },
        "breakout": brk,
        "macd": macd_sig,
        "vwap": vwap_sig,
        "volume_strong": bool(volume),
        "risk": risk
    }


# ==============================
# 🔥 FINAL SCANNER (FIXED)
# ==============================
@app.get("/scanner")
def scanner():
    results = []

    for sym in SYMBOLS[:40]:
        try:
            df = yf.download(sym, period="6mo", progress=False)

            if df is None or df.empty or len(df) < 50:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = indicators(df)
            df = df.dropna()

            if len(df) < 30:
                continue

            latest = df.iloc[-1]

            score = 0

            score += 20 if latest["Close"] > latest["EMA20"] else -20
            score += 10 if latest["RSI"] < 35 else -10 if latest["RSI"] > 65 else 0
            score += 15 if latest["MACD"] > latest["MACD_SIGNAL"] else -15
            score += 10 if latest["Close"] > latest["VWAP"] else -10

            signal = "BUY" if score > 20 else "SELL" if score < -20 else "HOLD"

            results.append({
                "symbol": sym,
                "signal": signal,
                "score": score
            })

            time.sleep(0.1)

        except Exception as e:
            print(f"{sym} error: {e}")
            continue

    return {
        "TOP_BUY": sorted(results, key=lambda x: -x["score"])[:5],
        "TOP_SELL": sorted(results, key=lambda x: x["score"])[:5],
        "TOTAL": len(results)
    }
