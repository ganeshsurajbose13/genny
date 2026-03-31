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
# FETCH
# ==============================
def get_df(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        if len(df) < 50:
            return None

        return indicators(df)

    except:
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

    # VWAP safe
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)

    df["Vol_Avg"] = df["Volume"].rolling(20).mean()

    return df


# ==============================
# TREND
# ==============================
def trend(df):
    if df is None or len(df) < 20:
        return "SIDEWAYS"

    if df["EMA5"].iloc[-1] > df["EMA20"].iloc[-1]:
        return "UPTREND"
    elif df["EMA5"].iloc[-1] < df["EMA20"].iloc[-1]:
        return "DOWNTREND"
    return "SIDEWAYS"


# ==============================
# ML MODEL
# ==============================
def train_ml(df):
    df = df.dropna().copy()

    if len(df) < 60:
        return 50.0

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    X = df[["RSI","EMA5","EMA20","ATR","MACD","Volume"]]
    y = df["target"]

    try:
        model = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42)
        model.fit(X[:-1], y[:-1])
        prob = model.predict_proba(X.iloc[-1:])[0][1] * 100
        return float(prob)
    except:
        return 50.0


# ==============================
# SIGNAL LOGIC (COMMON)
# ==============================
def calculate_signal(df):

    latest = df.iloc[-1]
    score = 0

    score += 20 if latest["Close"] > latest["EMA20"] else -20

    if latest["RSI"] < 30:
        score += 10
    elif latest["RSI"] > 70:
        score -= 10

    score += 15 if latest["MACD"] > latest["MACD_SIGNAL"] else -15
    score += 10 if latest["Close"] > latest["VWAP"] else -10

    if latest["Volume"] > latest["Vol_Avg"]:
        score += 5

    if score > 40:
        signal = "STRONG BUY"
    elif score > 15:
        signal = "BUY"
    elif score < -40:
        signal = "STRONG SELL"
    elif score < -15:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, score


# ==============================
# MULTI TIMEFRAME
# ==============================
def get_multi_tf(symbol):
    return {
        "1H": get_df(symbol, "1mo", "1h"),
        "4H": get_df(symbol, "3mo", "1h"),
        "DAILY": get_df(symbol, "6mo", "1d"),
        "WEEKLY": get_df(symbol, "1y", "1wk"),
        "MONTHLY": get_df(symbol, "5y", "1mo")
    }


# ==============================
# ANALYZE
# ==============================
@app.get("/analyze")
def analyze(symbol: str):

    tf_data = get_multi_tf(symbol)
    df = tf_data["DAILY"]

    if df is None:
        return {"error": "No Data"}

    latest = df.iloc[-1]

    price = float(latest["Close"])
    atr = float(latest["ATR"])

    signal, score = calculate_signal(df)
    ml_prob = train_ml(df)

    timeframes = {tf: trend(tf_data[tf]) for tf in tf_data}

    macd_sig = "BULLISH" if latest["MACD"] > latest["MACD_SIGNAL"] else "BEARISH"
    vwap_sig = "BULLISH" if latest["Close"] > latest["VWAP"] else "BEARISH"
    volume = latest["Volume"] > latest["Vol_Avg"]

    # RISK
    if signal in ["BUY","STRONG BUY","HOLD"]:
        risk = {
            "stop_loss_small": round(price - atr,2),
            "stop_loss_big": round(price - 2*atr,2),
            "target_small": round(price + 2*atr,2),
            "target_big": round(price + 4*atr,2)
        }
    else:
        risk = {
            "stop_loss_small": round(price + atr,2),
            "stop_loss_big": round(price + 2*atr,2),
            "target_small": round(price - 2*atr,2),
            "target_big": round(price - 4*atr,2)
        }

    return {
        "symbol": symbol,
        "signal": signal,
        "score": round(score,2),
        "price": price,
        "rsi": round(latest["RSI"],2),
        "ml_probability": round(ml_prob,2),
        "timeframes": timeframes,
        "macd": macd_sig,
        "vwap": vwap_sig,
        "breakout": "N/A",
        "volume_strong": bool(volume),
        "risk": risk
    }


# ==============================
# SCANNER (SAME LOGIC)
# ==============================
@app.get("/scanner")
def scanner():

    results = []

    for sym in SYMBOLS[:50]:
        try:
            df = get_df(sym, "3mo")

            if df is None:
                continue

            signal, score = calculate_signal(df)

            results.append({
                "symbol": sym,
                "signal": signal,
                "score": score
            })

            time.sleep(0.15)

        except:
            continue

    return {
        "TOP_BUY": sorted(results, key=lambda x: -x["score"])[:5],
        "TOP_SELL": sorted(results, key=lambda x: x["score"])[:5]
    }
