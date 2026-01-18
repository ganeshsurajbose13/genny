from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# ================= APP INIT =================
app = FastAPI(title="Market AI – Institutional Index Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= STOCK UNIVERSE =================
BASE_SYMBOLS = [
    "RELIANCE.NS","HDFCBANK.NS","BHARTIARTL.NS","TCS.NS","ICICIBANK.NS",
    "SBIN.NS","INFY.NS","BAJFINANCE.NS","ITC.NS","HINDUNILVR.NS",
    "LT.NS","MARUTI.NS","AXISBANK.NS","TECHM.NS","SUNPHARMA.NS",
    "NESTLEIND.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","POWERGRID.NS",
    "HCLTECH.NS","ONGC.NS","TATACONSUM.NS","ADANIENT.NS","BPCL.NS",
    "COALINDIA.NS","EICHERMOT.NS","DIVISLAB.NS","JSWSTEEL.NS","NTPC.NS",
    "GRASIM.NS","BRITANNIA.NS","CIPLA.NS","SHREECEM.NS","HDFCLIFE.NS",
    "ICICIPRULI.NS","UPL.NS","SBILIFE.NS","ICICIGI.NS","BAJAJFINSV.NS",
    "ADANIGREEN.NS","TECHM.NS","INDUSINDBK.NS","ONGC.NS","PIDILITIND.NS",
    "ADANIPORTS.NS","GAIL.NS","M&M.NS","ZEEL.NS","DLF.NS","TATAPOWER.NS",
    "VEDL.NS","COFORGE.NS","BEL.NS","TATAELXSI.NS","COLPAL.NS","HINDALCO.NS",
    "CERA.NS","ICICIBANK.NS","AUROPHARMA.NS","MUTHOOTFIN.NS","TORNTPOWER.NS",
    "BANDHANBNK.NS","LUPIN.NS","SRF.NS","PIDILITIND.NS","NMDC.NS","AMBUJACEM.NS",
    "HAL.NS","SIEMENS.NS","ADANIENT.NS","INDIGO.NS","ICICIBANK.NS",
    "M&MFIN.NS","BOSCHLTD.NS","PAGEIND.NS","BHEL.NS","TATASTEEL.NS","BANDHANBNK.NS"
]

ALL_SYMBOLS = list(set(BASE_SYMBOLS * 20))[:500]

# ================= DATA FETCH =================
def fetch_df(symbol, interval="1d", period="1y"):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df is None or df.empty or len(df) < 50:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        return df
    except Exception as e:
        print("Fetch error:", e)
        return None

# ================= INDICATORS =================
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = -delta.clip(upper=0).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def stochastic(df, k=14, d=3):
    low_min = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    k_percent = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(d).mean()
    return k_percent, d_percent

def atr(df, n=14):
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(abs(df["High"] - df["Close"].shift()),
                   abs(df["Low"] - df["Close"].shift()))
    )
    return tr.rolling(n).mean()

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

def reg_trend(series):
    if series is None or len(series) < 30:
        return "UNKNOWN"
    x = np.arange(len(series)).reshape(-1,1)
    y = series.values.reshape(-1,1)
    slope = LinearRegression().fit(x,y).coef_[0][0]
    return "UP" if slope > 0 else "DOWN"

def dow_theory(series):
    highs = series.rolling(5).max()
    lows = series.rolling(5).min()
    if highs.iloc[-1] > highs.iloc[-6] and lows.iloc[-1] > lows.iloc[-6]:
        return "UP"
    if highs.iloc[-1] < highs.iloc[-6] and lows.iloc[-1] < lows.iloc[-6]:
        return "DOWN"
    return "SIDEWAYS"

def wave_phase(series):
    slope = LinearRegression().fit(
        np.arange(len(series)).reshape(-1,1),
        series.values.reshape(-1,1)
    ).coef_[0][0]
    volatility = series.pct_change().rolling(10).std().iloc[-1]
    if slope > 0 and volatility < 0.03:
        return "IMPULSE"
    if slope < 0:
        return "CORRECTIVE"
    return "NEUTRAL"

# ================= AI ENGINE =================
def ai_engine(symbol):
    df = fetch_df(symbol,"1d","5y")
    if df is None or df.empty:
        return 50,"UNKNOWN"
    close = df["Close"]
    df["EMA20"]=ema(close,20)
    df["EMA50"]=ema(close,50)
    df["RSI"]=rsi(close)
    df["RET"]=close.pct_change()
    df["Target"]=(close.shift(-5)>close).astype(int)
    df.dropna(inplace=True)
    X=df[["EMA20","EMA50","RSI","RET"]]
    y=df["Target"]
    model=RandomForestClassifier(n_estimators=200,max_depth=6,random_state=42)
    model.fit(X[:-1],y[:-1])
    p=model.predict_proba(X.iloc[[-1]])[0][1]*100
    return round(p,2), "UP" if p>=50 else "DOWN"

# ================= CORE ENGINE =================
def institutional_engine(symbol: str):
    symbol = symbol.strip().upper()
    df = fetch_df(symbol,"1d","2y")
    if df is None or df.empty:
        return None

    close = df["Close"]

    # ===== Indicators =====
    df["EMA5"] = ema(close,5)
    df["EMA10"] = ema(close,10)
    df["EMA20"] = ema(close,20)
    df["VWAP"] = vwap(df)
    df["RSI"] = rsi(close)

    macd_line, macd_signal, macd_hist = macd(close)
    k_perc, d_perc = stochastic(df)

    daily = reg_trend(close[-60:])
    weekly = reg_trend(close.resample("W").last())
    monthly = reg_trend(close.resample("M").last())

    dow = dow_theory(close)
    wave = wave_phase(close)

    ai_prob, ai_tr = ai_engine(symbol)

    score = 50
    reasons = []

    for tf in [daily, weekly, monthly]:
        score += 12 if tf=="UP" else -12 if tf=="DOWN" else 0

    rsi_val = df["RSI"].iloc[-1]
    if rsi_val < 30:
        score += 10; reasons.append("RSI oversold")
    elif rsi_val > 70:
        score -= 10; reasons.append("RSI overbought")

    if macd_hist.iloc[-1] > 0:
        score += 8; reasons.append("MACD bullish")
    else:
        score -= 8; reasons.append("MACD bearish")

    if k_perc.iloc[-1] < 20:
        score += 6; reasons.append("Stochastic oversold")
    elif k_perc.iloc[-1] > 80:
        score -= 6; reasons.append("Stochastic overbought")

    if dow=="UP":
        score += 8; reasons.append("Dow higher-high")
    elif dow=="DOWN":
        score -= 8; reasons.append("Dow lower-low")

    if wave=="IMPULSE":
        score += 6; reasons.append("Impulse wave")
    elif wave=="CORRECTIVE":
        score -= 6; reasons.append("Corrective wave")

    if ai_prob >= 60:
        score += 10
    elif ai_prob <= 40:
        score -= 10

    final = (
        "STRONG BUY" if score >= 80 else
        "BUY" if score >= 65 else
        "SELL" if score <= 40 else
        "HOLD"
    )

    price = close.iloc[-1]
    volume = df["Volume"].iloc[-1]  # Added volume
    atr_v = atr(df).iloc[-1]

    # ===== ATR-dependent stop-loss and target & backtest equity =====
    if final in ["BUY","STRONG BUY"]:
        stop_loss = round(price - 1.5*atr_v,2)
        target = round(price + 3*atr_v,2)
        equity = [100000]
        for i in range(1,len(df)):
            position = equity[-1]/df["Close"].iloc[i-1]
            equity.append(round(position*df["Close"].iloc[i],2))

    elif final=="SELL":
        stop_loss = round(price + 1.5*atr_v,2)
        target = round(price - 3*atr_v,2)
        equity = [100000]
        for i in range(1,len(df)):
            position = equity[-1]/df["Close"].iloc[i-1]
            # Short selling: gain if price falls
            equity.append(round(position*(2*df["Close"].iloc[i-1]-df["Close"].iloc[i]),2))
    else:
        stop_loss = target = round(price,2)
        equity = [price for _ in df]

    primary_engine = {
        "symbol":symbol,
        "final_signal":final,
        "confidence_score_%":score,
        "confidence_band":"HIGH" if score>=75 else "MEDIUM" if score>=60 else "LOW",
        "ai_probability_%":ai_prob,
        "ai_trend":ai_tr,
        "ai_model":"RandomForest",
        "regression_trend":"MIXED" if len({daily,weekly,monthly})>1 else daily,
        "daily_trend":daily,
        "weekly_trend":weekly,
        "monthly_trend":monthly,
        "analysis":{
            "Daily":{
                "Price":round(price,2),
                "Volume":int(volume),  # Added volume to output
                "VWAP":round(df["VWAP"].iloc[-1],2),
                "EMA_5":round(df["EMA5"].iloc[-1],2),
                "EMA_10":round(df["EMA10"].iloc[-1],2),
                "EMA_20":round(df["EMA20"].iloc[-1],2),
                "Bias":"UP" if df["EMA5"].iloc[-1]>df["EMA20"].iloc[-1] else "DOWN"
            },
            "Indicators":{
                "RSI":round(rsi_val,2),
                "MACD_hist":round(macd_hist.iloc[-1],4),
                "Stochastic_%K":round(k_perc.iloc[-1],2),
                "Dow_Theory":dow,
                "Wave_Phase":wave,
                "ATR": round(atr_v,2)
            }
        },
        "risk":{
            "stop_loss": stop_loss,
            "target": target
        },
        "backtest_equity_curve": equity[-50:],
        "reason":reasons
    }

    combo_engine = {
        "symbol":symbol,
        "final_signal":final,
        "confidence_score_%":score,
        "ai_probability_%":ai_prob,
        "ai_trend":ai_tr,
        "ai_model":"RandomForest"
    }

    return {"primary_engine": primary_engine, "combo_engine": combo_engine}

# ================= API =================
@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    try:
        result = institutional_engine(symbol)
        if not result:
            return {"error": f"No data found for {symbol}"}
        return result
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

@app.get("/scanner/index")
def scanner():
    results = [institutional_engine(s) for s in ALL_SYMBOLS]
    results = [r for r in results if r]
    return {
        "TOP_BUY": sorted(
            [r["primary_engine"] for r in results if "BUY" in r["primary_engine"]["final_signal"]],
            key=lambda x:x["confidence_score_%"], reverse=True
        )[:10],
        "TOP_SELL": sorted(
            [r["primary_engine"] for r in results if r["primary_engine"]["final_signal"]=="SELL"],
            key=lambda x:x["confidence_score_%"]
        )[:10],
        "TOTAL_SCANNED": len(results)
    }

# ================= STATIC =================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html", headers={"Cache-Control":"no-cache"})
