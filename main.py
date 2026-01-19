from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# ================= APP INIT =================
app = FastAPI(title="Market AI – Unified Confidence Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= SYMBOL UNIVERSE =================
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

MODEL_DIR = "lstm_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= DATA FETCH =================
def fetch_df(symbol, period="8y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty or len(df) < 30:
        return None
    return df[["Open","High","Low","Close","Volume"]].dropna()

# ================= INDICATORS =================
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = -d.clip(upper=0).rolling(n).mean()
    rs = g/l
    return 100-(100/(1+rs))
def macd(s):
    m = ema(s,12)-ema(s,26)
    sig = ema(m,9)
    return m, sig, m-sig
def stochastic(df,n=14):
    lo = df["Low"].rolling(n).min()
    hi = df["High"].rolling(n).max()
    return 100*(df["Close"]-lo)/(hi-lo)
def atr(df,n=14):
    tr = np.maximum(df["High"]-df["Low"], np.maximum(abs(df["High"]-df["Close"].shift()), abs(df["Low"]-df["Close"].shift())))
    return tr.rolling(n).mean()
def vwap(df):
    return (df["Close"]*df["Volume"]).cumsum()/df["Volume"].cumsum()
def dow_wave(df): return "UPTREND", "Impulse"

# ================= REGRESSION =================
def regression_trend(series):
    if len(series)<60: return "UNKNOWN"
    x = np.arange(len(series)).reshape(-1,1)
    y = series.values.reshape(-1,1)
    slope = LinearRegression().fit(x,y).coef_[0][0]
    return "UP" if slope>0 else "DOWN" if slope<0 else "SIDEWAYS"

# ================= RANDOM FOREST =================
def rf_engine(df):
    df = df.copy()
    df["EMA20"] = ema(df["Close"],20)
    df["EMA50"] = ema(df["Close"],50)
    df["RSI"] = rsi(df["Close"])
    df["RET"] = df["Close"].pct_change()
    df["Target"] = (df["Close"].shift(-5) < df["Close"]).astype(int)
    df.dropna(inplace=True)

    X = df[["EMA20","EMA50","RSI","RET"]]
    y = df["Target"]
    if len(X) < 2:  # Not enough data
        return 50.0, "NEUTRAL"

    m = RandomForestClassifier(n_estimators=250, max_depth=6, random_state=42)
    m.fit(X[:-1],y[:-1])
    p = float(m.predict_proba(X.iloc[[-1]])[0][1]*100)
    return round(p,2),("DOWN" if p>55 else "UP")

# ================= LSTM =================
def lstm_engine(symbol, df):
    if "Close" not in df.columns or df["Close"].empty:
        return 50,"NEUTRAL"
    close = df["Close"].dropna()
    if len(close)<120: return 50,"NEUTRAL"
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.values.reshape(-1,1))
    X, y = [], []
    for i in range(60,len(scaled)):
        X.append(scaled[i-60:i,0])
        y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1],1))
    model_path = f"{MODEL_DIR}/{symbol.replace('.','_')}.keras"
    if not os.path.exists(model_path):
        model = Sequential([LSTM(64,return_sequences=True,input_shape=(60,1)),
                            LSTM(32),Dense(1)])
        model.compile(optimizer="adam",loss="mse")
        model.fit(X,y,epochs=5,batch_size=32,verbose=0)
        model.save(model_path)
    else:
        model = load_model(model_path,compile=False)
    pred = model.predict(X[-1:].reshape(1,60,1),verbose=0)
    prediction = float(scaler.inverse_transform(pred)[0][0])
    delta = prediction - float(close.iloc[-1])
    prob = float(max(0,min(100,50+delta*3)))
    trend = "UP" if delta>0 else "DOWN"
    return round(prob,2),trend

# ================= ANALYSIS PREP =================
def prepare_analysis(df):
    if df is None or df.empty: 
        return None

    close = df["Close"]
    price = float(close.iloc[-1])
    volume = int(df["Volume"].iloc[-1])
    df["EMA5"] = ema(close,5)
    df["EMA10"] = ema(close,10)
    df["EMA20"] = ema(close,20)
    
    rsi_val = float(rsi(close).iloc[-1]) if not rsi(close).empty else 50
    macd_hist = macd(close)[2]
    macd_val = float(macd_hist.iloc[-1]) if not macd_hist.empty else 0
    k_perc = stochastic(df)
    k_val = float(k_perc.iloc[-1]) if not k_perc.empty else 50
    atr_v = float(atr(df).iloc[-1]) if not atr(df).empty else 0
    dow, wave = dow_wave(df)
    bias = "UP" if df["EMA5"].iloc[-1] > df["EMA20"].iloc[-1] else "DOWN"

    stop_loss = round(price - atr_v*1.5,2)
    target = round(price + atr_v*3,2)

    # Backtest equity
    ret = df["Close"].pct_change().fillna(0)
    equity = (1 + ret).cumprod()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:,0]
    equity = equity.tolist()

    reasons = []
    if rsi_val < 30: reasons.append("RSI Oversold")
    elif rsi_val > 70: reasons.append("RSI Overbought")
    if dow == "UPTREND": reasons.append("Dow Theory Uptrend")
    elif dow == "DOWNTREND": reasons.append("Dow Theory Downtrend")
    if wave == "Impulse": reasons.append("Wave Phase Impulse")
    elif wave == "Correction": reasons.append("Wave Phase Correction")

    return {
        "Price": round(price,2),
        "Volume": volume,
        "VWAP": round(float(vwap(df).iloc[-1]),2),
        "EMA5": round(df["EMA5"].iloc[-1],2),
        "EMA10": round(df["EMA10"].iloc[-1],2),
        "EMA20": round(df["EMA20"].iloc[-1],2),
        "Bias": bias,
        "Indicators": {
            "RSI": round(rsi_val,2),
            "MACD_hist": round(macd_val,4),
            "Stochastic_%K": round(k_val,2),
            "Dow_Theory": dow,
            "Wave_Phase": wave,
            "ATR": round(atr_v,2)
        },
        "risk": {"stop_loss": stop_loss, "target": target},
        "reason": reasons,
        "backtest_equity_curve": equity[-50:]
    }

# ================= CORE ENGINE =================
def engine(symbol):
    df1d = fetch_df(symbol,period="8y",interval="1d")
    df4h = fetch_df(symbol,period="120d",interval="4h")
    df1h = fetch_df(symbol,period="60d",interval="60m")
    if df1d is None: return None

    analysis = {
        "1D": prepare_analysis(df1d),
        "4H": prepare_analysis(df4h),
        "1H": prepare_analysis(df1h)
    }

    daily_trend = regression_trend(df1d["Close"][-80:])
    weekly_trend = regression_trend(df1d["Close"].resample("W").last())
    monthly_trend = regression_trend(df1d["Close"].resample("M").last())

    rf_p, rf_t = rf_engine(df1d)
    lstm_p, lstm_t = lstm_engine(symbol,df1d)
    ai_prob={"RF": rf_p,"LSTM": lstm_p}
    ai_tr={"RF": rf_t,"LSTM": lstm_t}

    # Confidence score
    score = 50
    for tf in ["1D","4H","1H"]:
        if analysis.get(tf):
            bias = analysis[tf]["Bias"]
            score += 12 if bias=="UP" else -12 if bias=="DOWN" else 0
    for trend in [daily_trend, weekly_trend, monthly_trend]:
        score += 10 if trend=="UP" else -10 if trend=="DOWN" else 0
    score += 10 if analysis["1D"]["Indicators"]["RSI"]<35 else -10 if analysis["1D"]["Indicators"]["RSI"]>65 else 0
    score += 8 if analysis["1D"]["Price"]>analysis["1D"]["VWAP"] else -8
    score += (lstm_p-50)/2
    score += (50-rf_p)/2

    final_signal = ("STRONG BUY" if score>=80 else "BUY" if score>=65 else 
                    "STRONG SELL" if score<=25 else "SELL" if score<=40 else "HOLD")
    confidence_band = "HIGH" if score>=75 else "MEDIUM" if score>=60 else "LOW"

    return {
        "primary_engine":{
            "symbol": symbol,
            "final_signal": final_signal,
            "confidence_score_%": round(score,2),
            "confidence_band": confidence_band,
            "ai_probability_%": ai_prob,
            "ai_trend": ai_tr,
            "ai_model": "RandomForest + LSTM",
            "regression_trend": "MIXED" if len({daily_trend,weekly_trend,monthly_trend})>1 else daily_trend,
            "daily_trend": daily_trend,
            "weekly_trend": weekly_trend,
            "monthly_trend": monthly_trend,
            "analysis": analysis
        }
    }

# ================= API =================
@app.get("/analyze")
def analyze(symbol:str=Query(...)):
    r = engine(symbol.upper())
    return r if r else {"error":"No data"}

@app.get("/scanner")
def scanner():
    results = [engine(s) for s in SYMBOLS]
    results = [r for r in results if r]
    scored = [
        {
            "symbol": r["primary_engine"]["symbol"],
            "signal": r["primary_engine"]["final_signal"],
            "score": r["primary_engine"]["confidence_score_%"],
            "data": r
        } for r in results
    ]
    return {
        "TOP_BUY": sorted(scored, key=lambda x:x["score"], reverse=True)[:10],
        "TOP_SELL": sorted(scored, key=lambda x:x["score"])[:10],
        "TOTAL_SCANNED": len(scored)
    }

# ================= STATIC =================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
