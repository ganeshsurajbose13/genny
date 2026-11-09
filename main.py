"""
DeepGenny — Unified Market Analysis & Trading Signal Engine
Built by: Ganesh + AI assistant
====================================================
Local SQLite signal logging (signals.db).
For Render free: use Turso instead (see db.py notes).
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import pandas as pd
import numpy as np
import os
import time
import sqlite3
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import LSTM, Dense

from curl_cffi import requests as cffi_requests
SESSION = cffi_requests.Session(impersonate="chrome")

from option_chain import get_option_sentiment

app = FastAPI(title="DeepGenny – Unified Market AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "signals.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            symbol TEXT,
            signal TEXT,
            score REAL,
            scan_score REAL,
            price REAL,
            pcr REAL,
            max_pain_dist REAL,
            overbought INTEGER
        )
    """)
    conn.commit()
    conn.close()


def log_signal(result):
    p = result.get("primary_engine", result)
    opt = p.get("option_sentiment") or {}
    reasons = p.get("score_reasons") or []
    overbought = 1 if any("overbought" in r.lower() for r in reasons) else 0
    a1d = (p.get("analysis") or {}).get("1D") or {}
    price = a1d.get("Price")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO signals (ts, symbol, signal, score, scan_score, price, pcr, max_pain_dist, overbought) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            p.get("symbol"),
            p.get("final_signal"),
            p.get("confidence_score_%"),
            p.get("scan_score"),
            price,
            opt.get("pcr"),
            opt.get("max_pain_dist_pct"),
            overbought,
        ),
    )
    conn.commit()
    conn.close()


init_db()


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

FOREX_COMMODITY = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X", "USDINR": "INR=X",
    "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X",
    "GOLD": "GC=F", "SILVER": "SI=F", "COPPER": "HG=F",
    "PLATINUM": "PL=F", "PALLADIUM": "PA=F",
    "CRUDEOIL_WTI": "CL=F", "CRUDEOIL_BRENT": "BZ=F", "NATGAS": "NG=F",
    "HEATINGOIL": "HO=F", "GASOLINE": "RB=F",
    "CORN": "ZC=F", "WHEAT": "ZW=F", "SOYBEAN": "ZS=F",
    "SUGAR": "SB=F", "COFFEE": "KC=F", "COTTON": "CT=F",
    "SP500_FUT": "ES=F", "NASDAQ_FUT": "NQ=F",
    "DOW_FUT": "YM=F", "RUSSELL_FUT": "RTY=F",
}

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
MODEL_DIR = "lstm_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def fetch_yahoo_df(symbol, interval="1d", range_period="2y"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_period}"
    try:
        r = SESSION.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if "chart" not in data or "result" not in data["chart"] or data["chart"]["result"] is None:
            return None
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        ohlc = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Open": ohlc["open"], "High": ohlc["high"],
            "Low": ohlc["low"], "Close": ohlc["close"], "Volume": ohlc["volume"]
        }, index=pd.to_datetime(ts, unit="s"))
        df = df.dropna()
        return df if not df.empty else None
    except Exception:
        return None


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = -d.clip(upper=0).rolling(n).mean()
    rs = g / l
    return 100 - (100 / (1 + rs))


def macd(s):
    m = ema(s, 12) - ema(s, 26)
    sig = ema(m, 9)
    return m, sig, m - sig


def stochastic(df, n=14):
    lo = df["Low"].rolling(n).min()
    hi = df["High"].rolling(n).max()
    return 100 * (df["Close"] - lo) / (hi - lo)


def atr(df, n=14):
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(abs(df["High"] - df["Close"].shift()),
                   abs(df["Low"] - df["Close"].shift()))
    )
    return tr.rolling(n).mean()


def vwap(df, window=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol_sum = df["Volume"].rolling(window).sum()
    return (tp * df["Volume"]).rolling(window).sum() / vol_sum.replace(0, np.nan)


def pivot_points(df):
    if df is None or len(df) < 2:
        return None
    prev = df.iloc[-2]
    H, L, C = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    P = (H + L + C) / 3
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {"P": round(P, 2), "R1": round(R1, 2), "S1": round(S1, 2),
            "R2": round(R2, 2), "S2": round(S2, 2)}


def dow_wave(df):
    if df is None or len(df) < 10:
        return "SIDEWAYS", "Unknown"
    highs = df["High"].rolling(5).max()
    lows = df["Low"].rolling(5).min()
    if highs.iloc[-1] > highs.iloc[-5] and lows.iloc[-1] > lows.iloc[-5]:
        return "UPTREND", "Impulse"
    elif highs.iloc[-1] < highs.iloc[-5] and lows.iloc[-1] < lows.iloc[-5]:
        return "DOWNTREND", "Corrective"
    return "SIDEWAYS", "Consolidation"


def regression_trend(series, min_points=60):
    if len(series) < min_points:
        return "UNKNOWN"
    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)
    slope = LinearRegression().fit(x, y).coef_[0][0]
    if slope > 0: return "UP"
    if slope < 0: return "DOWN"
    return "SIDEWAYS"


def rf_engine(df, option_features=None):
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI"] = rsi(df["Close"])
    df["RET"] = df["Close"].pct_change()
    df["Target_DOWN"] = (df["Close"].shift(-5) < df["Close"]).astype(int)

    if option_features:
        spot = option_features.get("spot") or 1
        df["PCR"] = option_features.get("pcr") or 1.0
        df["MaxPainDist"] = option_features.get("max_pain_dist_pct") or 0.0
        df["CallWallDist"] = ((option_features.get("call_wall") or 0) - spot) / spot * 100
        df["PutWallDist"] = ((option_features.get("put_wall") or 0) - spot) / spot * 100
    else:
        df["PCR"] = 1.0
        df["MaxPainDist"] = 0.0
        df["CallWallDist"] = 0.0
        df["PutWallDist"] = 0.0

    df.dropna(inplace=True)
    if len(df) < 30:
        return 50.0, "NEUTRAL"

    feature_cols = ["EMA20", "EMA50", "RSI", "RET",
                    "PCR", "MaxPainDist", "CallWallDist", "PutWallDist"]
    X = df[feature_cols]
    y = df["Target_DOWN"]
    m = RandomForestClassifier(n_estimators=250, max_depth=6, random_state=42)
    m.fit(X[:-1], y[:-1])
    p_down = float(m.predict_proba(X.iloc[[-1]])[0][1] * 100)
    return round(p_down, 2), ("DOWN" if p_down > 55 else "UP")


def lstm_engine(symbol, df):
    if "Close" not in df.columns or df["Close"].empty:
        return 50, "NEUTRAL"
    close = df["Close"].dropna()
    if len(close) < 120:
        return 50, "NEUTRAL"
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.values.reshape(-1, 1))
    X = []
    for i in range(60, len(scaled)):
        X.append(scaled[i - 60:i, 0])
    X = np.array(X).reshape(-1, 60, 1)
    model_path = f"{MODEL_DIR}/{symbol.replace('.', '_')}.keras"

    if not os.path.exists(model_path):
        return 50, "NEUTRAL"

    try:
        model = load_model(model_path, compile=False)
    except TypeError:
        import h5py, json
        with h5py.File(model_path, "r") as f:
            config = json.loads(f.attrs["model_config"].decode("utf-8"))
        for layer in config["config"]["layers"]:
            layer["config"].pop("quantization_config", None)
        model = model_from_json(json.dumps(config))
        model.load_weights(model_path)

    pred = model.predict(X[-1:].reshape(1, 60, 1), verbose=0)
    prediction = float(scaler.inverse_transform(pred)[0][0])
    delta = prediction - float(close.iloc[-1])
    delta_pct = delta / float(close.iloc[-1]) * 100
    prob = float(max(0, min(100, 50 + delta_pct * 10)))
    if prob < 1 or prob > 99:
        return 50, "NEUTRAL"
    trend = "UP" if delta > 0 else "DOWN"
    return round(prob, 2), trend


def prepare_analysis(df):
    if df is None or df.empty:
        return None
    close = df["Close"]
    price = float(close.iloc[-1])
    volume = int(df["Volume"].iloc[-1]) if float(df["Volume"].iloc[-1]) > 0 else 0

    df["EMA5"] = ema(close, 5)
    df["EMA9"] = ema(close, 9)
    df["EMA10"] = ema(close, 10)
    df["EMA20"] = ema(close, 20)
    df["EMA21"] = ema(close, 21)
    df["VWAP"] = vwap(df)

    rsi_series = rsi(close)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50
    macd_hist = macd(close)[2]
    macd_val = float(macd_hist.iloc[-1]) if not macd_hist.empty else 0
    k_perc = stochastic(df)
    k_val = float(k_perc.iloc[-1]) if not k_perc.empty else 50
    atr_series = atr(df)
    atr_v = float(atr_series.iloc[-1]) if not atr_series.empty else 0

    dow, wave = dow_wave(df)
    bias = "UP" if float(df["EMA5"].iloc[-1]) > float(df["EMA20"].iloc[-1]) else "DOWN"

    vwap_cross_up = False
    vwap_cross_down = False
    if len(df) >= 2 and not pd.isna(df["VWAP"].iloc[-1]) and not pd.isna(df["VWAP"].iloc[-2]):
        vwap_cross_up = bool(
            float(df["Close"].iloc[-1]) > float(df["VWAP"].iloc[-1])
            and float(df["Close"].iloc[-2]) <= float(df["VWAP"].iloc[-2])
        )
        vwap_cross_down = bool(
            float(df["Close"].iloc[-1]) < float(df["VWAP"].iloc[-1])
            and float(df["Close"].iloc[-2]) >= float(df["VWAP"].iloc[-2])
        )

    ema_cross_up = bool(
        float(df["EMA9"].iloc[-1]) > float(df["EMA21"].iloc[-1])
        and float(df["EMA9"].iloc[-2]) <= float(df["EMA21"].iloc[-2])
    )
    ema_cross_down = bool(
        float(df["EMA9"].iloc[-1]) < float(df["EMA21"].iloc[-1])
        and float(df["EMA9"].iloc[-2]) >= float(df["EMA21"].iloc[-2])
    )

    piv = pivot_points(df)

    stop_loss = round(price - atr_v * 1.5, 2)
    target = round(price + atr_v * 3, 2)

    ret = df["Close"].pct_change().fillna(0)
    equity = (1 + ret).cumprod().tolist()[-50:]

    reasons = []
    if rsi_val < 30: reasons.append("RSI Oversold")
    elif rsi_val > 70: reasons.append("RSI Overbought")
    if dow == "UPTREND": reasons.append("Dow Theory Uptrend")
    if wave == "Impulse": reasons.append("Wave Phase Impulse")
    if vwap_cross_up: reasons.append("VWAP Cross Up")
    if vwap_cross_down: reasons.append("VWAP Cross Down")
    if ema_cross_up: reasons.append("EMA 9/21 Cross Up")
    if ema_cross_down: reasons.append("EMA 9/21 Cross Down")

    vwap_val = float(df["VWAP"].iloc[-1]) if not pd.isna(df["VWAP"].iloc[-1]) else price

    return {
        "Price": round(price, 2),
        "Volume": volume,
        "VWAP": round(vwap_val, 2),
        "EMA5": round(float(df["EMA5"].iloc[-1]), 2),
        "EMA9": round(float(df["EMA9"].iloc[-1]), 2),
        "EMA10": round(float(df["EMA10"].iloc[-1]), 2),
        "EMA20": round(float(df["EMA20"].iloc[-1]), 2),
        "EMA21": round(float(df["EMA21"].iloc[-1]), 2),
        "Bias": bias,
        "VWAP_cross_up": vwap_cross_up,
        "VWAP_cross_down": vwap_cross_down,
        "EMA_cross_up": ema_cross_up,
        "EMA_cross_down": ema_cross_down,
        "pivots": piv,
        "Indicators": {
            "RSI": round(rsi_val, 2),
            "MACD_hist": round(macd_val, 4),
            "Stochastic_%K": round(k_val, 2),
            "Dow_Theory": dow,
            "Wave_Phase": wave,
            "ATR": round(atr_v, 2)
        },
        "risk": {"stop_loss": stop_loss, "target": target},
        "reason": reasons,
        "backtest_equity_curve": equity
    }


def compute_score(analysis, daily_trend, weekly_trend, monthly_trend, rf_p, opt, is_index):
    score = 50.0
    reasons = []

    a1d = analysis.get("1D")
    a4h = analysis.get("4H")
    a1h = analysis.get("1H")

    bias_1d = a1d["Bias"] if a1d else "NEUTRAL"
    bias_4h = a4h["Bias"] if a4h else "NEUTRAL"
    bias_1h = a1h["Bias"] if a1h else "NEUTRAL"

    if bias_1d == "UP" and bias_4h == "UP" and bias_1h == "UP":
        score += 15
        reasons.append("Trend alignment: 1D + 4H + 1H all UP (+15)")
    elif bias_1d == "UP" and bias_4h == "UP" and bias_1h != "UP":
        score += 4
        reasons.append("Trend alignment: 1D + 4H UP, 1H diverges (+4)")
    elif bias_1d == "UP" and bias_4h != "UP":
        reasons.append("Trend alignment: 1D UP only, 4H mixed (0)")
    elif bias_1d == "DOWN" and bias_4h == "DOWN" and bias_1h == "DOWN":
        score -= 15
        reasons.append("Trend alignment: 1D + 4H + 1H all DOWN (-15)")
    elif bias_1d == "DOWN" and bias_4h == "DOWN" and bias_1h != "DOWN":
        score -= 4
        reasons.append("Trend alignment: 1D + 4H DOWN, 1H diverges (-4)")
    elif bias_1d == "DOWN" and bias_4h != "DOWN":
        reasons.append("Trend alignment: 1D DOWN only, 4H mixed (0)")

    if daily_trend == "UP" and weekly_trend == "UP" and monthly_trend == "UP":
        score += 10
        reasons.append("Long-term trends (D/W/M) all UP (+10)")
    elif daily_trend == "DOWN" and weekly_trend == "DOWN" and monthly_trend == "DOWN":
        score -= 10
        reasons.append("Long-term trends (D/W/M) all DOWN (-10)")

    if a1d:
        if a1d["VWAP_cross_up"]:
            score += 8
            reasons.append("VWAP cross UP today (+8)")
        if a1d["VWAP_cross_down"]:
            score -= 8
            reasons.append("VWAP cross DOWN today (-8)")
        if a1d["EMA_cross_up"]:
            score += 6
            reasons.append("EMA 9/21 cross UP today (+6)")
        if a1d["EMA_cross_down"]:
            score -= 6
            reasons.append("EMA 9/21 cross DOWN today (-6)")

    if opt:
        pcr = opt.get("pcr")
        if pcr is not None:
            if is_index:
                if pcr < 0.7:
                    score += 8
                    reasons.append(f"PCR {pcr:.2f} - index fear extreme, contrarian bullish (+8)")
                elif pcr < 0.9:
                    score += 4
                    reasons.append(f"PCR {pcr:.2f} - mild index fear (+4)")
                elif pcr > 1.5:
                    score -= 8
                    reasons.append(f"PCR {pcr:.2f} - index greed extreme, contrarian bearish (-8)")
                elif pcr > 1.3:
                    score -= 4
                    reasons.append(f"PCR {pcr:.2f} - mild index greed (-4)")
            else:
                if pcr < 0.7:
                    score -= 6
                    reasons.append(f"PCR {pcr:.2f} - call writers dominant, bearish (-6)")
                elif pcr < 0.9:
                    score -= 3
                    reasons.append(f"PCR {pcr:.2f} - call OI heavier (-3)")
                elif pcr > 1.3:
                    score += 6
                    reasons.append(f"PCR {pcr:.2f} - put writers dominant, bullish (+6)")
                elif pcr > 1.1:
                    score += 3
                    reasons.append(f"PCR {pcr:.2f} - put OI heavier (+3)")

        mpd = opt.get("max_pain_dist_pct")
        if mpd is not None:
            if mpd > 1.5:
                score += 5
                reasons.append(f"Spot {mpd:+.1f}% below Max Pain - upward pull (+5)")
            elif mpd < -1.5:
                score -= 5
                reasons.append(f"Spot {mpd:+.1f}% above Max Pain - downward pull (-5)")

    if a1d and a1d.get("pivots"):
        piv = a1d["pivots"]
        px = a1d["Price"]
        P, R1, S1 = piv["P"], piv["R1"], piv["S1"]
        if px > P:
            score += 3
            reasons.append(f"Price above daily P pivot {P} (+3)")
        else:
            score -= 3
            reasons.append(f"Price below daily P pivot {P} (-3)")
        if abs(px - R1) / px < 0.005:
            score -= 4
            reasons.append(f"Price near R1 resistance {R1} (-4)")
        if abs(px - S1) / px < 0.005:
            score += 4
            reasons.append(f"Price near S1 support {S1} (+4)")

    rf_contrib = (50 - rf_p) * 0.1
    score += rf_contrib
    if abs(rf_contrib) >= 1:
        d = "bullish" if rf_contrib > 0 else "bearish"
        reasons.append(f"RF P(down) {rf_p:.1f}% - {d} ({rf_contrib:+.1f})")

    if a1d:
        rsi_val = a1d["Indicators"]["RSI"]
        if rsi_val > 70:
            score -= 5
            reasons.append(f"1D RSI {rsi_val:.1f} - overbought (-5)")
        elif rsi_val < 30:
            score += 5
            reasons.append(f"1D RSI {rsi_val:.1f} - oversold (+5)")

    extremes = 0
    if a1d and a1d["Indicators"]["RSI"] > 70: extremes += 1
    if a1d and a1d["Indicators"]["Stochastic_%K"] > 90: extremes += 1
    if a4h and a4h["Indicators"]["RSI"] > 80: extremes += 1
    if a1h and a1h["Indicators"]["RSI"] > 80: extremes += 1
    if extremes >= 2:
        score -= 5
        reasons.append(f"Multi-timeframe overbought ({extremes} indicators) (-5)")

    oversold = 0
    if a1d and a1d["Indicators"]["RSI"] < 30: oversold += 1
    if a1d and a1d["Indicators"]["Stochastic_%K"] < 10: oversold += 1
    if a4h and a4h["Indicators"]["RSI"] < 20: oversold += 1
    if a1h and a1h["Indicators"]["RSI"] < 20: oversold += 1
    if oversold >= 2:
        score += 5
        reasons.append(f"Multi-timeframe oversold ({oversold} indicators) (+5)")

    if a1d:
        if a1d["VWAP_cross_up"] and monthly_trend == "DOWN":
            score -= 5
            reasons.append("VWAP cross UP against monthly DOWN - conflict (-5)")
        if a1d["VWAP_cross_down"] and monthly_trend == "UP":
            score -= 5
            reasons.append("VWAP cross DOWN against monthly UP - conflict (-5)")

    score = max(0.0, min(100.0, score))
    return round(score, 2), reasons


def compute_scan_score(analysis, opt, is_index):
    s = 0.0
    a1d = analysis.get("1D")
    if not a1d:
        return 0.0

    if a1d["VWAP_cross_up"]:   s += 20
    if a1d["EMA_cross_up"]:    s += 15
    if a1d["VWAP_cross_down"]: s -= 20
    if a1d["EMA_cross_down"]:  s -= 15

    if opt:
        pcr = opt.get("pcr")
        if pcr is not None:
            if is_index:
                if pcr < 0.7:   s += 15
                elif pcr > 1.5: s -= 15
            else:
                if pcr < 0.7:   s -= 15
                elif pcr > 1.3: s += 15

    rsi_val = a1d["Indicators"]["RSI"]
    if rsi_val < 30:  s += 10
    elif rsi_val > 70: s -= 10

    a4h = analysis.get("4H")
    if a4h and a1d["Bias"] == "UP" and a4h["Bias"] == "UP":
        s += 5
    if a4h and a1d["Bias"] == "DOWN" and a4h["Bias"] == "DOWN":
        s -= 5

    return round(s, 2)


def engine(symbol, option_features=None, is_index=False):
    df1d = fetch_yahoo_df(symbol, interval="1d", range_period="2y")
    df4h = fetch_yahoo_df(symbol, interval="4h", range_period="120d")
    df1h = fetch_yahoo_df(symbol, interval="60m", range_period="60d")
    if df1d is None:
        return None

    analysis = {
        "1D": prepare_analysis(df1d),
        "4H": prepare_analysis(df4h),
        "1H": prepare_analysis(df1h)
    }

    daily_trend = regression_trend(df1d["Close"][-80:], min_points=60)
    weekly_trend = regression_trend(df1d["Close"].resample("W").last(), min_points=20)
    monthly_trend = regression_trend(df1d["Close"].resample("ME").last(), min_points=12)

    rf_p, rf_t = rf_engine(df1d, option_features)
    lstm_p, lstm_t = lstm_engine(symbol, df1d)

    score, reasons = compute_score(analysis, daily_trend, weekly_trend, monthly_trend,
                                    rf_p, option_features, is_index)
    scan_score = compute_scan_score(analysis, option_features, is_index)

    final_signal = ("STRONG BUY" if score >= 80 else
                    "BUY" if score >= 65 else
                    "STRONG SELL" if score <= 25 else
                    "SELL" if score <= 40 else "HOLD")
    confidence_band = "HIGH" if score >= 75 else "MEDIUM" if score >= 60 else "LOW"

    bull_count = 0
    for tf_bias in [analysis["1D"]["Bias"] if analysis["1D"] else "NEUTRAL",
                    analysis["4H"]["Bias"] if analysis["4H"] else "NEUTRAL",
                    analysis["1H"]["Bias"] if analysis["1H"] else "NEUTRAL"]:
        if tf_bias == "UP": bull_count += 1
    for tr in [daily_trend, weekly_trend, monthly_trend]:
        if tr == "UP": bull_count += 1
    alignment_pct = round(bull_count / 6 * 100, 1)

    return {
        "primary_engine": {
            "symbol": symbol,
            "final_signal": final_signal,
            "confidence_score_%": score,
            "scan_score": scan_score,
            "confidence_band": confidence_band,
            "alignment_pct": alignment_pct,
            "score_reasons": reasons,
            "ai_probability_%": {"RF": rf_p, "LSTM": lstm_p},
            "ai_trend": {"RF": rf_t, "LSTM": lstm_t},
            "ai_model": "RandomForest + OptionOI (LSTM disabled in score)",
            "regression_trend": "MIXED" if len({daily_trend, weekly_trend, monthly_trend}) > 1 else daily_trend,
            "daily_trend": daily_trend,
            "weekly_trend": weekly_trend,
            "monthly_trend": monthly_trend,
            "analysis": analysis,
            "option_features_used": {
                "pcr": option_features.get("pcr") if option_features else None,
                "max_pain_dist_pct": option_features.get("max_pain_dist_pct") if option_features else None,
                "call_wall": option_features.get("call_wall") if option_features else None,
                "put_wall": option_features.get("put_wall") if option_features else None,
                "interpretation": "contrarian (index)" if is_index else "directional (stock)",
            } if option_features else None,
        }
    }


@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    base_sym = symbol.upper().replace(".NS", "").replace(".BO", "")
    is_idx = base_sym in INDEX_SYMBOLS
    opt = get_option_sentiment(base_sym, is_index=is_idx)

    r = engine(symbol.upper(), option_features=opt, is_index=is_idx)
    if not r:
        return {"error": "No data"}
    r["primary_engine"]["option_sentiment"] = opt
    log_signal(r)
    return r


@app.get("/history")
def history(symbol: str = Query(...), limit: int = 20):
    sym = symbol.upper()
    if not sym.endswith(".NS") and sym not in INDEX_SYMBOLS:
        sym = sym + ".NS"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT ts, symbol, signal, score, price, pcr, max_pain_dist "
        "FROM signals WHERE symbol = ? ORDER BY id DESC LIMIT ?",
        (sym, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return {
        "symbol": sym,
        "count": len(rows),
        "rows": [
            {
                "ts": r[0], "symbol": r[1], "signal": r[2], "score": r[3],
                "price": r[4], "pcr": r[5], "max_pain_dist": r[6],
            }
            for r in rows
        ],
    }


@app.get("/option-sentiment")
def option_sentiment_endpoint(symbol: str = Query(...), is_index: bool = False):
    sym = symbol.upper()
    is_idx = is_index or sym in INDEX_SYMBOLS
    data = get_option_sentiment(sym, is_index=is_idx)
    return data if data else {"error": "option chain unavailable"}


@app.get("/option-scan")
def option_scan(limit: int = Query(20, ge=1, le=74)):
    results = []
    scanned = 0
    for s in SYMBOLS:
        if scanned >= limit:
            break
        base = s.replace(".NS", "").replace(".BO", "")
        try:
            opt = get_option_sentiment(base, is_index=False)
            if opt and "error" not in opt:
                results.append({
                    "symbol": base,
                    "spot": opt.get("spot"),
                    "pcr": opt.get("pcr"),
                    "bias": opt.get("bias"),
                    "max_pain": opt.get("max_pain"),
                    "max_pain_dist_pct": opt.get("max_pain_dist_pct"),
                    "call_wall": opt.get("call_wall"),
                    "put_wall": opt.get("put_wall"),
                })
            scanned += 1
            time.sleep(1.0)
        except Exception as e:
            print(f"option_scan error {base}: {e}")
            continue

    results.sort(key=lambda x: x["pcr"] if x["pcr"] is not None else 1.0)
    return {
        "count": len(results),
        "limit": limit,
        "data": results,
        "note": "NSE rate-limits; use ?limit=10 for fast scan, ?limit=74 for full (~75 sec)",
    }


@app.get("/symbols")
def symbols_list():
    return {
        "stocks": [s.replace(".NS", "").replace(".BO", "") for s in SYMBOLS],
        "indexes": sorted(list(INDEX_SYMBOLS)),
    }


@app.get("/forex-commodity")
def forex_commodity():
    results = []
    for name, ticker in FOREX_COMMODITY.items():
        df = fetch_yahoo_df(ticker, "1d", "6mo")
        if df is None or len(df) < 20:
            continue
        df["EMA20"] = ema(df["Close"], 20)
        df["RSI"] = rsi(df["Close"])
        price = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else price
        change_pct = round((price - prev) / prev * 100, 2) if prev else 0
        results.append({
            "name": name,
            "ticker": ticker,
            "price": round(price, 4),
            "change_pct": change_pct,
            "rsi": round(float(df["RSI"].iloc[-1]), 2),
            "bias": "UP" if price > float(df["EMA20"].iloc[-1]) else "DOWN",
        })
        time.sleep(0.15)
    return {"count": len(results), "data": results}


@app.get("/scanner")
def scanner():
    results = []
    for s in SYMBOLS:
        base = s.replace(".NS", "").replace(".BO", "")
        opt = get_option_sentiment(base, is_index=False) if base not in INDEX_SYMBOLS else None
        r = engine(s, option_features=opt, is_index=False)
        if r:
            results.append(r)
        time.sleep(0.2)

    scored = [
        {
            "symbol": r["primary_engine"]["symbol"],
            "signal": r["primary_engine"]["final_signal"],
            "confidence_score": r["primary_engine"]["confidence_score_%"],
            "scan_score": r["primary_engine"]["scan_score"],
            "alignment_pct": r["primary_engine"]["alignment_pct"],
            "data": r
        } for r in results
    ]

    return {
        "TOP_BUY": sorted(scored, key=lambda x: x["scan_score"], reverse=True)[:10],
        "TOP_SELL": sorted(scored, key=lambda x: x["scan_score"])[:10],
        "TOP_BY_CONFIDENCE": sorted(scored, key=lambda x: x["confidence_score"], reverse=True)[:10],
        "TOTAL_SCANNED": len(scored)
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
