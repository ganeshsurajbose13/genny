"""
DeepGenny — Unified Market Analysis & Trading Signal Engine
Built by: Ganesh + AI assistant
====================================================
Combo score = rule_score + (stat_prob - rule_score)*0.4 + (hit_rate - 50)*0.2
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
import math
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
            overbought INTEGER,
            call_oi REAL,
            put_oi REAL,
            combo_score REAL,
            rule_score REAL,
            stat_prob_score REAL,
            hit_rate_score REAL
        )
    """)
    for col in ["call_oi", "put_oi", "combo_score", "rule_score", "stat_prob_score", "hit_rate_score"]:
        try:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass
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
        "INSERT INTO signals (ts, symbol, signal, score, scan_score, price, pcr, max_pain_dist, overbought, call_oi, put_oi, combo_score, rule_score, stat_prob_score, hit_rate_score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            p.get("symbol"),
            p.get("final_signal"),
            p.get("combo_score"),
            p.get("scan_score"),
            price,
            opt.get("pcr"),
            opt.get("max_pain_dist_pct"),
            overbought,
            opt.get("total_call_oi"),
            opt.get("total_put_oi"),
            p.get("combo_score"),
            p.get("rule_score"),
            p.get("stat_prob_score"),
            p.get("hit_rate_score"),
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


def fetch_yahoo_df(symbol, interval="1d", range_period="5y"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_period}"
    try:
        r = SESSION.get(url, timeout=20)
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


def supertrend_direction(df, atr_period=10, multiplier=3.0):
    if df is None or len(df) < atr_period + 2:
        return "UNKNOWN"
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1/atr_period, min_periods=atr_period).mean()
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr_s
    lower = hl2 - multiplier * atr_s
    st = [True] * len(df)
    for i in range(1, len(df)):
        if close.iloc[i] > upper.iloc[i-1]:
            st[i] = True
        elif close.iloc[i] < lower.iloc[i-1]:
            st[i] = False
        else:
            st[i] = st[i-1]
    return "GREEN" if st[-1] else "RED"


def detect_engulfing(df):
    if df is None or len(df) < 3:
        return False, False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    prev_open, prev_close = float(prev["Open"]), float(prev["Close"])
    curr_open, curr_close = float(curr["Open"]), float(curr["Close"])

    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open
    curr_bullish = curr_close > curr_open
    curr_bearish = curr_close < curr_open

    bullish_engulfing = bool(
        prev_bearish and curr_bullish
        and curr_open <= prev_close and curr_close >= prev_open
    )
    bearish_engulfing = bool(
        prev_bullish and curr_bearish
        and curr_open >= prev_close and curr_close <= prev_open
    )
    return bullish_engulfing, bearish_engulfing


def detect_breakout(df):
    if df is None or len(df) < 22:
        return False, False, None, None
    prev_20_high = float(df["High"].iloc[-22:-2].max())
    prev_20_low = float(df["Low"].iloc[-22:-2].min())
    prev_close = float(df["Close"].iloc[-2])
    curr_close = float(df["Close"].iloc[-1])
    breakout_up = bool(prev_close <= prev_20_high and curr_close > prev_20_high)
    breakout_down = bool(prev_close >= prev_20_low and curr_close < prev_20_low)
    return breakout_up, breakout_down, prev_20_high, prev_20_low


def compute_volume_score(analysis):
    v_score = 0.0
    v_reasons = []

    a1d = analysis.get("1D")
    a4h = analysis.get("4H")
    a1h = analysis.get("1H")

    if a1d:
        vr = a1d.get("Volume_ratio", 1.0)
        vt = a1d.get("Volume_trend_5d", 1.0)
        day_bull = a1d.get("Day_bullish", True)

        if vr >= 3.0:
            if day_bull:
                v_score += 6
                v_reasons.append(f"1D Volume spike {vr:.1f}x average on bullish day (+6)")
            else:
                v_score -= 6
                v_reasons.append(f"1D Volume spike {vr:.1f}x average on bearish day (-6)")
        elif vr >= 1.5:
            if day_bull:
                v_score += 4
                v_reasons.append(f"1D Volume {vr:.1f}x average on bullish day (+4)")
            else:
                v_score -= 4
                v_reasons.append(f"1D Volume {vr:.1f}x average on bearish day (-4)")
        elif vr < 0.5:
            v_score -= 3
            v_reasons.append(f"1D Volume only {vr:.1f}x average — weak participation (-3)")

        if vt >= 1.3 and day_bull:
            v_score += 3
            v_reasons.append(f"1D 5-day volume rising ({vt:.1f}x 20-day avg) on up move (+3)")
        elif vt < 0.7 and day_bull:
            v_score -= 2
            v_reasons.append(f"1D 5-day volume falling ({vt:.1f}x) on up move — exhaustion (-2)")

    if a4h:
        vr4 = a4h.get("Volume_ratio", 1.0)
        bull4 = a4h.get("Day_bullish", True)
        if vr4 >= 1.5 and bull4:
            v_score += 3
            v_reasons.append(f"4H Volume {vr4:.1f}x on bullish bar (+3)")
        elif vr4 >= 1.5 and not bull4:
            v_score -= 3
            v_reasons.append(f"4H Volume {vr4:.1f}x on bearish bar (-3)")
        elif vr4 < 0.5:
            v_score -= 1
            v_reasons.append(f"4H Volume only {vr4:.1f}x — weak bar (-1)")

    if a1h:
        vr1 = a1h.get("Volume_ratio", 1.0)
        bull1 = a1h.get("Day_bullish", True)
        if vr1 >= 2.0 and bull1:
            v_score += 2
            v_reasons.append(f"1H Volume surge {vr1:.1f}x on bullish bar (+2)")
        elif vr1 >= 2.0 and not bull1:
            v_score -= 2
            v_reasons.append(f"1H Volume surge {vr1:.1f}x on bearish bar (-2)")

    if a1d and a4h and a1h:
        bulls = [a1d.get("Day_bullish"), a4h.get("Day_bullish"), a1h.get("Day_bullish")]
        if all(b is True for b in bulls):
            v_score += 2
            v_reasons.append("All 3 timeframes closed above open (+2)")
        elif all(b is False for b in bulls):
            v_score -= 2
            v_reasons.append("All 3 timeframes closed below open (-2)")

    return round(v_score, 2), v_reasons


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
    open_price = float(df["Open"].iloc[-1]) if "Open" in df.columns else price
    volume = int(df["Volume"].iloc[-1]) if float(df["Volume"].iloc[-1]) > 0 else 0

    df["EMA5"] = ema(close, 5)
    df["EMA9"] = ema(close, 9)
    df["EMA10"] = ema(close, 10)
    df["EMA20"] = ema(close, 20)
    df["EMA21"] = ema(close, 21)
    df["VWAP"] = vwap(df)
    df["VOL_AVG"] = df["Volume"].rolling(20).mean()
    df["VOL_5D_AVG"] = df["Volume"].rolling(5).mean()

    rsi_series = rsi(close)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50
    macd_hist = macd(close)[2]
    macd_val = float(macd_hist.iloc[-1]) if not macd_hist.empty else 0
    k_perc = stochastic(df)
    k_val = float(k_perc.iloc[-1]) if not k_perc.empty else 50
    atr_series = atr(df)
    atr_v = float(atr_series.iloc[-1]) if not atr_series.empty else 0

    vol_avg_last = float(df["VOL_AVG"].iloc[-1]) if not pd.isna(df["VOL_AVG"].iloc[-1]) else 0.0
    vol_ratio = round(volume / vol_avg_last, 2) if vol_avg_last > 0 else 1.0
    vol_5d_avg = float(df["VOL_5D_AVG"].iloc[-1]) if not pd.isna(df["VOL_5D_AVG"].iloc[-1]) else vol_avg_last
    vol_trend = round(vol_5d_avg / vol_avg_last, 2) if vol_avg_last > 0 else 1.0
    vol_spike = bool(vol_ratio >= 3.0)

    day_bullish = bool(price > open_price)
    day_range_pct = round((float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1])) / open_price * 100, 2) if open_price > 0 else 0
    body_pct = round(abs(price - open_price) / open_price * 100, 2) if open_price > 0 else 0

    breakout_up, breakout_down, prev_20_high, prev_20_low = detect_breakout(df)
    bull_engulfing, bear_engulfing = detect_engulfing(df)

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
    if bull_engulfing: reasons.append("Bullish Engulfing")
    if bear_engulfing: reasons.append("Bearish Engulfing")
    if breakout_up: reasons.append(f"Fresh breakout above {round(prev_20_high,2)}")
    if breakout_down: reasons.append(f"Fresh breakdown below {round(prev_20_low,2)}")

    vwap_val = float(df["VWAP"].iloc[-1]) if not pd.isna(df["VWAP"].iloc[-1]) else price

    return {
        "Price": round(price, 2),
        "Open": round(open_price, 2),
        "Day_bullish": day_bullish,
        "Body_pct": body_pct,
        "Day_range_pct": day_range_pct,
        "Volume": volume,
        "Volume_avg": round(vol_avg_last, 0),
        "Volume_ratio": vol_ratio,
        "Volume_trend_5d": vol_trend,
        "Volume_spike": vol_spike,
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
        "Bull_engulfing": bull_engulfing,
        "Bear_engulfing": bear_engulfing,
        "Breakout_up": breakout_up,
        "Breakout_down": breakout_down,
        "Prev_20_high": round(prev_20_high, 2) if prev_20_high else None,
        "Prev_20_low": round(prev_20_low, 2) if prev_20_low else None,
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


def compute_statistical_probability(df, score, weekly_rsi, monthly_rsi, pcr, atr_pct):
    p_base = score / 100.0

    weekly_adj = -0.02 * (weekly_rsi - 50) / 10
    monthly_adj = -0.015 * (monthly_rsi - 50) / 10

    vol_adj = -0.01 * max(0, (atr_pct or 0) - 2)

    if pcr is not None:
        if pcr < 0.5:
            pcr_adj = 0.03
        elif pcr > 1.5:
            pcr_adj = -0.03
        else:
            pcr_adj = 0.0
    else:
        pcr_adj = 0.0

    p_adjusted = p_base + weekly_adj + monthly_adj + vol_adj + pcr_adj
    p_adjusted = max(0.05, min(0.95, p_adjusted))

    n_samples = min(len(df), 252)
    se = math.sqrt(p_adjusted * (1 - p_adjusted) / n_samples) if n_samples > 0 else 0
    ci_low = max(0, p_adjusted - 1.96 * se)
    ci_high = min(1, p_adjusted + 1.96 * se)

    return {
        "probability_up": round(p_adjusted * 100, 2),
        "confidence_interval_95": [round(ci_low * 100, 2), round(ci_high * 100, 2)],
        "adjustments": {
            "base_from_score": round(p_base * 100, 2),
            "weekly_rsi_adj": round(weekly_adj * 100, 2),
            "monthly_rsi_adj": round(monthly_adj * 100, 2),
            "volatility_adj": round(vol_adj * 100, 2),
            "pcr_adj": round(pcr_adj * 100, 2),
        },
        "sample_size": n_samples,
        "formula": "p = p_base + rsi_adj + vol_adj + pcr_adj, CI = p ± 1.96*sqrt(p(1-p)/n)"
    }


def compute_historical_hit_rate(df, lookback=252, forward=5):
    close = df["Close"]
    if len(close) < lookback + forward + 20:
        return None

    rsi_series = rsi(close)
    ema_series = ema(close, 20)

    hits = 0
    total = 0
    start = max(20, len(close) - lookback - forward)
    for i in range(start, len(close) - forward):
        r = rsi_series.iloc[i]
        if 40 <= r <= 60 and close.iloc[i] > ema_series.iloc[i]:
            total += 1
            if close.iloc[i + forward] > close.iloc[i]:
                hits += 1

    if total == 0:
        return None

    hit_rate = hits / total
    return {
        "hit_rate_pct": round(hit_rate * 100, 2),
        "sample_size": total,
        "lookback_days": lookback,
        "forward_days": forward,
        "note": f"Among {total} historical days with similar conditions, {hits} led to a rise in {forward} days."
    }


def compute_combo_score(rule_score, stat_prob_dict, hit_rate_dict):
    """
    Design B: rule_score is anchor. stat_prob measures DISAGREEMENT with rule,
    not its own level. This prevents double-counting.
    combo = rule_score + (stat_prob - rule_score)*0.4 + (hit_rate - 50)*0.2
    """
    stat_prob = stat_prob_dict.get("probability_up", rule_score) if stat_prob_dict else rule_score
    hit_rate = hit_rate_dict.get("hit_rate_pct", 50) if hit_rate_dict else 50

    stat_adj = (stat_prob - rule_score) * 0.4
    hit_adj = (hit_rate - 50) * 0.2

    combo_before = rule_score + stat_adj + hit_adj
    combo = max(0.0, min(100.0, combo_before))

    return round(combo, 2), {
        "rule_score": round(rule_score, 2),
        "stat_prob": round(stat_prob, 2),
        "hit_rate": round(hit_rate, 2),
        "stat_adjustment": round(stat_adj, 2),
        "hit_rate_adjustment": round(hit_adj, 2),
        "combo_before_clamp": round(combo_before, 2),
        "combo_after_clamp": round(combo, 2),
        "formula": "combo = rule_score + (stat_prob - rule_score)*0.4 + (hit_rate - 50)*0.2"
    }


def compute_score(analysis, daily_trend, weekly_trend, monthly_trend, rf_p, opt, is_index,
                  weekly_rsi=50, monthly_rsi=50, weekly_above_vwap=False, monthly_above_vwap=False,
                  st_1d="UNKNOWN", st_4h="UNKNOWN"):
    score = 50.0
    reasons = []

    a1d = analysis.get("1D")
    a4h = analysis.get("4H")
    a1h = analysis.get("1H")

    bias_1d = a1d["Bias"] if a1d else "NEUTRAL"
    bias_4h = a4h["Bias"] if a4h else "NEUTRAL"
    bias_1h = a1h["Bias"] if a1h else "NEUTRAL"

    # Layer 1: Trend alignment
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

    # Layer 2: Long-term context
    if daily_trend == "UP" and weekly_trend == "UP" and monthly_trend == "UP":
        score += 10
        reasons.append("Long-term trends (D/W/M) all UP (+10)")
    elif daily_trend == "DOWN" and weekly_trend == "DOWN" and monthly_trend == "DOWN":
        score -= 10
        reasons.append("Long-term trends (D/W/M) all DOWN (-10)")

    # Layer 2b: Supertrend filter
    if st_1d == "GREEN" and st_4h == "GREEN":
        score += 5
        reasons.append("Supertrend green on 1D + 4H (+5)")
    elif st_1d == "RED" and st_4h == "RED":
        score -= 5
        reasons.append("Supertrend red on 1D + 4H (-5)")
    elif st_1d == "GREEN" and st_4h == "RED":
        reasons.append("Supertrend divergence: 1D green, 4H red (0)")
    elif st_1d == "RED" and st_4h == "GREEN":
        reasons.append("Supertrend divergence: 1D red, 4H green (0)")

    # Layer 2c: Higher timeframe RSI regime (60/40)
    if weekly_rsi > 60:
        score += 4
        reasons.append(f"Weekly RSI {weekly_rsi:.1f} > 60 — bullish regime (+4)")
    elif weekly_rsi < 40:
        score -= 4
        reasons.append(f"Weekly RSI {weekly_rsi:.1f} < 40 — bearish regime (-4)")
    else:
        reasons.append(f"Weekly RSI {weekly_rsi:.1f} — neutral regime (0)")

    if monthly_rsi > 60:
        score += 5
        reasons.append(f"Monthly RSI {monthly_rsi:.1f} > 60 — long-term bullish (+5)")
    elif monthly_rsi < 40:
        score -= 5
        reasons.append(f"Monthly RSI {monthly_rsi:.1f} < 40 — long-term bearish (-5)")
    else:
        reasons.append(f"Monthly RSI {monthly_rsi:.1f} — neutral long-term (0)")

    # Layer 2d: Higher timeframe VWAP
    if weekly_above_vwap:
        score += 3
        reasons.append("Weekly close above weekly VWAP (+3)")
    else:
        score -= 3
        reasons.append("Weekly close below weekly VWAP (-3)")

    if monthly_above_vwap:
        score += 4
        reasons.append("Monthly close above monthly VWAP (+4)")
    else:
        score -= 4
        reasons.append("Monthly close below monthly VWAP (-4)")

    # Layer 3: Quick triggers
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

    # Layer 4: Precision breakout
    if a1d and a4h and a1h:
        d_break_up = a1d.get("Breakout_up", False)
        d_break_down = a1d.get("Breakout_down", False)
        d_vol_ok = a1d.get("Volume_ratio", 1.0) >= 1.3
        d_bull = a1d.get("Day_bullish", False)
        d_bias = a1d.get("Bias") == "UP"
        d_conf = d_bull and d_bias

        h4_break_up = a4h.get("Breakout_up", False)
        h4_break_down = a4h.get("Breakout_down", False)
        h4_bull = a4h.get("Day_bullish", False)
        h4_bias = a4h.get("Bias") == "UP"
        h4_conf = h4_bull and h4_bias

        h1_break_up = a1h.get("Breakout_up", False)
        h1_break_down = a1h.get("Breakout_down", False)

        if d_break_up and d_vol_ok and d_conf:
            score += 10
            reasons.append("1D breakout + volume + bullish structure (+10)")
            if h4_conf:
                score += 3
                reasons.append("4H confirms breakout (+3)")
            if h1_break_up:
                score += 2
                reasons.append("1H also broke out (+2)")
        elif h4_break_up and h4_conf and d_conf:
            score += 5
            reasons.append("4H breakout + 1D structure confirms (+5)")
        elif h1_break_up:
            if h4_conf:
                score += 3
                reasons.append("1H breakout + 4H confirms (weak, +3)")
            else:
                reasons.append("1H breakout but 4H bearish — no score (0)")

        if d_break_down and d_vol_ok and not d_conf:
            score -= 10
            reasons.append("1D breakdown + volume + bearish structure (-10)")
            if not h4_conf:
                score -= 3
                reasons.append("4H confirms breakdown (-3)")
            if h1_break_down:
                score -= 2
                reasons.append("1H also broke down (-2)")
        elif h4_break_down and not h4_conf and not d_conf:
            score -= 5
            reasons.append("4H breakdown + 1D structure confirms (-5)")
        elif h1_break_down:
            if not h4_conf:
                score -= 3
                reasons.append("1H breakdown + 4H confirms (weak, -3)")
            else:
                reasons.append("1H breakdown but 4H bullish — no score (0)")

    # Layer 5: Volume + open/close
    vol_score, vol_reasons = compute_volume_score(analysis)
    score += vol_score
    for r in vol_reasons:
        reasons.append(r)

    # Layer 6: Engulfing at pivots
    if a1d and a1d.get("pivots"):
        piv = a1d["pivots"]
        px = a1d["Price"]
        S1, R1 = piv["S1"], piv["R1"]
        near_s1 = abs(px - S1) / px < 0.005
        near_r1 = abs(px - R1) / px < 0.005
        if a1d.get("Bull_engulfing") and near_s1:
            score += 6
            reasons.append(f"Bullish Engulfing at S1 support {S1} (+6)")
        if a1d.get("Bear_engulfing") and near_r1:
            score -= 6
            reasons.append(f"Bearish Engulfing at R1 resistance {R1} (-6)")

    # Layer 7: Option sentiment
    if opt:
        pcr = opt.get("pcr")
        if pcr is not None:
            if is_index:
                if pcr < 0.7:
                    score += 8
                    reasons.append(f"PCR {pcr:.2f} — index fear extreme, contrarian bullish (+8)")
                elif pcr < 0.9:
                    score += 4
                    reasons.append(f"PCR {pcr:.2f} — mild index fear (+4)")
                elif pcr > 1.5:
                    score -= 8
                    reasons.append(f"PCR {pcr:.2f} — index greed extreme, contrarian bearish (-8)")
                elif pcr > 1.3:
                    score -= 4
                    reasons.append(f"PCR {pcr:.2f} — mild index greed (-4)")
            else:
                if pcr < 0.7:
                    score -= 6
                    reasons.append(f"PCR {pcr:.2f} — call writers dominant, bearish (-6)")
                elif pcr < 0.9:
                    score -= 3
                    reasons.append(f"PCR {pcr:.2f} — call OI heavier (-3)")
                elif pcr > 1.3:
                    score += 6
                    reasons.append(f"PCR {pcr:.2f} — put writers dominant, bullish (+6)")
                elif pcr > 1.1:
                    score += 3
                    reasons.append(f"PCR {pcr:.2f} — put OI heavier (+3)")

        mpd = opt.get("max_pain_dist_pct")
        if mpd is not None:
            if mpd > 1.5:
                score += 5
                reasons.append(f"Spot {mpd:+.1f}% below Max Pain — upward pull (+5)")
            elif mpd < -1.5:
                score -= 5
                reasons.append(f"Spot {mpd:+.1f}% above Max Pain — downward pull (-5)")

    # Layer 8: Pivot location
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

    # Layer 9: Machine probability
    rf_contrib = (50 - rf_p) * 0.1
    score += rf_contrib
    if abs(rf_contrib) >= 1:
        d = "bullish" if rf_contrib > 0 else "bearish"
        reasons.append(f"RF P(down) {rf_p:.1f}% — {d} ({rf_contrib:+.1f})")

    # Layer 10: RSI extreme on 1D
    if a1d:
        rsi_val = a1d["Indicators"]["RSI"]
        if rsi_val > 70:
            score -= 5
            reasons.append(f"1D RSI {rsi_val:.1f} — overbought (-5)")
        elif rsi_val < 30:
            score += 5
            reasons.append(f"1D RSI {rsi_val:.1f} — oversold (+5)")

    # Layer 11: Multi-timeframe overbought/oversold
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

    # Conflict penalties
    if a1d:
        if a1d["VWAP_cross_up"] and monthly_trend == "DOWN":
            score -= 5
            reasons.append("VWAP cross UP against monthly DOWN — conflict (-5)")
        if a1d["VWAP_cross_down"] and monthly_trend == "UP":
            score -= 5
            reasons.append("VWAP cross DOWN against monthly UP — conflict (-5)")

    score = max(0.0, min(100.0, score))
    return round(score, 2), reasons


def compute_scan_score(analysis, opt, is_index):
    s = 0.0
    a1d = analysis.get("1D")
    a4h = analysis.get("4H")
    if not a1d:
        return 0.0

    if a1d["VWAP_cross_up"]:   s += 20
    if a1d["EMA_cross_up"]:    s += 15
    if a1d["VWAP_cross_down"]: s -= 20
    if a1d["EMA_cross_down"]:  s -= 15

    d_vol_ok = a1d.get("Volume_ratio", 1.0) >= 1.3
    if a1d.get("Breakout_up") and d_vol_ok:
        s += 15
    if a1d.get("Breakout_down") and d_vol_ok:
        s -= 15

    vol_score, _ = compute_volume_score(analysis)
    s += vol_score

    if opt:
        pcr = opt.get("pcr")
        if pcr is not None:
            if is_index:
                if pcr < 0.5:
                    s += 10
                elif pcr > 1.5:
                    s -= 10
            else:
                if pcr < 0.6:
                    s -= 6
                elif pcr > 1.4:
                    s += 6

    rsi_val = a1d["Indicators"]["RSI"]
    if rsi_val < 30:  s += 10
    elif rsi_val > 70: s -= 10

    if a4h and a1d["Bias"] == "UP" and a4h["Bias"] == "UP":
        s += 5
    if a4h and a1d["Bias"] == "DOWN" and a4h["Bias"] == "DOWN":
        s -= 5

    return round(s, 2)


def engine(symbol, option_features=None, is_index=False):
    df1d = fetch_yahoo_df(symbol, interval="1d", range_period="5y")
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

    st_1d = supertrend_direction(df1d)
    st_4h = supertrend_direction(df4h)

    weekly_close = df1d["Close"].resample("W").last().dropna()
    monthly_close = df1d["Close"].resample("ME").last().dropna()

    weekly_rsi_series = rsi(weekly_close)
    weekly_rsi = float(weekly_rsi_series.iloc[-1]) if not weekly_rsi_series.empty else 50

    monthly_rsi_series = rsi(monthly_close)
    monthly_rsi = float(monthly_rsi_series.iloc[-1]) if not monthly_rsi_series.empty else 50

    weekly_df = df1d.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()
    weekly_vwap_series = vwap(weekly_df)
    weekly_vwap = float(weekly_vwap_series.iloc[-1]) if not weekly_vwap_series.empty and not pd.isna(weekly_vwap_series.iloc[-1]) else float(weekly_close.iloc[-1])

    monthly_df = df1d.resample("ME").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()
    monthly_vwap_series = vwap(monthly_df)
    monthly_vwap = float(monthly_vwap_series.iloc[-1]) if not monthly_vwap_series.empty and not pd.isna(monthly_vwap_series.iloc[-1]) else float(monthly_close.iloc[-1])

    weekly_above_vwap = bool(float(weekly_close.iloc[-1]) > weekly_vwap) if len(weekly_close) > 0 else False
    monthly_above_vwap = bool(float(monthly_close.iloc[-1]) > monthly_vwap) if len(monthly_close) > 0 else False

    rf_p, rf_t = rf_engine(df1d, option_features)
    lstm_p, lstm_t = lstm_engine(symbol, df1d)

    rule_score, reasons = compute_score(
        analysis, daily_trend, weekly_trend, monthly_trend,
        rf_p, option_features, is_index,
        weekly_rsi, monthly_rsi,
        weekly_above_vwap, monthly_above_vwap,
        st_1d, st_4h
    )
    scan_score = compute_scan_score(analysis, option_features, is_index)
    vol_score, vol_reasons = compute_volume_score(analysis)

    close = df1d["Close"]
    high = df1d["High"]
    low = df1d["Low"]
    price = float(close.iloc[-1])

    mean_20 = float(close.tail(20).mean())
    std_20 = float(close.tail(20).std())
    price_z = round((price - mean_20) / std_20, 2) if std_20 > 0 else 0.0

    rsi_series_all = rsi(close).dropna()
    rsi_1y = rsi_series_all.tail(252)
    rsi_current = float(rsi_series_all.iloc[-1]) if not rsi_series_all.empty else 50
    rsi_percentile = round(float((rsi_1y < rsi_current).mean() * 100), 1) if len(rsi_1y) > 0 else None

    atr_series_st = atr(df1d)
    atr_current = float(atr_series_st.iloc[-1]) if not atr_series_st.empty else 0
    atr_pct_val = round(atr_current / price * 100, 2) if price > 0 else None

    sma_20 = close.tail(20).mean()
    std_20_bb = close.tail(20).std()
    upper_bb = sma_20 + 2 * std_20_bb
    lower_bb = sma_20 - 2 * std_20_bb
    percent_b = round((price - lower_bb) / (upper_bb - lower_bb), 3) if (upper_bb - lower_bb) > 0 else 0.5

    high_52w = float(high.tail(252).max())
    low_52w = float(low.tail(252).min())
    dist_from_52w_high = round((price - high_52w) / high_52w * 100, 2) if high_52w > 0 else None
    dist_from_52w_low = round((price - low_52w) / low_52w * 100, 2) if low_52w > 0 else None

    stat_prob = compute_statistical_probability(
        df1d, rule_score, weekly_rsi, monthly_rsi,
        option_features.get("pcr") if option_features else None,
        atr_pct_val
    )
    stat_prob_score = stat_prob["probability_up"]

    hit_rate = compute_historical_hit_rate(df1d, lookback=252, forward=5)
    hit_rate_score = hit_rate["hit_rate_pct"] if hit_rate else 50

    combo_score, combo_meta = compute_combo_score(rule_score, stat_prob, hit_rate)

    final_signal = ("STRONG BUY" if combo_score >= 80 else
                    "BUY" if combo_score >= 65 else
                    "STRONG SELL" if combo_score <= 25 else
                    "SELL" if combo_score <= 40 else "HOLD")
    confidence_band = "HIGH" if combo_score >= 75 else "MEDIUM" if combo_score >= 60 else "LOW"

    bull_count = 0
    for tf_bias in [analysis["1D"]["Bias"] if analysis["1D"] else "NEUTRAL",
                    analysis["4H"]["Bias"] if analysis["4H"] else "NEUTRAL",
                    analysis["1H"]["Bias"] if analysis["1H"] else "NEUTRAL"]:
        if tf_bias == "UP": bull_count += 1
    for tr in [daily_trend, weekly_trend, monthly_trend]:
        if tr == "UP": bull_count += 1
    alignment_pct = round(bull_count / 6 * 100, 1)

    statistical = {
        "price_z_score_20d": price_z,
        "rsi_percentile_1y": rsi_percentile,
        "atr_pct_of_price": atr_pct_val,
        "bollinger_percent_b": percent_b,
        "dist_from_52w_high_pct": dist_from_52w_high,
        "dist_from_52w_low_pct": dist_from_52w_low,
        "formulas": {
            "price_z": "(price - mean_20) / std_20",
            "rsi_pct": "(count of days RSI < current) / total_days × 100",
            "atr_pct": "(ATR / price) × 100",
            "percent_b": "(price - lower_BB) / (upper_BB - lower_BB)",
            "dist_52w_high": "(price - high_52w) / high_52w × 100",
            "dist_52w_low": "(price - low_52w) / low_52w × 100",
        }
    }

    # Position sizing (direction-aware)
    a1d_data = analysis.get("1D") or {}
    long_stop = a1d_data.get("risk", {}).get("stop_loss", 0)
    long_target = a1d_data.get("risk", {}).get("target", 0)

    is_short = final_signal in ("SELL", "STRONG SELL")

    if is_short:
        atr_implied = (price - long_stop) / 1.5 if long_stop else 0
        stop_loss_val = round(price + atr_implied * 1.5, 2)
        target_val = round(price - atr_implied * 3, 2)
    else:
        stop_loss_val = long_stop
        target_val = long_target

    if is_short:
        risk_per_share = round(stop_loss_val - price, 2) if stop_loss_val else 0
        reward_per_share = round(price - target_val, 2) if target_val else 0
    else:
        risk_per_share = round(price - stop_loss_val, 2) if stop_loss_val else 0
        reward_per_share = round(target_val - price, 2) if target_val else 0
    rr_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0

    account_size = 100000
    risk_pct = 1.0
    risk_amount = account_size * risk_pct / 100
    suggested_qty = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    suggested_capital = round(suggested_qty * price, 2)

    position_sizing = {
        "entry": round(price, 2),
        "stop_loss": stop_loss_val,
        "target": target_val,
        "direction": "SHORT" if is_short else "LONG",
        "risk_per_share": risk_per_share,
        "reward_per_share": reward_per_share,
        "risk_reward_ratio": rr_ratio,
        "account_size": account_size,
        "risk_pct": risk_pct,
        "risk_amount": round(risk_amount, 2),
        "suggested_qty": suggested_qty,
        "suggested_capital": suggested_capital,
        "formula": "qty = (account × risk_pct/100) / |entry - stop_loss|"
    }

    bull_layers = sum(1 for r in reasons if "(+" in r)
    bear_layers = sum(1 for r in reasons if "(-" in r)
    neutral_layers = len(reasons) - bull_layers - bear_layers

    if final_signal == "STRONG BUY":
        summary = f"Strong bullish: rule {rule_score:.1f}, stat {stat_prob_score:.1f}, hit rate {hit_rate_score:.1f} → combo {combo_score:.1f}."
        action = f"LONG at Rs {price:.2f}, stop Rs {stop_loss_val:.2f}, target Rs {target_val:.2f}, qty {suggested_qty} (risk Rs {risk_amount:.0f})."
    elif final_signal == "BUY":
        summary = f"Bullish: rule {rule_score:.1f}, stat {stat_prob_score:.1f}, hit rate {hit_rate_score:.1f} → combo {combo_score:.1f}."
        action = f"Consider LONG at Rs {price:.2f}, stop Rs {stop_loss_val:.2f}, target Rs {target_val:.2f}, qty {suggested_qty}."
    elif final_signal == "STRONG SELL":
        summary = f"Strong bearish: rule {rule_score:.1f}, stat {stat_prob_score:.1f}, hit rate {hit_rate_score:.1f} → combo {combo_score:.1f}."
        action = f"SHORT at Rs {price:.2f}, stop Rs {stop_loss_val:.2f} (above), target Rs {target_val:.2f} (below), qty {suggested_qty} (risk Rs {risk_amount:.0f})."
    elif final_signal == "SELL":
        summary = f"Bearish: rule {rule_score:.1f}, stat {stat_prob_score:.1f}, hit rate {hit_rate_score:.1f} → combo {combo_score:.1f}."
        action = f"Consider SHORT at Rs {price:.2f}, stop Rs {stop_loss_val:.2f} (above), target Rs {target_val:.2f} (below), qty {suggested_qty}."
    else:
        summary = f"Mixed: rule {rule_score:.1f}, stat {stat_prob_score:.1f}, hit rate {hit_rate_score:.1f} → combo {combo_score:.1f}. {bull_layers} bull vs {bear_layers} bear layers."
        action = "No action — wait for a clearer signal."

    return {
        "primary_engine": {
            "symbol": symbol,
            "final_signal": final_signal,
            "combo_score": combo_score,
            "rule_score": rule_score,
            "stat_prob_score": stat_prob_score,
            "hit_rate_score": hit_rate_score,
            "combo_meta": combo_meta,
            "confidence_score_%": combo_score,
            "scan_score": scan_score,
            "volume_score": vol_score,
            "volume_reasons": vol_reasons,
            "confidence_band": confidence_band,
            "alignment_pct": alignment_pct,
            "score_reasons": reasons,
            "higher_tf": {
                "weekly_rsi": round(weekly_rsi, 2),
                "monthly_rsi": round(monthly_rsi, 2),
                "weekly_vwap": round(weekly_vwap, 2),
                "monthly_vwap": round(monthly_vwap, 2),
                "weekly_above_vwap": weekly_above_vwap,
                "monthly_above_vwap": monthly_above_vwap,
                "supertrend_1d": st_1d,
                "supertrend_4h": st_4h,
            },
            "statistical": statistical,
            "statistical_probability": stat_prob,
            "historical_hit_rate": hit_rate,
            "position_sizing": position_sizing,
            "layer_breakdown": {
                "bull_layers": bull_layers,
                "bear_layers": bear_layers,
                "neutral_layers": neutral_layers,
                "total_layers": len(reasons)
            },
            "signal_summary": summary,
            "suggested_action": action,
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
        "SELECT ts, symbol, signal, combo_score, rule_score, stat_prob_score, hit_rate_score, price, pcr, max_pain_dist "
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
                "ts": r[0], "symbol": r[1], "signal": r[2],
                "combo_score": r[3], "rule_score": r[4],
                "stat_prob_score": r[5], "hit_rate_score": r[6],
                "price": r[7], "pcr": r[8], "max_pain_dist": r[9],
            }
            for r in rows
        ],
    }


@app.get("/oi-change")
def oi_change(symbol: str = Query(...)):
    sym = symbol.upper()
    if not sym.endswith(".NS"):
        sym += ".NS"
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT ts, call_oi, put_oi FROM signals WHERE symbol = ? AND call_oi IS NOT NULL ORDER BY id DESC LIMIT 2",
        (sym,)
    ).fetchall()
    conn.close()
    if len(rows) < 2:
        return {"error": "Need at least 2 logged signals with OI data"}
    today, yesterday = rows[0], rows[1]
    call_delta = (today[1] - yesterday[1]) if today[1] is not None and yesterday[1] is not None else None
    put_delta = (today[2] - yesterday[2]) if today[2] is not None and yesterday[2] is not None else None
    return {
        "symbol": sym,
        "latest_ts": today[0],
        "previous_ts": yesterday[0],
        "call_oi_change": call_delta,
        "put_oi_change": put_delta,
        "note": "Positive call OI change = new call writing (bearish). Positive put OI change = new put writing (bullish)."
    }


@app.get("/pcr-change")
def pcr_change(symbol: str = Query(...)):
    sym = symbol.upper()
    if not sym.endswith(".NS") and sym not in INDEX_SYMBOLS:
        sym = sym + ".NS"
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT ts, pcr FROM signals WHERE symbol = ? AND pcr IS NOT NULL ORDER BY id DESC LIMIT 2",
        (sym,)
    ).fetchall()
    conn.close()
    if len(rows) < 2:
        return {"error": "Need at least 2 logged signals with PCR data. Run /analyze twice on different days."}
    today, yesterday = rows[0], rows[1]
    today_pcr = today[1]
    yesterday_pcr = yesterday[1]
    delta = round(today_pcr - yesterday_pcr, 4) if today_pcr is not None and yesterday_pcr is not None else None

    if delta is None:
        interpretation = "Insufficient PCR data"
    elif delta > 0.15:
        interpretation = "PCR rising fast — fear increasing, potential bottom forming"
    elif delta < -0.15:
        interpretation = "PCR falling fast — call writers aggressive, caution on longs"
    elif delta > 0.05:
        interpretation = "PCR rising mildly — sentiment improving"
    elif delta < -0.05:
        interpretation = "PCR falling mildly — sentiment weakening"
    else:
        interpretation = "PCR stable — positioning already priced in"

    return {
        "symbol": sym,
        "latest_ts": today[0],
        "previous_ts": yesterday[0],
        "latest_pcr": today_pcr,
        "previous_pcr": yesterday_pcr,
        "pcr_change": delta,
        "interpretation": interpretation,
        "note": "PCR change > 0.15 (rising) = fear building, contrarian bullish. < -0.15 (falling) = call writing aggressive, cautious.",
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
            "combo_score": r["primary_engine"]["combo_score"],
            "rule_score": r["primary_engine"]["rule_score"],
            "stat_prob_score": r["primary_engine"]["stat_prob_score"],
            "hit_rate_score": r["primary_engine"]["hit_rate_score"],
            "scan_score": r["primary_engine"]["scan_score"],
            "alignment_pct": r["primary_engine"]["alignment_pct"],
            "data": r
        } for r in results
    ]

    return {
        "TOP_BUY": sorted(scored, key=lambda x: x["combo_score"], reverse=True)[:10],
        "TOP_SELL": sorted(scored, key=lambda x: x["combo_score"])[:10],
        "TOP_BY_CONFIDENCE": sorted(scored, key=lambda x: x["combo_score"], reverse=True)[:10],
        "TOTAL_SCANNED": len(scored)
    }


@app.get("/index-option-sentiment")
def index_option_sentiment():
    """Fetch option sentiment for NIFTY, BANKNIFTY, FINNIFTY in one call."""
    results = {}
    for sym in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
        try:
            data = get_option_sentiment(sym, is_index=True)
            if data and not data.get("error"):
                results[sym] = data
            else:
                results[sym] = {"error": (data or {}).get("error", "no data")}
        except Exception as e:
            results[sym] = {"error": str(e)}
        time.sleep(2)
    return results


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
