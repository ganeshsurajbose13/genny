"""
Option chain analysis module.
Fetches NSE option chain via indiaopt, computes PCR, Max Pain, OI walls.
Caches results for 5 minutes to avoid hammering NSE.
"""
import asyncio
import time
from datetime import datetime
from indiaopt import NSEClient

_CACHE = {}
_CACHE_TTL = 300


def _pick_next_expiry(expiries):
    today = datetime.today().date()
    for e in expiries:
        try:
            d = datetime.strptime(e, "%d-%b-%Y").date()
        except ValueError:
            continue
        if d > today:
            return e
    return expiries[-1] if expiries else None


def _compute_metrics(result):
    total_call = result.total_call_oi or 0
    total_put = result.total_put_oi or 0
    pcr = round(total_put / total_call, 3) if total_call > 0 else None

    strikes = []
    for row in result.atm_window(n=20):
        strikes.append({
            "strike": float(row.strike),
            "call_oi": int(row.call_oi or 0),
            "put_oi": int(row.put_oi or 0),
        })

    if not strikes:
        return {
            "spot": result.spot_price,
            "atm": result.atm_strike,
            "pcr": pcr,
            "max_pain": None,
            "call_wall": None,
            "put_wall": None,
        }

    strikes.sort(key=lambda x: x["strike"])

    pain_values = []
    for cand in strikes:
        expiry = cand["strike"]
        loss = 0
        for s in strikes:
            if expiry < s["strike"]:
                loss += (s["strike"] - expiry) * s["call_oi"]
            if expiry > s["strike"]:
                loss += (expiry - s["strike"]) * s["put_oi"]
        pain_values.append(loss)
    max_pain = strikes[pain_values.index(min(pain_values))]["strike"]

    call_wall = max(strikes, key=lambda x: x["call_oi"])["strike"]
    put_wall = max(strikes, key=lambda x: x["put_oi"])["strike"]

    if pcr is None:
        bias = "UNKNOWN"
    elif pcr < 0.7:
        bias = "BEARISH"
    elif pcr > 1.3:
        bias = "BULLISH"
    else:
        bias = "NEUTRAL"

    spot = float(result.spot_price)
    max_pain_dist_pct = round((max_pain - spot) / spot * 100, 2) if spot > 0 else None

    return {
        "spot": spot,
        "atm": float(result.atm_strike),
        "expiry": getattr(result, "expiry_date", None),
        "pcr": pcr,
        "bias": bias,
        "max_pain": max_pain,
        "max_pain_dist_pct": max_pain_dist_pct,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "total_call_oi": total_call,
        "total_put_oi": total_put,
    }


async def _fetch(symbol, is_index):
    async with NSEClient() as nse:
        expiries = await nse.get_expiry_dates(symbol)
        if not expiries:
            return {"error": "no expiries"}
        next_exp = _pick_next_expiry(expiries)
        result = await nse.fetch_option_chain(
            symbol, is_index=is_index, expiry=next_exp
        )
        return _compute_metrics(result)


def get_option_sentiment(symbol, is_index=False):
    key = f"{symbol}:{is_index}"
    now = time.time()

    if key in _CACHE:
        ts, data = _CACHE[key]
        if now - ts < _CACHE_TTL:
            return data

    try:
        data = asyncio.run(_fetch(symbol, is_index))
    except Exception as e:
        print(f"option_chain error for {symbol}: {e}")
        return None

    _CACHE[key] = (now, data)
    return data
