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


def _pick_monthly_expiry(expiries):
    """Return the last expiry of the nearest future month."""
    from datetime import datetime
    today = datetime.today().date()
    future = []
    for e in expiries:
        try:
            d = datetime.strptime(e, "%d-%b-%Y").date()
        except ValueError:
            continue
        if d > today:
            future.append((d, e))
    if not future:
        return None
    future.sort()
    first_month = future[0][0].month
    same_month = [x for x in future if x[0].month == first_month]
    return same_month[-1][1] if same_month else future[0][1]


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

    # ---- Liquidity gate (added) ----
    total_volume = sum(int(getattr(r, "volume", 0) or 0) for r in result.atm_window(n=20))
    vol_data_available = total_volume > 0
    oi_min_ok = (total_call >= 50000) and (total_put >= 50000)
    # If the data source doesn't expose volume, don't penalize — use OI + walls only
    vol_min_ok = (total_volume >= 10000) if vol_data_available else True
    walls_separated = (max_pain != call_wall and max_pain != put_wall and call_wall != put_wall)

    # Effective gates: include volume gate ONLY if volume data was actually available
    effective_gates = [oi_min_ok, walls_separated]
    if vol_data_available:
        effective_gates.append(vol_min_ok)

    if all(effective_gates):
        liquidity_grade = "OK"
    elif oi_min_ok and not walls_separated:
        liquidity_grade = "PINNED"
    elif not any(effective_gates):
        liquidity_grade = "UNRELIABLE"
    else:
        liquidity_grade = "THIN"

    gates = {
        "oi_minimum_met": oi_min_ok,
        "volume_minimum_met": vol_min_ok,
        "volume_data_available": vol_data_available,
        "walls_separated": walls_separated,
    }

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
        "total_volume": total_volume,
        "liquidity_grade": liquidity_grade,
        "gates": gates,
    }


async def _fetch(symbol, is_index):
    """Weekly first for indices (with monthly fallback); monthly-only for stocks."""
    async with NSEClient() as nse:
        expiries = await nse.get_expiry_dates(symbol)
        if not expiries:
            return {"error": "no expiries"}

        if not is_index:
            monthly = _pick_monthly_expiry(expiries)
            if monthly is None:
                return {"error": "no monthly expiry"}
            result = await nse.fetch_option_chain(symbol, is_index=False, expiry=monthly)
            data = _compute_metrics(result)
            data["expiry_type"] = "monthly_stock"
            data["expiry_date"] = monthly
            return data

        weekly = _pick_next_expiry(expiries)
        result_w = await nse.fetch_option_chain(symbol, is_index=True, expiry=weekly)
        data_w = _compute_metrics(result_w)

        if data_w.get("gates", {}).get("walls_separated", True):
            data_w["expiry_type"] = "weekly"
            data_w["expiry_date"] = weekly
            return data_w

        monthly = _pick_monthly_expiry(expiries)
        if monthly and monthly != weekly:
            try:
                result_m = await nse.fetch_option_chain(symbol, is_index=True, expiry=monthly)
                data_m = _compute_metrics(result_m)
                if data_m.get("gates", {}).get("walls_separated", False):
                    data_m["expiry_type"] = "monthly"
                    data_m["expiry_date"] = monthly
                    return data_m
                data_w["expiry_type"] = "weekly"
                data_w["expiry_date"] = weekly
                data_w["liquidity_grade"] = "PINNED"
                data_w["note"] = "Walls collapsed on both weekly and monthly expiries"
                return data_w
            except Exception as e:
                print(f"monthly fetch error for {symbol}: {e}")

        data_w["expiry_type"] = "weekly"
        data_w["expiry_date"] = weekly
        return data_w

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
