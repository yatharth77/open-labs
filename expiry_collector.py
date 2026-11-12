#!/usr/bin/env python3
"""
expiry_collector.py - record far-OTM option behaviour on SENSEX / NIFTY expiry days.

On each index's expiry day (run any time after 15:30 IST, same day) it records:
  * the index's open (first 1-min candle: O/H/L/C) and the day's high/low/close
  * the ATM strike, taken from the index's 09:15 open
  * for every CE and PE whose strike is within the far-OTM band:
        SENSEX: 2000-3000 points from ATM     NIFTY: 1200-2000 points from ATM
    - the first 1-min candle (09:15) open, high, low, close, volume
    - the day's high and when it happened, the high after the first minute,
      the day's low and the last close
Rows are appended to expiry_options.csv (one row per option) and
expiry_index.csv (one row per index per expiry). Raw 1-min candles go to
candles/<date>_<INDEX>.csv for later backtests.

Expired contracts disappear from Kite after expiry day, so this must run on
the expiry day itself (after 15:30 and before the next morning).

Setup (once):
    uv add kiteconnect            # or:  pip install kiteconnect
    export KITE_API_KEY=xxxx
    export KITE_API_SECRET=yyyy   # never hard-code these in the script

Each trading morning (Kite access tokens expire daily):
    uv run expiry_collector.py --login

After 15:30 on any day (does nothing unless it's an expiry day):
    uv run expiry_collector.py
Options:
    --indices SENSEX,NIFTY   which indices to check (default both)
    --force                  collect even if today isn't expiry (uses nearest expiry)
    --out DIR                output folder (default: folder of this script)

Needs a Kite Connect plan that includes historical data (minute candles).
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

# index -> where its options trade, where the spot index lives, OTM band (points from ATM)
CONFIG = {
    "SENSEX": {"opt_exchange": "BFO", "opt_name": "SENSEX",
               "spot_exchange": "BSE", "spot_symbol": "SENSEX", "band": (2000, 3000)},
    "NIFTY": {"opt_exchange": "NFO", "opt_name": "NIFTY",
              "spot_exchange": "NSE", "spot_symbol": "NIFTY 50", "band": (1200, 2000)},
}
HIST_DELAY = 0.35  # historical API allows ~3 requests/second
TOKEN_FILE = ".kite_token.json"
OPTION_COLS = [
    "date", "index", "expiry", "index_open", "atm_strike", "symbol", "type", "strike",
    "points_otm", "pct_otm", "lot_size", "traded", "first_candle_time", "first_candle_is_0915",
    "first_open", "first_high", "first_low", "first_close", "first_volume",
    "day_high", "day_high_time", "high_after_first_min", "high_after_first_min_time",
    "day_low", "last_close", "day_high_vs_first_close_x",
]


# ------------------------------------------------------------------ auth
def kite_client(out_dir, login=False):
    from kiteconnect import KiteConnect
    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET")
    if not api_key:
        sys.exit("Set KITE_API_KEY (and KITE_API_SECRET) as environment variables first.")
    kite = KiteConnect(api_key=api_key)
    token_path = Path(out_dir) / TOKEN_FILE

    if login:
        if not api_secret:
            sys.exit("Set KITE_API_SECRET to log in.")
        print("1. Open this URL and log in to Kite:\n  ", kite.login_url())
        print("2. After login you'll land on your redirect URL. Paste that full URL "
              "(or just the request_token) here:")
        raw = input("> ").strip()
        req = raw.split("request_token=")[1].split("&")[0] if "request_token=" in raw else raw
        session = kite.generate_session(req, api_secret=api_secret)
        token_path.write_text(json.dumps({"access_token": session["access_token"],
                                          "date": today_ist().isoformat()}))
        try:
            os.chmod(token_path, 0o600)
        except OSError:
            pass
        print("Logged in; token saved for today.")

    if not token_path.exists():
        sys.exit("No saved login. Run:  expiry_collector.py --login")
    saved = json.loads(token_path.read_text())
    kite.set_access_token(saved["access_token"])
    try:
        kite.profile()
    except Exception as err:
        sys.exit(f"Saved login is no longer valid ({err}). Run:  expiry_collector.py --login")
    return kite


# ------------------------------------------------------------------ helpers
def today_ist():
    return dt.datetime.now(IST).date()


def hist(kite, token, day):
    """All 1-min candles for one session (09:15-15:30 IST)."""
    time.sleep(HIST_DELAY)
    start = dt.datetime.combine(day, dt.time(9, 15))
    end = dt.datetime.combine(day, dt.time(15, 30))
    for attempt in range(3):
        try:
            return kite.historical_data(token, start, end, "minute")
        except Exception as err:  # network blips / rate limit
            if attempt == 2:
                print(f"    ! candles failed for token {token}: {err}")
                return []
            time.sleep(1.5 * (attempt + 1))


def hhmm(ts):
    return ts.strftime("%H:%M") if ts else ""


def first_minute(candles):
    """Candle stamped 09:15; else the first candle of the day (flagged)."""
    for c in candles:
        if c["date"].hour == 9 and c["date"].minute == 15:
            return c, True
    return (candles[0], False) if candles else (None, False)


def append_csv(path, rows):
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def save_candles(path, by_symbol):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "time", "open", "high", "low", "close", "volume"])
        for sym, cs in by_symbol.items():
            for c in cs:
                w.writerow([sym, c["date"].strftime("%Y-%m-%d %H:%M"), c["open"], c["high"],
                            c["low"], c["close"], c.get("volume", "")])


# ------------------------------------------------------------------ core
def collect(kite, index, cfg, day, force, out_dir, instruments_cache):
    ex = cfg["opt_exchange"]
    if ex not in instruments_cache:
        instruments_cache[ex] = kite.instruments(ex)
    opts = [i for i in instruments_cache[ex]
            if i["name"] == cfg["opt_name"] and i["instrument_type"] in ("CE", "PE")
            and i["expiry"] and i["expiry"] >= day]
    if not opts:
        print(f"[{index}] no option contracts found on {ex}")
        return
    expiry = min(i["expiry"] for i in opts)
    if expiry != day and not force:
        print(f"[{index}] not an expiry day (next expiry {expiry}); nothing to do")
        return
    chain = [i for i in opts if i["expiry"] == expiry]

    # spot index
    sx = cfg["spot_exchange"]
    if sx not in instruments_cache:
        instruments_cache[sx] = kite.instruments(sx)
    spot = next((i for i in instruments_cache[sx]
                 if i["tradingsymbol"] == cfg["spot_symbol"]
                 and i.get("segment", "") == "INDICES"), None)
    if not spot:
        print(f"[{index}] couldn't find spot instrument {sx}:{cfg['spot_symbol']}")
        return
    sc = hist(kite, spot["instrument_token"], day)
    if not sc:
        print(f"[{index}] no index candles for {day} (market closed or plan lacks historical data)")
        return
    s1, exact = first_minute(sc)
    idx_open = s1["open"]

    strikes = sorted({i["strike"] for i in chain})
    atm = min(strikes, key=lambda k: abs(k - idx_open))
    lo, hi = cfg["band"]
    picked = [i for i in chain
              if (i["instrument_type"] == "CE" and lo <= i["strike"] - atm <= hi)
              or (i["instrument_type"] == "PE" and lo <= atm - i["strike"] <= hi)]
    picked.sort(key=lambda i: (i["instrument_type"], i["strike"]))
    print(f"[{index}] expiry {expiry}: open {idx_open}, ATM {atm:g}, "
          f"{len(picked)} options {lo}-{hi} pts OTM")

    day_hi = max(sc, key=lambda c: c["high"])
    day_lo = min(sc, key=lambda c: c["low"])
    index_row = {
        "date": day.isoformat(), "index": index, "expiry": expiry.isoformat(),
        "is_expiry_day": expiry == day,
        "index_open": idx_open, "first_min_high": s1["high"], "first_min_low": s1["low"],
        "first_min_close": s1["close"], "first_min_is_0915": exact,
        "day_high": day_hi["high"], "day_high_time": hhmm(day_hi["date"]),
        "day_low": day_lo["low"], "day_low_time": hhmm(day_lo["date"]),
        "day_close": sc[-1]["close"], "atm_strike": atm,
        "band_points": f"{lo}-{hi}", "options_recorded": len(picked),
    }

    rows, raw = [], {f"{index}_SPOT": sc}
    for n, ins in enumerate(picked, 1):
        cs = hist(kite, ins["instrument_token"], day)
        raw[ins["tradingsymbol"]] = cs
        dist = (ins["strike"] - atm) if ins["instrument_type"] == "CE" else (atm - ins["strike"])
        base = {
            "date": day.isoformat(), "index": index, "expiry": expiry.isoformat(),
            "index_open": idx_open, "atm_strike": atm,
            "symbol": ins["tradingsymbol"], "type": ins["instrument_type"],
            "strike": ins["strike"], "points_otm": dist,
            "pct_otm": round(dist / idx_open * 100, 2), "lot_size": ins["lot_size"],
        }
        if not cs:
            rows.append({**base, "traded": False})
            continue
        f1, f1_exact = first_minute(cs)
        dh = max(cs, key=lambda c: c["high"])
        after = [c for c in cs if c["date"] > f1["date"]]
        ah = max(after, key=lambda c: c["high"]) if after else None
        dl = min(cs, key=lambda c: c["low"])
        rows.append({
            **base, "traded": True,
            "first_candle_time": hhmm(f1["date"]), "first_candle_is_0915": f1_exact,
            "first_open": f1["open"], "first_high": f1["high"], "first_low": f1["low"],
            "first_close": f1["close"], "first_volume": f1.get("volume", ""),
            "day_high": dh["high"], "day_high_time": hhmm(dh["date"]),
            "high_after_first_min": ah["high"] if ah else "",
            "high_after_first_min_time": hhmm(ah["date"]) if ah else "",
            "day_low": dl["low"], "last_close": cs[-1]["close"],
            "day_high_vs_first_close_x": (round(dh["high"] / f1["close"], 2)
                                          if f1["close"] else ""),
        })
        if n % 10 == 0:
            print(f"    {n}/{len(picked)} options done")

    # fixed column order so every day's rows line up in the CSV
    rows = [{k: r.get(k, "") for k in OPTION_COLS} for r in rows]

    out = Path(out_dir)
    append_csv(out / "expiry_index.csv", [index_row])
    append_csv(out / "expiry_options.csv", rows)
    (out / "candles").mkdir(exist_ok=True)
    save_candles(out / "candles" / f"{day.isoformat()}_{index}.csv", raw)
    traded = [r for r in rows if r.get("traded")]
    print(f"[{index}] saved {len(traded)}/{len(rows)} options with trades "
          f"(index open {idx_open}, day range {day_lo['low']}-{day_hi['high']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--login", action="store_true", help="do the daily Kite login and save the token")
    ap.add_argument("--indices", default="SENSEX,NIFTY")
    ap.add_argument("--force", action="store_true", help="collect even if today isn't expiry")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()

    kite = kite_client(a.out, login=a.login)
    day = today_ist()
    now = dt.datetime.now(IST).time()
    if a.login and now < dt.time(15, 31):
        print("Login saved. Run again after 15:30 to collect today's data.")
        return
    if now < dt.time(15, 31) and not a.force:
        print("Market still open: run after 15:30 so the full day's candles exist "
              "(or use --force for a partial day).")
        return
    cache = {}
    for idx in [s.strip().upper() for s in a.indices.split(",") if s.strip()]:
        if idx not in CONFIG:
            print(f"Unknown index {idx}; choose from {', '.join(CONFIG)}")
            continue
        try:
            collect(kite, idx, CONFIG[idx], day, a.force, a.out, cache)
        except Exception as err:
            print(f"[{idx}] failed: {err}")


if __name__ == "__main__":
    main()
