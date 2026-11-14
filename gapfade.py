#!/usr/bin/env python3
"""
gapfade.py - 9:20 scanner for the "failed gap-up" short setup.

The setup, measured on NSE minute data from 2019 to April 2026:
    At 09:20, a stock qualifies when ALL THREE are true
      1. it opened >= 1.0% ABOVE yesterday's close          (the gap)
      2. it is now >= 0.4% BELOW today's open               (the failure)
      3. its first 5 minutes' volume is >= 1.2x its own     (the conviction)
         20-session average for those same 5 minutes
    Then: short it, stop-loss 1.5% above entry, cover at 10:20.

This script only SCANS and reports. It never places an order.
(Order placement through Kite Connect requires a static IP, and a trade this
size deserves a human pressing the button.)

--------------------------------------------------------------------------
SETUP (once)
    pip install kiteconnect
    Create an app at https://developers.kite.trade  (Kite Connect, Rs 500/month
    - the free "Personal" plan has NO market data and will not work here)
    export KITE_API_KEY=xxxx
    export KITE_API_SECRET=yyyy

DAILY USE
    python gapfade.py login          # once a day, before the market opens
    python gapfade.py prep           # ~2 min, builds the 20-day volume baseline
    python gapfade.py scan --watch   # start by 09:18, it fires at 09:20:05

    python gapfade.py backtest-check # sanity-check the baseline numbers
--------------------------------------------------------------------------
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, date

# ---------------------------------------------------------------- parameters
GAP_MIN = 1.0        # % above yesterday's close
DROP_MIN = 0.4       # % below today's open
RELVOL_MIN = 1.2     # multiple of the stock's own normal first-5-min volume
OPEN_MIN = 5         # minutes of the opening window used for the volume test
BASELINE_DAYS = 20   # sessions in the volume average
STOP_PCT = 1.5       # stop-loss, % above the entry price
POSITION = 2_000_000 # rupees per trade, for the suggested quantity
MAX_TRADES = 3       # most candidates to act on in one morning
SCAN_AT = "09:20"
COVER_AT = "10:20"

SYMBOLS = ("ABB,ADANIENSOL,ADANIENT,ADANIGREEN,ADANIPORTS,ADANIPOWER,AMBUJACEM,APOLLOHOSP,ASIANPAINT,"
 "ATGL,AXISBANK,BAJAJ-AUTO,BAJAJFINSV,BAJAJHLDNG,BAJFINANCE,BANKBARODA,BEL,BHARTIARTL,BHEL,BOSCHLTD,"
 "BPCL,BRITANNIA,CANBK,CGPOWER,CHOLAFIN,CIPLA,COALINDIA,CUMMINSIND,DABUR,DIVISLAB,DLF,DMART,DRREDDY,"
 "EICHERMOT,ETERNAL,GAIL,GODREJCP,GRASIM,HAL,HAVELLS,HCLTECH,HDFCAMC,HDFCBANK,HDFCLIFE,HEROMOTOCO,"
 "HINDALCO,HINDUNILVR,HINDZINC,ICICIBANK,ICICIGI,ICICIPRULI,INDHOTEL,INDIGO,INDUSINDBK,INFY,IOC,IRCTC,"
 "IRFC,ITC,JINDALSTEL,JIOFIN,JSWENERGY,JSWSTEEL,KOTAKBANK,LICI,LODHA,LT,LTIM,LTM,MARUTI,MAXHEALTH,MAZDOCK,"
 "M&M,MOTHERSON,MUTHOOTFIN,NAUKRI,NESTLEIND,NHPC,NTPC,ONGC,PFC,PIDILITIND,PNB,POWERGRID,RECLTD,RELIANCE,"
 "SBILIFE,SBIN,SHREECEM,SHRIRAMFIN,SIEMENS,SOLARINDS,SUNPHARMA,TATACONSUM,TMPV,TATAPOWER,TATASTEEL,"
 "TCS,TECHM,TITAN,TORNTPHARM,TRENT,TVSMOTOR,ULTRACEMCO,UNIONBANK,UNITDSPR,VBL,VEDL,WIPRO,ZYDUSLIFE").split(",")

HOME = os.path.expanduser("~/.gapfade")
TOKEN_FILE = os.path.join(HOME, "access_token.json")
BASE_FILE = os.path.join(HOME, "baseline.json")
LOG_FILE = os.path.join(HOME, "signals.csv")


# ------------------------------------------------------------------- helpers
def need(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"Set the {name} environment variable first (see the top of this file).")
    return v


def kite_client(require_token=True):
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        sys.exit("pip install kiteconnect")
    k = KiteConnect(api_key=need("KITE_API_KEY"))
    if require_token:
        if not os.path.exists(TOKEN_FILE):
            sys.exit("No access token. Run:  python gapfade.py login")
        tok = json.load(open(TOKEN_FILE))
        if tok.get("date") != str(date.today()):
            sys.exit(f"Token is from {tok.get('date')}. Kite tokens expire daily — run:  python gapfade.py login")
        k.set_access_token(tok["access_token"])
    return k


def cmd_login(_):
    os.makedirs(HOME, exist_ok=True)
    k = kite_client(require_token=False)
    print("\n1. Open this URL and log in to Zerodha:\n")
    print("   " + k.login_url())
    print("\n2. You land on a URL containing  request_token=XXXXX")
    rt = input("\n3. Paste the request_token here: ").strip()
    data = k.generate_session(rt, api_secret=need("KITE_API_SECRET"))
    json.dump({"access_token": data["access_token"], "date": str(date.today())}, open(TOKEN_FILE, "w"))
    os.chmod(TOKEN_FILE, 0o600)
    print(f"\nSaved. Valid until tomorrow morning.  ({TOKEN_FILE})")


def instrument_map(k):
    """tradingsymbol -> instrument_token for NSE equities."""
    out = {}
    for ins in k.instruments("NSE"):
        if ins.get("segment") == "NSE" and ins.get("instrument_type") == "EQ":
            out[ins["tradingsymbol"]] = ins["instrument_token"]
    return out


def open_window_volume(candles, minutes=OPEN_MIN):
    """Sum volume of the first `minutes` one-minute candles of each session."""
    by_day = {}
    for c in candles:
        t = c["date"]
        mins = t.hour * 60 + t.minute
        if 555 <= mins < 555 + minutes:          # 09:15 .. 09:15+minutes
            by_day.setdefault(t.date(), 0)
            by_day[t.date()] += c["volume"]
    return by_day


# ---------------------------------------------------------------------- prep
def cmd_prep(args):
    os.makedirs(HOME, exist_ok=True)
    k = kite_client()
    tokens = instrument_map(k)
    syms = args.symbols.split(",") if args.symbols else SYMBOLS
    to_d = date.today()
    from_d = to_d - timedelta(days=int(BASELINE_DAYS * 2.2))   # calendar days for ~20 sessions
    base, missing, failed = {}, [], []
    print(f"Building the {BASELINE_DAYS}-session opening-volume baseline for {len(syms)} stocks...")
    for i, s in enumerate(syms, 1):
        tok = tokens.get(s)
        if tok is None:
            missing.append(s)
            continue
        for attempt in range(3):
            try:
                candles = k.historical_data(tok, from_d, to_d, "minute")
                break
            except Exception as e:
                if attempt == 2:
                    failed.append(f"{s}: {e}")
                    candles = None
                time.sleep(1.5 * (attempt + 1))
        if not candles:
            continue
        vols = open_window_volume(candles)
        days = [d for d in sorted(vols) if d < date.today()][-BASELINE_DAYS:]   # never include today
        if len(days) < 10:
            failed.append(f"{s}: only {len(days)} sessions")
            continue
        vals = [vols[d] for d in days]
        base[s] = {"token": tok, "avg_open_vol": sum(vals) / len(vals), "sessions": len(vals),
                   "last_session": str(days[-1])}
        if i % 20 == 0:
            print(f"  {i}/{len(syms)}")
        time.sleep(0.34)                       # historical API allows 3 requests/second
    json.dump({"built": str(datetime.now()), "open_minutes": OPEN_MIN, "data": base}, open(BASE_FILE, "w"))
    print(f"\nSaved {len(base)} stocks to {BASE_FILE}")
    if missing:
        print(f"Not found on NSE (check the symbol): {', '.join(missing)}")
    if failed:
        print(f"Failed: {'; '.join(failed[:10])}")
    stale = [s for s, v in base.items() if v["last_session"] != max(x["last_session"] for x in base.values())]
    if stale:
        print(f"Note: {len(stale)} stocks have an older last session than the rest.")


# ---------------------------------------------------------------------- scan
def tick(x, step=0.05):
    """Round a stop price UP to the tick size so the order is accepted."""
    import math
    return math.ceil(round(x / step, 6)) * step


def hhmm_now():
    n = datetime.now()
    return n.hour * 60 + n.minute, n.strftime("%H:%M:%S")


def cmd_scan(args):
    if not os.path.exists(BASE_FILE):
        sys.exit("No baseline. Run:  python gapfade.py prep")
    blob = json.load(open(BASE_FILE))
    base = blob["data"]
    built = datetime.fromisoformat(blob["built"])
    age_days = (datetime.now() - built).days
    if age_days >= 5:
        print(f"WARNING: baseline is {age_days} days old. Re-run prep for accurate volume ratios.\n")
    k = kite_client()

    target_min = int(SCAN_AT[:2]) * 60 + int(SCAN_AT[3:])
    if args.watch:
        while True:
            mins, clock = hhmm_now()
            if mins >= target_min:
                break
            print(f"\r{clock}  waiting for {SCAN_AT} ...", end="", flush=True)
            time.sleep(5)
        print()
        time.sleep(5)                           # let the 09:19 minute settle

    mins, clock = hhmm_now()
    if not args.force and abs(mins - target_min) > 3:
        print(f"It is {clock}. This setup is defined at {SCAN_AT} exactly.")
        print("Readings taken at another time do not match what was tested. Use --force to run anyway.\n")
        if not args.watch:
            sys.exit(1)

    keys = [f"NSE:{s}" for s in base] + ["NSE:NIFTY 50"]
    quotes = {}
    for i in range(0, len(keys), 250):
        chunk = keys[i:i + 250]
        try:
            quotes.update(k.quote(chunk))
        except Exception as e:
            print(f"quote() failed for a batch: {e}")
    if not quotes:
        sys.exit("No quotes returned. Market closed, or your app lacks the data subscription "
                 "(the free Personal plan has no market data).")

    nif = quotes.get("NSE:NIFTY 50", {})
    nif_ohlc = nif.get("ohlc", {}) or {}
    nif_gap = None
    if nif_ohlc.get("open") and nif_ohlc.get("close"):
        nif_gap = (nif_ohlc["open"] / nif_ohlc["close"] - 1) * 100

    rows, skipped = [], 0
    for s, meta in base.items():
        q = quotes.get(f"NSE:{s}")
        if not q:
            skipped += 1
            continue
        o = q.get("ohlc") or {}
        op, prev_close, ltp = o.get("open"), o.get("close"), q.get("last_price")
        vol = q.get("volume", q.get("volume_traded"))
        if not op or not prev_close or not ltp or vol is None or not meta["avg_open_vol"]:
            skipped += 1
            continue
        gap = (op / prev_close - 1) * 100
        drop = (ltp / op - 1) * 100
        relvol = vol / meta["avg_open_vol"]
        rows.append({"sym": s, "gap": gap, "drop": drop, "relvol": relvol, "ltp": ltp,
                     "open": op, "prev_close": prev_close, "vol": vol,
                     "rel_gap": gap - nif_gap if nif_gap is not None else None,
                     "pass": gap >= GAP_MIN and drop <= -DROP_MIN and relvol >= RELVOL_MIN})

    hits = [r for r in rows if r["pass"]]
    # tie-break: the gap that is most specific to the stock rather than the market
    hits.sort(key=lambda r: (r["rel_gap"] if r["rel_gap"] is not None else r["gap"]), reverse=True)

    print("=" * 108)
    print(f"FAILED GAP-UP SCAN   {datetime.now():%Y-%m-%d %H:%M:%S}"
          f"   |   Nifty gap: {nif_gap:+.2f}%" if nif_gap is not None else "")
    print(f"filters: gap >= +{GAP_MIN}%   now <= -{DROP_MIN}% from open   first-{OPEN_MIN}min volume >= {RELVOL_MIN}x normal")
    print("=" * 108)
    if not hits:
        print("\nNo candidates. Do nothing today.\n")
        near = [r for r in rows if r["gap"] >= GAP_MIN and r["drop"] < 0]
        if near:
            near.sort(key=lambda r: r["drop"])
            print("Closest misses (gapped up and red, but failed another check):")
            for r in near[:5]:
                why = []
                if r["drop"] > -DROP_MIN:
                    why.append(f"only {r['drop']:+.2f}% from open")
                if r["relvol"] < RELVOL_MIN:
                    why.append(f"volume {r['relvol']:.1f}x")
                print(f"   {r['sym']:<13} gap {r['gap']:+5.2f}%  {' , '.join(why)}")
    else:
        print(f"\n{len(hits)} candidate(s). Ranked by how much the gap beat the market's gap.\n")
        hdr = f"{'#':<3}{'STOCK':<13}{'GAP':>7}{'FROM OPEN':>11}{'VOLUME':>9}{'vs NIFTY':>10}{'LTP':>10}{'QTY':>7}{'STOP':>10}"
        print(hdr); print("-" * len(hdr))
        for i, r in enumerate(hits, 1):
            qty = int(POSITION // r["ltp"])
            stop = tick(r["ltp"] * (1 + STOP_PCT / 100))
            rg = f"{r['rel_gap']:+.2f}%" if r["rel_gap"] is not None else "   -"
            flag = "" if i <= MAX_TRADES else "  (over daily limit)"
            print(f"{i:<3}{r['sym']:<13}{r['gap']:>+6.2f}%{r['drop']:>+10.2f}%{r['relvol']:>8.1f}x"
                  f"{rg:>10}{r['ltp']:>10.2f}{qty:>7}{stop:>10.2f}{flag}")
        print("\nOrders to place by hand, for each one you take:")
        for i, r in enumerate(hits[:MAX_TRADES], 1):
            qty = int(POSITION // r["ltp"])
            stop = tick(r["ltp"] * (1 + STOP_PCT / 100))
            print(f"  {i}. SELL {qty} {r['sym']} LIMIT near {r['ltp']:.2f} (not market), product MIS")
            print(f"     then SL-M BUY {qty} {r['sym']} trigger {stop:.2f}   (stop, {STOP_PCT}% above)")
            print(f"     cover the rest at {COVER_AT}")
        print(f"\nPosition {POSITION:,} per trade. A stop costs about Rs {POSITION*STOP_PCT/100:,.0f} plus charges.")
        print(f"Set a {COVER_AT} alarm now — the exit time is part of the setup.")

    os.makedirs(HOME, exist_ok=True)
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if new:
            f.write("timestamp,symbol,gap_pct,drop_pct,relvol,ltp,open,prev_close,nifty_gap,passed\n")
        for r in rows:
            if r["gap"] >= GAP_MIN or r["pass"]:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S},{r['sym']},{r['gap']:.3f},{r['drop']:.3f},"
                        f"{r['relvol']:.3f},{r['ltp']},{r['open']},{r['prev_close']},"
                        f"{'' if nif_gap is None else round(nif_gap,3)},{int(r['pass'])}\n")
    print(f"\nScanned {len(rows)} stocks ({skipped} skipped). Logged to {LOG_FILE}")
    print("Not investment advice. Past patterns do not guarantee future results.")


# --------------------------------------------------------------------- check
def cmd_check(_):
    """Print the baseline so you can eyeball whether the volume numbers look sane."""
    if not os.path.exists(BASE_FILE):
        sys.exit("Run prep first.")
    blob = json.load(open(BASE_FILE))
    d = blob["data"]
    print(f"baseline built {blob['built']}, opening window {blob['open_minutes']} minutes, {len(d)} stocks\n")
    rows = sorted(d.items(), key=lambda kv: -kv[1]["avg_open_vol"])
    print(f"{'STOCK':<14}{'AVG FIRST-5-MIN VOL':>22}{'SESSIONS':>10}{'LAST':>13}")
    for s, v in rows[:8]:
        print(f"{s:<14}{v['avg_open_vol']:>22,.0f}{v['sessions']:>10}{v['last_session']:>13}")
    print("   ...")
    for s, v in rows[-5:]:
        print(f"{s:<14}{v['avg_open_vol']:>22,.0f}{v['sessions']:>10}{v['last_session']:>13}")
    thin = [s for s, v in d.items() if v["avg_open_vol"] < 20000]
    if thin:
        print(f"\nThin opening volume (a 20 lakh order may move these): {', '.join(thin)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login", help="daily Kite login").set_defaults(func=cmd_login)
    pp = sub.add_parser("prep", help="build the 20-session opening-volume baseline")
    pp.add_argument("--symbols", default="", help="comma-separated override of the stock list")
    pp.set_defaults(func=cmd_prep)
    ps = sub.add_parser("scan", help="run the 09:20 scan")
    ps.add_argument("--watch", action="store_true", help="wait until 09:20 and fire automatically")
    ps.add_argument("--force", action="store_true", help="run even if it is not 09:20")
    ps.set_defaults(func=cmd_scan)
    sub.add_parser("backtest-check", help="show the stored volume baseline").set_defaults(func=cmd_check)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
