#!/usr/bin/env python3
"""
Signals engine for the open-labs "Signals" dashboard.

Runs the strategies we backtested on 11 years of NSE data (2015 - Apr 2026) against
live prices, writes the results to signals/data/latest.json and keeps a permanent
log (signals/data/log.json) so every past call is scored against what really happened.

    python signals/engine.py --section intraday   # 9:20 run  (gap-fade, first-minute fades)
    python signals/engine.py --section long       # 11:00 run (S9, 52w breakout, dip-buy)
    python signals/engine.py --section overnight  # 15:00 run (momentum carry, panic rebound)
    python signals/engine.py --section score      # re-score open calls only
    python signals/engine.py --section all        # everything (what the Refresh button runs)

Add --wait to sleep until the section's scan time (09:20:30 / 11:00 / 15:00) when started early.
Prices come from Yahoo Finance's public chart API (NSE feed, ~1 minute delay).
It only scans and reports - it never places orders.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IST = timezone(timedelta(hours=5, minutes=30))
PRIORS = json.load(open(os.path.join(HERE, "priors.json")))
VOL_PROFILE = PRIORS.pop("_volume_profile")  # cumulative share of day volume by minute (0 = 9:15)

UNIVERSE = ("ABB,ADANIENSOL,ADANIENT,ADANIGREEN,ADANIPORTS,ADANIPOWER,AMBUJACEM,APOLLOHOSP,ASIANPAINT,ATGL,"
            "AXISBANK,BAJAJ-AUTO,BAJAJFINSV,BAJAJHFL,BAJAJHLDNG,BAJFINANCE,BANKBARODA,BEL,BHARTIARTL,BHEL,"
            "BOSCHLTD,BPCL,BRITANNIA,CANBK,CGPOWER,CHOLAFIN,CIPLA,COALINDIA,CUMMINSIND,DABUR,DIVISLAB,DLF,"
            "DMART,DRREDDY,EICHERMOT,ENRIN,ETERNAL,GAIL,GODREJCP,GRASIM,HAL,HAVELLS,HCLTECH,HDFCAMC,HDFCBANK,"
            "HDFCLIFE,HEROMOTOCO,HINDALCO,HINDUNILVR,HINDZINC,HYUNDAI,ICICIBANK,ICICIGI,INDHOTEL,INDIGO,"
            "INDUSINDBK,INFY,IOC,IRCTC,IRFC,ITC,JINDALSTEL,JIOFIN,JSWENERGY,JSWSTEEL,KOTAKBANK,LICI,LODHA,LTM,"
            "LT,MARUTI,MAXHEALTH,MAZDOCK,M&M,MOTHERSON,MUTHOOTFIN,NAUKRI,NESTLEIND,NHPC,NTPC,ONGC,PFC,"
            "PIDILITIND,PNB,POWERGRID,RECLTD,RELIANCE,SBILIFE,SBIN,SHREECEM,SHRIRAMFIN,SIEMENS,SOLARINDS,"
            "SUNPHARMA,TATACONSUM,TATAPOWER,TATASTEEL,TCS,TECHM,TITAN,TMPV,TORNTPHARM,TRENT,TVSMOTOR,"
            "ULTRACEMCO,UNIONBANK,UNITDSPR,VBL,VEDL,WIPRO,ZYDUSLIFE").split(",")
NIFTY = "^NSEI"
MAX_GF_PER_DAY = 3
OVERNIGHT_COST = 0.25  # % round trip, delivery


# --------------------------------------------------------------------------- data
class Yahoo:
    HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
    UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

    def __init__(self):
        self.errors = []

    def chart(self, sym: str, interval: str, rng: str):
        ysym = sym if sym.startswith("^") else sym + ".NS"
        path = f"/v8/finance/chart/{urllib.parse.quote(ysym)}?interval={interval}&range={rng}&includePrePost=false"
        last = None
        for attempt in range(4):
            host = self.HOSTS[attempt % 2]
            try:
                req = urllib.request.Request(host + path, headers=self.UA)
                with urllib.request.urlopen(req, timeout=20) as r:
                    j = json.load(r)
                res = j["chart"]["result"][0]
                q = res["indicators"]["quote"][0]
                bars = []
                for i, t in enumerate(res.get("timestamp") or []):
                    c = q["close"][i]
                    if c is None or q["open"][i] is None:
                        continue
                    bars.append(dict(t=datetime.fromtimestamp(t, IST), o=q["open"][i], h=q["high"][i],
                                     l=q["low"][i], c=c, v=q["volume"][i] or 0))
                return bars
            except Exception as e:  # network, 429, bad symbol
                last = e
                time.sleep(1.5 * (attempt + 1))
        self.errors.append(f"{sym} {interval}: {last}")
        return []

    def daily(self, sym):
        return self.chart(sym, "1d", "3y")

    def m5(self, sym):
        return self.chart(sym, "5m", "60d")

    def m1(self, sym):
        return self.chart(sym, "1m", "1d")

    def m1_week(self, sym):  # used to score calls from earlier days (Yahoo keeps ~7 days of 1-minute bars)
        return self.chart(sym, "1m", "5d")


def fetch_all(fn, syms, workers=2):
    with ThreadPoolExecutor(workers) as ex:
        return dict(zip(syms, ex.map(fn, syms)))


# --------------------------------------------------------------------------- helpers
def now_ist():
    return datetime.now(IST)


def mean(a):
    a = [x for x in a if x is not None]
    return sum(a) / len(a) if a else float("nan")


def r2(x, n=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, n)


def tick(x):
    return round(round(x / 0.05) * 0.05, 2)


def minute_idx(t: datetime):
    return (t.hour - 9) * 60 + t.minute - 15


def vol_frac(t: datetime):
    k = max(14, min(374, minute_idx(t)))
    return VOL_PROFILE[k]


def stock_hist(strategy, sym):
    v = PRIORS[strategy]["by_stock"].get(sym)
    return v  # [n, win%, avg%] or None


def prob_note(strategy, sym):
    h = stock_hist(strategy, sym)
    if not h:
        return "No past signals on this stock."
    return f"This stock: {h[0]} past signal{'s' if h[0] != 1 else ''}, {h[1]:.0f}% won, avg {h[2]:+.1f}%."


def today_bars(bars, d: date):
    return [b for b in bars if b["t"].date() == d]


def split_daily(bars, d: date, completed_today: bool):
    """returns (completed bars, today's partial bar or None)"""
    if bars and bars[-1]["t"].date() == d and not completed_today:
        return bars[:-1], bars[-1]
    return bars, None


def sma(closes, n):
    return mean(closes[-n:]) if len(closes) >= n else float("nan")


# --------------------------------------------------------------------------- log
def load_json(name, default):
    p = os.path.join(DATA, name)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return default


def save_json(name, obj):
    os.makedirs(DATA, exist_ok=True)
    tmp = os.path.join(DATA, name + ".tmp")
    json.dump(obj, open(tmp, "w"), separators=(",", ":"), default=str)
    os.replace(tmp, os.path.join(DATA, name))


def make_signal(strategy, sym, d, side, entry, stop, target, why, entry_note, exit_note, extra=None):
    p = PRIORS[strategy]
    s = dict(id=f"{strategy}|{d}|{sym}", strategy=strategy, strategy_name=p["name"], section=p["section"],
             sym=sym, date=str(d), side=side, entry=r2(entry), entry_note=entry_note, stop=r2(stop) if stop else None,
             target=r2(target) if target else None, exit_note=exit_note, prob=p["win"], prob_n=p["n"],
             prob_oos=p.get("oos_win"), stock_hist=stock_hist(strategy, sym), why=why,
             status="open", result=None, created=now_ist().isoformat(timespec="seconds"))
    if extra:
        s.update(extra)
    return s


# --------------------------------------------------------------------------- 9:20 intraday
def scan_intraday(src, d, daily, nifty_daily):
    """Failed gap-up short (GF) + first-minute spike fades (OFL/OFS). Uses bars up to 9:19."""
    # Yahoo reports 0 volume for the 9:15 minute, so opening volume is measured on 9:16-9:19
    # (today vs the previous sessions in Yahoo's ~5-day 1-minute history).
    wk = fetch_all(src.m1_week if hasattr(src, "m1_week") else src.m1, UNIVERSE)
    m1 = {k: today_bars(v or [], d) for k, v in wk.items()}
    rows, watch, fades = [], [], []
    for s in UNIVERSE:
        db = daily.get(s) or []
        done, _ = split_daily(db, d, False)
        if not done:
            continue
        pc = done[-1]["c"]
        tb = today_bars(m1.get(s) or [], d)
        first = [b for b in tb if minute_idx(b["t"]) <= 4]
        if not first or minute_idx(first[0]["t"]) != 0:
            continue
        op = first[0]["o"]
        px = first[-1]["c"]  # 9:19 close = 9:20 price
        # opening 5-minute volume vs 20-session norm, measured on 5-min bars (consistent today vs history)
        fv = {}
        for b in wk.get(s) or []:
            if 1 <= minute_idx(b["t"]) <= 4:
                fv[b["t"].date()] = fv.get(b["t"].date(), 0) + b["v"]
        hist = [fv[k] for k in sorted(fv) if k < d][-20:]
        base = mean(hist) if len(hist) >= 3 else float("nan")
        rv = fv.get(d, float("nan")) / base if base and not math.isnan(base) else float("nan")
        gap = (op / pc - 1) * 100
        drop = (px / op - 1) * 100
        row = dict(sym=s, prev_close=r2(pc), open=r2(op), px=r2(px), gap=r2(gap), drop=r2(drop), rv=r2(rv))
        if gap >= 1.0 and drop <= -0.4 and rv >= 1.2:
            rows.append(row)
        elif gap >= 0.7 and drop <= 0 and (rv >= 1.0 or math.isnan(rv)):
            need = []
            if gap < 1.0:
                need.append(f"gap {gap:+.1f}% (needs ≥+1%)")
            if drop > -0.4:
                need.append(f"{drop:+.2f}% from open (needs ≤−0.4%)")
            if not (rv >= 1.2):
                need.append(f"volume {rv:.2f}× (needs ≥1.2×)" if not math.isnan(rv) else "volume baseline missing")
            watch.append(dict(sym=s, strategy="GF", px=r2(px), note="; ".join(need),
                              closeness=r2(abs(drop + 0.4) + max(0, 1 - gap)), extra=row))
        # first-minute spike (9:15 candle) - signal fires at 9:16
        c0 = first[0]["c"]
        mv = (c0 / op - 1) * 100
        if abs(mv) >= 3:
            fades.append(dict(sym=s, move=mv, open=op, c0=c0, entry=(tb[1]["o"] if len(tb) > 1 else c0)))
    sigs = []
    rows.sort(key=lambda r: -r["gap"])
    for i, r in enumerate(rows):
        e = r["px"]
        sig = make_signal("GF", r["sym"], d, "SHORT", e, tick(e * 1.015),
                          tick(e * (1 - PRIORS["GF"]["med_win"] / 100)),
                          f"Gap {r['gap']:+.1f}%, now {r['drop']:+.2f}% from open, opening volume {r['rv']:.1f}× normal. "
                          + prob_note("GF", r["sym"]),
                          "Short at the 9:20 price", "Cover at 10:20 (or stop)",
                          dict(rank=i + 1, over_limit=i >= MAX_GF_PER_DAY, exit_due=f"{d} 10:20"))
        sigs.append(sig)
    if fades:
        f = max(fades, key=lambda x: abs(x["move"]))  # one trade a day: the largest spike
        code = "OFL" if f["move"] < 0 else "OFS"
        e = f["entry"]
        side = "BUY" if code == "OFL" else "SHORT"
        sg = 1 if side == "BUY" else -1
        sigs.append(make_signal(code, f["sym"], d, side, e, tick(e * (1 - sg * 0.01)),
                                tick(e * (1 + sg * PRIORS[code]["med_win"] / 100)),
                                f"First 1-minute candle {f['move']:+.1f}%. Already live at 9:16 — this is a record, "
                                f"not a fresh entry. " + prob_note(code, f["sym"]),
                                "9:16 open", "Trailing stop 0.5% or 9:30", dict(exit_due=f"{d} 09:30", late=True)))
    watch.sort(key=lambda w: w["closeness"])
    return sigs, watch[:25], m1


# --------------------------------------------------------------------------- 11:00 long term
def tier_of(h, c_i, i):
    """prior-high tiers (2y > 1y > 6m) computed on highs before index i"""
    out = None
    for name, L in (("2-year+", 504), ("1-year", 252), ("6-month", 126)):
        if i - L < 0:
            continue
        window = h[i - L:i]
        m = max(window)
        if c_i > m:
            age = i - (i - L + window.index(m))
            out = (name, m, age)
            break
    return out


def scan_long(d, daily, nifty_daily, t_now, completed_today):
    sigs, watch = [], []
    nd, ntoday = split_daily(nifty_daily, d, completed_today)
    ncl = [b["c"] for b in nd] + ([ntoday["c"]] if ntoday else [])
    n_last = ncl[-1]
    healthy = n_last > sma(ncl, 50) and n_last > sma(ncl, 200)
    frac = vol_frac(t_now) if not completed_today else 1.0
    for s in UNIVERSE:
        bars = daily.get(s) or []
        done, part = split_daily(bars, d, completed_today)
        if len(done) < 270:
            continue
        o = [b["o"] for b in done]; h = [b["h"] for b in done]; c = [b["c"] for b in done]
        v = [b["v"] for b in done]; dts = [b["t"].date() for b in done]
        n = len(done)
        cur = part["c"] if part else c[-1]
        today_open = part["o"] if part else None
        pv = (part["v"] / frac) if part and frac > 0 else None  # projected full-day volume

        def feat(i):
            v50 = mean(v[i - 50:i]); base = mean(v[i - 59:i - 9])
            return dict(vr=v[i] / v50 if v50 else 0, v10=mean(v[i - 9:i + 1]) / base if base else 0, base=base,
                        d200=mean(c[i - 199:i + 1]), r5=c[i] / c[i - 5] - 1, hi252=max(h[i - 252:i]))

        # ---- S9: breakouts in the last 14 completed sessions
        for i in range(max(270, n - 14), n):
            t = tier_of(h, c[i], i)
            if not t:
                continue
            f = feat(i)
            name, lvl, age = t
            if not (age >= 20 and f["vr"] >= 2 and f["v10"] >= 1.5):
                continue
            j = i + 10
            post = v[i + 1:min(j, n - 1) + 1]
            vpost = mean(post) / f["base"] if post else float("nan")
            info = (f"{name} high {lvl:,.1f} broken on {dts[i]} with {f['vr']:.1f}× volume; "
                    f"since then volume {vpost:.2f}× normal (needs ≥1.3×), price {(cur / lvl - 1) * 100:+.1f}% vs level")
            if j <= n - 1:
                if c[j] > lvl and vpost >= 1.3 and j >= n - 5:
                    entry_day = dts[j + 1] if j + 1 < n else d
                    entry = o[j + 1] if j + 1 < n else today_open
                    if entry is None:
                        continue
                    sigs.append(make_signal("S9", s, entry_day, "BUY", entry, None,
                                            entry * (1 + PRIORS["S9"]["med_win"] / 100),
                                            info + ". Confirmed " + str(dts[j]) + ". " + prob_note("S9", s),
                                            f"Open on {entry_day}", "Sell after 63 sessions",
                                            dict(level=r2(lvl), hold=63)))
            else:
                left = j - (n - 1)
                ok = cur > lvl and (vpost >= 1.3 or math.isnan(vpost))
                watch.append(dict(sym=s, strategy="S9", px=r2(cur),
                                  note=(f"Confirms {'at the next close' if left == 1 else f'in {left} sessions'}"
                                        f" — {'on track' if ok else 'currently failing'}. " + info),
                                  closeness=left + (0 if ok else 20)))
        # ---- B4 (yesterday's completed bar -> buy today's open) and A2/A5 dip
        i = n - 1
        f = feat(i)
        if healthy and c[i] > f["hi252"] and f["vr"] > 2 and part:
            sigs.append(make_signal("B4", s, d, "BUY", today_open, None,
                                    today_open * (1 + PRIORS["B4"]["med_win"] / 100),
                                    f"Closed {dts[i]} at {c[i]:,.1f}, above the 52-week high {f['hi252']:,.1f}, on {f['vr']:.1f}× volume; NIFTY healthy. "
                                    + prob_note("B4", s), "Today's open", "Sell after 63 sessions", dict(hold=63)))
        if f["r5"] < -0.12 and c[i] > f["d200"] and part:
            for code in ("A2", "A5"):
                tgt = today_open * 1.05 if code == "A5" else today_open * (1 + PRIORS["A2"]["med_win"] / 100)
                stp = today_open * 0.90 if code == "A5" else None
                sigs.append(make_signal(code, s, d, "BUY", today_open, stp, tgt,
                                        f"Fell {f['r5'] * 100:.1f}% in 5 days to {dts[i]} while above its 200-day average {f['d200']:,.1f}. "
                                        + prob_note(code, s), "Today's open",
                                        "Day 21 close" if code == "A2" else "+5% / −10% / day 21", dict(hold=21)))
        # ---- forming today / near misses (could fire in a day or two)
        if part:
            hi252 = max(h[-252:])
            d200 = mean((c + [cur])[-200:])
            vr_proj = pv / mean(v[-50:]) if pv else 0
            dist = (cur / hi252 - 1) * 100
            if healthy and dist > -2.0 and vr_proj >= 1.2:
                state = ("above the 52-week high now" if dist > 0 else f"{dist:.1f}% below its 52-week high {hi252:,.1f}")
                watch.append(dict(sym=s, strategy="B4", px=r2(cur),
                                  note=f"{state}; projected volume {vr_proj:.1f}× (needs >2× at the close)",
                                  closeness=abs(min(dist, 0)) + max(0, 2 - vr_proj)))
            r5 = cur / c[-5] - 1
            if r5 < -0.09 and cur > d200:
                watch.append(dict(sym=s, strategy="A2", px=r2(cur),
                                  note=f"down {r5 * 100:.1f}% over 5 sessions (triggers below −12% at a close), still above 200-day avg {d200:,.0f}",
                                  closeness=abs(r5 * 100 + 12)))
    watch.sort(key=lambda w: (w["closeness"]))
    return sigs, watch[:40], healthy


# --------------------------------------------------------------------------- 15:00 overnight
def scan_overnight(src, d, daily, nifty_daily, m1=None):
    m1 = m1 or fetch_all(src.m1, UNIVERSE + [NIFTY])
    if NIFTY not in m1:
        m1[NIFTY] = src.m1(NIFTY)
    nd, _ = split_daily(nifty_daily, d, False)
    ntb = today_bars(m1.get(NIFTY) or [], d)
    if not ntb or not nd:
        return [], [], None
    nret = (ntb[-1]["c"] / nd[-1]["c"] - 1) * 100
    sigs, watch = [], []
    for s in UNIVERSE:
        done, _ = split_daily(daily.get(s) or [], d, False)
        tb = today_bars(m1.get(s) or [], d)
        if len(done) < 210 or not tb:
            continue
        pc = done[-1]["c"]; px = tb[-1]["c"]
        r = (px / pc - 1) * 100
        frac = vol_frac(tb[-1]["t"])
        cumv = sum(b["v"] for b in tb)
        v20 = mean([b["v"] for b in done[-20:]])
        vr = (cumv / frac) / v20 if v20 else 0
        d200 = mean([b["c"] for b in done[-199:]] + [px])
        if r >= 5 and vr >= 3:
            sigs.append(make_signal("O1", s, d, "BUY", px, None, px * (1 + PRIORS["O1"]["med_win"] / 100),
                                    f"Up {r:+.1f}% on {vr:.1f}× projected volume. " + prob_note("O1", s),
                                    "~15:00 price", "Sell at tomorrow's open"))
        elif r >= 4 and vr >= 2:
            watch.append(dict(sym=s, strategy="O1", px=r2(px), note=f"up {r:+.1f}% (needs ≥5%), volume {vr:.1f}× (needs ≥3×)",
                              closeness=max(0, 5 - r) + max(0, 3 - vr)))
        if r <= -3 and nret <= -1.5 and px > d200:
            sigs.append(make_signal("O2", s, d, "BUY", px, None, px * (1 + PRIORS["O2"]["med_win"] / 100),
                                    f"Down {r:.1f}% with NIFTY {nret:+.1f}%, still above its 200-day average {d200:,.0f}. "
                                    + prob_note("O2", s), "~15:00 price", "Sell at tomorrow's close"))
        elif r <= -2.5 and nret <= -1.0 and px > d200:
            watch.append(dict(sym=s, strategy="O2", px=r2(px),
                              note=f"down {r:.1f}% (needs ≤−3%), NIFTY {nret:+.1f}% (needs ≤−1.5%)",
                              closeness=max(0, r + 3) + max(0, nret + 1.5)))
    watch.sort(key=lambda w: w["closeness"])
    return sigs, watch[:25], r2(nret)


# --------------------------------------------------------------------------- scoring
def score(sig, src, daily, m1_cache, d_now, t_now):
    """Update sig['result'] / sig['status'] from prices. Returns True if changed."""
    s, k, side = sig["sym"], sig["strategy"], sig["side"]
    sg = 1 if side == "BUY" else -1
    e = sig["entry"]
    if not e:
        return False
    sd = date.fromisoformat(sig["date"])

    def done(pnl, note, exit_px):
        sig["status"] = "closed"
        sig["result"] = dict(pnl=r2(pnl), outcome="win" if pnl > 0 else "loss", note=note, exit=r2(exit_px),
                             scored=t_now.isoformat(timespec="minutes"))

    def live(pnl, note):
        sig["result"] = dict(pnl=r2(pnl), outcome="open", note=note, scored=t_now.isoformat(timespec="minutes"))

    if k in ("GF", "OFL", "OFS"):
        if sd == d_now:
            bars = m1_cache.get(s)
            if bars is None:
                bars = m1_cache[s] = src.m1(s)
        else:
            key = s + "|week"
            bars = m1_cache.get(key)
            if bars is None:
                bars = m1_cache[key] = (src.m1_week(s) if hasattr(src, "m1_week") else [])
        tb = today_bars(bars, sd)
        if not tb:
            if d_now > sd:  # can't see that day any more (1m data only covers today)
                sig["status"] = "closed"
                sig["result"] = sig.get("result") or dict(pnl=None, outcome="unscored", note="minute data no longer available")
                return True
            return False
        start = 5 if k == "GF" else 1
        end = 64 if k == "GF" else 14  # last minute index held (10:19 / 9:29)
        held = [b for b in tb if start <= minute_idx(b["t"]) <= end]
        if not held:
            return False
        stop = e * (1 + 0.015) if k == "GF" else e * (1 - sg * 0.01)
        best = e
        for b in held:
            adverse = b["h"] if sg < 0 else b["l"]
            if (sg < 0 and adverse >= stop) or (sg > 0 and adverse <= stop):
                pnl = sg * (stop / e - 1) * 100 - 0.1
                done(pnl, "stopped" if abs(stop / e - 1) >= 0.0099 or k == "GF" else "trailing stop", stop)
                return True
            if k != "GF":
                fav = b["h"] if sg > 0 else b["l"]
                best = max(best, fav) if sg > 0 else min(best, fav)
                ts = best * (1 - sg * 0.005)
                stop = max(stop, ts) if sg > 0 else min(stop, ts)
        last = held[-1]
        pnl = sg * (last["c"] / e - 1) * 100 - 0.1
        if minute_idx(last["t"]) >= end or d_now > sd:
            done(pnl, "time exit 10:20" if k == "GF" else "time exit 9:30", last["c"])
        else:
            live(pnl, f"live at {last['t']:%H:%M}")
        return True

    bars = daily.get(s)
    if bars is None:
        bars = daily[s] = src.daily(s)
    after = [b for b in bars if b["t"].date() >= sd]
    if k in ("O1", "O2"):
        nxt = [b for b in bars if b["t"].date() > sd]
        if not nxt:
            return False
        b = nxt[0]
        if k == "O1":
            if minute_idx(t_now) < 0 and b["t"].date() == d_now:
                return False
            done((b["o"] / e - 1) * 100 - OVERNIGHT_COST, f"sold at {b['t']:%d %b} open", b["o"])
            return True
        closed = b["t"].date() < d_now or t_now.hour * 60 + t_now.minute >= 15 * 60 + 35
        if closed:
            done((b["c"] / e - 1) * 100 - OVERNIGHT_COST, f"sold at {b['t']:%d %b} close", b["c"])
        else:
            live((b["c"] / e - 1) * 100, "live (sells at today's close)")
        return True
    # long-term: S9/B4/A2/A5 (entry = open of signal date)
    if not after:
        return False
    hold = sig.get("hold") or PRIORS[k].get("hold_n", 63)
    closed_bars = [b for b in after if b["t"].date() < d_now or t_now.hour * 60 + t_now.minute >= 15 * 60 + 35]
    if k == "A5":
        for i, b in enumerate(after):
            if b["l"] <= e * 0.90:
                done(-10.0 - 0.25, f"stop hit {b['t']:%d %b}", e * 0.9); return True
            if b["h"] >= e * 1.05:
                done(5.0 - 0.25, f"target hit {b['t']:%d %b}", e * 1.05); return True
            if i + 1 >= hold and b in closed_bars:
                done((b["c"] / e - 1) * 100 - 0.25, f"day-{hold} exit {b['t']:%d %b}", b["c"]); return True
    elif len(closed_bars) >= hold:
        b = closed_bars[hold - 1]
        done((b["c"] / e - 1) * 100 - 0.25, f"{hold}-session exit {b['t']:%d %b}", b["c"])
        return True
    cur = after[-1]["c"]
    live((cur / e - 1) * 100, f"day {len(after)}/{hold}")
    return True


def accuracy(log):
    out = {}
    for sec in ("intraday", "long", "overnight"):
        cl = [s for s in log.values() if s["section"] == sec and s["status"] == "closed"
              and s.get("result") and s["result"].get("pnl") is not None and not s.get("over_limit") and not s.get("late")]
        opn = [s for s in log.values() if s["section"] == sec and s["status"] == "open"]
        if cl:
            wins = sum(1 for s in cl if s["result"]["pnl"] > 0)
            out[sec] = dict(closed=len(cl), wins=wins, hit=round(wins / len(cl) * 100, 1),
                            expected=round(mean([s["prob"] for s in cl]), 1),
                            avg_pnl=round(mean([s["result"]["pnl"] for s in cl]), 2), open=len(opn))
        else:
            out[sec] = dict(closed=0, wins=0, hit=None, expected=None, avg_pnl=None, open=len(opn))
    return out


# --------------------------------------------------------------------------- main
SCAN_TIME = {"intraday": (9, 20, 30), "long": (11, 0, 0), "overnight": (15, 0, 0)}


def wait_until(section):
    if section not in SCAN_TIME:
        return
    h, m, s = SCAN_TIME[section]
    target = now_ist().replace(hour=h, minute=m, second=s, microsecond=0)
    delay = (target - now_ist()).total_seconds()
    if 0 < delay < 3 * 3600:
        print(f"waiting {delay:.0f}s until {target:%H:%M:%S} IST", flush=True)
        time.sleep(delay)


def run(section, src=None, t_now=None):
    src = src or Yahoo()
    t_now = t_now or now_ist()
    d = t_now.date()
    log = load_json("log.json", {})
    latest = load_json("latest.json", {"sections": {}})
    nifty_daily = src.daily(NIFTY)
    if not nifty_daily:  # feed unreachable: say so instead of pretending the market is shut
        latest["generated"] = t_now.isoformat(timespec="seconds")
        latest["errors"] = (src.errors if hasattr(src, "errors") else [])[-20:]
        latest["data_problem"] = "The price feed could not be reached, so nothing was updated."
        latest["market_open_today"] = None
        for v in latest.get("sections", {}).values():
            v.pop("note", None)
        save_json("latest.json", latest)
        print("price feed unavailable:", latest["errors"][:3])
        return latest
    latest.pop("data_problem", None)
    market_open_today = nifty_daily[-1]["t"].date() == d
    completed_today = t_now.hour * 60 + t_now.minute >= 15 * 60 + 35
    daily = fetch_all(src.daily, UNIVERSE)
    m1_cache = {}
    sections = [section] if section in SCAN_TIME else (list(SCAN_TIME) if section == "all" else [])
    mins = t_now.hour * 60 + t_now.minute
    for sec in sections:
        h, m, _ = SCAN_TIME[sec]
        if not market_open_today:
            latest["sections"].setdefault(sec, {})
            latest["sections"][sec].update(run_at=t_now.isoformat(timespec="minutes"), note="Market closed today — showing the last trading day's calls.")
            continue
        if mins < h * 60 + m - 1 and sec == "intraday":
            latest["sections"].setdefault(sec, {})
            latest["sections"][sec].update(run_at=t_now.isoformat(timespec="minutes"),
                                           note=f"Too early — this scan runs at {h:02d}:{m:02d}.")
            continue
        extra = {}
        if sec == "intraday":
            sigs, watch, m1 = scan_intraday(src, d, daily, nifty_daily)
            m1_cache.update(m1)
        elif sec == "long":
            sigs, watch, healthy = scan_long(d, daily, nifty_daily, t_now, completed_today)
            extra["nifty_healthy"] = healthy
        else:
            if not m1_cache:
                m1_cache.update(fetch_all(src.m1, UNIVERSE + [NIFTY]))
            sigs, watch, nret = scan_overnight(src, d, daily, nifty_daily, m1_cache)
            extra["nifty_ret"] = nret
            if mins < 15 * 60:
                extra["preview"] = True
                extra["note"] = "Preview before 15:00 — calls are only logged by the 15:00 scan."
        if sec == "overnight" and mins < 15 * 60:
            pass  # preview: list but don't log before 15:00
        for s in sigs:
            if sec == "overnight" and mins < 15 * 60:
                s["preview"] = True
                continue
            old = log.get(s["id"])
            if old:  # keep the first call (entry price, created time) - never rewrite history
                continue
            log[s["id"]] = s
        latest["sections"][sec] = dict(run_at=t_now.isoformat(timespec="minutes"), date=str(d),
                                       preview_signals=[s for s in sigs if s.get("preview")],
                                       watch=watch, **extra)
    # score everything still open
    for sig in log.values():
        if sig["status"] == "open":
            try:
                score(sig, src, daily, m1_cache, d, t_now)
            except Exception as e:
                sig.setdefault("errors", []).append(str(e)[:120])
    latest["generated"] = t_now.isoformat(timespec="seconds")
    latest["market_open_today"] = market_open_today
    latest["accuracy"] = accuracy(log)
    latest["priors"] = {k: {kk: v[kk] for kk in ("name", "section", "side", "rule", "entry", "exit", "stop", "target_pct",
                                                    "n", "win", "avg", "med_win", "med_loss", "p10", "oos_n", "oos_win", "oos_avg")}
                        for k, v in PRIORS.items()}
    latest["errors"] = src.errors[-20:] if hasattr(src, "errors") else []
    # recent window of the log for the page (all open + last 60 days closed)
    cutoff = str(d - timedelta(days=90))
    latest["signals"] = sorted([s for s in log.values() if s["status"] == "open" or s["date"] >= cutoff],
                               key=lambda s: (s["date"], s["section"], s["sym"]), reverse=True)
    save_json("log.json", log)
    save_json("latest.json", latest)
    print(f"{t_now:%Y-%m-%d %H:%M} section={section} market_open={market_open_today} "
          f"signals_logged={len(log)} errors={len(latest['errors'])}")
    return latest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", choices=["intraday", "long", "overnight", "score", "all"])
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    if a.wait:
        wait_until(a.section)
    run(a.section)
