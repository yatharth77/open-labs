#!/usr/bin/env python3
"""
export_for_claude.py - turn the multi-GB 1-minute dataset
(thetrademarkk/india-index-options-1m) into a small, git-friendly folder of
CSV + Markdown files that Claude (or you) can read directly and analyse.

Output folder (default ./claude_data):
  README.md                       data dictionary + how to ask Claude questions
  SUMMARY.md                      headline stats per index (computed, not guessed)
  {SYM}_daily.csv                 one row per trading day: gap, range, intraday extremes, expiry flag
  {SYM}_expiry_days.csv           one row per expiry session: moves from 9:15/9:20, break-even touch/close flags
  {SYM}_expiry_strikes.csv        one row per OTM strike on each expiry day: 9:15 candle,
                                  premium snapshots through the day, day high + time, settlement
  {SYM}_expiry_index_5min.csv     index 5-minute path on every expiry session
  {SYM}_expiry_strikes_5min.csv   (only with --strike-5min) 5-min premium path per strike

Rules applied everywhere (same as options_lab.py):
  * timestamps converted to IST; pre-open bars (< 09:15) dropped
  * "settle" = average of the last 30 one-minute closes (approximates the exchange's settlement)
  * snapshot "at HH:MM" = last traded close at or before that minute (no look-ahead)
  * only OTM strikes kept, distance measured from the 09:15 index open

Usage:
  # from Hugging Face (cached after first download)
  python3 export_for_claude.py
  # from a local copy made with `hf download ... --local-dir ~/india-options-data`
  python3 export_for_claude.py --local ~/india-options-data
  # limit scope
  python3 export_for_claude.py --symbols SENSEX,NIFTY --since 2024-01-01 --max-otm 6
"""
import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "thetrademarkk/india-index-options-1m"
FIRST_TOD = 9 * 60 + 15
CLOSE_TOD = 15 * 60 + 30
SNAP_TIMES = ["09:15", "09:16", "09:20", "09:30", "09:45", "10:00", "10:30", "11:00",
              "11:30", "12:00", "13:00", "14:00", "14:30", "15:00", "15:15", "15:29"]
MOVES = [1.0, 1.5, 2.0, 2.5, 3.0]
GIT_SOFT_LIMIT_MB = 45  # GitHub warns at 50 MB and rejects at 100 MB


# ---------------------------------------------------------------- data access
def list_files(local):
    if local:
        return sorted(p.relative_to(local).as_posix() for p in Path(local).rglob("*.parquet"))
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(REPO, repo_type="dataset")


def fetch(path, local):
    if local:
        return str(Path(local) / path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, path, repo_type="dataset")


def to_ist(ts):
    ts = pd.to_datetime(ts)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC") if ts.dt.hour.min() < 5 else ts.dt.tz_localize("Asia/Kolkata")
    return ts.dt.tz_convert("Asia/Kolkata")


def add_time_cols(df):
    df["ts"] = to_ist(df["timestamp"])
    local = df["ts"].dt.tz_localize(None)
    df["day"] = local.dt.normalize()
    df["tod"] = local.dt.hour * 60 + local.dt.minute
    return df


def hm(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def tod_str(t):
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return ""
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def load_index(path, local):
    df = pd.read_parquet(fetch(path, local)).rename(columns=str.lower)
    df = add_time_cols(df)
    df = df[(df.tod >= FIRST_TOD) & (df.tod <= CLOSE_TOD)]
    return df.sort_values("ts")[["day", "tod", "open", "high", "low", "close"]].reset_index(drop=True)


def norm_opt(df):
    df = df.rename(columns=str.lower)
    sym = df["symbol"].astype(str).str.upper().str.strip() if "symbol" in df else None
    ot = (df["option_type"].astype(str).str.upper().str.strip() if "option_type" in df
          else pd.Series("", index=df.index))
    ot = ot.replace({"CALL": "CE", "C": "CE", "PUT": "PE", "P": "PE"})
    if sym is not None:
        bad = ~ot.isin(["CE", "PE"])
        if bad.any():
            ot[bad] = sym[bad].str.extract(r"(CE|PE)$")[0]
    df["option_type"] = ot
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce") if "strike" in df else np.nan
    df = df.dropna(subset=["option_type", "strike"])
    df = df[df.option_type.isin(["CE", "PE"])]
    df = add_time_cols(df)
    for col in ("open", "high", "low"):
        if col not in df:
            df[col] = df["close"]
    for col in ("volume", "open_interest"):
        if col not in df:
            df[col] = np.nan
    df = df[(df.tod >= FIRST_TOD) & (df.tod <= CLOSE_TOD)]
    return df[["day", "tod", "strike", "option_type", "open", "high", "low", "close",
               "volume", "open_interest"]]


def expiry_session(exp, opt_days, spot_days):
    days = sorted(d for d in opt_days if d <= exp)
    if not days:
        return None
    if exp in spot_days:
        return exp if days[-1] == exp else None
    d = days[-1]  # expiry fell on a holiday -> previous session
    return d if (exp - d).days <= 3 else None


# ---------------------------------------------------------------- daily index table
def daily_table(sp, expiry_sessions):
    rows = []
    prev_close = np.nan
    exp_sorted = sorted(expiry_sessions)
    for d, g in sp.groupby("day", sort=True):
        g = g.sort_values("tod")
        o, c = g.open.iloc[0], g.close.iloc[-1]
        hi, lo = g.high.max(), g.low.min()
        t_hi = g.tod.iloc[int(g.high.values.argmax())]
        t_lo = g.tod.iloc[int(g.low.values.argmin())]
        at920 = g[g.tod <= hm("09:20")].close
        nxt = next((e for e in exp_sorted if e >= d), None)
        rows.append({
            "date": d.date(), "weekday": d.day_name()[:3],
            "prev_close": round(prev_close, 2) if prev_close == prev_close else "",
            "open": round(o, 2), "high": round(hi, 2), "low": round(lo, 2), "close": round(c, 2),
            "spot_0920": round(at920.iloc[-1], 2) if len(at920) else "",
            "gap_pct": round((o / prev_close - 1) * 100, 3) if prev_close == prev_close else "",
            "open_to_close_pct": round((c / o - 1) * 100, 3),
            "close_to_close_pct": round((c / prev_close - 1) * 100, 3) if prev_close == prev_close else "",
            "range_pct": round((hi - lo) / o * 100, 3),
            "max_up_from_open_pct": round((hi / o - 1) * 100, 3),
            "max_down_from_open_pct": round((lo / o - 1) * 100, 3),
            "time_of_high": tod_str(t_hi), "time_of_low": tod_str(t_lo),
            "is_expiry": int(d in expiry_sessions),
            "sessions_to_next_expiry": "",  # filled below
            "next_expiry_session": nxt.date() if nxt is not None else "",
            "minutes": len(g),
        })
        prev_close = c
    df = pd.DataFrame(rows)
    # trading sessions (not calendar days) until the next expiry session
    dates = list(df["date"])
    pos = {dd: i for i, dd in enumerate(dates)}
    df["sessions_to_next_expiry"] = [
        (pos.get(n) - pos[dd]) if n != "" and n in pos else "" for dd, n in zip(dates, df.next_expiry_session)]
    return df


# ---------------------------------------------------------------- one expiry session
def snap(g, t):
    """last close at or before minute t (no look-ahead)."""
    x = g[g.tod <= t]
    return round(float(x.close.iloc[-1]), 2) if len(x) else np.nan


def process_expiry(sym, exp, opt, sp_day, prev_close, cfg):
    sess = sp_day.sort_values("tod")
    o915 = sess.open.iloc[0]
    s920 = sess[sess.tod <= hm("09:20")].close.iloc[-1]
    close = sess.close.iloc[-1]
    settle_idx = sess.close.iloc[-30:].mean()
    after = sess[sess.tod >= hm("09:20")]
    hi_after, lo_after = after.high.max(), after.low.min()

    day = {
        "expiry_date": exp.date(), "session": sess.day.iloc[0].date(),
        "weekday": sess.day.iloc[0].day_name()[:3],
        "prev_close": round(prev_close, 2) if prev_close == prev_close else "",
        "open_0915": round(o915, 2), "spot_0920": round(s920, 2),
        "high": round(sess.high.max(), 2), "low": round(sess.low.min(), 2),
        "close": round(close, 2), "settle_avg30": round(settle_idx, 2),
        "gap_pct": round((o915 / prev_close - 1) * 100, 3) if prev_close == prev_close else "",
        "max_up_from_0920_pct": round((hi_after / s920 - 1) * 100, 3),
        "max_down_from_0920_pct": round((lo_after / s920 - 1) * 100, 3),
        "close_vs_0920_pct": round((close / s920 - 1) * 100, 3),
        "time_of_high": tod_str(sess.tod.iloc[int(sess.high.values.argmax())]),
        "time_of_low": tod_str(sess.tod.iloc[int(sess.low.values.argmin())]),
    }
    for m in MOVES:
        k = str(m).replace(".0", "")
        day[f"touched_{k}pct_from_0920"] = int(hi_after >= s920 * (1 + m / 100) or lo_after <= s920 * (1 - m / 100))
        day[f"closed_beyond_{k}pct"] = int(abs(settle_idx / s920 - 1) * 100 >= m)

    # ---- strikes
    rows, paths = [], []
    o = opt[opt.day == sess.day.iloc[0]]
    lim = o915 * cfg.max_otm / 100
    o = o[(o.strike >= o915 - lim) & (o.strike <= o915 + lim)]
    snaps = [(t, hm(t)) for t in SNAP_TIMES]
    for (k, cp), g in o.groupby(["strike", "option_type"], sort=False):
        dist = (k - o915) if cp == "CE" else (o915 - k)
        dist_pct = dist / o915 * 100
        if dist < 0 or dist_pct > cfg.max_otm:
            continue
        g = g.sort_values("tod")
        first = g[g.tod == FIRST_TOD]
        f_o = f_h = f_l = f_c = np.nan
        if len(first):
            f = first.iloc[0]
            f_o, f_h, f_l, f_c = f.open, f.high, f.low, f.close
        entry_mid = (f_h + f_l) / 2 if f_h == f_h else np.nan
        if not (entry_mid == entry_mid) or entry_mid < cfg.min_prem:
            continue
        later = g[g.tod > FIRST_TOD]
        if len(later):
            i_hi = int(later.high.values.argmax())
            day_hi_after, t_hi = later.high.iloc[i_hi], later.tod.iloc[i_hi]
        else:
            day_hi_after, t_hi = np.nan, np.nan
        day_hi = g.high.max()
        t_day_hi = g.tod.iloc[int(g.high.values.argmax())]
        settle = g.close.iloc[-30:].mean() if g.tod.iloc[-1] >= CLOSE_TOD - 15 else np.nan
        intrinsic = max(0.0, (settle_idx - k) if cp == "CE" else (k - settle_idx))
        r = {
            "expiry_date": exp.date(), "session": sess.day.iloc[0].date(), "type": cp,
            "strike": int(k) if float(k).is_integer() else k,
            "dist_pts": round(dist, 1), "dist_pct": round(dist_pct, 3),
            "dist_bucket_pct": f"{math.floor(dist_pct * 2) / 2:.1f}-{math.floor(dist_pct * 2) / 2 + 0.5:.1f}",
            "spot_0915_open": round(o915, 2),
            "breakeven_mid": round(k + entry_mid, 2) if cp == "CE" else round(k - entry_mid, 2),
            "o_0915": round(f_o, 2), "h_0915": round(f_h, 2), "l_0915": round(f_l, 2), "c_0915": round(f_c, 2),
            "entry_mid_0915": round(entry_mid, 2),
        }
        for name, t in snaps:
            r[f"px_{name.replace(':', '')}"] = snap(g, t)
        r.update({
            "high_after_0915": round(day_hi_after, 2) if day_hi_after == day_hi_after else np.nan,
            "time_high_after_0915": tod_str(t_hi),
            "max_rise_vs_entry_pts": round(day_hi_after - entry_mid, 2) if day_hi_after == day_hi_after else np.nan,
            "day_high": round(day_hi, 2), "time_day_high": tod_str(t_day_hi),
            "day_low": round(g.low.min(), 2),
            "last_close": round(g.close.iloc[-1], 2), "last_trade_time": tod_str(g.tod.iloc[-1]),
            "settle_avg30": round(settle, 2) if settle == settle else np.nan,
            "intrinsic_at_settle": round(intrinsic, 2),
            "expired_itm": int(intrinsic > 0),
            "short_pnl_pts_hold": round(entry_mid - intrinsic, 2),
            "volume_day": int(g.volume.sum()) if g.volume.notna().any() else np.nan,
            "minutes_traded": len(g),
        })
        # first minute (after 09:15) the premium traded >= entry + X  (stop-loss study)
        for sl in cfg.sl_points:
            hit = later[later.high >= entry_mid + sl]
            r[f"sl{sl:g}_hit_time"] = tod_str(hit.tod.iloc[0]) if len(hit) else ""
        rows.append(r)

        if cfg.strike_5min:
            gg = g.copy()
            gg["bar"] = (gg.tod - FIRST_TOD) // 5
            b = gg.groupby("bar").agg(open=("open", "first"), high=("high", "max"),
                                      low=("low", "min"), close=("close", "last"))
            for bar, x in b.iterrows():
                paths.append({"expiry_date": exp.date(), "type": cp, "strike": r["strike"],
                              "dist_pct": r["dist_pct"], "time": tod_str(FIRST_TOD + bar * 5),
                              "open": x.open, "high": x.high, "low": x.low, "close": x.close})

    # index 5-min path for the session
    s = sess.copy()
    s["bar"] = (s.tod - FIRST_TOD) // 5
    b = s.groupby("bar").agg(open=("open", "first"), high=("high", "max"),
                             low=("low", "min"), close=("close", "last"))
    idx5 = [{"expiry_date": exp.date(), "time": tod_str(FIRST_TOD + bar * 5),
             "open": round(x.open, 2), "high": round(x.high, 2), "low": round(x.low, 2),
             "close": round(x.close, 2), "pct_vs_0920": round((x.close / s920 - 1) * 100, 3)}
            for bar, x in b.iterrows()]
    return day, rows, idx5, paths


# ---------------------------------------------------------------- output helpers
def write_csv(df, path):
    """Write CSV; if it would be too big for git, split by year."""
    df.to_csv(path, index=False)
    mb = path.stat().st_size / 1e6
    if mb <= GIT_SOFT_LIMIT_MB:
        return [path.name]
    path.unlink()
    year_col = next(c for c in ("expiry_date", "date", "session") if c in df)
    out = []
    for y, part in df.groupby(pd.to_datetime(df[year_col]).dt.year):
        p = path.with_name(f"{path.stem}_{y}.csv")
        part.to_csv(p, index=False)
        out.append(p.name)
    return out


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


def summary_md(sym, daily, E, S):
    out = [f"## {sym}", ""]
    if len(daily):
        out.append(f"- Trading days: **{len(daily)}** ({daily.date.min()} -> {daily.date.max()}); "
                   f"expiry sessions with option data: **{len(E)}**")
        cc = pd.to_numeric(daily.close_to_close_pct, errors="coerce").abs()
        out.append(f"- Days with |close-to-close| >= 2%: {int((cc >= 2).sum())}, >= 3%: {int((cc >= 3).sum())}")
    if len(E):
        out += ["", "### Expiry-day index moves measured from 09:20 spot", ""]
        t = []
        for m in MOVES:
            k = str(m).replace(".0", "")
            t.append({"move": f"{k}%", "touched": int(E[f"touched_{k}pct_from_0920"].sum()),
                      "closed_beyond": int(E[f"closed_beyond_{k}pct"].sum()), "of_sessions": len(E)})
        out.append(md_table(pd.DataFrame(t)))
        big = E[E["touched_2pct_from_0920"] == 1][["session", "weekday", "gap_pct",
                                                  "max_up_from_0920_pct", "max_down_from_0920_pct",
                                                  "close_vs_0920_pct"]]
        if len(big):
            out += ["", "Sessions that touched 2% from 09:20:", "", md_table(big)]
    if len(S):
        out += ["", "### Typical 09:15 premium (entry_mid) by distance from ATM", ""]
        g = S.groupby(["type", "dist_bucket_pct"]).agg(
            n=("entry_mid_0915", "size"), median_entry=("entry_mid_0915", "median"),
            median_settle=("settle_avg30", "median"),
            pct_expired_itm=("expired_itm", lambda x: round(x.mean() * 100, 1)),
            most_common_peak_time=("time_high_after_0915", lambda x: x[x != ""].mode().iloc[0] if (x != "").any() else ""),
        ).reset_index()
        g["median_entry"] = g.median_entry.round(2)
        g["median_settle"] = g.median_settle.round(2)
        out.append(md_table(g[g.n >= 5]))
        out += ["", "### When far-OTM premium peaks (after 09:15), share of strike-days", ""]
        tt = S[S.time_high_after_0915 != ""].copy()
        tt["slot"] = tt.time_high_after_0915.map(lambda s: _slot(hm(s)))
        sl = (tt.slot.value_counts(normalize=True).sort_index() * 100).round(1)
        out.append(md_table(sl.rename("pct").reset_index().rename(columns={"index": "slot"})))
    out.append("")
    return "\n".join(out)


def _slot(t):
    for a, b, lab in [(hm("09:16"), hm("09:25"), "a 09:16-09:25"), (hm("09:26"), hm("09:59"), "b 09:26-09:59"),
                      (hm("10:00"), hm("10:30"), "c 10:00-10:30"), (hm("10:31"), hm("11:59"), "d 10:31-11:59"),
                      (hm("12:00"), hm("13:59"), "e 12:00-13:59"), (hm("14:00"), CLOSE_TOD, "f 14:00-close")]:
        if a <= t <= b:
            return lab
    return "z other"


README = """# India index options - analysis-ready export

Generated by `export_for_claude.py` on {when} from the Hugging Face dataset
[`{repo}`](https://huggingface.co/datasets/{repo}) (1-minute index + option candles).
Raw data is several GB; these files are compact derivations small enough for git and for Claude to read.

Symbols: {syms}. Expiries since: {since}. OTM strikes kept: 0-{max_otm}% from the 09:15 open, entry premium >= {min_prem}.

## Conventions
- All times IST. Pre-open bars (before 09:15) removed. Session = 09:15-15:30.
- `entry_mid_0915` = (high + low) / 2 of the 09:15 one-minute option candle (the realistic first-minute fill).
- `px_HHMM` = last traded option price at or before HH:MM (no look-ahead).
- `settle_avg30` = mean of the last 30 one-minute closes (approximation of the exchange settlement).
- `dist_pts` / `dist_pct` = how far OTM the strike is from the 09:15 index open (CE: strike - open, PE: open - strike).
- `short_pnl_pts_hold` = premium collected at entry_mid minus intrinsic at settlement, per unit, if held to expiry with no stop.
  Multiply by quantity (SENSEX lot 20, NIFTY lot 65, BANKNIFTY lot 30) for rupees; subtract charges.
- `slX_hit_time` = first minute after 09:15 when the option traded at >= entry_mid + X points (blank = never). Use to test stop-losses.
- Break-even of a short strangle = CE strike + total premium on the upside and PE strike - total premium on the downside.
  `breakeven_mid` is the single-leg version (strike +/- that leg's premium).

## Files
| file | one row per | key columns |
|---|---|---|
| `SUMMARY.md` | - | headline stats, computed |
| `*_daily.csv` | trading day | gap_pct, range_pct, close_to_close_pct, max_up/down_from_open_pct, time_of_high/low, is_expiry, sessions_to_next_expiry |
| `*_expiry_days.csv` | expiry session | open_0915, spot_0920, max_up/down_from_0920_pct, touched_Xpct_from_0920, closed_beyond_Xpct |
| `*_expiry_strikes.csv` | strike x type x expiry session | 09:15 OHLC, entry_mid_0915, px_0916...px_1529, high_after_0915 + time, settle_avg30, expired_itm, short_pnl_pts_hold, slX_hit_time |
| `*_expiry_index_5min.csv` | 5-min bar on expiry sessions | OHLC, pct_vs_0920 |
| `*_expiry_strikes_5min.csv` | 5-min bar per strike (optional) | OHLC |

Files larger than ~{lim} MB are split by year (`..._2024.csv`).

## Example questions to ask Claude
- "Using SENSEX_expiry_strikes.csv, backtest selling the CE and PE closest to 2% break-even at 09:15 entry_mid with a 7-point stop
  (use sl7_hit_time; if one leg stops, exit the other at its px at that minute). Show win rate, average, worst day, max drawdown."
- "From SENSEX_expiry_days.csv, list every expiry where the index touched 2.5% from 09:20 and what the gap was."
- "Which time slot gives the highest premium for strikes 2-3% OTM on NIFTY expiry days? Compare px_0916 vs px_1000 vs px_1030."
- "Is 09:15 premium higher after a gap-up? Join *_expiry_strikes.csv with *_daily.csv on session/date."
"""


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", default=None, help="local copy of the dataset (else download from Hugging Face)")
    ap.add_argument("--symbols", default="SENSEX,NIFTY,BANKNIFTY")
    ap.add_argument("--since", default=None, help="only expiries on/after YYYY-MM-DD")
    ap.add_argument("--max-otm", type=float, default=6.0, help="keep strikes up to this %% OTM (default 6)")
    ap.add_argument("--min-prem", type=float, default=0.3, help="drop strikes whose 09:15 entry_mid < this")
    ap.add_argument("--sl-points", default="3,5,7,10,15,20,30", help="stop-loss points to pre-compute hit times for")
    ap.add_argument("--strike-5min", action="store_true", help="also write 5-min premium path per strike (bigger)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="claude_data")
    cfg = ap.parse_args()
    cfg.sl_points = [float(x) for x in cfg.sl_points.split(",") if x.strip()]
    if cfg.local:
        cfg.local = str(Path(cfg.local).expanduser())
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    files = list_files(cfg.local)
    syms = [s.strip().upper() for s in cfg.symbols.split(",") if s.strip()]
    summary, written = ["# Summary (computed from data)", ""], []

    # extra index files (e.g. INDIA VIX) -> daily table only
    idx_files = {Path(f).stem.upper(): f for f in files if f.lower().startswith("index/")}

    for sym in syms:
        if sym not in idx_files:
            print(f"[{sym}] no index file, skipped")
            continue
        print(f"[{sym}] loading index ...")
        sp = load_index(idx_files[sym], cfg.local)
        spot_days = set(sp.day.unique())
        exp_files = {}
        for f in files:
            if f.startswith(f"options/{sym}/") and f.endswith(".parquet"):
                try:
                    exp_files[pd.Timestamp(Path(f).stem)] = f
                except ValueError:
                    pass
        exps = sorted(e for e in exp_files if not cfg.since or e >= pd.Timestamp(cfg.since))
        print(f"[{sym}] {len(exps)} expiry files")

        days_by = {d: g for d, g in sp.groupby("day")}
        sorted_days = sorted(days_by)
        prev_close = {d: days_by[sorted_days[i - 1]].close.iloc[-1] if i else np.nan
                      for i, d in enumerate(sorted_days)}

        E, S, I5, P5, sessions = [], [], [], [], set()
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            futs = {ex.submit(fetch, exp_files[e], cfg.local): e for e in exps}
            for n, fu in enumerate(as_completed(futs), 1):
                e = futs[fu]
                try:
                    opt = norm_opt(pd.read_parquet(fu.result()))
                    d = expiry_session(e, set(opt.day.unique()), spot_days)
                    if d is None or d not in days_by:
                        continue
                    day, rows, idx5, paths = process_expiry(sym, e, opt, days_by[d], prev_close[d], cfg)
                    E.append(day); S += rows; I5 += idx5; P5 += paths; sessions.add(d)
                except Exception as err:  # keep going; report at the end
                    print(f"  ! {e.date()}: {err}")
                if n % 25 == 0 or n == len(exps):
                    print(f"  {n}/{len(exps)}")

        daily = daily_table(sp, sessions)
        E = pd.DataFrame(E).sort_values("session") if E else pd.DataFrame()
        S = pd.DataFrame(S).sort_values(["session", "type", "dist_pts"]) if S else pd.DataFrame()
        written += write_csv(daily, out / f"{sym}_daily.csv")
        if len(E):
            written += write_csv(E, out / f"{sym}_expiry_days.csv")
            written += write_csv(S, out / f"{sym}_expiry_strikes.csv")
            written += write_csv(pd.DataFrame(I5).sort_values(["expiry_date", "time"]),
                                 out / f"{sym}_expiry_index_5min.csv")
        if P5:
            written += write_csv(pd.DataFrame(P5), out / f"{sym}_expiry_strikes_5min.csv")
        summary.append(summary_md(sym, daily, E, S))
        print(f"[{sym}] {len(daily)} days, {len(E)} expiry sessions, {len(S)} strike rows")

    # VIX or other index-only files
    for name, f in idx_files.items():
        if "VIX" in name:
            v = load_index(f, cfg.local)
            dv = v.groupby("day").agg(open=("open", "first"), high=("high", "max"),
                                      low=("low", "min"), close=("close", "last")).reset_index()
            dv["date"] = dv.day.dt.date
            written += write_csv(dv[["date", "open", "high", "low", "close"]].round(2), out / f"{name}_daily.csv")

    (out / "SUMMARY.md").write_text("\n".join(summary))
    (out / "README.md").write_text(README.format(
        when=datetime.now().strftime("%Y-%m-%d %H:%M"), repo=REPO, syms=", ".join(syms),
        since=cfg.since or "start of data", max_otm=cfg.max_otm, min_prem=cfg.min_prem, lim=GIT_SOFT_LIMIT_MB))
    total = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"\nWrote {len(written) + 2} files to {out}/ ({total:.1f} MB):")
    for w in ["README.md", "SUMMARY.md"] + written:
        print("  ", w)


if __name__ == "__main__":
    main()
