#!/usr/bin/env python3
"""
options_lab.py - expiry-day short-strangle lab with an HTML report.

Replays every past expiry day minute by minute from the 1-min data in
huggingface.co/datasets/thetrademarkk/india-index-options-1m and answers:
  * How much premium can I collect at X% out of the money on expiry morning?
  * How often does the index move far enough to hit a stop-loss, and when?
  * What does my exact rule (entry time, strikes, SL, exit-both) earn over history?
  * Which distance / stop-loss combination worked best?
  * Does a high ATM straddle at entry warn of bad days?
  * How fast does premium decay through the day?

Setup:  pip install --index-url https://pypi.org/simple huggingface_hub pandas pyarrow
Run:    python options_lab.py                      # SENSEX, your trade settings
        python options_lab.py --symbols SENSEX,NIFTY --since 2024-01-01
        python options_lab.py --ce-dist 3.8 --pe-dist 4.1 --sl-points 7.14 --qty 420
Output: options_lab_report.html (open in a browser). Every trade and every
        strike row is browsable inside it; add --csv to also save CSV files.

Three parts per index, expiry days only:
  1. Your original rule (CE/PE a set % away at --entry, stop-loss premium + N points).
  2. Short strangles by break-even (default 2, 2.5 and 3 % from the 09:15 open),
     sold at the high-low midpoint of each leg's first 1-minute candle, with a
     7-point stop-loss per option and a grid of other stop-losses. Reports
     average and total profit, maximum drawdown, recovery time and more.
  3. Every traded strike 1500-3000 points (SENSEX) or 800-2000 points (NIFTY)
     from the 09:15 ATM strike: its first candle, how far it ran later, and what
     sell limit on the first candle would realistically have filled.

Useful flags: --breakevens 2,2.5,3  --be-sl 7  --be-sl-grid 3,5,7,10,15,20,30
              --sl-mode both|independent  --bands "SENSEX:1500-3000,NIFTY:800-2000"
              --since 2024-01-01  --symbols SENSEX

What is read from the data and what is a modelling choice is listed at the
bottom of the report.
"""
import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "thetrademarkk/india-index-options-1m"
CLOSE_TOD = 15 * 60 + 30  # 15:30


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


def load_index(sym, files, local):
    path = next((f for f in files if f.lower() == f"index/{sym.lower()}.parquet"), None)
    if not path:
        return None
    df = pd.read_parquet(fetch(path, local)).rename(columns=str.lower)
    df = add_time_cols(df)
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
    if sym is not None and df["strike"].isna().any():
        m = df["strike"].isna()
        df.loc[m, "strike"] = pd.to_numeric(
            sym[m].str.extract(r"(\d+(?:\.\d+)?)(?:CE|PE)$")[0], errors="coerce")
    df = df.dropna(subset=["option_type", "strike"])
    df = df[df.option_type.isin(["CE", "PE"])]
    df = add_time_cols(df)
    for col in ("open", "low"):
        if col not in df:
            df[col] = df["close"]
    if "volume" not in df:
        df["volume"] = np.nan
    return df[["day", "tod", "strike", "option_type", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------- one expiry day
FIRST_TOD = 9 * 60 + 15


def tod_str(t):
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return ""
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def expiry_session(exp, opt, spot):
    """The trading session on which this contract expired, or (None, reason)."""
    days = sorted(d for d in opt["day"].unique() if d <= exp)
    if not days:
        return None, "option file has no rows on or before its expiry date"
    if (spot.day == exp).any():
        if days[-1] != exp:
            return None, "no option trades on the expiry date"
        return exp, None
    d = days[-1]  # expiry date is a holiday: accept the session just before it
    if (exp - d).days > 3 or spot.day.max() < exp - pd.Timedelta(days=3):
        return None, "expiry date is outside the index data"
    return d, None


def minute_path(g, start_tod, seed):
    """Per-minute highs and closes from start_tod to the close. Minutes without a
    trade keep the last traded price; the start minute itself holds `seed`."""
    minutes = np.arange(start_tod, CLOSE_TOD)
    x = (g[g.tod > start_tod].groupby("tod")
         .agg(high=("high", "max"), close=("close", "last")).reindex(minutes))
    x["close"] = x.close.ffill().fillna(seed)
    x["high"] = x.high.fillna(x.close)
    return x.high.values, x.close.values


def sim_strangle(legs, sl, mode, settle, slip):
    """legs: (side, strike, entry, highs, closes) for each leg that was sold.
    sl is points above entry. mode 'both' exits every leg when the first
    stop-loss hits; 'independent' leaves the other leg running.
    Returns (pnl points, legs stopped, minute index of first stop, exits)."""
    if not legs:
        return 0.0, "", None, 0
    hits = []
    for cp, k, e, hi, cl in legs:
        idx = np.flatnonzero(hi >= e + sl) if math.isfinite(sl) else []
        hits.append(int(idx[0]) if len(idx) else None)

    def intrinsic(cp, k):
        return max(0.0, settle - k) if cp == "CE" else max(0.0, k - settle)

    pnl, names = 0.0, []
    live = [h for h in hits if h is not None]
    first = min(live) if live else None
    for (cp, k, e, hi, cl), h in zip(legs, hits):
        if mode == "both" and first is not None:
            if h == first:
                pnl += e - (e + sl) - slip
                names.append(cp)
            else:
                pnl += e - float(cl[first]) - slip
        elif mode != "both" and h is not None:
            pnl += e - (e + sl) - slip
            names.append(cp)
        else:
            pnl += e - intrinsic(cp, k)
    return pnl, "+".join(names), first, len(names)


def breakeven_trades(sym, d, o, sp, settle, cfg):
    """Short strangles entered on the 09:15 candle whose break-evens sit a set
    percentage from the index's 09:15 open."""
    rows, skips = [], {}

    def skip_all(why):
        return rows, {x: why for x in cfg.breakevens}

    if sp.empty:
        return skip_all("index has no bar at or after 09:15")
    S = float(sp.open.iloc[0])
    otod = FIRST_TOD if (o.tod == FIRST_TOD).any() else (int(o.tod.min()) if len(o) else None)
    if otod is None or otod > FIRST_TOD + cfg.max_stale:
        return skip_all(f"no option traded within {cfg.max_stale} minutes of 09:15")
    f = o[o.tod == otod]
    if f.empty:
        return skip_all("no option traded in the entry minute")
    fc = (f.groupby(["strike", "option_type"])
          .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
               close=("close", "last")).reset_index())
    fc["mid"] = (fc.high + fc.low) / 2
    ce = fc[(fc.option_type == "CE") & (fc.strike > S) & (fc.mid >= cfg.min_prem)]
    pe = fc[(fc.option_type == "PE") & (fc.strike < S) & (fc.mid >= cfg.min_prem)]
    if ce.empty or pe.empty:
        return skip_all("no out-of-the-money CE or PE with a 09:15 candle above the minimum premium")
    grp = {k: g for k, g in o.groupby(["strike", "option_type"])}
    paths = {}
    kc, mc = ce.strike.values[:, None], ce.mid.values[:, None]
    kp, mp = pe.strike.values[None, :], pe.mid.values[None, :]
    up = ((kc + mc + mp) / S - 1) * 100       # upper break-even, % above the open
    dn = (1 - (kp - mc - mp) / S) * 100       # lower break-even, % below the open
    for x in cfg.breakevens:
        err = np.abs(up - x) + np.abs(dn - x)
        i, j = np.unravel_index(int(err.argmin()), err.shape)
        if abs(up[i, j] - x) > cfg.be_tol or abs(dn[i, j] - x) > cfg.be_tol:
            skips[x] = (f"no strike pair with a 09:15 candle put both break-evens within "
                        f"{cfg.be_tol:g} percentage points of the target")
            continue
        base = []
        for r, cp in ((ce.iloc[i], "CE"), (pe.iloc[j], "PE")):
            key = (r.strike, cp)
            if key not in paths:
                paths[key] = minute_path(grp[key], otod, float(r.close))
            base.append((cp, float(r.strike), r, paths[key][0], paths[key][1]))

        def legs_for(price_of, need_fill=False):
            out = []
            for cp, k, r, hi, cl in base:
                price = float(price_of(r))
                if need_fill and not (r.high > price):   # a sell limit fills only if it traded higher
                    continue
                out.append((cp, k, price, hi, cl))
            return out

        variants = [("mid", legs_for(lambda r: r.mid))]
        variants.append(("open", legs_for(lambda r: r.open)))
        variants.append(("close", legs_for(lambda r: r.close)))
        for lim in cfg.entry_limits:
            variants.append((f"limit +{lim:g}%",
                             legs_for(lambda r, m=lim: r.open * (1 + m / 100), need_fill=True)))
        for name, legs in variants:
            if name == "mid":
                combos = [(sl, mode) for sl in cfg.be_sl_grid + [math.inf]
                          for mode in ("both", "independent")]
            else:
                combos = [(cfg.be_sl, cfg.sl_mode)]
            ent = {l[0]: l[2] for l in legs}
            for sl, mode in combos:
                pnl, stopped, first, exits = sim_strangle(legs, sl, mode, settle, cfg.slippage)
                rows.append({
                    "symbol": sym, "date": d.date().isoformat(), "breakeven": x,
                    "entry": name, "sl": "none" if math.isinf(sl) else f"{sl:g}",
                    "mode": mode, "index_open": S, "index_settle": round(settle, 2),
                    "ce_strike": base[0][1], "pe_strike": base[1][1],
                    "ce_entry": ent.get("CE", np.nan), "pe_entry": ent.get("PE", np.nan),
                    "be_up_pct": round(float(up[i, j]), 3), "be_dn_pct": round(float(dn[i, j]), 3),
                    "legs": len(legs), "pnl_pts": round(pnl, 2), "sl_hit": stopped,
                    "entry_minute": tod_str(otod),
                    "sl_time": tod_str(otod + first) if first is not None and stopped else "",
                    "exits": exits,
                })
    return rows, skips


def perf_stats(dates, rs):
    """Equity-curve statistics for one ₹-per-expiry series (dates ascending)."""
    rs = np.asarray(rs, float)
    n = len(rs)
    if not n:
        return {}
    dts = pd.to_datetime(list(dates))
    eq = np.cumsum(rs)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    dd = eq - peak
    wins, losses = rs[rs > 0], rs[rs <= 0]
    out = {
        "days": n, "total": float(eq[-1]), "avg": float(rs.mean()), "median": float(np.median(rs)),
        "win_rate": float((rs > 0).mean() * 100),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "best": float(rs.max()), "worst": float(rs.min()),
        "std": float(rs.std(ddof=1)) if n > 1 else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.nan,
        "max_dd": float(dd.min()), "dd_text": "none", "recovery_text": "-",
        "underwater_text": "-", "eq": eq, "dd": dd,
        "rec_exp": 0.0, "rec_days": 0.0, "recovered": True,
        "uw_exp": 0.0, "uw_days": 0.0,
    }
    out["avg_over_std"] = out["avg"] / out["std"] if out["std"] and out["std"] > 0 else np.nan
    streak = best = 0
    for v in rs:
        streak = streak + 1 if v <= 0 else 0
        best = max(best, streak)
    out["losing_streak"] = best
    if dd.min() < 0:
        t = int(dd.argmin())
        pk = peak[t]
        before = np.flatnonzero(eq[:t] >= pk - 1e-9)
        p_date = dts[before[-1]].date().isoformat() if len(before) else "start"
        out["dd_text"] = f"{p_date} → {dts[t].date().isoformat()}"
        rec = np.flatnonzero(eq[t + 1:] >= pk - 1e-9)
        if len(rec):
            r = t + 1 + int(rec[0])
            out["rec_exp"], out["rec_days"] = float(r - t), float((dts[r] - dts[t]).days)
            e_, d_ = r - t, (dts[r] - dts[t]).days
            out["recovery_text"] = (f"{e_} expir{'y' if e_ == 1 else 'ies'} "
                                    f"({d_} day{'' if d_ == 1 else 's'}, "
                                    f"by {dts[r].date().isoformat()})")
        else:
            out["rec_exp"] = float(n - 1 - t)
            out["rec_days"] = float((dts[-1] - dts[t]).days)
            out["recovered"] = False
            out["recovery_text"] = f"not recovered by {dts[-1].date().isoformat()}"
        # longest stretch below a previous peak
        longest, start = (0, None, None), None
        for i in range(n + 1):
            under = i < n and dd[i] < 0
            if under and start is None:
                start = i
            if not under and start is not None:
                a = dts[start - 1] if start > 0 else dts[start]
                b = dts[i] if i < n else dts[-1]
                if i - start > longest[0]:
                    longest = (i - start, (b - a).days, i < n)
                start = None
        out["uw_exp"], out["uw_days"] = float(longest[0]), float(longest[1] or 0)
        out["underwater_text"] = (
            f"{longest[0]} expir{'y' if longest[0] == 1 else 'ies'} ({longest[1]} days"
            f"{'' if longest[2] else ', still below'})")
    return out


def strike_tags(sym, d, o, sp, settle, band, prev_close=None):
    """Every traded strike inside the points-from-ATM band, tagged by its distance."""
    if o.empty or sp.empty:
        return []
    lo, hi = band
    idx_open = float(sp.open.iloc[0])
    strikes = np.sort(o.strike.unique())
    if len(strikes) < 3:
        return []
    step = float(pd.Series(np.diff(strikes)).round(2).mode().min())  # strike interval, from the data
    atm = round(idx_open / step) * step
    rows = []
    for (k, cp), g in o.groupby(["strike", "option_type"]):
        dist = (k - atm) if cp == "CE" else (atm - k)
        if dist < lo or dist > hi:
            continue
        ft = int(g.tod.iloc[0])
        fc, later = g[g.tod == ft], g[g.tod > ft]
        f_close = float(fc.close.iloc[-1])
        f_high = float(fc.high.max())
        if len(later):
            j = int(np.argmax(later.high.values))
            lh, lht = float(later.high.values[j]), int(later.tod.values[j])
        else:
            lh, lht = np.nan, None
        vol = fc.volume.sum(min_count=1)
        rows.append({
            "symbol": sym, "date": d.date().isoformat(),
            "index_open": idx_open, "index_open_time": tod_str(int(sp.tod.iloc[0])),
            "atm_strike": atm, "strike_step": step, "type": cp, "strike": float(k),
            "tag": int(round(dist)), "pct_otm": round(dist / idx_open * 100, 2),
            "first_time": tod_str(ft), "first_is_0915": ft == FIRST_TOD,
            "first_open": float(fc.open.iloc[0]), "first_high": f_high,
            "first_low": float(fc.low.min()), "first_close": f_close,
            "first_volume": None if pd.isna(vol) else float(vol),
            "prev_close": (prev_close or {}).get((k, cp), np.nan),
            "high_after_first": lh, "high_after_first_time": tod_str(lht),
            "day_high": f_high if math.isnan(lh) else max(f_high, lh),
            "day_low": float(g.low.min()), "last_close": float(g.close.iloc[-1]),
            "last_trade_time": tod_str(int(g.tod.iloc[-1])),
            "mult_after_first": (lh / f_close) if (f_close > 0 and not math.isnan(lh)) else np.nan,
            "index_settle": round(settle, 2),
            "settle_value": round(max(0.0, settle - k) if cp == "CE" else max(0.0, k - settle), 2),
        })
    return rows


def process_expiry(sym, exp, opt, spot, cfg):
    d, why = expiry_session(exp, opt, spot)
    if d is None:
        return {"skip": why}
    # bars before 09:15 are the pre-open session: not part of the trading day
    sp = spot[(spot.day == d) & (spot.tod >= FIRST_TOD)]
    o = opt[(opt.day == d) & (opt.tod >= FIRST_TOD)].sort_values("tod", kind="stable")
    if sp.empty:
        return {"skip": "index has no bar at or after 09:15"}
    last30 = sp[sp.tod >= CLOSE_TOD - 30]
    settle = float(last30.close.mean()) if len(last30) else float(sp.close.iloc[-1])
    band = cfg.bands.get(sym)
    prev_close = {}
    earlier = spot.day[spot.day < d]
    if len(earlier):
        pv = opt[(opt.day == earlier.max()) & (opt.tod >= FIRST_TOD)].sort_values(
            "tod", kind="stable")
        if len(pv):
            prev_close = pv.groupby(["strike", "option_type"]).close.last().to_dict()
    tags = strike_tags(sym, d, o, sp, settle, band, prev_close) if band else []
    be_rows, be_skip = breakeven_trades(sym, d, o, sp, settle, cfg)
    extra = {"tags": tags, "be": be_rows, "be_skip": be_skip, "session": d.date().isoformat()}

    sp_at = sp[sp.tod >= cfg.entry_tod]
    if sp_at.empty:
        return {"skip": "index has no bar at or after the entry time", **extra}
    entry_tod = int(sp_at.tod.iloc[0])
    S0 = float(sp_at.close.iloc[0])
    S_last = float(sp.close.iloc[-1])
    minutes = np.arange(entry_tod, CLOSE_TOD)

    # only what was on the screen at entry: trades at or before the entry minute
    seen = o[(o.tod <= entry_tod) & (o.tod >= entry_tod - cfg.max_stale)]
    if seen.empty:
        return {"skip": "no option traded in the minutes before entry", **extra}
    avail = {cp: np.sort(seen[seen.option_type == cp].strike.unique()) for cp in ("CE", "PE")}
    grp = {k: g for k, g in o.groupby(["strike", "option_type"])}

    def pick(cp, target):
        a = avail[cp]
        return None if len(a) == 0 else float(a[np.argmin(np.abs(a - target))])

    cache = {}

    def leg(k, cp):
        """(entry price, per-minute highs, per-minute closes, age of entry quote in minutes)."""
        if (k, cp) in cache:
            return cache[(k, cp)]
        g = grp.get((k, cp))
        res = None
        if g is not None:
            before = g[(g.tod <= entry_tod) & (g.tod >= entry_tod - cfg.max_stale)]
            if not before.empty:
                last = before.iloc[-1]
                entry = float(last.close)
                # minutes strictly after entry: the entry minute's own high is not a stop-loss
                x = (g[g.tod > entry_tod].groupby("tod")
                     .agg(high=("high", "max"), close=("close", "last")).reindex(minutes))
                x["close"] = x.close.ffill().fillna(entry)
                x["high"] = x.high.fillna(x.close)
                res = (entry, x.high.values, x.close.values, int(entry_tod - last.tod))
        cache[(k, cp)] = res
        return res

    after = sp[sp.tod > entry_tod]
    day = {
        "symbol": sym, "date": d.date().isoformat(), "spot_entry": S0, "spot_close": S_last,
        "spot_settle": settle,
        "close_move_pct": (S_last / S0 - 1) * 100,
        "max_up_pct": float((after.high.max() / S0 - 1) * 100) if len(after) else 0.0,
        "max_dn_pct": float((after.low.min() / S0 - 1) * 100) if len(after) else 0.0,
    }
    k_atm_c, k_atm_p = pick("CE", S0), pick("PE", S0)
    ce_atm = leg(k_atm_c, "CE") if k_atm_c else None
    pe_atm = leg(k_atm_p, "PE") if k_atm_p else None
    day["atm_straddle_pct"] = ((ce_atm[0] + pe_atm[0]) / S0 * 100) if ce_atm and pe_atm else np.nan

    checkpoints = np.arange(0, len(minutes), 15)
    trades, decay, prem = [], [], []
    configs = [(f"{x:g}%", x, x) for x in cfg.dists] + [("yours", cfg.ce_dist, cfg.pe_dist)]
    for label, dc, dp in configs:
        kc, kp = pick("CE", S0 * (1 + dc / 100)), pick("PE", S0 * (1 - dp / 100))
        if kc is None or kp is None:
            continue
        c, p = leg(kc, "CE"), leg(kp, "PE")
        if not c or not p or c[0] < cfg.min_prem or p[0] < cfg.min_prem:
            continue
        pc, pp = c[0], p[0]
        prem.append({"symbol": sym, "date": day["date"], "config": label,
                     "ce_prem": pc, "pe_prem": pp, "total_prem": pc + pp,
                     "total_prem_pct_spot": (pc + pp) / S0 * 100})
        comb = (c[2] + p[2]) / (pc + pp) * 100
        decay.append({"symbol": sym, "config": label,
                      "minutes": (entry_tod + checkpoints).tolist(),
                      "pct": comb[checkpoints].round(2).tolist()})
        for rule, slc, slp in sl_rules(pc, pp, cfg):
            r = simulate(pc, pp, c, p, slc, slp, kc, kp, settle, cfg.slippage)
            hit_i = r["i"]
            trades.append({
                "symbol": sym, "date": day["date"], "config": label, "sl_rule": rule,
                "ce_strike": kc, "pe_strike": kp, "ce_prem": pc, "pe_prem": pp,
                "ce_quote_age_min": c[3], "pe_quote_age_min": p[3],
                "pnl_pts": round(r["pnl"], 2),
                "pnl_pts_fill_close": round(r["pnl_close"], 2),
                "pnl_pts_fill_high": round(r["pnl_high"], 2),
                "sl_hit": r["leg"] or "",
                "sl_time": tod_str(entry_tod + hit_i) if r["leg"] else "",
                "sl_tod": (entry_tod + hit_i) if r["leg"] else np.nan,
                "sl_trigger": r["trigger"], "sl_bar_close": r["bar_close"],
                "sl_bar_high": r["bar_high"],
                "atm_straddle_pct": day["atm_straddle_pct"],
                "max_up_pct": day["max_up_pct"], "max_dn_pct": day["max_dn_pct"],
                "close_move_pct": day["close_move_pct"],
            })
    return {"day": day, "trades": trades, "decay": decay, "prem": prem, **extra}


def sl_rules(pc, pp, cfg):
    out = [(f"{m:g}x", pc * m, pp * m) for m in cfg.sl_mults]
    out.append(("none", math.inf, math.inf))
    out.append((f"+{cfg.sl_points:g} pts", pc + cfg.sl_points, pp + cfg.sl_points))
    return out


def simulate(pc, pp, c, p, slc, slp, kc, kp, settle, slip):
    """P&L in points under three stop-loss fills: at the trigger, at the trigger
    minute's close (never better than the trigger), and at that minute's high."""
    hc = np.flatnonzero(c[1] >= slc)
    hp = np.flatnonzero(p[1] >= slp)
    ic = hc[0] if len(hc) else None
    ip = hp[0] if len(hp) else None
    if ic is None and ip is None:
        pnl = pc + pp - max(0.0, settle - kc) - max(0.0, kp - settle)
        return {"pnl": pnl, "pnl_close": pnl, "pnl_high": pnl, "leg": None, "i": None,
                "trigger": np.nan, "bar_close": np.nan, "bar_high": np.nan}
    if ip is None or (ic is not None and ic <= ip):
        name, hit, other, trig, i = "CE", c, p, slc, int(ic)
    else:
        name, hit, other, trig, i = "PE", p, c, slp, int(ip)
    base = pc + pp - float(other[2][i]) - 2 * slip
    bar_close, bar_high = float(hit[2][i]), float(hit[1][i])
    return {"pnl": base - trig, "pnl_close": base - max(trig, bar_close),
            "pnl_high": base - bar_high, "leg": name, "i": i,
            "trigger": round(trig, 2), "bar_close": bar_close, "bar_high": bar_high}


# ---------------------------------------------------------------- stats
def max_drawdown(x):
    eq = np.cumsum(x)
    peak = np.maximum.accumulate(np.concatenate([[0], eq]))[1:]
    return float((eq - peak).min()) if len(eq) else 0.0


def summarise(t, qty, charges, col="pnl_pts"):
    if t.empty:
        return {}
    rs = t[col] * qty - charges - np.where(t.sl_hit != "", charges, 0)
    wins, losses = rs[rs > 0], rs[rs <= 0]
    return {
        "trades": int(len(t)),
        "win_rate": float((rs > 0).mean() * 100),
        "sl_rate": float((t.sl_hit != "").mean() * 100),
        "avg_rs": float(rs.mean()), "median_rs": float(rs.median()),
        "worst_rs": float(rs.min()), "best_rs": float(rs.max()),
        "total_rs": float(rs.sum()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
        "max_dd_rs": max_drawdown(rs.values),
        "avg_prem_pts": float((t.ce_prem + t.pe_prem).mean()),
    }


# ---------------------------------------------------------------- report
PALETTE = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100",
           "s5": "#e87ba4", "s6": "#008300", "s7": "#6250d6", "s8": "#e34948",
           "good": "#0ca30c", "critical": "#d03b3b"}
SERIES = [PALETTE[f"s{i}"] for i in range(1, 9)]


def fmt_rs(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    sign = "−" if x < 0 else ""
    return f"{sign}₹{abs(x):,.0f}"


def build_symbol_section(sym, days, trades, decay, prem, cfg, meta):
    q, ch = cfg.qty, cfg.charges
    yours = trades[(trades.config == "yours") & (trades.sl_rule == f"+{cfg.sl_points:g} pts")]
    ys = summarise(yours, q, ch)
    figs, insights = [], []
    n_days = len(days)
    rng = f"{days.date.min()} to {days.date.max()}"

    # ---- headline insights
    if ys:
        insights.append(
            f"Over <b>{ys['trades']}</b> expiry days ({rng}), your rule "
            f"(sell ~{cfg.ce_dist:g}% OTM CE / ~{cfg.pe_dist:g}% OTM PE at {cfg.entry}, "
            f"SL +{cfg.sl_points:g} pts per leg, exit both) made money on "
            f"<b>{ys['win_rate']:.0f}%</b> of days and averaged <b>{fmt_rs(ys['avg_rs'])}</b> per day "
            f"for {q} qty. Worst day {fmt_rs(ys['worst_rs'])}; total {fmt_rs(ys['total_rs'])}.")
        insights.append(
            f"Average premium collected at your strikes: <b>{ys['avg_prem_pts']:.1f} pts</b> "
            f"(≈{fmt_rs(ys['avg_prem_pts'] * q)}). The stop-loss fired on "
            f"<b>{ys['sl_rate']:.0f}%</b> of days.")
        hit_m = yours.sl_hit != ""
        rs_y = yours.pnl_pts * q - ch - np.where(hit_m, ch, 0)
        n_hit = int(hit_m.sum())
        if n_hit and (~hit_m).any():
            w, l = float(rs_y[~hit_m].mean()), float(-rs_y[hit_m].mean())
            if l > 0:
                insights.append(
                    f"Break-even, from the data: days without a stop-loss made {fmt_rs(w)} on average "
                    f"and the {n_hit} stop-loss day(s) lost {fmt_rs(l)} on average, so the rule stops "
                    f"making money if stop-losses hit on more than about <b>{w / (w + l) * 100:.0f}%</b> "
                    f"of days. They hit on {ys['sl_rate']:.1f}%. With {n_hit} stop-loss day(s) the loss "
                    f"figure is a thin sample.")
        elif not n_hit:
            insights.append("The stop-loss never fired in this history, so the data cannot say what "
                            "a stop-loss day costs.")

    else:
        insights.append(
            f"No expiry day had both of your legs (~{cfg.ce_dist:g}% / ~{cfg.pe_dist:g}% OTM) listed "
            f"with a premium of at least {cfg.min_prem:g} pts at {cfg.entry}, so your exact rule "
            f"couldn't be replayed. Try a lower --min-prem or closer distances.")

    # best config by expectancy (min 20 trades)
    grid = (trades[trades.config != "yours"].groupby(["config", "sl_rule"])
            .apply(lambda g: pd.Series(summarise(g, q, ch)), include_groups=False).reset_index())
    grid = grid[grid.trades >= 20] if not grid.empty else grid
    if not grid.empty:
        b = grid.sort_values("avg_rs", ascending=False).iloc[0]
        insights.append(
            f"Best grid combination by average P&L: <b>{b.config} OTM with SL {b.sl_rule}</b> "
            f"→ {fmt_rs(b.avg_rs)}/day, win rate {b.win_rate:.0f}%, worst day {fmt_rs(b.worst_rs)} "
            f"({int(b.trades)} days). Check its worst day before preferring it.")

    # straddle filter
    st = yours.dropna(subset=["atm_straddle_pct"])
    st_hits = int((st.sl_hit != "").sum()) if len(st) else 0
    straddle_ok = len(st) >= 20 and st_hits >= 5
    if len(st) >= 20 and not straddle_ok:
        insights.append(
            f"Straddle filter: only {st_hits} stop-loss day(s) in this history, too few to tell "
            f"whether an expensive ATM straddle at entry warns of trouble.")
    if straddle_ok:
        cut = st.atm_straddle_pct.quantile(0.75)
        hi, lo = st[st.atm_straddle_pct >= cut], st[st.atm_straddle_pct < cut]
        insights.append(
            f"Straddle filter: on days when the 9:20 ATM straddle was ≥ {cut:.2f}% of spot "
            f"(top quarter), your SL fired on <b>{(hi.sl_hit != '').mean() * 100:.0f}%</b> of days vs "
            f"<b>{(lo.sl_hit != '').mean() * 100:.0f}%</b> on the rest; avg P&L "
            f"{fmt_rs(summarise(hi, q, ch)['avg_rs'])} vs {fmt_rs(summarise(lo, q, ch)['avg_rs'])}.")

    # timing of SL
    hits = yours[yours.sl_hit != ""]
    if len(hits) >= 3:
        early = (hits.sl_tod < 12 * 60).mean() * 100
        insights.append(f"Timing: <b>{early:.0f}%</b> of stop-loss hits came before 12:00. "
                        f"CE leg hit {int((hits.sl_hit == 'CE').sum())}×, PE leg "
                        f"{int((hits.sl_hit == 'PE').sum())}×.")

    # move probability
    exc = np.maximum(days.max_up_pct, -days.max_dn_pct)
    p_ce = (days.max_up_pct >= cfg.ce_dist).mean() * 100
    p_pe = (days.max_dn_pct <= -cfg.pe_dist).mean() * 100
    insights.append(
        f"From {cfg.entry} to the close of expiry day, {sym} went up ≥{cfg.ce_dist:g}% on "
        f"<b>{p_ce:.1f}%</b> of days and down ≥{cfg.pe_dist:g}% on <b>{p_pe:.1f}%</b>. "
        f"It moved ≥1% either way on {(exc >= 1).mean() * 100:.0f}% of days and ≥2% on "
        f"{(exc >= 2).mean() * 100:.0f}%.")

    # ---- KPI tiles
    tiles = []
    if ys:
        tiles = [("Expiry days tested", f"{ys['trades']}"),
                 ("Win rate", f"{ys['win_rate']:.0f}%"),
                 ("Avg P&L / day", fmt_rs(ys["avg_rs"])),
                 ("Worst day", fmt_rs(ys["worst_rs"])),
                 ("SL hit rate", f"{ys['sl_rate']:.0f}%"),
                 ("Max drawdown", fmt_rs(ys["max_dd_rs"]))]

    # ---- 1. equity curves: yours vs. same strikes with other SLs
    eq_series = []
    yy = trades[trades.config == "yours"]
    rules = [f"+{cfg.sl_points:g} pts"] + [r for r in yy.sl_rule.unique() if r != f"+{cfg.sl_points:g} pts"]
    for i, r in enumerate(rules[:6]):
        t = yy[yy.sl_rule == r].sort_values("date")
        rs = t.pnl_pts * q - ch - np.where(t.sl_hit != "", ch, 0)
        eq_series.append({"x": t.date.tolist(), "y": np.cumsum(rs).round(0).tolist(),
                          "name": ("Your SL " if i == 0 else "SL ") + r,
                          "type": "scatter", "mode": "lines",
                          "line": {"width": 2.5 if i == 0 else 1.5, "color": SERIES[i]},
                          "hovertemplate": "%{x}<br>₹%{y:,.0f}<extra>%{fullData.name}</extra>"})
    figs.append(("Cumulative P&L at your strikes, by stop-loss rule (₹, after charges)",
                 "Same entries every day; only the stop-loss differs. A steady line beats a "
                 "higher line with deep drops.",
                 eq_series, {"yaxis": {"title": "₹ cumulative", "tickformat": ",.0f"}}))

    # ---- 2. heatmap avg P&L by distance x SL
    if not grid.empty:
        grid["d"] = grid.config.str.rstrip("%").astype(float)
        order_sl = [f"{m:g}x" for m in cfg.sl_mults] + [f"+{cfg.sl_points:g} pts", "none"]
        pv = grid.pivot(index="sl_rule", columns="d", values="avg_rs").reindex(order_sl)
        pw = grid.pivot(index="sl_rule", columns="d", values="win_rate").reindex(order_sl)
        pk = grid.pivot(index="sl_rule", columns="d", values="worst_rs").reindex(order_sl)
        z = pv.values
        lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1
        text = [[("" if np.isnan(v) else fmt_rs(v)) for v in row] for row in z]
        custom = np.dstack([pw.values, pk.values]).tolist()
        figs.append(("Average P&L per day (₹) by strike distance and stop-loss",
                     "Blue = made money on average, red = lost. Hover for win rate and worst day. "
                     "Uses symmetric distances on both sides.",
                     [{"type": "heatmap", "x": [f"{c:g}%" for c in pv.columns], "y": list(pv.index),
                       "z": np.where(np.isnan(z), None, np.round(z, 0)).tolist(),
                       "text": text, "texttemplate": "%{text}", "customdata": custom,
                       "zmin": -lim, "zmax": lim, "xgap": 2, "ygap": 2,
                       "colorscale": [[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#2a78d6"]],
                       "colorbar": {"title": "₹/day", "thickness": 10},
                       "hovertemplate": "%{x} OTM, SL %{y}<br>Avg %{text}/day<br>"
                                        "Win rate %{customdata[0]:.0f}%<br>Worst day ₹%{customdata[1]:,.0f}"
                                        "<extra></extra>"}],
                     {"xaxis": {"title": "Distance from spot at entry (both legs)", "type": "category"},
                      "yaxis": {"title": "Stop-loss", "autorange": "reversed"}}))

    # ---- 3. P&L distribution (yours)
    if not yours.empty:
        rs = (yours.pnl_pts * q - ch - np.where(yours.sl_hit != "", ch, 0)).round(0)
        figs.append(("How your daily results are spread (₹ per day)",
                     "Most days sit at the small premium you collect; the left tail is the "
                     "stop-loss days.",
                     [{"type": "histogram", "x": rs.tolist(), "nbinsx": 40,
                       "marker": {"color": PALETTE["s1"], "line": {"color": "#fcfcfb", "width": 1}},
                       "hovertemplate": "₹%{x}<br>%{y} days<extra></extra>"}],
                     {"xaxis": {"title": "₹ per day", "tickformat": ",.0f"},
                      "yaxis": {"title": "Days"}, "bargap": 0.05}))

    # ---- 4. premium collected vs distance
    if not prem.empty:
        pg = prem[prem.config != "yours"].copy()
        pg["d"] = pg.config.str.rstrip("%").astype(float)
        agg = pg.groupby("d").total_prem.describe(percentiles=[.25, .5, .75])
        figs.append(("Premium you can collect at entry (CE + PE, points)",
                     "Median with the middle 50% of days shaded. Far strikes pay very little "
                     "on expiry morning.",
                     [{"x": agg.index.tolist(), "y": agg["75%"].round(2).tolist(), "type": "scatter",
                       "mode": "lines", "line": {"width": 0}, "showlegend": False, "hoverinfo": "skip"},
                      {"x": agg.index.tolist(), "y": agg["25%"].round(2).tolist(), "type": "scatter",
                       "mode": "lines", "fill": "tonexty", "fillcolor": "rgba(42,120,214,0.15)",
                       "line": {"width": 0}, "name": "Middle 50% of days", "hoverinfo": "skip"},
                      {"x": agg.index.tolist(), "y": agg["50%"].round(2).tolist(), "type": "scatter",
                       "mode": "lines+markers", "name": "Median",
                       "line": {"color": PALETTE["s1"], "width": 2}, "marker": {"size": 8},
                       "hovertemplate": "%{x}% OTM<br>Median %{y} pts<extra></extra>"}],
                     {"xaxis": {"title": "Distance from spot (%)"},
                      "yaxis": {"title": "Points (CE + PE)"}}))

    # ---- 5. decay through the day
    if not decay.empty:
        tr = []
        cfgs = [c for c in decay.config.unique() if c != "yours"]
        pick_cfgs = cfgs[::max(1, len(cfgs) // 4)][:4] + ["yours"]
        for i, c in enumerate(pick_cfgs):
            dd = decay[decay.config == c]
            if dd.empty:
                continue
            L = min(len(x) for x in dd.pct)
            mat = np.array([x[:L] for x in dd.pct])
            mins = dd.minutes.iloc[0][:L]
            labels = [f"{m // 60:02d}:{m % 60:02d}" for m in mins]
            tr.append({"x": labels, "y": np.median(mat, axis=0).round(1).tolist(),
                       "type": "scatter", "mode": "lines", "name": f"{c} OTM" if c != "yours" else "Your strikes",
                       "line": {"color": SERIES[i], "width": 2.5 if c == "yours" else 1.8},
                       "hovertemplate": "%{x}: %{y}% of entry premium left<extra>%{fullData.name}</extra>"})
        figs.append(("How fast the premium melts on expiry day (median, % of entry premium left)",
                     "Typical day only. On a trending day the side being approached rises instead.",
                     tr, {"yaxis": {"title": "% of entry premium", "rangemode": "tozero"},
                          "xaxis": {"title": "Time"}}))

    # ---- 6. move probability curves
    xs = np.round(np.arange(0.25, 5.01, 0.25), 2)
    up = [(days.max_up_pct >= x).mean() * 100 for x in xs]
    dn = [(days.max_dn_pct <= -x).mean() * 100 for x in xs]
    cl = [((days.close_move_pct.abs()) >= x).mean() * 100 for x in xs]
    figs.append((f"How often {sym} travels X% from the {cfg.entry} price on expiry day",
                 "‘Touched’ counts any intraday high/low; this is what triggers stop-losses. "
                 "‘Closed beyond’ is what decides expiry settlement.",
                 [{"x": xs.tolist(), "y": np.round(up, 1).tolist(), "name": "Touched +X% (up)",
                   "type": "scatter", "mode": "lines+markers", "line": {"color": SERIES[0], "width": 2},
                   "marker": {"size": 8}, "hovertemplate": "+%{x}%: %{y}% of days<extra></extra>"},
                  {"x": xs.tolist(), "y": np.round(dn, 1).tolist(), "name": "Touched −X% (down)",
                   "type": "scatter", "mode": "lines+markers", "line": {"color": SERIES[1], "width": 2},
                   "marker": {"size": 8}, "hovertemplate": "−%{x}%: %{y}% of days<extra></extra>"},
                  {"x": xs.tolist(), "y": np.round(cl, 1).tolist(), "name": "Closed beyond ±X%",
                   "type": "scatter", "mode": "lines+markers", "line": {"color": SERIES[2], "width": 2},
                   "marker": {"size": 8}, "hovertemplate": "±%{x}% at close: %{y}% of days<extra></extra>"}],
                 {"xaxis": {"title": "Move from entry (%)"},
                  "yaxis": {"title": "% of expiry days", "type": "log"}}))

    # ---- 7. straddle filter bars
    if straddle_ok:
        st = st.copy()
        st["bucket"] = pd.qcut(st.atm_straddle_pct, 4, duplicates="drop")
        gb = st.groupby("bucket", observed=True)
        labels = [f"Q{i + 1}: {iv.left:.3f}–{iv.right:.3f}%" for i, iv in enumerate(gb.groups.keys())]
        sl_rate = (gb.apply(lambda g: (g.sl_hit != "").mean() * 100, include_groups=False)).round(1).tolist()
        avg = gb.apply(lambda g: summarise(g, q, ch)["avg_rs"], include_groups=False).round(0).tolist()
        figs.append(("Does an expensive 9:20 ATM straddle warn of trouble? (stop-loss hit rate)",
                     "Days split into four equal groups by ATM straddle as % of spot at entry. "
                     "Hover for average P&L in each group.",
                     [{"type": "bar", "x": labels, "y": sl_rate, "customdata": avg,
                       "marker": {"color": PALETTE["s2"]},
                       "hovertemplate": "Straddle %{x}<br>SL hit on %{y}% of days<br>"
                                        "Avg P&L ₹%{customdata:,.0f}<extra></extra>"}],
                     {"xaxis": {"title": "ATM straddle at entry (% of spot), Q1 = cheapest", "type": "category"},
                      "yaxis": {"title": "SL hit rate (%)", "rangemode": "tozero"}}))

    # ---- 8. SL time of day
    if len(hits) >= 3:
        figs.append(("When the stop-loss fires (your rule)",
                     "Count of stop-loss hits by half hour.",
                     [{"type": "histogram", "x": [f"{int(m) // 60:02d}:{(int(m) % 60) // 30 * 30:02d}"
                                                  for m in hits.sl_tod.sort_values()],
                       "marker": {"color": PALETTE["s8"], "line": {"color": "#fcfcfb", "width": 1}},
                       "hovertemplate": "%{x}<br>%{y} hits<extra></extra>"}],
                     {"xaxis": {"title": "Time (half-hour bucket)", "type": "category"},
                      "yaxis": {"title": "Stop-loss hits"}}))

    # ---- worst days table
    worst = yours.copy()
    worst["rs"] = worst.pnl_pts * q - ch - np.where(worst.sl_hit != "", ch, 0)
    worst = worst.sort_values("rs").head(10)

    def sl_minute(r):
        if not r.sl_hit:
            return "-"
        txt = f"{r.sl_trigger:.2f} → {r.sl_bar_close:.2f}"
        return txt + " (isolated spike)" if r.sl_bar_close < 0.5 * r.sl_trigger else txt

    rows = "".join(
        f"<tr><td>{r.date}</td><td>{fmt_rs(r.rs)}</td><td>{r.sl_hit or 'held'}</td>"
        f"<td>{r.sl_time or '-'}</td><td>{sl_minute(r)}</td>"
        f"<td>{r.ce_prem:.2f} / {r.pe_prem:.2f}</td>"
        f"<td>{r.max_up_pct:+.2f}% / {r.max_dn_pct:+.2f}%</td>"
        f"<td>{r.atm_straddle_pct:.2f}%</td></tr>" for r in worst.itertuples())
    table = ("<table><thead><tr><th>Date</th><th>P&L</th><th>SL leg</th><th>SL time</th>"
             "<th>SL trigger → price at end of that minute</th>"
             "<th>Entry CE / PE</th><th>Max up / down</th><th>ATM straddle</th></tr></thead>"
             f"<tbody>{rows}</tbody></table>")

    cards = []
    # ---- stop-loss fill sensitivity (all three fills are read from the candles)
    if not yours.empty and (yours.sl_hit != "").any():
        fills = [("At the trigger price", "pnl_pts"),
                 ("At that minute's close (never better than the trigger)", "pnl_pts_fill_close"),
                 ("At that minute's high (worst trade of the minute)", "pnl_pts_fill_high")]
        trs = ""
        for name, col in fills:
            m = summarise(yours, q, ch, col)
            trs += (f"<tr><td>{name}</td><td>{fmt_rs(m['avg_rs'])}</td><td>{fmt_rs(m['worst_rs'])}</td>"
                    f"<td>{fmt_rs(m['total_rs'])}</td><td>{m['win_rate']:.0f}%</td></tr>")
        cards.append(("If the stop-loss buyback filled at a different price",
                      "The headline numbers assume a fill at the trigger. The other rows use prices "
                      "that actually traded in the trigger minute.",
                      "<table><thead><tr><th>Stop-loss fill</th><th>Avg / day</th><th>Worst day</th>"
                      f"<th>Total</th><th>Win rate</th></tr></thead><tbody>{trs}</tbody></table>"))

    # ---- data coverage and quality, all counted from the files
    ql = [f"{meta['n_files']} {sym} expiry files in the dataset; {n_days} expiry sessions replayed "
          f"({rng})."]
    for why, n in sorted(meta["skips"].items(), key=lambda kv: -kv[1]):
        ql.append(f"{n} expiry file(s) skipped: {why}.")
    ql.append(f"Index data: {meta['first_bar']}.")
    if ys:
        ages = pd.concat([yours.ce_quote_age_min, yours.pe_quote_age_min])
        ql.append(f"Your exact strikes could be priced on {ys['trades']} of {n_days} sessions."
                  + ("" if ys["trades"] == n_days else
                     f" The rest had no trade in one of the legs in the {cfg.max_stale} minutes "
                     f"before entry, or a premium under {cfg.min_prem:g} pts."))
        ql.append(f"{(ages == 0).mean() * 100:.0f}% of your entry prices come from a trade in the entry "
                  f"minute itself; the oldest quote used was {int(ages.max())} min old.")
        hits_y = yours[yours.sl_hit != ""]
        if len(hits_y):
            sp_n = int((hits_y.sl_bar_close < 0.5 * hits_y.sl_trigger).sum())
            ql.append(f"{sp_n} of {len(hits_y)} stop-loss hit(s) were isolated spikes: the option "
                      f"touched the trigger and closed the same minute below half of it.")
    cards.append(("Data coverage and quality", "Counted from the dataset files on this run.",
                  "<ul>" + "".join(f"<li>{x}</li>" for x in ql) + "</ul>"))
    return {"sym": sym, "tiles": tiles, "insights": insights, "figs": figs, "table": table,
            "cards": cards, "n_days": n_days}


def be_rupees(df, cfg):
    return (df.pnl_pts * cfg.qty - np.where(df.legs > 0, cfg.charges, 0)
            - np.where(df.exits > 0, cfg.charges, 0))


def build_be_section(sym, B, n_sessions, be_skips, cfg):
    """Short strangles by break-even distance, entered on the 09:15 candle."""
    figs, insights, cards = [], [], []
    sl_lab = f"{cfg.be_sl:g}"
    mode_txt = {"both": "exit both legs when one stop-loss hits",
                "independent": "each leg keeps its own stop-loss"}
    bes = sorted(B.breakeven.unique())
    mid = B[B.entry == "mid"]
    head = mid[(mid.sl == sl_lab) & (mid["mode"] == cfg.sl_mode)]

    def stats(df):
        df = df.sort_values("date")
        st = perf_stats(df.date, be_rupees(df, cfg))
        if st:
            st["sl_rate"] = float((df.exits > 0).mean() * 100)
            st["avg_pts"] = float(df.pnl_pts.mean())
            st["dates"] = df.date.tolist()
        return st

    H = {x: stats(head[head.breakeven == x]) for x in bes}

    # ---- headline table: one column per break-even
    def row(label, fn):
        return f"<tr><td>{label}</td>" + "".join(f"<td>{fn(x)}</td>" for x in bes) + "</tr>"

    def g(x):
        return head[head.breakeven == x]

    num = lambda v, f="{:.2f}": "-" if pd.isna(v) else f.format(v)
    body = "".join([
        row("Expiries traded", lambda x: f"{H[x]['days']} of {n_sessions}"),
        row("Median actual break-even (up / down)",
            lambda x: f"+{g(x).be_up_pct.median():.2f}% / −{g(x).be_dn_pct.median():.2f}%"),
        row("Median strikes from the open (CE / PE, points)",
            lambda x: f"{(g(x).ce_strike - g(x).index_open).median():.0f} / "
                      f"{(g(x).index_open - g(x).pe_strike).median():.0f}"),
        row("Median premium collected (CE + PE, points)",
            lambda x: f"{(g(x).ce_entry + g(x).pe_entry).median():.2f}"),
        row("Average profit per expiry", lambda x: f"{fmt_rs(H[x]['avg'])} ({H[x]['avg_pts']:+.2f} pts)"),
        row("Median profit per expiry", lambda x: fmt_rs(H[x]["median"])),
        row("Total profit", lambda x: fmt_rs(H[x]["total"])),
        row("Winning expiries", lambda x: f"{H[x]['win_rate']:.0f}%"),
        row("Expiries with a stop-loss", lambda x: f"{H[x]['sl_rate']:.0f}%"),
        row("Average win / average loss", lambda x: f"{fmt_rs(H[x]['avg_win'])} / {fmt_rs(H[x]['avg_loss'])}"),
        row("Best / worst expiry", lambda x: f"{fmt_rs(H[x]['best'])} / {fmt_rs(H[x]['worst'])}"),
        row("Profit factor (gross wins ÷ gross losses)", lambda x: num(H[x]["profit_factor"])),
        row("Maximum drawdown", lambda x: fmt_rs(H[x]["max_dd"])),
        row("Drawdown ran from → to", lambda x: H[x]["dd_text"]),
        row("Time to recover from it", lambda x: H[x]["recovery_text"]),
        row("Longest stretch below a previous peak", lambda x: H[x]["underwater_text"]),
        row("Longest run of losing expiries", lambda x: str(H[x]["losing_streak"])),
        row("Total profit ÷ maximum drawdown",
            lambda x: num(H[x]["total"] / -H[x]["max_dd"]) if H[x]["max_dd"] < 0 else "-"),
        row("Average ÷ standard deviation per expiry", lambda x: num(H[x]["avg_over_std"])),
    ])
    cards.append((
        f"Results with a {sl_lab}-point stop-loss per option",
        f"Entry at the high-low midpoint of each leg's 09:15 candle; {mode_txt[cfg.sl_mode]}. "
        f"₹ for {cfg.qty} quantity after ₹{cfg.charges:g} charges per trade.",
        "<table><thead><tr><th>Measure</th>"
        + "".join(f"<th>{x:g}% break-even</th>" for x in bes) + f"</tr></thead><tbody>{body}</tbody></table>"))

    for x in bes:
        h = H[x]
        insights.append(
            f"<b>{x:g}% break-even</b>, {sl_lab}-point stop-loss: {h['days']} expiries, average "
            f"<b>{fmt_rs(h['avg'])}</b> per expiry, total <b>{fmt_rs(h['total'])}</b>, "
            f"{h['win_rate']:.0f}% winners, stop-loss on {h['sl_rate']:.0f}%. Maximum drawdown "
            f"<b>{fmt_rs(h['max_dd'])}</b> ({h['dd_text']}); recovery: {h['recovery_text']}.")

    # ---- equity and drawdown
    eq_tr, dd_tr = [], []
    for i, x in enumerate(bes):
        h = H[x]
        common = {"x": h["dates"], "type": "scatter", "mode": "lines", "name": f"{x:g}% break-even",
                  "line": {"color": SERIES[i], "width": 2}}
        eq_tr.append({**common, "y": np.round(h["eq"], 0).tolist(),
                      "hovertemplate": "%{x}<br>₹%{y:,.0f}<extra>%{fullData.name}</extra>"})
        dd_tr.append({**common, "y": np.round(h["dd"], 0).tolist(),
                      "hovertemplate": "%{x}<br>₹%{y:,.0f} below the peak<extra>%{fullData.name}</extra>"})
    figs.append((f"Cumulative profit by break-even ({sl_lab}-point stop-loss)",
                 "One point per expiry, after charges.",
                 eq_tr, {"yaxis": {"title": "₹ cumulative", "tickformat": ",.0f"}}))
    bar_tr, buttons = [], []
    for i, x in enumerate(bes):
        d = g(x).sort_values("date")
        rs = be_rupees(d, cfg).round(0)
        bar_tr.append({
            "type": "bar", "x": d.date.tolist(), "y": rs.tolist(), "visible": i == 0,
            "name": f"{x:g}% break-even",
            "marker": {"color": ["#2a78d6" if v > 0 else "#d03b3b" for v in rs]},
            "customdata": np.column_stack([d.ce_strike, d.pe_strike,
                                           d.sl_hit.replace("", "none"),
                                           d.sl_time.replace("", "-")]).tolist(),
            "hovertemplate": "%{x}<br>₹%{y:,.0f}<br>Sold %{customdata[0]} CE / %{customdata[1]} PE"
                             "<br>Stop-loss: %{customdata[2]} %{customdata[3]}<extra></extra>"})
        buttons.append({"label": f"{x:g}% break-even", "method": "update",
                        "args": [{"visible": [j == i for j in range(len(bes))]}]})
    figs.append((f"Profit on each expiry ({sl_lab}-point stop-loss)",
                 "One bar per expiry: blue made money, red lost. Pick the break-even from the menu "
                 "and hover a bar for the strikes and the stop-loss.",
                 bar_tr, {"updatemenus": [{"buttons": buttons, "x": 0, "xanchor": "left", "y": 1.22,
                                           "yanchor": "top", "showactive": True}],
                          "margin": {"t": 64}, "_height": 440, "showlegend": False,
                          "yaxis": {"title": "₹ per expiry", "tickformat": ",.0f"}}))
    figs.append(("Drawdown: how far below its previous peak each curve sat",
                 "Zero means a new high. The deepest dip is the maximum drawdown; the width of a "
                 "dip is the time it took to recover.",
                 dd_tr, {"yaxis": {"title": "₹ below peak", "tickformat": ",.0f"}}))

    # ---- stop-loss grid
    sls = [f"{v:g}" for v in cfg.be_sl_grid] + ["none"]

    def lab(sl):
        return "no stop-loss" if sl == "none" else f"{sl} pts"

    ylab = [lab(v) for v in sls]
    grid = {}
    for sl in sls:
        for x in bes:
            d = mid[(mid.sl == sl) & (mid["mode"] == cfg.sl_mode) & (mid.breakeven == x)]
            grid[(sl, x)] = stats(d) if len(d) else {}

    def mat(key):
        return np.array([[grid[(sl, x)].get(key, np.nan) for x in bes] for sl in sls], float)

    rec_ok = np.array([[grid[(sl, x)].get("recovered", True) for x in bes] for sl in sls], bool)
    hover_extra = ("<br>Total %{customdata[0]}<br>Max drawdown %{customdata[1]}"
                   "<br>Recovery: %{customdata[2]}<br>Win rate %{customdata[3]:.0f}%, "
                   "stop-loss on %{customdata[4]:.0f}% of expiries<extra></extra>")
    cd = [[[fmt_rs(grid[(sl, x)].get("total", np.nan)),
            fmt_rs(grid[(sl, x)].get("max_dd", np.nan)),
            grid[(sl, x)].get("recovery_text", "-"),
            grid[(sl, x)].get("win_rate", np.nan),
            grid[(sl, x)].get("sl_rate", np.nan)] for x in bes] for sl in sls]

    MEASURES = [
        ("avg", "Average profit per expiry (₹)", "rs", "div"),
        ("max_dd", "Maximum drawdown (₹)", "rs", "bad"),
        ("rec_exp", "Expiries taken to recover from it", "plus", "bad_lo"),
        ("rec_days", "Calendar days taken to recover", "plus", "bad_lo"),
        ("uw_exp", "Longest stretch below a previous peak (expiries)", "int", "bad_lo"),
        ("total", "Total profit (₹)", "rs", "div"),
        ("ratio", "Total profit ÷ maximum drawdown", "x1", "good"),
        ("win_rate", "Winning expiries (%)", "pct0", "good"),
        ("sl_rate", "Expiries with a stop-loss (%)", "pct0", "bad_lo"),
        ("profit_factor", "Profit factor (gross wins ÷ gross losses)", "x1", "good"),
        ("worst", "Worst single expiry (₹)", "rs", "bad"),
        ("avg_over_std", "Average ÷ standard deviation per expiry", "x2", "good"),
    ]
    RED = [[0, "#f0efec"], [1, "#d03b3b"]]
    BLUE = [[0, "#f0efec"], [1, "#2a78d6"]]
    traces, buttons = [], []
    for mi, (key, label, fmt, kind) in enumerate(MEASURES):
        if key == "ratio":
            z = np.where(mat("max_dd") < 0, mat("total") / -mat("max_dd"), np.nan)
        else:
            z = mat(key)
        fin = z[np.isfinite(z)]
        if kind == "div":
            limv = float(np.nanmax(np.abs(fin))) if fin.size else 1.0
            zmin, zmax = -limv, limv
            scale = [[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#2a78d6"]]
        elif kind == "bad":          # negative numbers: deeper red is worse
            zmin, zmax, scale = (float(fin.min()) if fin.size else -1.0), 0.0, \
                [[0, "#d03b3b"], [1, "#f0efec"]]
        elif kind == "bad_lo":       # positive numbers where lower is better
            zmin, zmax, scale = 0.0, (float(fin.max()) if fin.size else 1.0), RED
        else:
            zmin, zmax, scale = 0.0, (float(fin.max()) if fin.size else 1.0), BLUE
        txt = []
        for i in range(z.shape[0]):
            row = []
            for j in range(z.shape[1]):
                v = z[i, j]
                if not np.isfinite(v):
                    row.append("")
                elif fmt == "rs":
                    row.append(fmt_rs(v))
                elif fmt == "pct0":
                    row.append(f"{v:.0f}%")
                elif fmt == "x1":
                    row.append(f"{v:.1f}")
                elif fmt == "x2":
                    row.append(f"{v:.2f}")
                elif fmt == "plus":
                    row.append(f"{v:.0f}" + ("" if rec_ok[i, j] else "+"))
                else:
                    row.append(f"{v:.0f}")
            txt.append(row)
        traces.append({
            "type": "heatmap", "x": [f"{x:g}%" for x in bes], "y": ylab, "visible": mi == 0,
            "z": np.where(np.isfinite(z), np.round(z, 3), None).tolist(),
            "text": txt, "texttemplate": "%{text}", "customdata": cd,
            "zmin": zmin, "zmax": zmax, "xgap": 2, "ygap": 2, "colorscale": scale,
            "colorbar": {"thickness": 10},
            "hovertemplate": "%{x} break-even, %{y}<br>" + label + ": %{text}" + hover_extra})
        buttons.append({"label": label, "method": "update",
                        "args": [{"visible": [i == mi for i in range(len(MEASURES))]}]})
    figs.append(("Every stop-loss against every break-even",
                 "Pick the measure from the menu. A stop-loss is that many points above each "
                 "option's entry price. “+” on a recovery figure means it had not recovered by the "
                 "end of the data. Hover any cell for the rest of its numbers.",
                 traces,
                 {"updatemenus": [{"buttons": buttons, "x": 0, "xanchor": "left", "y": 1.18,
                                   "yanchor": "top", "showactive": True}],
                  "xaxis": {"type": "category", "title": "Break-even distance"},
                  "yaxis": {"autorange": "reversed", "type": "category"},
                  "margin": {"l": 110, "t": 78}, "_height": 480}))

    # ---- equity and drawdown curves, one stop-loss per line
    for which, title, note, ytitle in (
            ("eq", "Cumulative profit by stop-loss", "Pick a break-even from the menu. Each line "
             "is the same entries with a different stop-loss.", "₹ cumulative"),
            ("dd", "Drawdown by stop-loss", "How far each line sat below its own previous peak. "
             "The deepest point is the maximum drawdown; the width of a dip is the recovery time.",
             "₹ below peak")):
        tr, btns = [], []
        for xi, x in enumerate(bes):
            for si, sl in enumerate(sls):
                st = grid[(sl, x)]
                if not st:
                    continue
                tr.append({"x": st["dates"], "y": np.round(st[which], 0).tolist(),
                           "type": "scatter", "mode": "lines", "name": lab(sl),
                           "visible": xi == 0, "_be": xi,
                           "line": {"color": SERIES[si % len(SERIES)],
                                    "width": 2.5 if sl == f"{cfg.be_sl:g}" else 1.5},
                           "hovertemplate": "%{x}<br>₹%{y:,.0f}<extra>%{fullData.name}</extra>"})
        for xi, x in enumerate(bes):
            btns.append({"label": f"{x:g}% break-even", "method": "update",
                         "args": [{"visible": [t["_be"] == xi for t in tr]}]})
        for t in tr:
            t.pop("_be")
        figs.append((title, note, tr,
                     {"updatemenus": [{"buttons": btns, "x": 0, "xanchor": "left", "y": 1.2,
                                       "yanchor": "top", "showactive": True}],
                      "margin": {"t": 72}, "_height": 450,
                      "yaxis": {"title": ytitle, "tickformat": ",.0f"}}))

    # ---- return against risk
    tr = []
    for xi, x in enumerate(bes):
        pts = [(sl, grid[(sl, x)]) for sl in sls if grid[(sl, x)]]
        tr.append({"type": "scatter", "mode": "markers+text",
                   "x": [-v["max_dd"] for _, v in pts], "y": [v["avg"] for _, v in pts],
                   "text": [("none" if sl == "none" else sl) for sl, _ in pts],
                   "textposition": "top center", "textfont": {"size": 10},
                   "name": f"{x:g}% break-even",
                   "marker": {"size": 11, "color": SERIES[xi % len(SERIES)]},
                   "customdata": [[fmt_rs(v["total"]), v["win_rate"], v["recovery_text"]]
                                  for _, v in pts],
                   "hovertemplate": "Stop-loss %{text}<br>Avg ₹%{y:,.0f} per expiry<br>"
                                    "Max drawdown ₹%{x:,.0f}<br>Total %{customdata[0]}<br>"
                                    "Win rate %{customdata[1]:.0f}%<br>Recovery: %{customdata[2]}"
                                    "<extra>%{fullData.name}</extra>"})
    figs.append(("Return against risk: every stop-loss on one picture",
                 "Each dot is one stop-loss, labelled in points. Up is more profit per expiry; "
                 "right is a deeper drawdown. Top-left is the better corner.",
                 tr, {"xaxis": {"title": "Maximum drawdown (₹, deeper to the right)",
                                "tickformat": ",.0f"},
                      "yaxis": {"title": "Average profit per expiry (₹)", "tickformat": ",.0f",
                                "zeroline": True, "zerolinecolor": "#c3c2b7"}}))

    # ---- the whole grid as a table
    trs = ""
    for x in bes:
        for sl in sls:
            v = grid[(sl, x)]
            if not v:
                continue
            ratio = (v["total"] / -v["max_dd"]) if v["max_dd"] < 0 else None
            trs += (f"<tr><td>{x:g}%</td><td>{lab(sl)}</td><td>{v['days']}</td>"
                    f"<td>{fmt_rs(v['avg'])}</td><td>{fmt_rs(v['median'])}</td>"
                    f"<td>{fmt_rs(v['total'])}</td><td>{v['win_rate']:.0f}%</td>"
                    f"<td>{v['sl_rate']:.0f}%</td><td>{fmt_rs(v['worst'])}</td>"
                    f"<td>{fmt_rs(v['max_dd'])}</td><td>{v['dd_text']}</td>"
                    f"<td>{v['recovery_text']}</td><td>{v['underwater_text']}</td>"
                    f"<td>{num(v['profit_factor'])}</td>"
                    f"<td>{'-' if ratio is None else f'{ratio:.1f}'}</td>"
                    f"<td>{num(v['avg_over_std'])}</td></tr>")
    cards.append((
        "Every stop-loss and break-even, in full",
        f"Entry at the 09:15 candle's midpoint, {mode_txt[cfg.sl_mode]}, {cfg.qty} quantity after "
        f"₹{cfg.charges:g} charges per trade.",
        "<div class='scroll tall'><table><thead><tr><th>Break-even</th><th>Stop-loss</th>"
        "<th>Expiries</th><th>Avg / expiry</th><th>Median / expiry</th><th>Total</th>"
        "<th>Win rate</th><th>Stopped out</th><th>Worst expiry</th><th>Max drawdown</th>"
        "<th>Drawdown ran</th><th>Recovery</th><th>Longest below peak</th><th>Profit factor</th>"
        f"<th>Total ÷ drawdown</th><th>Avg ÷ std dev</th></tr></thead><tbody>{trs}</tbody></table></div>"))

    # ---- insights on the grid
    for x in bes:
        ok = {sl: grid[(sl, x)] for sl in sls if grid[(sl, x)]}
        if not ok:
            continue
        shallow = max(ok, key=lambda k: ok[k]["max_dd"])
        quick = {k: v for k, v in ok.items() if v["max_dd"] < 0 and v["recovered"]}
        fastest = min(quick, key=lambda k: quick[k]["rec_exp"]) if quick else None
        insights.append(
            f"{x:g}% break-even, drawdown: the shallowest was <b>{lab(shallow)}</b> at "
            f"{fmt_rs(ok[shallow]['max_dd'])}"
            + (f", and the quickest to recover was <b>{lab(fastest)}</b> "
               f"({quick[fastest]['recovery_text']})." if fastest else ".")
            + f" Your {sl_lab}-point stop-loss drew down {fmt_rs(ok[sl_lab]['max_dd'])}"
            + (f" and took {ok[sl_lab]['recovery_text']}." if ok[sl_lab]["max_dd"] < 0 else "."))

    trs = ""
    for x in bes:
        ok = {sl: grid[(sl, x)] for sl in sls if grid[(sl, x)]}
        if not ok:
            continue
        by_avg = max(ok, key=lambda k: ok[k]["avg"])
        by_dd = max(ok, key=lambda k: ok[k]["max_dd"])
        ratio = {k: (v["total"] / -v["max_dd"]) for k, v in ok.items() if v["max_dd"] < 0}
        by_ratio = max(ratio, key=ratio.get) if ratio else None
        trs += (f"<tr><td>{x:g}%</td>"
                f"<td>{lab(by_avg)} ({fmt_rs(ok[by_avg]['avg'])} per expiry)</td>"
                f"<td>{lab(by_dd)} ({fmt_rs(ok[by_dd]['max_dd'])})</td>"
                f"<td>{(lab(by_ratio) + f' ({ratio[by_ratio]:.1f})') if by_ratio else '-'}</td>"
                f"<td>{fmt_rs(ok[sl_lab]['avg']) if sl_lab in ok else '-'} per expiry, "
                f"{fmt_rs(ok[sl_lab]['max_dd']) if sl_lab in ok else '-'} drawdown</td></tr>")
        insights.append(
            f"{x:g}% break-even: the stop-loss with the highest average was <b>{lab(by_avg)}</b> "
            f"({fmt_rs(ok[by_avg]['avg'])} per expiry, drawdown {fmt_rs(ok[by_avg]['max_dd'])}); "
            f"the smallest drawdown came with <b>{lab(by_dd)}</b>.")
    cards.append(("Which stop-loss did best",
                  "Chosen after the fact from the same history, so the winner is flattered: treat "
                  "small differences between neighbouring stop-losses as noise.",
                  "<table><thead><tr><th>Break-even</th><th>Highest average</th>"
                  "<th>Smallest maximum drawdown</th><th>Best total ÷ drawdown</th>"
                  f"<th>Your {sl_lab} pts, for comparison</th></tr></thead><tbody>{trs}</tbody></table>"))

    # ---- exit both vs independent legs
    trs = ""
    for x in bes:
        for mode in ("both", "independent"):
            d = mid[(mid.sl == sl_lab) & (mid["mode"] == mode) & (mid.breakeven == x)]
            m = stats(d)
            if m:
                trs += (f"<tr><td>{x:g}%</td><td>{mode_txt[mode].capitalize()}</td>"
                        f"<td>{fmt_rs(m['avg'])}</td><td>{fmt_rs(m['total'])}</td>"
                        f"<td>{fmt_rs(m['worst'])}</td><td>{fmt_rs(m['max_dd'])}</td>"
                        f"<td>{m['win_rate']:.0f}%</td></tr>")
    cards.append((f"What happens to the other leg when a {sl_lab}-point stop-loss hits",
                  "Same entries, two ways of handling the leg that was not stopped.",
                  "<table><thead><tr><th>Break-even</th><th>Rule</th><th>Avg / expiry</th><th>Total</th>"
                  f"<th>Worst expiry</th><th>Max drawdown</th><th>Win rate</th></tr></thead><tbody>{trs}</tbody></table>"))

    # ---- entry price comparison
    names = {"mid": "First-candle midpoint (high + low) ÷ 2", "open": "First-candle open",
             "close": "First-candle close"}
    trs = ""
    for x in bes:
        for ent in ["mid", "open", "close"] + [f"limit +{v:g}%" for v in cfg.entry_limits]:
            d = B[(B.entry == ent) & (B.sl == sl_lab) & (B["mode"] == cfg.sl_mode) & (B.breakeven == x)]
            m = stats(d)
            if not m:
                continue
            both, one, none = (d.legs == 2).sum(), (d.legs == 1).sum(), (d.legs == 0).sum()
            nm = names[ent] if ent in names else f"Sell limit {ent.split()[1]} above the open"
            trs += (f"<tr><td>{x:g}%</td><td>{nm}</td><td>{both} / {one} / {none}</td>"
                    f"<td>{num((d.ce_entry.fillna(0) + d.pe_entry.fillna(0))[d.legs > 0].median())}</td>"
                    f"<td>{fmt_rs(m['avg'])}</td><td>{fmt_rs(m['total'])}</td>"
                    f"<td>{fmt_rs(m['worst'])}</td><td>{fmt_rs(m['max_dd'])}</td></tr>")
    cards.append(("Does the entry price hold up? Midpoint against prices you could actually get",
                  "The midpoint is only known once the candle has finished, so it cannot be an order. "
                  "The open and close are prices that traded. A limit leg counts only when the candle "
                  "traded above it; when one leg fills and the other does not, the filled leg is "
                  "carried alone with its stop-loss. Averages are per expiry, including expiries "
                  "where nothing filled.",
                  "<div class='scroll'><table><thead><tr><th>Break-even</th><th>Entry</th>"
                  "<th>Expiries with both / one / no legs filled</th><th>Median premium sold</th>"
                  "<th>Avg / expiry</th><th>Total</th><th>Worst expiry</th><th>Max drawdown</th>"
                  f"</tr></thead><tbody>{trs}</tbody></table></div>"))

    # ---- coverage
    ql = []
    for x in bes:
        ql.append(f"{x:g}% break-even: traded on {H[x]['days']} of {n_sessions} expiry sessions.")
    for (x, why), n in sorted(be_skips.items(), key=lambda kv: (kv[0][0], -kv[1])):
        ql.append(f"{x:g}% break-even: {n} session(s) skipped, {why}.")
    ex = mid[mid.entry_minute != "09:15"]
    if len(ex):
        ql.append(f"{ex.date.nunique()} session(s) had no option trade in the 09:15 minute itself; "
                  f"the first traded minute after it was used instead.")
    ql.append("Index and option bars before 09:15 (the pre-open session) are ignored.")
    ql.append("The entry minute's own high and low are not used to trigger a stop-loss, because "
              "the data does not say whether they came before or after the midpoint. Stop-losses "
              "are checked from the candle's close onward.")
    cards.append(("Coverage of this backtest", "Counted from the dataset files on this run.",
                  "<ul>" + "".join(f"<li>{t}</li>" for t in ql) + "</ul>"))
    return {"insights": insights, "figs": figs, "cards": cards}


def build_tag_section(sym, tg, n_sessions, band, cfg):
    """Far-OTM strikes tagged by their distance (points) from the 09:15 ATM strike."""
    lo, hi = band
    tags = sorted(tg.tag.unique())
    labels = [f"{t:g}" for t in tags]
    color = {"CE": SERIES[0], "PE": SERIES[1]}
    nice = {"CE": "CE (above ATM)", "PE": "PE (below ATM)"}
    cat_axis = {"type": "category", "categoryorder": "array", "categoryarray": labels,
                "title": "Points from ATM at 09:15"}

    def stat(g):
        m = g.mult_after_first
        has = m.notna().any()
        return pd.Series({
            "days": len(g), "pct_0915": g.first_is_0915.mean() * 100,
            "pct_otm": g.pct_otm.median(),
            "o": g.first_open.median(), "h": g.first_high.median(),
            "l": g.first_low.median(), "c": g.first_close.median(),
            "c_q1": g.first_close.quantile(.25), "c_q3": g.first_close.quantile(.75),
            "later_high": g.high_after_first.median(),
            "traded_again": m.notna().mean() * 100,
            "mult_med": m.median(), "mult_p90": m.quantile(.9), "mult_max": m.max(),
            "mult_max_date": g.loc[m.idxmax(), "date"] if has else "",
            "ge2": (m >= 2).mean() * 100, "ge3": (m >= 3).mean() * 100,
            "itm": (g.settle_value > 0).mean() * 100,
        })

    st = tg.groupby(["type", "tag"]).apply(stat, include_groups=False).reset_index()
    figs, insights = [], []

    # ---- insights, all counted from the rows
    insights.append(
        f"<b>{tg.date.nunique()}</b> of {n_sessions} expiry sessions had at least one traded strike "
        f"{lo:g}–{hi:g} points from ATM, giving <b>{len(tg):,}</b> option-days. "
        f"{tg.first_is_0915.mean() * 100:.0f}% of them traded in the 09:15 minute; for the others the "
        f"first trade of the day is used as the first candle.")
    for cp in ("CE", "PE"):
        x = st[st.type == cp].sort_values("tag")
        if len(x) >= 2:
            a, b = x.iloc[0], x.iloc[-1]
            insights.append(
                f"{nice[cp]}: the median first-candle close falls from <b>{a.c:.2f}</b> at "
                f"{a.tag:g} points (about {a.pct_otm:.1f}% away) to <b>{b.c:.2f}</b> at {b.tag:g} "
                f"points (about {b.pct_otm:.1f}% away).")
    m = tg.mult_after_first
    insights.append(
        f"After its first candle, an option later traded at 2× that candle's close or more on "
        f"<b>{(m >= 2).mean() * 100:.0f}%</b> of option-days, 3× or more on "
        f"<b>{(m >= 3).mean() * 100:.0f}%</b> and 5× or more on <b>{(m >= 5).mean() * 100:.0f}%</b> "
        f"(CE {(tg[tg.type == 'CE'].mult_after_first >= 2).mean() * 100:.0f}% / "
        f"PE {(tg[tg.type == 'PE'].mult_after_first >= 2).mean() * 100:.0f}% for 2×).")
    if m.notna().any():
        w = tg.loc[m.idxmax()]
        insights.append(
            f"Largest jump: {w.date}, {w.strike:g} {w.type} ({w.tag:g} points away) went from a "
            f"first-candle close of {w.first_close:.2f} to {w.high_after_first:.2f} at "
            f"{w.high_after_first_time}, <b>{w.mult_after_first:.0f}×</b>.")
    itm = tg[tg.settle_value > 0]
    if len(itm):
        ds = sorted(itm.date.unique())
        insights.append(
            f"{len(itm)} option-day(s) on {len(ds)} session(s) settled in the money "
            f"({', '.join(ds[:6])}{'…' if len(ds) > 6 else ''}).")
    else:
        insights.append("No strike in this band settled in the money on any session in the data.")

    # ---- 1. first-candle premium by tag
    tr = []
    for cp in ("CE", "PE"):
        x = st[st.type == cp].sort_values("tag")
        tr.append({"x": [f"{t:g}" for t in x.tag], "y": x.c.round(2).tolist(),
                   "customdata": np.column_stack([x.days, x.c_q1.round(2), x.c_q3.round(2),
                                                  x.o.round(2), x.h.round(2), x.l.round(2)]).tolist(),
                   "type": "scatter", "mode": "lines+markers", "name": nice[cp],
                   "line": {"color": color[cp], "width": 2}, "marker": {"size": 8},
                   "hovertemplate": "%{x} pts from ATM<br>Median first candle: O %{customdata[3]} "
                                    "H %{customdata[4]} L %{customdata[5]} C %{y}<br>"
                                    "Middle half of days: %{customdata[1]}–%{customdata[2]}<br>"
                                    "%{customdata[0]} days<extra>%{fullData.name}</extra>"})
    figs.append((f"First-candle premium by distance from ATM ({sym})",
                 "Median close of each strike's first 1-minute candle on expiry day. Hover for the "
                 "median open, high and low and the number of days behind each point.",
                 tr, {"xaxis": cat_axis, "yaxis": {"title": "Points", "rangemode": "tozero"}}))

    # ---- 2. how far each tag ran later in the day
    tr = []
    for cp in ("CE", "PE"):
        x = tg[(tg.type == cp) & tg.mult_after_first.notna()]
        tr.append({"type": "box", "x": [f"{t:g}" for t in x.tag],
                   "y": x.mult_after_first.round(2).tolist(),
                   "text": [f"{d} · {k:g} {cp}" for d, k in zip(x.date, x.strike)],
                   "name": nice[cp], "marker": {"color": color[cp], "size": 4},
                   "line": {"width": 1.5}, "boxpoints": "outliers",
                   "hovertemplate": "%{text}<br>%{y}× first-candle close<extra></extra>"})
    figs.append(("How far each strike ran later the same day",
                 "Highest price after the first candle, as a multiple of the first candle's close. "
                 "1× means it never traded higher. Boxes show the middle half of days; dots are the "
                 "unusual days. Log scale.",
                 tr, {"boxmode": "group", "xaxis": cat_axis,
                      "yaxis": {"title": "× first-candle close", "type": "log"}}))

    # ---- 3/4. how often it multiplied
    thr = [1.5, 2, 3, 5, 10]
    grids = {}
    for cp in ("CE", "PE"):
        z, n = [], []
        for t in tags:
            x = tg[(tg.type == cp) & (tg.tag == t)]
            n.append(len(x))
            z.append([(x.mult_after_first >= k).mean() * 100 if len(x) else np.nan for k in thr])
        grids[cp] = (np.array(z).T, n)
    zmax = max([np.nanmax(g[0]) for g in grids.values() if np.isfinite(g[0]).any()] + [1.0])
    for cp in ("CE", "PE"):
        z, n = grids[cp]
        figs.append((f"How often a {cp} multiplied after its first candle (% of days)",
                     "Share of expiry days on which the strike later traded at the given multiple "
                     "of its first-candle close. Days it never traded again count as not reaching it.",
                     [{"type": "heatmap", "x": labels, "y": [f"{k:g}× or more" for k in thr],
                       "z": np.where(np.isnan(z), None, np.round(z, 1)).tolist(),
                       "text": [["" if np.isnan(v) else f"{v:.0f}%" for v in row] for row in z],
                       "texttemplate": "%{text}", "customdata": [n] * len(thr),
                       "zmin": 0, "zmax": float(zmax), "xgap": 2, "ygap": 2,
                       "colorscale": [[0, "#f0efec"], [1, color[cp]]],
                       "colorbar": {"title": "% of days", "thickness": 10},
                       "hovertemplate": "%{x} pts from ATM, %{y}<br>%{text} of %{customdata} days"
                                        "<extra></extra>"}],
                     {"xaxis": cat_axis, "yaxis": {"autorange": "reversed"},
                      "margin": {"l": 110}}))

    # ---- 5. one chart per tag, picked from a menu
    tr, buttons = [], []
    for ti, t in enumerate(tags):
        for cp in ("CE", "PE"):
            x = tg[(tg.type == cp) & (tg.tag == t)].sort_values("date")
            for col, nm, dash in (("first_close", "first-candle close", "solid"),
                                  ("high_after_first", "highest later that day", "dot")):
                tr.append({"x": x.date.tolist(), "y": x[col].round(2).tolist(),
                           "customdata": x.strike.tolist(), "visible": ti == 0,
                           "type": "scatter", "mode": "lines+markers", "name": f"{cp} {nm}",
                           "line": {"color": color[cp], "width": 1.5, "dash": dash},
                           "marker": {"size": 5},
                           "hovertemplate": "%{x} · strike %{customdata}<br>%{y} pts"
                                            "<extra>%{fullData.name}</extra>"})
        buttons.append({"label": f"{t:g} points from ATM", "method": "update",
                        "args": [{"visible": [i // 4 == ti for i in range(4 * len(tags))]}]})
    figs.append(("Every expiry, one distance at a time",
                 "Pick a distance from the menu. Solid lines are the first-candle close; dotted "
                 "lines are the highest price later that day. Log scale.",
                 tr, {"updatemenus": [{"buttons": buttons, "x": 0, "xanchor": "left", "y": 1.22,
                                       "yanchor": "top", "showactive": True}],
                      "margin": {"t": 64}, "_height": 440,
                      "yaxis": {"title": "Option price (points)", "type": "log"}}))

    # ---- limit orders on the first candle (only real 09:15 candles)
    extra_cards = []
    L = tg[tg.first_is_0915 & (tg.first_open > 0)].copy()
    if len(L):
        tgt = cfg.limit_fill_target
        L["up"] = (L.first_high / L.first_open - 1) * 100
        L["mid_up"] = ((L.first_high + L.first_low) / 2 / L.first_open - 1) * 100
        L["close_up"] = (L.first_close / L.first_open - 1) * 100
        L["held_high"] = L.high_after_first.isna() | (L.first_high >= L.high_after_first)
        L["vs_prev"] = np.where(L.prev_close > 0, L.first_high / L.prev_close, np.nan)

        def lim(g):
            return pd.Series({
                "days": len(g), "open": g.first_open.median(),
                "mid_up": g.mid_up.median(), "close_up": g.close_up.median(),
                "q_t": g.up.quantile(1 - tgt / 100), "q_50": g.up.quantile(.5),
                "f10": (g.up > 10).mean() * 100, "f20": (g.up > 20).mean() * 100,
                "held": g.held_high.mean() * 100,
                "prev_n": g.vs_prev.notna().sum(),
                "prev_t": g.vs_prev.quantile(1 - tgt / 100), "prev_50": g.vs_prev.quantile(.5),
            })

        ls = L.groupby(["type", "tag"]).apply(lim, include_groups=False).reset_index()
        insights.append(
            f"First-candle limit orders ({len(L):,} option-days with a real 09:15 candle): a sell limit "
            f"10% above the candle's open would have filled on <b>{(L.up > 10).mean() * 100:.0f}%</b> "
            f"of option-days and 20% above on <b>{(L.up > 20).mean() * 100:.0f}%</b>. The candle's "
            f"high-low midpoint sat a median <b>{L.mid_up.median():+.1f}%</b> from its open, and the "
            f"first candle held the day's highest price on <b>{L.held_high.mean() * 100:.0f}%</b> "
            f"of option-days.")

        tr = []
        for cp in ("CE", "PE"):
            x = ls[ls.type == cp].sort_values("tag")
            for col, nm, dash in (("q_t", f"filled on {tgt:g}% of days", "solid"),
                                  ("q_50", "filled on 50% of days", "dot")):
                tr.append({"x": [f"{t:g}" for t in x.tag], "y": x[col].round(1).tolist(),
                           "customdata": np.column_stack([x.days, x.open.round(2)]).tolist(),
                           "type": "scatter", "mode": "lines+markers", "name": f"{cp}: {nm}",
                           "line": {"color": color[cp], "width": 2, "dash": dash},
                           "marker": {"size": 7},
                           "hovertemplate": "%{x} pts from ATM<br>Limit %{y}% above the open<br>"
                                            "Median open %{customdata[1]} pts, %{customdata[0]} days"
                                            "<extra>%{fullData.name}</extra>"})
        figs.append(("Highest sell limit on the first candle that would have filled",
                     "How far above the 09:15 candle's open a sell limit could sit and still be "
                     "traded through inside that candle. A limit counts as filled only if the "
                     "candle's high was above it.",
                     tr, {"xaxis": cat_axis, "yaxis": {"title": "% above the candle's open",
                                                       "rangemode": "tozero"}}))

        lv = [0, 5, 10, 20, 30, 50]
        for cp in ("CE", "PE"):
            z, n = [], []
            for t in tags:
                x = L[(L.type == cp) & (L.tag == t)]
                n.append(len(x))
                z.append([(x.up > k).mean() * 100 if len(x) else np.nan for k in lv])
            z = np.array(z).T
            figs.append((f"Fill rate of a {cp} sell limit placed above the first candle's open",
                         "Share of expiry days on which the first candle traded above the limit.",
                         [{"type": "heatmap", "x": labels,
                           "y": ["above the open" if k == 0 else f"open +{k}%" for k in lv],
                           "z": np.where(np.isnan(z), None, np.round(z, 1)).tolist(),
                           "text": [["" if np.isnan(v) else f"{v:.0f}%" for v in row] for row in z],
                           "texttemplate": "%{text}", "customdata": [n] * len(lv),
                           "zmin": 0, "zmax": 100, "xgap": 2, "ygap": 2,
                           "colorscale": [[0, "#f0efec"], [1, color[cp]]],
                           "colorbar": {"title": "% of days", "thickness": 10},
                           "hovertemplate": "%{x} pts from ATM, limit %{y}<br>filled on %{text} of "
                                            "%{customdata} days<extra></extra>"}],
                         {"xaxis": cat_axis, "yaxis": {"autorange": "reversed"},
                          "margin": {"l": 120}}))

        if ls.prev_n.sum() > 0:
            tr = []
            for cp in ("CE", "PE"):
                x = ls[(ls.type == cp) & (ls.prev_n > 0)].sort_values("tag")
                for col, nm, dash in (("prev_t", f"filled on {tgt:g}% of days", "solid"),
                                      ("prev_50", "filled on 50% of days", "dot")):
                    tr.append({"x": [f"{t:g}" for t in x.tag], "y": x[col].round(2).tolist(),
                               "customdata": x.prev_n.tolist(),
                               "type": "scatter", "mode": "lines+markers", "name": f"{cp}: {nm}",
                               "line": {"color": color[cp], "width": 2, "dash": dash},
                               "marker": {"size": 7},
                               "hovertemplate": "%{x} pts from ATM<br>Limit at %{y}× the previous "
                                                "close<br>%{customdata} days"
                                                "<extra>%{fullData.name}</extra>"})
            figs.append(("The same limit, set before the open from the previous day's close",
                         "A limit you could place before 09:15: the multiple of the strike's previous "
                         "close that the first candle still traded above. Strikes are tagged from "
                         "the expiry-day open, so the exact strike is known only at 09:15.",
                         tr, {"xaxis": cat_axis, "yaxis": {"title": "× previous day's close",
                                                           "rangemode": "tozero"}}))

        def f1(v, suf="%"):
            return "-" if pd.isna(v) else f"{v:+.1f}{suf}"

        lrows = "".join(
            f"<tr><td>{r.tag:g}</td><td>{r.type}</td><td>{int(r.days)}</td><td>{r.open:.2f}</td>"
            f"<td>{f1(r.mid_up)}</td><td>{f1(r.close_up)}</td>"
            f"<td>{f1(r.q_t)} (≈{r.open * (1 + r.q_t / 100):.2f})</td><td>{f1(r.q_50)}</td>"
            f"<td>{r.f10:.0f}%</td><td>{r.f20:.0f}%</td>"
            f"<td>{'-' if pd.isna(r.prev_t) else f'{r.prev_t:.2f}×'}</td><td>{r.held:.0f}%</td></tr>"
            for r in ls.sort_values(["tag", "type"]).itertuples())
        extra_cards.append((
            "First-candle limit orders, every tag",
            "Percentages are relative to the 09:15 candle's open. The figure in brackets applies "
            "the limit to the median open.",
            "<div class='scroll'><table><thead><tr><th>Points from ATM</th><th>Side</th>"
            "<th>Days with a 09:15 candle</th><th>Median open</th><th>Median midpoint vs open</th>"
            "<th>Median close vs open</th>"
            f"<th>Limit filled on {tgt:g}% of days</th><th>Limit filled on 50% of days</th>"
            "<th>Fill rate at open +10%</th><th>Fill rate at open +20%</th>"
            f"<th>Pre-open limit filled on {tgt:g}% of days (× previous close)</th>"
            "<th>First candle held the day's high</th></tr></thead>"
            f"<tbody>{lrows}</tbody></table></div>"))

    # ---- table view
    def f2(v):
        return "-" if pd.isna(v) else f"{v:.2f}"

    rows = "".join(
        f"<tr><td>{r.tag:g}</td><td>{r.type}</td><td>{r.pct_otm:.1f}%</td><td>{int(r.days)}</td>"
        f"<td>{r.pct_0915:.0f}%</td><td>{f2(r.o)} / {f2(r.h)} / {f2(r.l)} / {f2(r.c)}</td>"
        f"<td>{f2(r.later_high)}</td><td>{f2(r.mult_med)}×</td><td>{f2(r.mult_p90)}×</td>"
        f"<td>{f2(r.mult_max)}× ({r.mult_max_date})</td><td>{r.ge2:.0f}%</td><td>{r.ge3:.0f}%</td>"
        f"<td>{r.itm:.1f}%</td></tr>"
        for r in st.sort_values(["tag", "type"]).itertuples())
    table = ("<div class='scroll'><table><thead><tr><th>Points from ATM</th><th>Side</th>"
             "<th>Median % away</th><th>Days</th><th>Traded at 09:15</th>"
             "<th>Median first candle O / H / L / C</th><th>Median later high</th>"
             "<th>Median multiple</th><th>90th pct multiple</th><th>Largest multiple</th>"
             "<th>2× or more</th><th>3× or more</th><th>Settled in the money</th></tr></thead>"
             f"<tbody>{rows}</tbody></table></div>")
    return {"insights": insights, "figs": figs, "table": table, "band": band,
            "extra_cards": extra_cards}


def explorer(tid, title, note, df, cols, filters, defaults, sum_col=None):
    """One browsable table: cols = [(column, label, kind)], kind in text/num/rs."""
    keep = [c for c, _, _ in cols]
    d = df[keep].copy()
    for c, _, kind in cols:
        if kind != "text":
            d[c] = pd.to_numeric(d[c], errors="coerce").round(0 if kind == "rs" else 2)
        else:
            d[c] = d[c].fillna("").astype(str)
    return {"id": tid, "title": title, "note": note,
            "cols": [{"label": lab, "kind": kind} for _, lab, kind in cols],
            "rows": json.loads(d.to_json(orient="values")),
            "filters": [keep.index(f) for f in filters],
            "defaults": {str(keep.index(k)): str(v) for k, v in defaults.items()},
            "sum": keep.index(sum_col) if sum_col else None}


def build_explorers(sym, T, BEF, TG, cfg):
    out = []
    if BEF is not None and len(BEF):
        b = BEF.sort_values(["date", "breakeven"]).copy()
        b["pnl_rs"] = be_rupees(b, cfg)
        b["breakeven"] = b.breakeven.map(lambda v: f"{v:g}%")
        b["rule"] = b["mode"].map({"both": "exit both", "independent": "legs independent"})
        out.append(explorer(
            f"x_{sym}_be", "Every break-even strangle trade",
            "One row per expiry, per break-even, entry price and stop-loss. Use the filters to "
            "change which set you are looking at; click a column heading to sort.",
            b, [("date", "Date", "text"), ("entry_minute", "Entry minute", "text"),
                ("breakeven", "Break-even", "text"),
                ("entry", "Entry price", "text"), ("sl", "Stop-loss (pts)", "text"),
                ("rule", "Other leg", "text"), ("index_open", "Index open", "num"),
                ("index_settle", "Index settle", "num"), ("ce_strike", "CE strike", "num"),
                ("pe_strike", "PE strike", "num"), ("ce_entry", "CE sold at", "num"),
                ("pe_entry", "PE sold at", "num"), ("be_up_pct", "Break-even up %", "num"),
                ("be_dn_pct", "Break-even down %", "num"), ("legs", "Legs filled", "num"),
                ("pnl_pts", "Profit (pts)", "num"), ("pnl_rs", "Profit (₹)", "rs"),
                ("sl_hit", "Leg stopped", "text"), ("sl_time", "Stop time", "text")],
            ["breakeven", "entry", "sl", "rule"],
            {"breakeven": b.breakeven.iloc[0], "entry": "mid", "sl": f"{cfg.be_sl:g}",
             "rule": "exit both" if cfg.sl_mode == "both" else "legs independent"}, "pnl_rs"))
    if T is not None and len(T):
        t = T.sort_values(["date", "config"]).copy()
        t["pnl_rs"] = (t.pnl_pts * cfg.qty - cfg.charges
                       - np.where(t.sl_hit != "", cfg.charges, 0))
        t["config"] = t.config.replace({"yours": f"yours ({cfg.ce_dist:g}% / {cfg.pe_dist:g}%)"})
        out.append(explorer(
            f"x_{sym}_tr", f"Every trade of the original rule ({cfg.entry} entry)",
            "One row per expiry, per strike distance and stop-loss rule.",
            t, [("date", "Date", "text"), ("config", "Strike distance", "text"),
                ("sl_rule", "Stop-loss rule", "text"), ("ce_strike", "CE strike", "num"),
                ("pe_strike", "PE strike", "num"), ("ce_prem", "CE sold at", "num"),
                ("pe_prem", "PE sold at", "num"), ("pnl_pts", "Profit (pts)", "num"),
                ("pnl_rs", "Profit (₹)", "rs"), ("sl_hit", "Leg stopped", "text"),
                ("sl_time", "Stop time", "text"), ("sl_trigger", "Stop trigger", "num"),
                ("sl_bar_close", "Price at end of stop minute", "num"),
                ("atm_straddle_pct", "ATM straddle %", "num"),
                ("max_up_pct", "Index max up %", "num"), ("max_dn_pct", "Index max down %", "num")],
            ["config", "sl_rule"],
            {"config": f"yours ({cfg.ce_dist:g}% / {cfg.pe_dist:g}%)",
             "sl_rule": f"+{cfg.sl_points:g} pts"}, "pnl_rs"))
    if TG is not None and len(TG):
        g = TG.sort_values(["date", "type", "tag"]).copy()
        g["tag"] = g.tag.map(lambda v: f"{v:g}")
        out.append(explorer(
            f"x_{sym}_tg", "Every far strike on every expiry",
            "One row per strike per expiry: its first candle, the previous close, and what it "
            "did for the rest of the day.",
            g, [("date", "Date", "text"), ("type", "Side", "text"), ("tag", "Points from ATM", "text"),
                ("strike", "Strike", "num"), ("pct_otm", "% away", "num"),
                ("index_open", "Index open", "num"), ("atm_strike", "ATM", "num"),
                ("first_time", "First candle at", "text"), ("first_open", "Open", "num"),
                ("first_high", "High", "num"), ("first_low", "Low", "num"),
                ("first_close", "Close", "num"), ("first_volume", "Volume", "num"),
                ("prev_close", "Previous close", "num"),
                ("high_after_first", "Highest later", "num"),
                ("high_after_first_time", "At", "text"),
                ("mult_after_first", "× first close", "num"), ("day_low", "Day low", "num"),
                ("last_close", "Last price", "num"), ("settle_value", "Settled at", "num")],
            ["type", "tag"], {}, None))
    return out


HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Expiry-day strangle lab</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--border:rgba(11,11,11,.10);--good:#006300;--bad:#d03b3b}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}
main{max-width:1100px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 12px}
h3{font-size:15px;margin:0 0 2px}.sub{color:var(--ink2);margin:0 0 20px}
.note{color:var(--muted);font-size:13px;margin:0 0 8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--ink2);font-size:13px}.tile .v{font-size:24px;font-weight:600}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 18px 8px;margin:16px 0}
ul.ins{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 16px 16px 34px}
ul.ins li{margin:6px 0}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--grid)}th{color:var(--ink2);font-weight:600}
.assume{font-size:13px;color:var(--ink2)}
.xp-bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:end;margin:8px 0 10px;font-size:13px;color:var(--ink2)}
.xp-bar label{display:flex;flex-direction:column;gap:2px}
.xp-bar select,.xp-bar input{font:inherit;padding:5px 8px;border:1px solid var(--border);border-radius:6px;background:var(--surface);color:var(--ink)}
.xp-wrap{overflow:auto;max-height:560px;border:1px solid var(--grid);border-radius:8px}
.xp-wrap table{font-size:13px}.xp-wrap td,.xp-wrap th{white-space:nowrap;padding:6px 10px}
.xp-wrap th{position:sticky;top:0;background:var(--surface);cursor:pointer;user-select:none}
.xp-wrap td.n{text-align:right}.xp-wrap td.neg{color:var(--bad)}
.xp-foot{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;font-size:13px;color:var(--ink2);margin:10px 0 12px}
.xp-foot button{font:inherit;padding:4px 10px;border:1px solid var(--border);border-radius:6px;background:var(--surface);color:var(--ink);cursor:pointer}
.xp-foot button:disabled{opacity:.4;cursor:default}
.scroll{overflow-x:auto}.scroll td{white-space:nowrap}.card ul{margin:6px 0 12px;padding-left:20px;font-size:14px}
</style></head><body><main>
<h1>Expiry-day short strangle lab</h1>
<p class="sub">__SUB__</p>
__BODY__
<h2>Assumptions</h2>
<ul class="assume">__ASSUME__</ul>
<p class="note">Educational analysis on a community dataset (verify against exchange data). Not investment advice; the author is not SEBI-registered.</p>
</main>
<script>
const FIGS = __FIGS__;
const base = {paper_bgcolor:"#fcfcfb",plot_bgcolor:"#fcfcfb",
  font:{family:'system-ui,-apple-system,"Segoe UI",sans-serif',color:"#52514e",size:12},
  margin:{l:64,r:20,t:10,b:56},hovermode:"closest",
  legend:{orientation:"h",y:-0.22},
  xaxis:{gridcolor:"#e1e0d9",linecolor:"#c3c2b7",zeroline:false},
  yaxis:{gridcolor:"#e1e0d9",linecolor:"#c3c2b7",zeroline:false}};
for (const f of FIGS){
  const lay = JSON.parse(JSON.stringify(base));
  for (const k in f.layout){
    const v = f.layout[k];
    lay[k] = (v && typeof v === "object" && !Array.isArray(v)) ? Object.assign({}, lay[k]||{}, v) : v;
  }
  Plotly.newPlot(f.id, f.data, lay, {displaylogo:false,responsive:true});
}
const TABLES = __TABLES__;
const PAGE = 50;
function cell(v, kind){
  if (v === null || v === undefined || v === "") return "–";
  if (kind === "rs") return (v < 0 ? "−₹" : "₹") + Math.abs(v).toLocaleString("en-IN", {maximumFractionDigits: 0});
  if (kind === "num") return v.toLocaleString("en-IN", {maximumFractionDigits: 2});
  return String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;");
}
for (const t of TABLES){
  const root = document.getElementById(t.id);
  if (!root) continue;
  const st = {sort: null, dir: 1, page: 0, f: Object.assign({}, t.defaults), q: ""};
  const bar = document.createElement("div"); bar.className = "xp-bar";
  for (const ci of t.filters){
    const vals = [...new Set(t.rows.map(r => String(r[ci])))];
    const numeric = vals.every(v => v !== "" && !isNaN(parseFloat(v)));
    vals.sort((a, b) => numeric ? parseFloat(a) - parseFloat(b) : a.localeCompare(b));
    const lab = document.createElement("label"); lab.textContent = t.cols[ci].label;
    const sel = document.createElement("select");
    sel.innerHTML = '<option value="">All</option>' + vals.map(v => `<option>${v}</option>`).join("");
    if (st.f[ci] !== undefined && vals.includes(st.f[ci])) sel.value = st.f[ci]; else delete st.f[ci];
    sel.onchange = () => { if (sel.value) st.f[ci] = sel.value; else delete st.f[ci]; st.page = 0; draw(); };
    lab.appendChild(sel); bar.appendChild(lab);
  }
  const sl = document.createElement("label"); sl.textContent = "Search (date, strike…)";
  const inp = document.createElement("input"); inp.type = "search"; inp.placeholder = "e.g. 2025-04";
  inp.oninput = () => { st.q = inp.value.trim().toLowerCase(); st.page = 0; draw(); };
  sl.appendChild(inp); bar.appendChild(sl);
  const wrap = document.createElement("div"); wrap.className = "xp-wrap";
  const foot = document.createElement("div"); foot.className = "xp-foot";
  root.append(bar, wrap, foot);
  function draw(){
    let rows = t.rows.filter(r => {
      for (const k in st.f) if (String(r[k]) !== st.f[k]) return false;
      return !st.q || r.some(v => v !== null && String(v).toLowerCase().includes(st.q));
    });
    if (st.sort !== null){
      const i = st.sort, num = t.cols[i].kind !== "text";
      rows = rows.slice().sort((a, b) => {
        const x = a[i], y = b[i];
        if (x === null || x === "") return 1; if (y === null || y === "") return -1;
        return st.dir * (num ? x - y : String(x).localeCompare(String(y)));
      });
    }
    const pages = Math.max(1, Math.ceil(rows.length / PAGE));
    st.page = Math.min(st.page, pages - 1);
    const view = rows.slice(st.page * PAGE, (st.page + 1) * PAGE);
    const head = t.cols.map((c, i) => `<th data-i="${i}">${c.label}${st.sort === i ? (st.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("");
    const body = view.map(r => "<tr>" + r.map((v, i) => {
      const k = t.cols[i].kind;
      return `<td class="${k !== "text" ? "n" : ""}${k === "rs" && v < 0 ? " neg" : ""}">${cell(v, k)}</td>`;
    }).join("") + "</tr>").join("");
    wrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    wrap.querySelectorAll("th").forEach(th => th.onclick = () => {
      const i = +th.dataset.i; st.dir = st.sort === i ? -st.dir : 1; st.sort = i; draw();
    });
    let sum = "";
    if (t.sum !== null && rows.length){
      const tot = rows.reduce((a, r) => a + (r[t.sum] || 0), 0);
      sum = ` · total ${cell(Math.round(tot), "rs")} · average ${cell(Math.round(tot / rows.length), "rs")} per row`;
    }
    foot.innerHTML = `<span>${rows.length.toLocaleString("en-IN")} rows${sum}</span><span><button class="pv">Previous</button> page ${st.page + 1} of ${pages} <button class="nx">Next</button></span>`;
    const pv = foot.querySelector(".pv"), nx = foot.querySelector(".nx");
    pv.disabled = st.page === 0; nx.disabled = st.page >= pages - 1;
    pv.onclick = () => { st.page--; draw(); }; nx.onclick = () => { st.page++; draw(); };
  }
  draw();
}
</script></body></html>"""


def render(sections, cfg, out):
    body, all_figs, all_tables = [], [], []
    for s in sections:
        body.append(f"<h2>{s['sym']}</h2>")
        if s["tiles"]:
            body.append('<div class="tiles">' + "".join(
                f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>'
                for k, v in s["tiles"]) + "</div>")
        body.append("<h3>What the data says</h3><ul class='ins'>" +
                    "".join(f"<li>{i}</li>" for i in s["insights"]) + "</ul>")
        def add_figs(figs, prefix):
            for j, (title, note, data, layout) in enumerate(figs):
                fid = f"{prefix}_{s['sym']}_{j}"
                layout = dict(layout)
                h = layout.pop("_height", 380)
                body.append(f'<div class="card"><h3>{title}</h3><p class="note">{note}</p>'
                            f'<div id="{fid}" style="height:{h}px"></div></div>')
                all_figs.append({"id": fid, "data": data, "layout": layout})

        add_figs(s["figs"], "f")
        for title, note, html in s["cards"]:
            body.append(f'<div class="card"><h3>{title}</h3><p class="note">{note}</p>{html}</div>')
        body.append(f'<div class="card"><h3>Your 10 worst days</h3>'
                    f'<div class="scroll">{s["table"]}</div></div>')
        b = s.get("be")
        if b:
            body.append(f"<h2>{s['sym']}: short strangles by break-even, entered at 09:15</h2>"
                        "<p class='sub'>On each expiry day the CE and PE are chosen so that strike ± "
                        "total premium sits the stated percentage from the index's 09:15 open. Each "
                        "leg is sold at the midpoint of its first 1-minute candle's high and low.</p>")
            body.append("<ul class='ins'>" + "".join(f"<li>{i}</li>" for i in b["insights"]) + "</ul>")
            title, note, html = b["cards"][0]
            body.append(f'<div class="card"><h3>{title}</h3><p class="note">{note}</p>{html}</div>')
            add_figs(b["figs"], "b")
            for title, note, html in b["cards"][1:]:
                body.append(f'<div class="card"><h3>{title}</h3><p class="note">{note}</p>{html}</div>')
        t = s.get("tag")
        if t:
            lo, hi = t["band"]
            body.append(f"<h2>{s['sym']}: strikes {lo:g}–{hi:g} points from ATM</h2>"
                        "<p class='sub'>ATM is the index's 09:15 open rounded to the strike interval. "
                        "Each traded strike is tagged by how many points it sits from ATM "
                        "(CE above, PE below).</p>")
            body.append("<ul class='ins'>" + "".join(f"<li>{i}</li>" for i in t["insights"]) + "</ul>")
            add_figs(t["figs"], "t")
            body.append(f'<div class="card"><h3>Every tag in one table</h3>'
                        f'<p class="note">Multiples compare the highest price after the first candle '
                        f'with the first candle\'s close.</p>{t["table"]}</div>')
            for title, note, html in t.get("extra_cards", []):
                body.append(f'<div class="card"><h3>{title}</h3><p class="note">{note}</p>{html}</div>')
        if s.get("explore"):
            body.append(f"<h2>{s['sym']}: browse every row</h2>"
                        "<p class='sub'>The full detail behind the charts. Filter, search and sort "
                        "here instead of opening a spreadsheet.</p>")
            for x in s["explore"]:
                body.append(f'<div class="card"><h3>{x["title"]}</h3><p class="note">{x["note"]}</p>'
                            f'<div id="{x["id"]}"></div></div>')
                all_tables.append({k: x[k] for k in ("id", "cols", "rows", "filters", "defaults", "sum")})
    assume = [
        "<b>Read from the data:</b> entry prices (the last trade at or before the entry minute, never a "
        "later one), every stop-loss trigger (a later 1-minute high reaching the level), the other "
        "leg's buyback price that minute, expiry dates, strike intervals and ATM.",
        "<b>Settlement:</b> held legs settle against the average of the index's 1-minute closes in the "
        "last 30 minutes, the closest the data gets to the exchange's final settlement price.",
        f"<b>Modelling choice:</b> a stop-loss is assumed to fill at its trigger price + "
        f"{cfg.slippage:g} pts. The fill table shows the result at prices that actually traded.",
        "<b>Read from the data:</b> the 09:15 candle. Bars before 09:15 (the pre-open session) are "
        "dropped, so the 'open' is the first bar of the trading session, not of the file.",
        "<b>Modelling choice:</b> in minutes with no trade an option keeps its last traded price. "
        "If both legs reach their stop-loss in the same minute, the CE is treated as first.",
        f"<b>Your inputs, not from the data:</b> quantity {cfg.qty}, charges ₹{cfg.charges:g} per trade "
        f"(another ₹{cfg.charges:g} when stopped out), minimum premium {cfg.min_prem:g} pts, entry "
        f"quotes at most {cfg.max_stale} minutes old.",
        "<b>Not in the data:</b> bid and ask prices, so real fills on thinly traded far strikes can be "
        "worse than the last traded price, and margin, which is not modelled.",
    ]
    sub = (f"Expiry days only. Three parts per index: your original rule (CE ~{cfg.ce_dist:g}% / "
           f"PE ~{cfg.pe_dist:g}% away at {cfg.entry}, stop-loss +{cfg.sl_points:g} pts), strangles by "
           f"break-even entered at 09:15, and every far strike tagged by its distance from ATM. "
           f"Qty {cfg.qty}.")
    html = (HTML.replace("__SUB__", sub).replace("__BODY__", "\n".join(body))
            .replace("__ASSUME__", "".join(f"<li>{a}</li>" for a in assume))
            .replace("__FIGS__", json.dumps(all_figs, default=float))
            .replace("__TABLES__", json.dumps(all_tables, ensure_ascii=False,
                                              separators=(",", ":")).replace("</", "<\\/")))
    Path(out).write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="SENSEX,NIFTY")
    ap.add_argument("--since", default=None, help="only expiries on/after YYYY-MM-DD")
    ap.add_argument("--entry", default="09:20")
    ap.add_argument("--ce-dist", type=float, default=3.8, help="your CE distance %% OTM")
    ap.add_argument("--pe-dist", type=float, default=4.1, help="your PE distance %% OTM")
    ap.add_argument("--sl-points", type=float, default=7.14, help="your SL: premium + N pts")
    ap.add_argument("--dists", default="1.5,2,2.5,3,3.5,4,4.5", help="grid of distances %%")
    ap.add_argument("--sl-mults", default="2,3,4,5", help="grid of SL multiples")
    ap.add_argument("--qty", type=int, default=420)
    ap.add_argument("--charges", type=float, default=80, help="₹ per trade (entry)")
    ap.add_argument("--slippage", type=float, default=0.0, help="pts added to each buyback")
    ap.add_argument("--min-prem", type=float, default=0.3)
    ap.add_argument("--max-stale", type=int, default=10,
                    help="oldest trade (minutes before entry) accepted as an entry price")
    ap.add_argument("--bands", default="SENSEX:1500-3000,NIFTY:800-2000",
                    help="points-from-ATM band per index for the strike-tag section")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads")
    ap.add_argument("--local", default=None, help="local copy of the dataset repo")
    ap.add_argument("--out", default="options_lab_report.html")
    ap.add_argument("--csv", action="store_true",
                    help="also save the row-level data as CSV files (it is already in the report)")
    ap.add_argument("--breakevens", default="2,2.5,3",
                    help="break-even distances (%% from the 09:15 index open) to backtest")
    ap.add_argument("--be-sl", type=float, default=7, help="stop-loss in points per option")
    ap.add_argument("--be-sl-grid", default="3,5,7,10,15,20,30",
                    help="other stop-losses (points) to compare")
    ap.add_argument("--sl-mode", choices=["both", "independent"], default="both",
                    help="when one leg's stop-loss hits: exit both legs, or leave the other running")
    ap.add_argument("--be-tol", type=float, default=0.25,
                    help="skip a day if no strike pair gets within this many %% points of the break-even")
    ap.add_argument("--entry-limits", default="10,20",
                    help="sell limits to test, as %% above the first candle's open")
    ap.add_argument("--limit-fill-target", type=float, default=80,
                    help="fill rate (%% of days) used for the 'realistic limit' line")
    cfg = ap.parse_args()
    cfg.breakevens = [float(x) for x in cfg.breakevens.split(",")]
    cfg.be_sl_grid = sorted({float(x) for x in cfg.be_sl_grid.split(",")} | {cfg.be_sl})
    cfg.entry_limits = [float(x) for x in cfg.entry_limits.split(",") if x.strip()]
    cfg.bands_raw = cfg.bands
    h, m = cfg.entry.split(":")
    cfg.entry_tod = int(h) * 60 + int(m)
    cfg.dists = [float(x) for x in cfg.dists.split(",")]
    cfg.sl_mults = [float(x) for x in cfg.sl_mults.split(",")]
    cfg.bands = {}
    for part in cfg.bands_raw.split(","):
        if ":" in part:
            name, rng = part.split(":")
            lo, hi = rng.split("-")
            cfg.bands[name.strip().upper()] = (float(lo), float(hi))

    files = list_files(cfg.local)
    sections, all_trades, all_tags, all_be = [], [], [], []
    for sym in [s.strip().upper() for s in cfg.symbols.split(",")]:
        spot = load_index(sym, files, cfg.local)
        if spot is None:
            print(f"[{sym}] no index file, skipped")
            continue
        exp_files = {}
        for f in files:
            if f.startswith(f"options/{sym}/"):
                try:
                    exp_files[pd.Timestamp(Path(f).stem)] = f
                except Exception:
                    pass
        exps = sorted(e for e in exp_files if not cfg.since or e >= pd.Timestamp(cfg.since))
        print(f"[{sym}] {len(exps)} expiries to replay (downloads are cached after the first run)")
        days, trades, decay, prem, tags, skips = [], [], [], [], [], {}
        be_rows, be_skips, n_sess = [], {}, 0
        done = 0
        with ThreadPoolExecutor(cfg.workers) as ex:
            futs = {ex.submit(fetch, exp_files[e], cfg.local): e for e in exps}
            for fu in as_completed(futs):
                e = futs[fu]
                done += 1
                try:
                    opt = norm_opt(pd.read_parquet(fu.result()))
                    r = process_expiry(sym, e, opt, spot, cfg)
                except Exception as err:
                    print(f"  ! {e.date()}: {err}")
                    continue
                tags += r.get("tags", [])
                be_rows += r.get("be", [])
                n_sess += "session" in r
                for x, why in r.get("be_skip", {}).items():
                    be_skips[(x, why)] = be_skips.get((x, why), 0) + 1
                if "skip" in r:
                    skips[r["skip"]] = skips.get(r["skip"], 0) + 1
                else:
                    days.append(r["day"]); trades += r["trades"]
                    decay += r["decay"]; prem += r["prem"]
                if done % 20 == 0 or done == len(exps):
                    print(f"  {done}/{len(exps)} expiries processed")
        if not days:
            print(f"[{sym}] nothing usable")
            continue
        D, T = pd.DataFrame(days).sort_values("date"), pd.DataFrame(trades)
        g0 = spot.groupby("day").tod
        pre = float((g0.min() < FIRST_TOD).mean() * 100)
        has915 = float(spot[spot.tod == FIRST_TOD].day.nunique() / spot.day.nunique() * 100)
        meta = {"n_files": len(exps), "skips": skips,
                "first_bar": (f"{has915:.0f}% of sessions have a 09:15 bar; {pre:.0f}% also carry "
                              f"pre-open bars before 09:15, which are ignored")}
        sec = build_symbol_section(sym, D, T, pd.DataFrame(decay), pd.DataFrame(prem), cfg, meta)
        TG = BEF = None
        if tags:
            TG = pd.DataFrame(tags)
            sec["tag"] = build_tag_section(sym, TG, n_sess, cfg.bands[sym], cfg)
            all_tags.append(TG)
        elif sym in cfg.bands:
            print(f"  [{sym}] no traded strikes inside the {cfg.bands[sym]} band")
        if be_rows:
            BEF = pd.DataFrame(be_rows)
            sec["be"] = build_be_section(sym, BEF, n_sess, be_skips, cfg)
            all_be.append(BEF)
        else:
            print(f"  [{sym}] no break-even strangle could be built on any session")
        sec["explore"] = build_explorers(sym, T, BEF, TG, cfg)
        sections.append(sec)
        all_trades.append(T)

    if not sections:
        print("No results.")
        return
    render(sections, cfg, cfg.out)
    print(f"\nReport: {Path(cfg.out).resolve()}  (every row is browsable inside it)")
    if cfg.csv:
        for name, frames in (("trades", all_trades), ("breakeven_trades", all_be),
                             ("strike_tags", all_tags)):
            if frames:
                pd.concat(frames).to_csv(f"options_lab_{name}.csv", index=False)
                print(f"CSV: {Path(f'options_lab_{name}.csv').resolve()}")


if __name__ == "__main__":
    main()
