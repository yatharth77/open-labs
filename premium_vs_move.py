#!/usr/bin/env python3
"""
How much did option premiums move on days the index moved 2.5-3%?

Data: huggingface.co/datasets/thetrademarkk/india-index-options-1m
  index/{SYMBOL}.parquet            1-min spot
  options/{SYMBOL}/{EXPIRY}.parquet 1-min option bars (strike, option_type, expiry)

Setup:  pip install huggingface_hub pandas pyarrow openpyxl
Run:    python premium_vs_move.py                       # NIFTY, BANKNIFTY, SENSEX, 2.5-3%
        python premium_vs_move.py --symbols NIFTY --lo 2 --hi 3.5
        python premium_vs_move.py --local /path/to/clone  # if you already downloaded the repo

What it does, per event day (|close-to-close| move within [lo, hi] %):
  - takes the nearest weekly expiry (and the next one) on/after the event day
  - picks CE/PE at ATM and 1/2/3% OTM, chosen off the PREVIOUS day's close
  - compares prev-day close premium vs event-day close premium (and intraday peak)
  - records ATM straddle (% of spot) and ATM IV before/after = what was "priced in"
Outputs: events.csv, detail.csv, summary.csv (+ premium_vs_move.xlsx if openpyxl present)
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "thetrademarkk/india-index-options-1m"
RATE = 0.065  # risk-free for IV calc
MONEYNESS = [0, 1, 2, 3]  # % OTM
GAP = {"up": 0.5, "down": None}  # set from --max-gap-up / --max-gap-down


# ---------------- data access ----------------
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
        # naive: dataset preview shows 03:45 for the 09:15 open => UTC
        ts = ts.dt.tz_localize("UTC") if ts.dt.hour.min() < 5 else ts.dt.tz_localize("Asia/Kolkata")
    return ts.dt.tz_convert("Asia/Kolkata")


def prep(df):
    df = df.copy()
    df["ts"] = to_ist(df["timestamp"])
    df["day"] = df["ts"].dt.tz_localize(None).dt.normalize()
    return df.sort_values("ts")


# ---------------- maths ----------------
def bs_price(S, K, T, r, sig, cp):
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if cp == "CE" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if cp == "CE":
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def implied_vol(px, S, K, T, cp):
    if not (px and px > 0 and T > 0):
        return np.nan
    lo, hi = 0.01, 3.0
    if px <= bs_price(S, K, T, RATE, lo, cp) or px >= bs_price(S, K, T, RATE, hi, cp):
        return np.nan
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, RATE, mid, cp) > px:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def years_to_expiry(day, expiry):
    # from day's 15:30 close to expiry's 15:30
    return max((expiry - day).days, 0) / 365.0 + 1e-6


# ---------------- core ----------------
def daily_bars(spot):
    g = spot.groupby("day")
    d = pd.DataFrame({"open": g.open.first(), "high": g.high.max(),
                      "low": g.low.min(), "close": g.close.last()}).sort_index()
    d["prev_close"] = d.close.shift(1)
    d["prev_day"] = pd.Series(d.index, index=d.index).shift(1)
    d["ret_pct"] = (d.close / d.prev_close - 1) * 100
    d["gap_pct"] = (d.open / d.prev_close - 1) * 100
    d["max_up_pct"] = (d.high / d.prev_close - 1) * 100
    d["max_dn_pct"] = (d.low / d.prev_close - 1) * 100
    return d.dropna(subset=["prev_close"])


def day_snapshot(opt, day):
    x = opt[opt.day == day]
    if x.empty:
        return None
    g = x.groupby(["strike", "option_type"])
    return pd.DataFrame({"close": g.close.last(), "high": g.high.max(),
                         "low": g.low.min(), "volume": g.volume.sum()})


def nearest_strike(snap, cp, target):
    if snap is None or cp not in snap.index.get_level_values("option_type"):
        return None
    strikes = snap.xs(cp, level="option_type").index.values.astype(float)
    if len(strikes) == 0:
        return None
    return float(strikes[np.argmin(np.abs(strikes - target))])


_shown = set()


def norm_opt(df, label=""):
    df = df.rename(columns=str.lower)
    sym = df["symbol"].astype(str).str.upper().str.strip() if "symbol" in df else None
    if "option_type" in df:
        ot = df["option_type"].astype(str).str.upper().str.strip()
    else:
        ot = pd.Series("", index=df.index)
    ot = ot.replace({"CALL": "CE", "C": "CE", "CALLS": "CE", "PUT": "PE", "P": "PE", "PUTS": "PE"})
    # fall back to trading-symbol suffix, e.g. NIFTY24JAN22000CE
    if sym is not None:
        bad = ~ot.isin(["CE", "PE"])
        ot[bad] = sym[bad].str.extract(r"(CE|PE)$")[0]
    df["option_type"] = ot
    if "strike" in df:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    else:
        df["strike"] = np.nan
    if sym is not None and df["strike"].isna().any():
        m = df["strike"].isna()
        df.loc[m, "strike"] = pd.to_numeric(sym[m].str.extract(r"(\d+(?:\.\d+)?)(?:CE|PE)$")[0],
                                            errors="coerce")
    if label and label not in _shown:
        _shown.add(label)
        print(f"  [{label}] columns={list(df.columns)} option_type values="
              f"{df['option_type'].dropna().unique()[:6].tolist()}")
    df = df.dropna(subset=["option_type", "strike"])
    df = df[df.option_type.isin(["CE", "PE"])]
    if "volume" not in df:
        df["volume"] = 0
    return prep(df)


def analyse_symbol(sym, files, local, lo, hi, n_exp):
    idx_path = next((f for f in files if f.lower() == f"index/{sym.lower()}.parquet"), None)
    if not idx_path:
        print(f"[{sym}] no index file, skipping")
        return [], []
    spot = prep(pd.read_parquet(fetch(idx_path, local)).rename(columns=str.lower))
    daily = daily_bars(spot)
    ev = daily[(daily.ret_pct.abs() >= lo) & (daily.ret_pct.abs() <= hi)].copy()
    n_all = len(ev)
    ev = ev[ev.gap_pct <= GAP["up"]]
    if GAP["down"] is not None:
        ev = ev[ev.gap_pct >= -GAP["down"]]
    ev["symbol"] = sym
    print(f"[{sym}] {len(daily)} days, {n_all} event days in [{lo},{hi}]%, "
          f"{n_all - len(ev)} gap days removed, {len(ev)} kept")

    exp_files = {}
    for f in files:
        if f.startswith(f"options/{sym}/"):
            try:
                exp_files[pd.Timestamp(Path(f).stem)] = f
            except Exception:
                pass
    expiries = sorted(exp_files)

    rows, events = [], []
    cache = {}
    for day, e in ev.iterrows():
        prev_day, S0, S1 = e.prev_day, e.prev_close, e.close
        direction = "UP" if e.ret_pct > 0 else "DOWN"
        ev_row = {"symbol": sym, "date": day.date(), "ret_pct": round(e.ret_pct, 2),
                  "gap_pct": round(e.gap_pct, 2), "max_up_pct": round(e.max_up_pct, 2),
                  "max_dn_pct": round(e.max_dn_pct, 2), "direction": direction}
        upcoming = [x for x in expiries if x >= day][:n_exp]
        for rank, exp in enumerate(upcoming):
            if exp not in cache:
                try:
                    cache[exp] = norm_opt(pd.read_parquet(fetch(exp_files[exp], local)),
                                          label=sym)
                except Exception as ex:
                    print(f"  ! {exp_files[exp]}: {ex}")
                    cache[exp] = None
            opt = cache[exp]
            if opt is None:
                continue
            p0, p1 = day_snapshot(opt, prev_day), day_snapshot(opt, day)
            if p0 is None or p1 is None:
                continue
            dte = (exp - day).days
            tag = "near" if rank == 0 else "next"
            # ATM straddle / IV (priced-in move)
            k_atm = nearest_strike(p0, "CE", S0)
            try:
                ce0, pe0 = p0.loc[(k_atm, "CE"), "close"], p0.loc[(k_atm, "PE"), "close"]
                ev_row[f"{tag}_expiry"] = exp.date()
                ev_row[f"{tag}_dte"] = dte
                ev_row[f"{tag}_straddle_pct_prev"] = round((ce0 + pe0) / S0 * 100, 2)
                T0 = years_to_expiry(prev_day, exp)
                ev_row[f"{tag}_atm_iv_prev"] = round(implied_vol(ce0, S0, k_atm, T0, "CE") * 100, 1)
                k1 = nearest_strike(p1, "CE", S1)
                if (k1, "CE") in p1.index:
                    T1 = years_to_expiry(day, exp)
                    ev_row[f"{tag}_atm_iv_after"] = round(
                        implied_vol(p1.loc[(k1, "CE"), "close"], S1, k1, T1, "CE") * 100, 1)
            except KeyError:
                pass

            for cp in ("CE", "PE"):
                side = "with_move" if (cp == "CE") == (direction == "UP") else "against_move"
                for m in MONEYNESS:
                    target = S0 * (1 + m / 100) if cp == "CE" else S0 * (1 - m / 100)
                    k = nearest_strike(p0, cp, target)
                    if k is None or (k, cp) not in p0.index or (k, cp) not in p1.index:
                        continue
                    a, b = p0.loc[(k, cp)], p1.loc[(k, cp)]
                    if a.close <= 0:
                        continue
                    peak = b.high if side == "with_move" else b.low
                    rows.append({
                        "symbol": sym, "date": day.date(), "direction": direction,
                        "ret_pct": round(e.ret_pct, 2), "expiry": exp.date(), "expiry_rank": tag,
                        "dte": dte, "type": cp, "side": side, "otm_pct": m, "strike": k,
                        "spot_prev": S0, "spot_close": S1,
                        "prem_prev": a.close, "prem_close": b.close,
                        "chg_pts": round(b.close - a.close, 2),
                        "chg_pct": round((b.close / a.close - 1) * 100, 1),
                        "chg_x": round(b.close / a.close, 2),
                        "intraday_extreme": peak,
                        "extreme_pct": round((peak / a.close - 1) * 100, 1),
                        "prem_prev_pct_spot": round(a.close / S0 * 100, 3),
                        "vol_event_day": int(b.volume),
                    })
        events.append(ev_row)
    return events, rows


def dte_bucket(d):
    return "0 (expiry day)" if d == 0 else "1" if d == 1 else "2-3" if d <= 3 else "4-7" if d <= 7 else "8+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="NIFTY,BANKNIFTY,SENSEX")
    ap.add_argument("--lo", type=float, default=2.5)
    ap.add_argument("--hi", type=float, default=3.0)
    ap.add_argument("--expiries", type=int, default=2, help="near + next")
    ap.add_argument("--local", default=None, help="local clone of the dataset repo")
    ap.add_argument("--out", default=".")
    ap.add_argument("--max-gap-up", type=float, default=0.5,
                    help="drop event days that opened more than this %% above prev close")
    ap.add_argument("--max-gap-down", type=float, default=None,
                    help="also drop days that opened more than this %% below prev close")
    a = ap.parse_args()
    GAP["up"], GAP["down"] = a.max_gap_up, a.max_gap_down

    files = list_files(a.local)
    all_ev, all_rows = [], []
    for s in a.symbols.split(","):
        e, r = analyse_symbol(s.strip().upper(), files, a.local, a.lo, a.hi, a.expiries)
        all_ev += e
        all_rows += r

    out = Path(a.out)
    events, detail = pd.DataFrame(all_ev), pd.DataFrame(all_rows)
    if detail.empty:
        print("No option rows matched — check that options/ files exist for event dates.")
        events.to_csv(out / "events.csv", index=False)
        return
    detail["dte_bucket"] = detail.dte.map(dte_bucket)
    summary = (detail.groupby(["symbol", "expiry_rank", "dte_bucket", "side", "type", "otm_pct"])
               .agg(n=("chg_pct", "size"),
                    median_chg_pct=("chg_pct", "median"),
                    p25_chg_pct=("chg_pct", lambda x: x.quantile(.25)),
                    p75_chg_pct=("chg_pct", lambda x: x.quantile(.75)),
                    median_multiple=("chg_x", "median"),
                    median_chg_pts=("chg_pts", "median"),
                    median_intraday_extreme_pct=("extreme_pct", "median"),
                    median_prem_prev_pct_spot=("prem_prev_pct_spot", "median"))
               .round(2).reset_index())

    events.to_csv(out / "events.csv", index=False)
    detail.to_csv(out / "detail.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    try:
        with pd.ExcelWriter(out / "premium_vs_move.xlsx") as xw:
            summary.to_excel(xw, "Summary", index=False)
            events.to_excel(xw, "Events", index=False)
            detail.to_excel(xw, "Detail", index=False)
    except Exception as ex:
        print("xlsx skipped:", ex)

    pd.set_option("display.width", 200, "display.max_rows", 200)
    near = summary[summary.expiry_rank == "near"]
    print("\n=== Near-expiry: median premium change, prev close -> event-day close ===")
    print(near.pivot_table(index=["symbol", "dte_bucket", "side"], columns="otm_pct",
                           values="median_chg_pct").round(0))
    ev_cols = [c for c in events.columns if "straddle" in c or "iv" in c]
    if ev_cols:
        print("\n=== Priced-in move (median across events) ===")
        print(events.groupby("symbol")[ev_cols].median().round(2))
    print(f"\nWrote events.csv, detail.csv, summary.csv to {out.resolve()}")


if __name__ == "__main__":
    main()
