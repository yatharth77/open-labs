#!/usr/bin/env python3
"""
open_minute_lab.py - how out-of-the-money index options behave in the first
minute of trading (the 09:15 candle), with an HTML report.

For NIFTY, SENSEX and BANKNIFTY, on every trading day in
huggingface.co/datasets/thetrademarkk/india-index-options-1m, it takes every
out-of-the-money option of the nearest expiries and records:
  * the 09:15 candle: open, high, low, close, volume
  * previous day's close -> today's open (%), and open -> close of the candle (%)
and groups them by
  * days to expiry
  * distance of the strike from ATM (ATM = index 09:15 open rounded to the strike step)
  * the index's own gap (previous close -> 09:15 open)
It also measures India VIX over the same minute and how it relates to the options.

Setup:  pip install --index-url https://pypi.org/simple huggingface_hub pandas pyarrow
Run:    python open_minute_lab.py
        python open_minute_lab.py --symbols NIFTY --since 2024-01-01
        python open_minute_lab.py --vix-file ~/Downloads/india_vix_1min.csv
Output: open_minute_report.html (open in a browser). Add --csv for row-level files.

India VIX: the script looks for a file with "vix" in its name inside the
dataset's index/ folder. If there is none, pass --vix-file with a 1-minute
CSV or parquet (columns: timestamp/date, open, high, low, close).
"""
import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "thetrademarkk/india-index-options-1m"
FIRST_TOD = 9 * 60 + 15
CLOSE_TOD = 15 * 60 + 30

DTE_EDGES = [(0, 0, "0 (expiry day)"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"),
             (4, 7, "4–7"), (8, 14, "8–14"), (15, 10 ** 6, "15+")]
DTE_ORDER = [x[2] for x in DTE_EDGES]
DIST_EDGES = [0, 0.5, 1, 1.5, 2, 3, 4, 100]
DIST_ORDER = ["0–0.5%", "0.5–1%", "1–1.5%", "1.5–2%", "2–3%", "3–4%", "4%+"]
GAP_EDGES = [-100, -1, -0.5, -0.2, 0.2, 0.5, 1, 100]
GAP_ORDER = ["down 1%+", "down 0.5–1%", "down 0.2–0.5%", "flat (±0.2%)",
             "up 0.2–0.5%", "up 0.5–1%", "up 1%+"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SIDE = {"CE": SERIES[0], "PE": SERIES[1]}
NICE = {"CE": "CE (calls)", "PE": "PE (puts)"}


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
    tcol = next((c for c in ("timestamp", "datetime", "date", "time") if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"no timestamp column in {list(df.columns)}")
    df["ts"] = to_ist(df[tcol])
    local = df["ts"].dt.tz_localize(None)
    df["day"] = local.dt.normalize()
    df["tod"] = local.dt.hour * 60 + local.dt.minute
    return df


def load_bars(path):
    """1-minute bars for an index (or VIX), trading session only."""
    p = str(path)
    df = pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_parquet(p)
    df = add_time_cols(df.rename(columns=str.lower))
    df = df[(df.tod >= FIRST_TOD) & (df.tod <= CLOSE_TOD)].sort_values("ts")
    return df[["day", "tod", "open", "high", "low", "close"]].reset_index(drop=True)


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
    if "volume" not in df:
        df["volume"] = np.nan
    return df[["day", "tod", "strike", "option_type", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------- daily tables
def daily_table(bars):
    """One row per session: the 09:15 candle, the previous session's last close and the gap."""
    days = sorted(bars.day.unique())
    first = bars[bars.tod == FIRST_TOD].drop_duplicates("day").set_index("day")
    d = pd.DataFrame({"open": first.open, "high": first.high, "low": first.low,
                      "close1": first.close}).reindex(days)
    d["last"] = bars.groupby("day").close.last()
    d["prev_close"] = d["last"].shift(1)
    d["gap_pct"] = (d.open / d.prev_close - 1) * 100
    d["oc_pct"] = (d.close1 / d.open - 1) * 100
    return d


def bucket_dte(n):
    for lo, hi, lab in DTE_EDGES:
        if lo <= n <= hi:
            return lab
    return None


def cut(series, edges, labels):
    return pd.cut(series, edges, labels=labels, right=False).astype(object)


def process_file(path, exp, daily, next_day, cfg):
    """09:15 candles of every OTM strike in one expiry file, one row per strike per day."""
    df = norm_opt(pd.read_parquet(path))
    df = df[(df.tod >= FIRST_TOD) & (df.tod <= CLOSE_TOD) & (df.day <= exp)
            & (df.day >= exp - pd.Timedelta(days=cfg.max_dte))]
    if df.empty:
        return None
    df = df.sort_values("tod", kind="stable")
    keys = ["day", "strike", "option_type"]
    strikes = np.sort(df.strike.unique())
    if len(strikes) < 3:
        return None
    step = float(pd.Series(np.diff(strikes)).round(2).mode().min())
    first = (df[df.tod == FIRST_TOD].groupby(keys, as_index=False)
             .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                  close=("close", "last"), volume=("volume", lambda v: v.sum(min_count=1))))
    if first.empty:
        return None
    last = df.groupby(keys, as_index=False).agg(prev_close=("close", "last"))
    last["day"] = last.day.map(next_day)           # yesterday's close belongs to today's row
    m = first.merge(last.dropna(subset=["day"]), on=keys, how="left")
    m = m.merge(daily[["open", "gap_pct", "oc_pct"]].rename(
        columns={"open": "idx_open", "gap_pct": "idx_gap_pct", "oc_pct": "idx_oc_pct"}),
        left_on="day", right_index=True, how="inner")
    m = m.dropna(subset=["idx_open"])
    m["atm"] = (m.idx_open / step).round() * step
    m["dist_pts"] = np.where(m.option_type == "CE", m.strike - m.atm, m.atm - m.strike)
    m["dist_pct"] = m.dist_pts / m.idx_open * 100
    m = m[(m.dist_pts > 0) & (m.dist_pct <= cfg.max_otm)]
    if m.empty:
        return None
    m["expiry"] = exp
    m["dte"] = (exp - m.day).dt.days
    m["strike_step"] = step
    return m


# ---------------------------------------------------------------- statistics
def cell_stats(df, keys):
    g = df.groupby(keys, observed=True)
    out = g.agg(n=("oc_pct", "size"), days=("date", "nunique"),
                gap=("opt_gap_pct", "median"), oc=("oc_pct", "median"),
                fell=("fell", "mean"), rose=("rose", "mean"),
                hi=("hi_pct", "median"), lo=("lo_pct", "median"),
                prev=("prev_close", "median"), open=("open", "median"),
                high=("high", "median"), low=("low", "median"), close=("close", "median"),
                dist_pts=("dist_pts", "median")).reset_index()
    out["fell"] *= 100
    out["rose"] *= 100
    return out


def spearman(a, b):
    ok = a.notna() & b.notna()
    if ok.sum() < 30:
        return np.nan, int(ok.sum())
    return float(a[ok].rank().corr(b[ok].rank())), int(ok.sum())


def strength(r):
    if pd.isna(r):
        return "not enough days to say"
    a = abs(r)
    word = ("no visible relation" if a < 0.1 else "a weak relation" if a < 0.3
            else "a moderate relation" if a < 0.5 else "a strong relation")
    return f"{word} ({r:+.2f})"


def pct(v, d=1):
    return "-" if v is None or pd.isna(v) else f"{v:+.{d}f}%"


def num(v, d=2):
    return "-" if v is None or pd.isna(v) else f"{v:,.{d}f}"


# ---------------------------------------------------------------- charts
METRICS = [("gap", "Previous close → open (%)", "div"),
           ("oc", "First minute: open → close (%)", "div"),
           ("fell", "Share that fell in the first minute (%)", "share"),
           ("hi", "First-minute high vs open (%)", "up"),
           ("lo", "First-minute low vs open (%)", "down"),
           ("open", "Median open price (points)", "seq"),
           ("n", "Number of option-days", "seq")]


def heat_group(df, row, rows, col, cols, ytitle, xtitle, min_n):
    """CE and PE heatmaps side by side, with a menu to switch the measure."""
    st = cell_stats(df, ["type", row, col])
    traces, buttons = [], []
    for mi, (key, label, kind) in enumerate(METRICS):
        mats = {}
        for cp in ("CE", "PE"):
            x = st[st.type == cp]
            z = x.pivot(index=row, columns=col, values=key).reindex(index=rows, columns=cols)
            n = x.pivot(index=row, columns=col, values="n").reindex(index=rows, columns=cols)
            z = z.astype(float).values
            n = n.fillna(0).values
            if key != "n":
                z = np.where(n >= min_n, z, np.nan)
            mats[cp] = (z, n)
        allv = np.concatenate([mats["CE"][0].ravel(), mats["PE"][0].ravel()])
        allv = allv[np.isfinite(allv)]
        if kind == "div":
            lim = float(np.percentile(np.abs(allv), 95)) if len(allv) else 1.0
            lim = max(lim, 0.5)
            zmin, zmax, scale = -lim, lim, [[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#2a78d6"]]
        elif kind == "share":
            zmin, zmax, scale = 0, 100, [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#d03b3b"]]
        elif kind == "down":
            zmin, zmax = (float(allv.min()) if len(allv) else -1.0), 0
            scale = [[0, "#d03b3b"], [1, "#f0efec"]]
        else:
            zmin, zmax = 0, (float(np.percentile(allv, 98)) if len(allv) else 1.0)
            scale = [[0, "#f0efec"], [1, "#2a78d6"]]
        for si, cp in enumerate(("CE", "PE")):
            z, n = mats[cp]
            if key in ("gap", "oc", "hi", "lo"):
                text = [["" if np.isnan(v) else f"{v:+.1f}%" for v in r] for r in z]
            elif key == "fell":
                text = [["" if np.isnan(v) else f"{v:.0f}%" for v in r] for r in z]
            elif key == "n":
                text = [["" if np.isnan(v) or v == 0 else f"{int(v):,}" for v in r] for r in z]
            else:
                text = [["" if np.isnan(v) else f"{v:.1f}" for v in r] for r in z]
            traces.append({
                "type": "heatmap", "x": cols, "y": rows, "visible": mi == 0,
                "z": np.where(np.isnan(z), None, np.round(z, 2)).tolist(),
                "text": text, "texttemplate": "%{text}", "customdata": n.astype(int).tolist(),
                "xaxis": "x" if si == 0 else "x2", "yaxis": "y" if si == 0 else "y2",
                "zmin": zmin, "zmax": zmax, "colorscale": scale, "xgap": 2, "ygap": 2,
                "showscale": si == 1, "colorbar": {"thickness": 10},
                "hovertemplate": f"{NICE[cp]}<br>{ytitle}: %{{y}}<br>{xtitle}: %{{x}}<br>"
                                 f"{label}: %{{text}}<br>%{{customdata:,}} option-days<extra></extra>"})
        buttons.append({"label": label, "method": "update",
                        "args": [{"visible": [i // 2 == mi for i in range(2 * len(METRICS))]}]})
    cat = {"type": "category"}
    layout = {
        "updatemenus": [{"buttons": buttons, "x": 0, "xanchor": "left", "y": 1.2,
                         "yanchor": "top", "showactive": True}],
        "margin": {"t": 84, "l": 150, "r": 30, "b": 56}, "_height": 470,
        "xaxis": {**cat, "domain": [0, 0.47], "title": xtitle},
        "xaxis2": {**cat, "domain": [0.53, 1], "title": xtitle},
        "yaxis": {**cat, "autorange": "reversed", "title": {"text": ytitle, "standoff": 12}},
        "yaxis2": {**cat, "autorange": "reversed", "anchor": "x2", "showticklabels": False},
        "annotations": [
            {"text": "<b>CE (calls)</b>", "x": 0.235, "y": 1.06, "xref": "paper", "yref": "paper",
             "showarrow": False},
            {"text": "<b>PE (puts)</b>", "x": 0.765, "y": 1.06, "xref": "paper", "yref": "paper",
             "showarrow": False}],
    }
    return traces, layout


def box_by(df, value, by, order, ytitle):
    tr = []
    for cp in ("CE", "PE"):
        x = df[df.type == cp].dropna(subset=[value])
        q = x.groupby(by, observed=True)[value].quantile([.05, .25, .5, .75, .95]).unstack()
        q = q.reindex(order).dropna()
        n = x.groupby(by, observed=True)[value].size().reindex(q.index)
        if q.empty:
            continue
        tr.append({"type": "box", "name": NICE[cp], "x": q.index.tolist(),
                   "lowerfence": q[.05].round(2).tolist(), "q1": q[.25].round(2).tolist(),
                   "median": q[.5].round(2).tolist(), "q3": q[.75].round(2).tolist(),
                   "upperfence": q[.95].round(2).tolist(),
                   "marker": {"color": SIDE[cp]}, "line": {"width": 1.5},
                   "customdata": n.tolist(),
                   "hovertemplate": "%{x}<extra>" + NICE[cp] + "</extra>"})
    return tr, {"boxmode": "group", "xaxis": {"type": "category", "categoryorder": "array",
                                               "categoryarray": order},
                "yaxis": {"title": ytitle, "zeroline": True, "zerolinecolor": "#c3c2b7"}}


def fmt_date_range(df):
    return f"{df.date.min()} to {df.date.max()}"


def build_section(sym, R, D, has_vix, cfg):
    """R: option rows used for statistics. D: one row per trading day."""
    figs, insights, cards = [], [], []
    ce, pe = R[R.type == "CE"], R[R.type == "PE"]
    tiles = [("Trading days", f"{R.date.nunique():,}"),
             ("Option-days analysed", f"{len(R):,}"),
             ("CE: median first minute", pct(ce.oc_pct.median())),
             ("PE: median first minute", pct(pe.oc_pct.median())),
             ("CE: fell in first minute", f"{ce.fell.mean() * 100:.0f}%"),
             ("PE: fell in first minute", f"{pe.fell.mean() * 100:.0f}%")]

    # ---- insights, all counted from the rows
    insights.append(
        f"<b>{len(R):,}</b> option-days on <b>{R.date.nunique():,}</b> trading days "
        f"({fmt_date_range(R)}): every out-of-the-money strike up to {cfg.max_otm:g}% from ATM in "
        f"the {cfg.expiries} nearest expiries that traded in the 09:15 minute at "
        f"{cfg.min_prem:g} points or more.")
    insights.append(
        f"Previous close → open: the median CE opened <b>{pct(ce.opt_gap_pct.median())}</b> from "
        f"its previous close and the median PE <b>{pct(pe.opt_gap_pct.median())}</b>. "
        f"First minute (open → close): CE <b>{pct(ce.oc_pct.median())}</b>, PE "
        f"<b>{pct(pe.oc_pct.median())}</b>; the candle closed below its open on "
        f"{ce.fell.mean() * 100:.0f}% of CE and {pe.fell.mean() * 100:.0f}% of PE option-days.")
    e0, e1 = R[R.dte == 0], R[R.dte > 0]
    if len(e0) >= cfg.min_n and len(e1) >= cfg.min_n:
        insights.append(
            f"Time to expiry: on expiry day the median first-minute move was "
            f"<b>{pct(e0.oc_pct.median())}</b> ({len(e0):,} option-days, {e0.fell.mean() * 100:.0f}% "
            f"fell) against <b>{pct(e1.oc_pct.median())}</b> on other days "
            f"({e1.fell.mean() * 100:.0f}% fell). The median candle range (high to low) was "
            f"{(e0.hi_pct - e0.lo_pct).median():.0f}% of the open on expiry day and "
            f"{(e1.hi_pct - e1.lo_pct).median():.0f}% otherwise.")
    near, far = R[R.dist_pct < 1], R[R.dist_pct >= 3]
    if len(near) >= cfg.min_n and len(far) >= cfg.min_n:
        insights.append(
            f"Distance from ATM: strikes within 1% of ATM moved a median "
            f"<b>{pct(near.oc_pct.median())}</b> in the first minute ({near.fell.mean() * 100:.0f}% "
            f"fell); strikes 3% or more away moved <b>{pct(far.oc_pct.median())}</b> "
            f"({far.fell.mean() * 100:.0f}% fell).")
    for name, mask in (("gapped up 0.5% or more", R.idx_gap_pct >= 0.5),
                       ("gapped down 0.5% or more", R.idx_gap_pct <= -0.5)):
        x = R[mask]
        if x.date.nunique() >= 5:
            c, p = x[x.type == "CE"], x[x.type == "PE"]
            insights.append(
                f"When the index {name} ({x.date.nunique()} days): CEs opened a median "
                f"<b>{pct(c.opt_gap_pct.median())}</b> from their previous close and then moved "
                f"<b>{pct(c.oc_pct.median())}</b> in the first minute; PEs opened "
                f"<b>{pct(p.opt_gap_pct.median())}</b> and then moved <b>{pct(p.oc_pct.median())}</b>.")
    if has_vix:
        r1, n1 = spearman(D.vix_oc_pct, D.both_oc)
        r2, n2 = spearman(D.vix_gap_pct, D.both_gap)
        insights.append(
            f"India VIX, first minute: against the average first-minute move of CEs and PEs "
            f"together there is {strength(r1)} over {n1:,} days. India VIX previous close → open "
            f"against how options opened: {strength(r2)} over {n2:,} days. Figures are rank "
            f"correlations from −1 to +1.")
    else:
        insights.append("India VIX: no VIX file was found, so the VIX charts are left out. "
                        "Pass --vix-file to add them.")

    # ---- heatmaps
    tr, lay = heat_group(R, "dte_b", DTE_ORDER, "dist_b", DIST_ORDER,
                         "Days to expiry", "Strike distance from ATM", cfg.min_n)
    figs.append(("By days to expiry and distance from ATM",
                 "Median across option-days. Pick the measure from the menu. Blue is a rise, red a "
                 f"fall; empty cells have fewer than {cfg.min_n} option-days.", tr, lay))
    tr, lay = heat_group(R, "gap_b", GAP_ORDER, "dist_b", DIST_ORDER,
                         "Index gap at the open", "Strike distance from ATM", cfg.min_n)
    figs.append(("By the index's gap and distance from ATM",
                 "The index gap is its previous close to its 09:15 open. On a gap up, CEs are the "
                 "side the index moved towards.", tr, lay))
    tr, lay = heat_group(R, "gap_b", GAP_ORDER, "dte_b", DTE_ORDER,
                         "Index gap at the open", "Days to expiry", cfg.min_n)
    figs.append(("By the index's gap and days to expiry",
                 "The same measures, with expiry day and the days before it across the columns.",
                 tr, lay))

    # ---- typical candle by DTE
    st = cell_stats(R, ["type", "dte_b"])
    tr = []
    for cp in ("CE", "PE"):
        x = st[(st.type == cp) & (st.n >= cfg.min_n)].set_index("dte_b").reindex(DTE_ORDER).dropna(
            subset=["oc"])
        tr.append({"type": "scatter", "mode": "markers", "name": NICE[cp], "x": x.index.tolist(),
                   "y": x.oc.round(2).tolist(),
                   "error_y": {"type": "data", "symmetric": False, "width": 6, "thickness": 2,
                               "array": (x.hi - x.oc).round(2).tolist(),
                               "arrayminus": (x.oc - x.lo).round(2).tolist()},
                   "marker": {"color": SIDE[cp], "size": 10},
                   "customdata": np.column_stack([x.hi.round(1), x.lo.round(1), x.open.round(2),
                                                  x.n.astype(int)]).tolist(),
                   "hovertemplate": "%{x} days to expiry<br>Close %{y:+.1f}% vs open<br>"
                                    "High %{customdata[0]:+.1f}%, low %{customdata[1]:+.1f}%<br>"
                                    "Median open %{customdata[2]} pts, %{customdata[3]:,} option-days"
                                    "<extra>" + NICE[cp] + "</extra>"})
    figs.append(("The typical first-minute candle, by days to expiry",
                 "Dot: median close relative to the open. Whiskers: median high and low relative "
                 "to the open.",
                 tr, {"xaxis": {"type": "category", "categoryorder": "array",
                                "categoryarray": DTE_ORDER, "title": "Days to expiry"},
                      "yaxis": {"title": "% from the candle's open", "zeroline": True,
                                "zerolinecolor": "#c3c2b7"}}))

    # ---- spread of outcomes by index gap
    tr, lay = box_by(R, "opt_gap_pct", "gap_b", GAP_ORDER, "Previous close → open (%)")
    lay["xaxis"]["title"] = "Index gap at the open"
    figs.append(("How options opened, by the index's gap",
                 "Option's previous close to its 09:15 open. Box: middle half of option-days. "
                 "Whiskers: 5th to 95th percentile.", tr, lay))
    tr, lay = box_by(R, "oc_pct", "gap_b", GAP_ORDER, "First minute: open → close (%)")
    lay["xaxis"]["title"] = "Index gap at the open"
    figs.append(("What they did in the first minute, by the index's gap",
                 "Open to close of the 09:15 candle. Box: middle half of option-days. Whiskers: "
                 "5th to 95th percentile.", tr, lay))

    # ---- day-level: index gap vs how options opened
    tr = []
    for cp, col in (("CE", "ce_gap"), ("PE", "pe_gap")):
        x = D.dropna(subset=["idx_gap_pct", col])
        tr.append({"type": "scattergl", "mode": "markers", "name": NICE[cp],
                   "x": x.idx_gap_pct.round(3).tolist(), "y": x[col].round(1).tolist(),
                   "text": x.date.tolist(), "marker": {"color": SIDE[cp], "size": 5, "opacity": 0.55},
                   "hovertemplate": "%{text}<br>Index gap %{x:+.2f}%<br>Options opened %{y:+.1f}%"
                                    "<extra>" + NICE[cp] + "</extra>"})
    figs.append(("Each day: the index's gap against how its options opened",
                 "One dot per trading day and side: the median previous close → open of that "
                 "day's out-of-the-money options in the nearest expiry.",
                 tr, {"xaxis": {"title": "Index: previous close → 09:15 open (%)", "zeroline": True,
                                "zerolinecolor": "#c3c2b7"},
                      "yaxis": {"title": "Options: previous close → open (%)", "zeroline": True,
                                "zerolinecolor": "#c3c2b7"}}))

    # ---- India VIX
    if has_vix:
        drivers = [("vix_oc_pct", "India VIX: first minute"),
                   ("vix_gap_pct", "India VIX: previous close → open"),
                   ("idx_oc_pct", f"{sym}: first minute"),
                   ("idx_gap_pct", f"{sym}: previous close → open")]
        targets = [("ce_oc", "CE first minute"), ("pe_oc", "PE first minute"),
                   ("both_oc", "CE and PE first minute"), ("ce_gap", "CE open vs prev close"),
                   ("pe_gap", "PE open vs prev close"), ("both_gap", "CE and PE open vs prev close")]
        z, nn = [], []
        for dcol, _ in drivers:
            row, nrow = [], []
            for tcol, _ in targets:
                r, n = spearman(D[dcol], D[tcol])
                row.append(r)
                nrow.append(n)
            z.append(row)
            nn.append(nrow)
        z = np.array(z, float)
        figs.append(("What moves with what: rank correlations across days",
                     "+1 means the two always rise and fall together, −1 always opposite, 0 no "
                     "relation. The index rows are there for comparison with India VIX.",
                     [{"type": "heatmap", "x": [t[1] for t in targets], "y": [d[1] for d in drivers],
                       "z": np.where(np.isnan(z), None, np.round(z, 2)).tolist(),
                       "text": [["" if np.isnan(v) else f"{v:+.2f}" for v in r] for r in z],
                       "texttemplate": "%{text}", "customdata": nn, "zmin": -1, "zmax": 1,
                       "xgap": 2, "ygap": 2,
                       "colorscale": [[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#2a78d6"]],
                       "colorbar": {"thickness": 10},
                       "hovertemplate": "%{y}<br>against %{x}<br>%{text} over %{customdata:,} days"
                                        "<extra></extra>"}],
                     {"xaxis": {"type": "category"}, "yaxis": {"type": "category",
                                                               "autorange": "reversed"},
                      "margin": {"l": 230, "b": 150, "r": 60}, "_height": 460}))

        x = D.dropna(subset=["vix_oc_pct"]).copy()
        if len(x) >= 50:
            try:
                x["q"] = pd.qcut(x.vix_oc_pct, 5, duplicates="drop")
                g = x.groupby("q", observed=True)
                labs = [f"{iv.left:+.2f}% to {iv.right:+.2f}%" for iv in g.groups.keys()]
                tr = []
                for cp, col in (("CE", "ce_oc"), ("PE", "pe_oc")):
                    tr.append({"type": "bar", "name": NICE[cp], "x": labs,
                               "y": g[col].median().round(2).tolist(),
                               "customdata": g.size().tolist(), "marker": {"color": SIDE[cp]},
                               "hovertemplate": "VIX first minute %{x}<br>Median option move "
                                                "%{y:+.2f}%<br>%{customdata} days<extra>"
                                                + NICE[cp] + "</extra>"})
                figs.append(("Option first-minute move, by India VIX's first-minute move",
                             "Days split into five equal groups by how much India VIX moved from "
                             "its 09:15 open to that candle's close.",
                             tr, {"barmode": "group", "bargap": 0.25,
                                  "xaxis": {"type": "category",
                                            "title": "India VIX first-minute move (lowest to highest)"},
                                  "yaxis": {"title": "Median option open → close (%)",
                                            "zeroline": True, "zerolinecolor": "#c3c2b7"}}))
            except ValueError:
                pass
        x = D.dropna(subset=["vix_oc_pct", "both_oc"])
        figs.append(("Each day: India VIX's first minute against the options' first minute",
                     "One dot per trading day. Vertical axis: average of the median CE move and "
                     "the median PE move, so a rise means both sides got dearer.",
                     [{"type": "scattergl", "mode": "markers", "x": x.vix_oc_pct.round(3).tolist(),
                       "y": x.both_oc.round(2).tolist(), "text": x.date.tolist(),
                       "marker": {"color": SERIES[6], "size": 5, "opacity": 0.55},
                       "hovertemplate": "%{text}<br>VIX %{x:+.2f}%<br>Options %{y:+.2f}%"
                                        "<extra></extra>"}],
                     {"xaxis": {"title": "India VIX: 09:15 open → close (%)", "zeroline": True,
                                "zerolinecolor": "#c3c2b7"},
                      "yaxis": {"title": "Options: open → close (%)", "zeroline": True,
                                "zerolinecolor": "#c3c2b7"}}))

    # ---- table view of the main grid
    st = cell_stats(R, ["type", "dte_b", "dist_b"])
    st = st[st.n >= cfg.min_n]
    st["o1"] = st.dte_b.map({v: i for i, v in enumerate(DTE_ORDER)})
    st["o2"] = st.dist_b.map({v: i for i, v in enumerate(DIST_ORDER)})
    rows = "".join(
        f"<tr><td>{r.dte_b}</td><td>{r.dist_b}</td><td>{r.type}</td><td>{int(r.n):,}</td>"
        f"<td>{num(r.dist_pts, 0)}</td><td>{num(r.prev)}</td>"
        f"<td>{num(r.open)} / {num(r.high)} / {num(r.low)} / {num(r.close)}</td>"
        f"<td>{pct(r.gap)}</td><td>{pct(r.oc)}</td><td>{pct(r.hi)}</td><td>{pct(r.lo)}</td>"
        f"<td>{r.fell:.0f}%</td></tr>"
        for r in st.sort_values(["o1", "o2", "type"]).itertuples())
    cards.append(("Every combination of days to expiry and distance",
                  "Medians across option-days.",
                  "<div class='scroll tall'><table><thead><tr><th>Days to expiry</th>"
                  "<th>Distance from ATM</th><th>Side</th><th>Option-days</th>"
                  "<th>Median points from ATM</th><th>Median previous close</th>"
                  "<th>Median 09:15 candle O / H / L / C</th><th>Prev close → open</th>"
                  "<th>Open → close</th><th>High vs open</th><th>Low vs open</th>"
                  f"<th>Fell in the minute</th></tr></thead><tbody>{rows}</tbody></table></div>"))
    return {"sym": sym, "tiles": tiles, "insights": insights, "figs": figs, "cards": cards}


# ---------------------------------------------------------------- explorers
def explorer(tid, title, note, df, cols, filters, defaults):
    keep = [c for c, _, _ in cols]
    d = df[keep].copy()
    for c, _, kind in cols:
        if kind != "text":
            d[c] = pd.to_numeric(d[c], errors="coerce").round(2)
        else:
            d[c] = d[c].fillna("").astype(str)
    return {"id": tid, "title": title, "note": note,
            "cols": [{"label": lab, "kind": kind} for _, lab, kind in cols],
            "rows": json.loads(d.to_json(orient="values")),
            "filters": [keep.index(f) for f in filters],
            "defaults": {str(keep.index(k)): str(v) for k, v in defaults.items()}, "sum": None}


HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>First-minute option lab</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--border:rgba(11,11,11,.10);--bad:#d03b3b}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}
main{max-width:1160px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:21px;margin:44px 0 12px}
h3{font-size:15px;margin:0 0 2px}.sub{color:var(--ink2);margin:0 0 20px}
.note{color:var(--muted);font-size:13px;margin:0 0 8px}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 0}
nav a{font-size:13px;padding:5px 12px;border:1px solid var(--border);border-radius:999px;
background:var(--surface);color:var(--ink);text-decoration:none}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--ink2);font-size:13px}.tile .v{font-size:24px;font-weight:600}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 18px 8px;margin:16px 0}
ul.ins{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 16px 16px 34px}
ul.ins li{margin:6px 0}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--grid)}th{color:var(--ink2);font-weight:600}
.assume{font-size:13px;color:var(--ink2)}
.scroll{overflow:auto;margin-bottom:12px}.scroll td,.scroll th{white-space:nowrap}
.scroll.tall{max-height:520px}.scroll.tall th{position:sticky;top:0;background:var(--surface)}
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
</style></head><body><main>
<h1>First-minute option lab</h1>
<p class="sub">__SUB__</p>
__BODY__
<h2>How to read this, and what is assumed</h2>
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
  if (kind === "pct") return (v > 0 ? "+" : "") + v.toLocaleString("en-IN", {maximumFractionDigits: 2}) + "%";
  if (kind === "num") return v.toLocaleString("en-IN", {maximumFractionDigits: 2});
  return String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;");
}
for (const t of TABLES){
  const root = document.getElementById(t.id);
  if (!root) continue;
  const st = {sort: null, dir: 1, page: 0, f: Object.assign({}, t.defaults), q: ""};
  const bar = document.createElement("div"); bar.className = "xp-bar";
  for (const ci of t.filters){
    const vals = [...new Set(t.rows.map(r => String(r[ci])))].sort();
    const lab = document.createElement("label"); lab.textContent = t.cols[ci].label;
    const sel = document.createElement("select");
    sel.innerHTML = '<option value="">All</option>' + vals.map(v => `<option>${v}</option>`).join("");
    if (st.f[ci] !== undefined && vals.includes(st.f[ci])) sel.value = st.f[ci]; else delete st.f[ci];
    sel.onchange = () => { if (sel.value) st.f[ci] = sel.value; else delete st.f[ci]; st.page = 0; draw(); };
    lab.appendChild(sel); bar.appendChild(lab);
  }
  const sl = document.createElement("label"); sl.textContent = "Search (date…)";
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
      const i = st.sort, isNum = t.cols[i].kind !== "text";
      rows = rows.slice().sort((a, b) => {
        const x = a[i], y = b[i];
        if (x === null || x === "") return 1; if (y === null || y === "") return -1;
        return st.dir * (isNum ? x - y : String(x).localeCompare(String(y)));
      });
    }
    const pages = Math.max(1, Math.ceil(rows.length / PAGE));
    st.page = Math.min(st.page, pages - 1);
    const view = rows.slice(st.page * PAGE, (st.page + 1) * PAGE);
    const head = t.cols.map((c, i) => `<th data-i="${i}">${c.label}${st.sort === i ? (st.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("");
    const body = view.map(r => "<tr>" + r.map((v, i) => {
      const k = t.cols[i].kind;
      return `<td class="${k !== "text" ? "n" : ""}${k === "pct" && v < 0 ? " neg" : ""}">${cell(v, k)}</td>`;
    }).join("") + "</tr>").join("");
    wrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    wrap.querySelectorAll("th").forEach(th => th.onclick = () => {
      const i = +th.dataset.i; st.dir = st.sort === i ? -st.dir : 1; st.sort = i; draw();
    });
    foot.innerHTML = `<span>${rows.length.toLocaleString("en-IN")} rows</span><span><button class="pv">Previous</button> page ${st.page + 1} of ${pages} <button class="nx">Next</button></span>`;
    const pv = foot.querySelector(".pv"), nx = foot.querySelector(".nx");
    pv.disabled = st.page === 0; nx.disabled = st.page >= pages - 1;
    pv.onclick = () => { st.page--; draw(); }; nx.onclick = () => { st.page++; draw(); };
  }
  draw();
}
</script></body></html>"""


def render(sections, compare, tables, quality, cfg, vix_note):
    body, figs = [], []
    body.append("<nav>" + "".join(f'<a href="#{s["sym"]}">{s["sym"]}</a>' for s in sections)
                + '<a href="#browse">Browse the rows</a></nav>')
    body.append('<div class="card"><h3>The three indices side by side</h3>'
                '<p class="note">Medians across all option-days analysed.</p>'
                f'<div class="scroll">{compare}</div></div>')
    for s in sections:
        body.append(f'<h2 id="{s["sym"]}">{s["sym"]}</h2>')
        body.append('<div class="tiles">' + "".join(
            f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>'
            for k, v in s["tiles"]) + "</div>")
        body.append("<ul class='ins'>" + "".join(f"<li>{i}</li>" for i in s["insights"]) + "</ul>")
        for j, (title, note, data, layout) in enumerate(s["figs"]):
            fid = f"f_{s['sym']}_{j}"
            layout = dict(layout)
            h = layout.pop("_height", 400)
            body.append(f'<div class="card"><h3>{title}</h3><p class="note">{note}</p>'
                        f'<div id="{fid}" style="height:{h}px"></div></div>')
            figs.append({"id": fid, "data": data, "layout": layout})
        for title, note, html in s["cards"]:
            body.append(f'<div class="card"><h3>{title}</h3><p class="note">{note}</p>{html}</div>')
    body.append('<h2 id="browse">Browse the rows</h2>'
                "<p class='sub'>Filter, search and sort. Click a column heading to sort.</p>")
    for x in tables:
        body.append(f'<div class="card"><h3>{x["title"]}</h3><p class="note">{x["note"]}</p>'
                    f'<div id="{x["id"]}"></div></div>')
    body.append('<div class="card"><h3>Data coverage</h3><p class="note">Counted from the files on '
                'this run.</p><ul>' + "".join(f"<li>{q}</li>" for q in quality) + "</ul></div>")
    assume = [
        "<b>First minute:</b> the 1-minute candle stamped 09:15, covering 09:15:00 to 09:15:59. "
        "Bars before 09:15 (the pre-open session) are ignored.",
        "<b>ATM and distance:</b> ATM is the index's 09:15 open rounded to the strike interval found "
        "in each expiry file. Distance is how far the strike sits beyond ATM, as a share of the "
        "index, so the three indices and different years can be compared. Only out-of-the-money "
        f"strikes up to {cfg.max_otm:g}% away are kept.",
        "<b>Previous close:</b> the last traded price of the same contract in the previous session, "
        "and for the index and India VIX the last 1-minute close of the previous session. These can "
        "differ slightly from the exchange's official closing prices.",
        f"<b>Which expiries:</b> on each day, the {cfg.expiries} nearest expiries. Days to expiry "
        "are calendar days to the expiry date in the file name.",
        f"<b>Small premiums:</b> options that opened below {cfg.min_prem:g} points are left out of "
        "every statistic, because one tick (0.05) is a large percentage of such a price. Medians "
        "are used throughout for the same reason.",
        "<b>Only what traded:</b> a strike appears on a day only if it traded in the 09:15 minute. "
        "Far strikes that did not trade are missing, so far-strike figures describe the liquid ones.",
        "<b>India VIX:</b> " + vix_note,
        "<b>Correlation is not cause:</b> the index gap, India VIX and option prices all react to "
        "the same overnight news. The charts show what moved together, not what drove what.",
    ]
    sub = (f"Out-of-the-money options of {', '.join(s['sym'] for s in sections)} in the 09:15 "
           "candle: how they open against the previous close and what they do in the first minute, "
           "by days to expiry, distance from ATM, the index's gap and India VIX.")
    html = (HTML.replace("__SUB__", sub).replace("__BODY__", "\n".join(body))
            .replace("__ASSUME__", "".join(f"<li>{a}</li>" for a in assume))
            .replace("__FIGS__", json.dumps(figs, default=float).replace("</", "<\\/"))
            .replace("__TABLES__", json.dumps(tables, ensure_ascii=False,
                                              separators=(",", ":")).replace("</", "<\\/")))
    Path(cfg.out).write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- main
def find_index_file(files, sym):
    want = sym.lower().replace(" ", "")
    for f in files:
        if f.lower().startswith("index/") and Path(f).stem.lower().replace(" ", "").replace(
                "_", "") == want:
            return f
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="NIFTY,SENSEX,BANKNIFTY")
    ap.add_argument("--since", default=None, help="only trading days on/after YYYY-MM-DD")
    ap.add_argument("--expiries", type=int, default=2, help="nearest expiries to include each day")
    ap.add_argument("--max-otm", type=float, default=5, help="furthest strike, %% from ATM")
    ap.add_argument("--max-dte", type=int, default=45, help="ignore contracts further from expiry")
    ap.add_argument("--min-prem", type=float, default=1.0,
                    help="leave out options that opened below this many points")
    ap.add_argument("--min-n", type=int, default=30,
                    help="fewest option-days a cell needs before it is shown")
    ap.add_argument("--vix-file", default=None, help="1-minute India VIX CSV or parquet")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--local", default=None, help="local copy of the dataset repo")
    ap.add_argument("--csv", action="store_true", help="also save row-level CSV files")
    ap.add_argument("--out", default="open_minute_report.html")
    cfg = ap.parse_args()
    since = pd.Timestamp(cfg.since) if cfg.since else None

    files = list_files(cfg.local)
    quality = []

    # ---- India VIX
    vix, vix_note = None, ""
    try:
        if cfg.vix_file:
            vix = daily_table(load_bars(Path(cfg.vix_file).expanduser()))
            vix_note = f"read from {Path(cfg.vix_file).name}."
        else:
            cand = [f for f in files if f.lower().startswith("index/") and "vix" in f.lower()]
            if cand:
                vix = daily_table(load_bars(fetch(cand[0], cfg.local)))
                vix_note = f"read from {cand[0]} in the dataset."
    except Exception as err:
        print(f"! could not read India VIX: {err}")
    if vix is None:
        vix_note = ("no VIX file was found in the dataset's index/ folder and none was passed with "
                    "--vix-file, so the VIX charts are missing.")
        print("! India VIX not found in the dataset. Pass --vix-file to include it.")
    else:
        ok = int(vix.open.notna().sum())
        vix_note += (f" {ok:,} of {len(vix):,} sessions have a 09:15 VIX bar. As far as the author "
                     "knows, India VIX is calculated by NSE from NIFTY option quotes, so some "
                     "relation with NIFTY options is built in.")
        quality.append(f"India VIX: {ok:,} of {len(vix):,} sessions have a 09:15 bar.")

    sections, tables_day, tables_cell, comp_rows, all_rows = [], [], [], [], []
    for sym in [s.strip().upper() for s in cfg.symbols.split(",") if s.strip()]:
        ipath = find_index_file(files, sym)
        if not ipath:
            print(f"[{sym}] no index file in the dataset, skipped")
            quality.append(f"{sym}: no index file in the dataset, so it is left out.")
            continue
        daily = daily_table(load_bars(fetch(ipath, cfg.local)))
        days = list(daily.index)
        next_day = dict(zip(days[:-1], days[1:]))
        exp_files = {}
        for f in files:
            if f.startswith(f"options/{sym}/"):
                try:
                    exp_files[pd.Timestamp(Path(f).stem)] = f
                except Exception:
                    pass
        exps = sorted(e for e in exp_files if since is None or e >= since)
        print(f"[{sym}] {len(daily):,} sessions, {len(exps)} expiry files "
              f"(downloads are cached after the first run)")
        parts, failed, done = [], 0, 0
        with ThreadPoolExecutor(cfg.workers) as ex:
            futs = {ex.submit(fetch, exp_files[e], cfg.local): e for e in exps}
            for fu in as_completed(futs):
                e = futs[fu]
                done += 1
                try:
                    r = process_file(fu.result(), e, daily, next_day, cfg)
                    if r is not None:
                        parts.append(r)
                except Exception as err:
                    failed += 1
                    print(f"  ! {e.date()}: {err}")
                if done % 25 == 0 or done == len(exps):
                    print(f"  {done}/{len(exps)} expiry files processed")
        if not parts:
            print(f"[{sym}] nothing usable")
            continue
        A = pd.concat(parts, ignore_index=True)
        if since is not None:
            A = A[A.day >= since]
        A["rank"] = A.groupby("day").expiry.rank(method="dense")
        A = A[A["rank"] <= cfg.expiries].copy()
        A["symbol"] = sym
        A["date"] = A.day.dt.strftime("%Y-%m-%d")
        A = A.rename(columns={"option_type": "type"})
        A["opt_gap_pct"] = np.where(A.prev_close >= cfg.min_prem,
                                    (A.open / A.prev_close - 1) * 100, np.nan)
        A["oc_pct"] = (A.close / A.open - 1) * 100
        A["hi_pct"] = (A.high / A.open - 1) * 100
        A["lo_pct"] = (A.low / A.open - 1) * 100
        A["fell"] = (A.close < A.open).astype(float)
        A["rose"] = (A.close > A.open).astype(float)
        A["dte_b"] = A.dte.map(bucket_dte)
        A["dist_b"] = cut(A.dist_pct, DIST_EDGES, DIST_ORDER)
        A["gap_b"] = cut(A.idx_gap_pct, GAP_EDGES, GAP_ORDER)
        n_all = len(A)
        R = A[(A.open >= cfg.min_prem) & A.idx_gap_pct.notna()].copy()
        if R.empty:
            print(f"[{sym}] no option opened at {cfg.min_prem:g} points or more")
            continue

        # ---- one row per day (nearest expiry only)
        N = R[R["rank"] == 1]
        piv = N.pivot_table(index="day", columns="type", values=["oc_pct", "opt_gap_pct"],
                            aggfunc="median")
        D = pd.DataFrame(index=sorted(R.day.unique()))
        for cp, low in (("CE", "ce"), ("PE", "pe")):
            D[f"{low}_oc"] = piv["oc_pct"][cp] if cp in piv["oc_pct"] else np.nan
            D[f"{low}_gap"] = piv["opt_gap_pct"][cp] if cp in piv["opt_gap_pct"] else np.nan
        D["both_oc"] = (D.ce_oc + D.pe_oc) / 2
        D["both_gap"] = (D.ce_gap + D.pe_gap) / 2
        D["dte"] = N.groupby("day").dte.first()
        D["options"] = N.groupby("day").size()
        D = D.join(daily[["prev_close", "open", "gap_pct", "oc_pct"]].rename(columns={
            "prev_close": "idx_prev_close", "open": "idx_open", "gap_pct": "idx_gap_pct",
            "oc_pct": "idx_oc_pct"}))
        if vix is not None:
            D = D.join(vix[["open", "gap_pct", "oc_pct"]].rename(columns={
                "open": "vix_open", "gap_pct": "vix_gap_pct", "oc_pct": "vix_oc_pct"}))
        else:
            D["vix_open"] = D["vix_gap_pct"] = D["vix_oc_pct"] = np.nan
        D["date"] = D.index.strftime("%Y-%m-%d")
        D["symbol"] = sym
        has_vix = vix is not None and D.vix_oc_pct.notna().sum() >= 30

        sections.append(build_section(sym, R, D, has_vix, cfg))
        ce, pe = R[R.type == "CE"], R[R.type == "PE"]
        comp_rows.append(
            f"<tr><td>{sym}</td><td>{R.date.nunique():,}</td><td>{len(R):,}</td>"
            f"<td>{pct(ce.opt_gap_pct.median())} / {pct(pe.opt_gap_pct.median())}</td>"
            f"<td>{pct(ce.oc_pct.median())} / {pct(pe.oc_pct.median())}</td>"
            f"<td>{ce.fell.mean() * 100:.0f}% / {pe.fell.mean() * 100:.0f}%</td>"
            f"<td>{pct(R[R.dte == 0].oc_pct.median())}</td>"
            f"<td>{pct(R[R.dte > 0].oc_pct.median())}</td>"
            f"<td>{'-' if not has_vix else f'{spearman(D.vix_oc_pct, D.both_oc)[0]:+.2f}'}</td></tr>")
        quality.append(
            f"{sym}: {len(exps)} expiry files read ({failed} failed), {n_all:,} out-of-the-money "
            f"09:15 candles found, {len(R):,} kept after dropping those that opened below "
            f"{cfg.min_prem:g} points or had no index gap. "
            f"{R.opt_gap_pct.notna().mean() * 100:.0f}% of kept rows have a previous close for the "
            f"same contract. {int(daily.open.notna().sum()):,} of {len(daily):,} index sessions "
            f"have a 09:15 bar.")
        tables_day.append(D)
        C = cell_stats(R, ["symbol", "type", "dte_b", "dist_b", "gap_b"])
        tables_cell.append(C[C.n >= 5])
        all_rows.append(A)

    if not sections:
        print("No results.")
        return
    compare = ("<table><thead><tr><th>Index</th><th>Trading days</th><th>Option-days</th>"
               "<th>Prev close → open (CE / PE)</th><th>First minute (CE / PE)</th>"
               "<th>Fell in the first minute (CE / PE)</th><th>First minute on expiry day</th>"
               "<th>First minute on other days</th><th>VIX vs options, first minute</th>"
               f"</tr></thead><tbody>{''.join(comp_rows)}</tbody></table>")
    DD = pd.concat(tables_day).sort_values(["date", "symbol"])
    CC = pd.concat(tables_cell)
    tables = [
        explorer("x_days", "Every trading day",
                 "The index, India VIX and the median out-of-the-money option of the nearest "
                 "expiry, in the 09:15 candle.",
                 DD, [("date", "Date", "text"), ("symbol", "Index", "text"),
                      ("dte", "Days to expiry", "num"), ("idx_prev_close", "Index prev close", "num"),
                      ("idx_open", "Index open", "num"), ("idx_gap_pct", "Index gap", "pct"),
                      ("idx_oc_pct", "Index first minute", "pct"), ("vix_open", "VIX open", "num"),
                      ("vix_gap_pct", "VIX gap", "pct"), ("vix_oc_pct", "VIX first minute", "pct"),
                      ("ce_gap", "CE prev close → open", "pct"), ("ce_oc", "CE first minute", "pct"),
                      ("pe_gap", "PE prev close → open", "pct"), ("pe_oc", "PE first minute", "pct"),
                      ("options", "Options counted", "num")],
                 ["symbol"], {}),
        explorer("x_cells", "Every combination of index, side, expiry, distance and gap",
                 "Medians across option-days for each combination with at least 5 option-days. "
                 "Small groups are noisy: check the count.",
                 CC, [("symbol", "Index", "text"), ("type", "Side", "text"),
                      ("dte_b", "Days to expiry", "text"), ("dist_b", "Distance from ATM", "text"),
                      ("gap_b", "Index gap", "text"), ("n", "Option-days", "num"),
                      ("days", "Days", "num"), ("dist_pts", "Points from ATM", "num"),
                      ("prev", "Prev close", "num"), ("open", "Open", "num"),
                      ("high", "High", "num"), ("low", "Low", "num"), ("close", "Close", "num"),
                      ("gap", "Prev close → open", "pct"), ("oc", "Open → close", "pct"),
                      ("hi", "High vs open", "pct"), ("lo", "Low vs open", "pct"),
                      ("fell", "Fell (% of cases)", "num")],
                 ["symbol", "type", "dte_b", "dist_b", "gap_b"], {}),
    ]
    render(sections, compare, tables, quality, cfg, vix_note)
    print(f"\nReport: {Path(cfg.out).resolve()}")
    if cfg.csv:
        keep = ["symbol", "date", "expiry", "dte", "type", "strike", "atm", "dist_pts", "dist_pct",
                "idx_open", "idx_gap_pct", "idx_oc_pct", "prev_close", "open", "high", "low",
                "close", "volume", "opt_gap_pct", "oc_pct", "hi_pct", "lo_pct"]
        pd.concat(all_rows)[keep].to_csv("open_minute_options.csv", index=False)
        DD.to_csv("open_minute_days.csv", index=False)
        print("CSV: open_minute_options.csv, open_minute_days.csv")


if __name__ == "__main__":
    main()
