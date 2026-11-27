#!/usr/bin/env python3
"""
Recent-performance check: replays the intraday and overnight rules on the days collected in
nifty-data-live/ (fresh data, after the 2015-Apr 2026 backtest) and compares the win rates
with the backtest. Writes signals/data/recent_stats.json, which the dashboard shows.

    python signals/recent.py            # all collected days
    python signals/recent.py --days 60  # last 60 collected days

Long-term rules (21-63 session holds) need months of new data; their live record is the
signal log itself (signals/data/log.json).
"""
import argparse
import csv
import glob
import gzip
import json
import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.join(ROOT, "nifty-data-live")
NAME = {"^NSEI": "NIFTY 50", "^NSEBANK": "NIFTY BANK"}


def load_day(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            t = datetime.fromisoformat(r["date"])
            out.setdefault(r["sym"], []).append(dict(t=t, o=float(r["open"]), h=float(r["high"]),
                                                     l=float(r["low"]), c=float(r["close"]), v=float(r["volume"])))
    return out


class Replay:
    """Data source frozen at `asof`, built from collected minute files + live daily bars."""

    def __init__(self, days, files, daily, asof):
        self.days, self.files, self.daily_all, self.asof, self.errors = days, files, daily, asof, []

    def _minutes(self, sym, day):
        return [b for b in self.files[day].get(NAME.get(sym, sym), []) if b["t"] <= self.asof]

    def m1(self, sym):
        return self._minutes(sym, self.asof.date())

    def m1_week(self, sym):
        k = self.days.index(self.asof.date())
        out = []
        for d in self.days[max(0, k - 5):k + 1]:
            out += self._minutes(sym, d)
        return out

    def m5(self, sym):
        return []

    def daily(self, sym):
        return self.daily_all.get(sym) or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(LIVE, "*", "*", "*.csv.gz")))
    if a.days:
        paths = paths[-a.days:]
    if len(paths) < 6:
        print("not enough collected days yet:", len(paths))
        return
    days = [date.fromisoformat(os.path.basename(p)[:10]) for p in paths]
    files = {d: load_day(p) for d, p in zip(days, paths)}
    yh = E.Yahoo()
    daily = E.fetch_all(yh.daily, E.UNIVERSE + [E.NIFTY])
    nifty_daily = daily.pop(E.NIFTY)
    sigs = []
    for d in days[5:]:  # need a few prior days for the opening-volume baseline
        upto = {k: [b for b in v if b["t"].date() <= d] for k, v in daily.items()}  # no look-ahead
        n_upto = [b for b in nifty_daily if b["t"].date() <= d]
        t920 = datetime(d.year, d.month, d.day, 9, 20, tzinfo=E.IST)
        t1500 = datetime(d.year, d.month, d.day, 15, 0, tzinfo=E.IST)
        s1, _, _ = E.scan_intraday(Replay(days, files, upto, t920), d, upto, n_upto)
        s2, _, _ = E.scan_overnight(Replay(days, files, upto, t1500), d, upto, n_upto,
                                    {**{s: Replay(days, files, daily, t1500).m1(s) for s in E.UNIVERSE},
                                     E.NIFTY: Replay(days, files, daily, t1500).m1(E.NIFTY)})
        end = Replay(days, files, daily, datetime(d.year, d.month, d.day, 15, 30, tzinfo=E.IST))
        for s in s1 + s2:
            if s.get("over_limit") or s.get("late"):
                continue
            t_score = datetime.combine(days[-1], datetime.min.time()).replace(hour=15, minute=47, tzinfo=E.IST)
            E.score(s, end, daily, {s["sym"]: end.m1(s["sym"]), s["sym"] + "|week": end.m1(s["sym"])}, days[-1], t_score)
            if s["status"] == "closed" and s["result"].get("pnl") is not None:
                sigs.append(s)
        print(d, len(s1), len(s2), flush=True)
    stats = {}
    for k in ("GF", "OFL", "OFS", "O1", "O2"):
        x = [s["result"]["pnl"] for s in sigs if s["strategy"] == k]
        p = E.PRIORS[k]
        st = dict(n=len(x), backtest_win=p["win"], backtest_avg=p["avg"])
        if x:
            st.update(win=round(100 * sum(v > 0 for v in x) / len(x), 1), avg=round(sum(x) / len(x), 2))
            st["drift"] = bool(len(x) >= 20 and st["win"] < p["win"] - 10)
        stats[k] = st
    out = dict(generated=E.now_ist().isoformat(timespec="minutes"), first_day=str(days[5]), last_day=str(days[-1]),
               days=len(days) - 5, stats=stats,
               note="Replay of the rules on data collected after the backtest period. Small samples swing a lot; "
                    "a drift flag needs ≥20 trades and a win rate 10+ points below the backtest.")
    E.save_json("recent_stats.json", out)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
