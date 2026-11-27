#!/usr/bin/env python3
"""
Daily 1-minute data collector (runs on the Mac after the close).

Saves one gzip CSV per trading day:  nifty-data-live/YYYY/MM/YYYY-MM-DD.csv.gz
columns: sym,date,open,high,low,close,volume   (date = 'YYYY-MM-DD HH:MM:SS+05:30', IST)

    python signals/collect.py            # today (after 15:30)
    python signals/collect.py --backfill # every trading day Yahoo still has (~last 29 days)

Notes
- Same universe as the scanner + NIFTY 50 (^NSEI) and NIFTY BANK (^NSEBANK).
- Yahoo reports 0 volume for the 9:15 minute; kept as-is (the old dataset has real 9:15 volume).
- Yahoo only serves 1-minute bars for the last ~30 days, 8 days per request.
"""
import argparse
import csv
import gzip
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "nifty-data-live")
INDICES = {"^NSEI": "NIFTY 50", "^NSEBANK": "NIFTY BANK"}


def fetch_1m(sym, p1, p2):
    ysym = sym if sym.startswith("^") else sym + ".NS"
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ysym)}"
           f"?interval=1m&period1={int(p1)}&period2={int(p2)}&includePrePost=false")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=E.Yahoo.UA)
            with urllib.request.urlopen(req, timeout=30, context=E._ssl_context()) as r:
                j = json.load(r)
            res = j["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            rows = []
            for i, t in enumerate(res.get("timestamp") or []):
                if q["close"][i] is None or q["open"][i] is None:
                    continue
                dt = datetime.fromtimestamp(t, E.IST)
                if not (9 * 60 + 15 <= dt.hour * 60 + dt.minute <= 15 * 60 + 29):
                    continue
                rows.append((dt, q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i] or 0))
            return rows
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    print(f"  ! {sym}: {last}")
    return []


def path_for(day):
    return os.path.join(OUT, f"{day:%Y}", f"{day:%m}", f"{day:%Y-%m-%d}.csv.gz")


def write_day(day, by_sym):
    p = path_for(day)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sym", "date", "open", "high", "low", "close", "volume"])
    n = 0
    for sym in sorted(by_sym):
        for dt, o, h, l, c, v in by_sym[sym]:
            w.writerow([sym, dt.isoformat(sep=" "), round(o, 2), round(h, 2), round(l, 2), round(c, 2), int(v)])
            n += 1
    with gzip.open(p, "wt") as f:
        f.write(buf.getvalue())
    return n


def collect(p1, p2, only_missing=True):
    syms = E.UNIVERSE + list(INDICES)
    days = {}
    for k, s in enumerate(syms):
        for row in fetch_1m(s, p1, p2):
            name = INDICES.get(s, s)
            days.setdefault(row[0].date(), {}).setdefault(name, []).append(row)
        if k % 20 == 0:
            print(f"  {k}/{len(syms)}", flush=True)
    written = []
    for day in sorted(days):
        if only_missing and os.path.exists(path_for(day)):
            continue
        now = E.now_ist()
        if day == now.date() and now.hour * 60 + now.minute < 15 * 60 + 31:
            continue  # today isn't finished yet
        if len(days[day]) < len(syms) * 0.8:  # incomplete day (e.g. still trading) - skip
            print(f"  skip {day}: only {len(days[day])} symbols")
            continue
        n = write_day(day, days[day])
        written.append((str(day), n))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    now = E.now_ist()
    if a.backfill:
        out = []
        end = now
        start = now - timedelta(days=29)
        while start < end:  # Yahoo: max 8 days per 1-minute request
            stop = min(start + timedelta(days=7), end)
            print(f"window {start:%d %b} - {stop:%d %b}")
            out += collect(start.timestamp(), stop.timestamp())
            start = stop
    else:
        day0 = now.replace(hour=9, minute=0, second=0, microsecond=0)
        out = collect(day0.timestamp(), now.timestamp(), only_missing=False)
    print("written:", out)


if __name__ == "__main__":
    main()
