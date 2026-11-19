# Market patterns research (NIFTY + Nifty 100, 2015 – Apr 2026)

Builds `patterns_report.html` data from the 1-minute CSVs in `../nifty-data`.

Run in order from this folder (paths at the top of `build.py` / `intraday.py` point at nifty-data; edit SRC/P to your local path):

    python3 build.py          # 1-min CSVs -> daily.parquet (~1 min)
    python3 scan_index.py     # NIFTY/BANKNIFTY signals, regimes, breadth
    python3 scan_stocks.py    # stock features
    python3 eval_stocks.py    # stock pattern scan
    python3 strategies.py     # stock strategy backtests
    python3 idx_strats.py     # index strategy backtests
    python3 intraday.py       # opening-range breakout
    python3 export.py         # report tables -> out/*.csv, out/*.json

Needs: pandas, numpy, pyarrow. Not investment advice.

## Volume Breakout strategy (volume_breakout_report.html)

    python3 breakout_lab.py      # breakout levels (6m/1y/2y/3y), breakout-day and 2-week volume features
    python3 breakout_eval.py     # grid of breakout x volume filters (63-day forward returns)
    python3 breakout_strats.py   # strategy versions S0-S6 and exit variants
    python3 breakout_extra.py    # volume buckets, tiers, situation/year tables
    python3 breakout_combo.py    # confirmation versions S7-S9 (S9 = final rule)
    python3 breakout_export.py   # data for the report

Needs `build.py` and `scan_index.py` to have run first.
