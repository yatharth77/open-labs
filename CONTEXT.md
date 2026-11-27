# open-labs: context and decisions

Read this first in any new chat ("read CONTEXT.md in yatharth77/open-labs").
It records what was tested, what was decided and why, so decisions don't get lost.
Personal positions and money details are not kept here (this repo is public).

_Last updated: 27 Nov 2026_

## 1. Data
- `nifty-data/`: 1-minute OHLCV, 116 NIFTY 100 stocks + NIFTY/BANKNIFTY, Feb 2015 – 8 Apr 2026 (5.3 GB).
- `claude_data/`: options export from the HF dataset `thetrademarkk/india-index-options-1m`.
- `research/`: backtest scripts (copied from the analysis sessions).
- `nifty-data-live/YYYY/MM/YYYY-MM-DD.csv.gz`: **new 1-minute data collected daily by the Mac at ~15:50** (Yahoo; same universe + NIFTY 50 / NIFTY BANK), plus a one-time backfill of Yahoo's last ~30 days. Gap 9 Apr – ~27 Oct 2026 is not available from Yahoo (only Kite Connect historical could fill it). Yahoo's 9:15 volume is 0.
- `signals/recent.py` (monthly + after backfill): replays intraday/overnight rules on the fresh data → `signals/data/recent_stats.json`; dashboard flags a strategy when ≥20 fresh trades win 10+ points below the backtest.
- Costs assumed: 0.1% intraday round trip, 0.25% delivery round trip.

## 2. Strategies: what's tradable (backtest, NIFTY 100, 2015–26)

| ID | Rule | Entry / exit | Win | Avg | Notes |
|---|---|---|---|---|---|
| **S9** Volume Breakout | Close above 6m/1y/2y high (high ≥20 sessions old) on ≥2× 50-day volume, 10-day volume already ≥1.5×; **confirm 10 sessions later**: still above level, post volume ≥1.3× | Buy next open after confirmation, hold 63 sessions, no stop | **70.3%** | +13.3% | 118 trades (~1/month). Best tier: 2-year+ highs (74%, +16%). PSU banks: only 2 trades in sample |
| **A2** Dip in uptrend | 5-day fall >12% while above 200-DMA | Buy next open, hold 21 sessions | 65% | +5.4% | Bunched in selloffs |
| **A5** Dip, target/stop | Same | +5% target / −10% stop / day 21 | 73.7% | +1.3% | Shallowest drawdown |
| **B4** 52-week breakout | Close above prior 52w high on >2× volume, NIFTY above 50 & 200-DMA | Buy next open, hold 63 | 62.6% | +7.1% | Random 63-day hold = +5.1%, so small edge |
| **GF** Failed gap-up short | Gap ≥+1%, 9:20 price ≥0.4% below open, opening volume ≥1.2× | Short at 9:20, stop +1.5%, cover 10:20; max 3/day | 52.4% | +0.08% | Recomputed (old figure 56.5% was wrong). Thin edge |
| **OFL/OFS** First-minute fade | 9:15 candle ±3% | Fade at 9:16, 1% stop + 0.5% trail, exit 9:30 | 58% / 55% | small | Long side only **44.8% since 2022**: edge faded |
| **O1** Momentum carry | Up ≥5% at 15:00 on ≥3× volume | Buy ~15:00, sell next open | 57% net | +0.39% net | Positive all 12 years |
| **O2** Panic rebound | Stock ≤−3% on a NIFTY ≤−1.5% day, stock above 200-DMA | Buy ~15:00, sell next close | 62% net | +0.74% net | ~1 day/month, many stocks at once |

## 3. Tested and rejected (don't retest without a new idea)
- Intraday breakouts 9:30–10:30: 34–42% won.
- No hour of the day is reliably predictable; candle patterns ≈ random.
- More volume = bigger swings, not a direction.
- **Gap-down continuation short at 9:20: 37–41% won, −0.2%/trade, lost in 11 of 12 years.** Buying it instead ≈ coin flip. (27 Nov 2026)
- "Falling / far below 52w high" in large caps is not a sell signal (52–60% rose later).
- Same-stock history matters: PNB's own plain breakouts mostly failed (4 of 6) — S9's confirmation filter removed them.

## 4. Live system (decided 25–26 Nov 2026)
- Dashboard: https://yatharth77.github.io/open-labs/signals/ (also "Live signals" tab on the site).
- Engine: `signals/engine.py`; priors in `signals/priors.json`; every call logged in `signals/data/log.json` and scored.
- **Price feed: Yahoo, fetched on Yatharth's Mac** (`signals/mac/`, launchd every minute). Yahoo blocks GitHub's servers (HTTP 429) — decided *not* to work around that. Alternative if the Mac route fails: Zerodha Kite Connect (official, paid, daily login).
- Schedule (IST, weekdays): 09:21 intraday, 11:00 long term, 15:00 overnight, 15:46 scoring, 15:48 post-close long-term check ("✅ buy at the next open" rows for S9/B4/dip that confirmed at the close), 15:50 data collection. Refresh button → GitHub workflow records a request → Mac scans within ~1 min.
- Office-network TLS: Python 3.13 strict-X509 rule relaxed; certificate verification stays on.
- Intraday section always uses the 9:15–9:19 snapshot; refresh only updates results. Overnight before 15:00 = preview, not logged.
- Yahoo quirk: 9:15 1-minute volume is reported as 0 → opening volume measured on 9:16–9:19.

## 5. Working agreements
- Paper-trade first; ~30 trades before judging a 55–60% edge.
- Set the stop **before** entering; size from the daily loss limit.
- Words: "S9 entry today (breakout <date>)" — never call the entry day "the breakout".
- Every signal shows backtest win rate + this stock's own history.
- Claude scans and explains; it never places orders. Not SEBI-registered.

## 6. Chart setup (Kite, daily)
Donchian 252 (+126, 504), Volume with MA 50 (+10), SMA 200, ROC 5; NIFTY chart with SMA 50 + 200.

## 7. Related work
- HVPlatform audit (friend's repo Mr-K10/HVPlatform): 17 fixes on branch `audit/low-ram-fixes`, PDF report delivered 24 Nov.
- Reports here: patterns_report, volume_breakout_report, open_minute_report, nifty_strangle, options_lab, intraday site.

## 8. Open items
- INDIGO S9: borderline volume (1.25–1.37×); confirmation decided at 27 Nov close.
- NIFTY 100 list is fixed as of Apr 2026; refresh twice a year.
