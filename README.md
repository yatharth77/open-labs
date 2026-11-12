For expiry_collector:

# Expiry-day far-OTM option collector (SENSEX & NIFTY)

`expiry_collector.py` runs after market close and, **only on SENSEX or NIFTY expiry days**, records how far out-of-the-money options behave through the day, using the [Kite Connect](https://kite.trade/docs/connect/v3/) API.

## What it records

| Index | Options exchange | Strikes captured (from ATM) |
|---|---|---|
| SENSEX | BFO | CE and PE 2,000–3,000 points away |
| NIFTY | NFO | CE and PE 1,200–2,000 points away |

ATM is the strike nearest to the index's 09:15 open.

For every option in the band:
- First 1-minute candle (09:15): open, high, low, close, volume
- Day's high and its time, high after the first minute and its time
- Day's low, last close, and day high ÷ first-minute close

For each index: open, first-minute candle, and the day's high and low (with times) and close.

### Output files
| File | Contents |
|---|---|
| `expiry_options.csv` | One row per option per expiry (appended) |
| `expiry_index.csv` | One row per index per expiry (appended) |
| `candles/<date>_<INDEX>.csv` | Raw 1-minute candles for the index and every option |

## Setup

1. **Kite Connect app.** Create one at [developers.kite.trade](https://developers.kite.trade/) and note the API key and secret. The plan must include **historical data** (minute candles).
2. **Install**
   ```bash
   uv add kiteconnect          # or: pip install kiteconnect
   ```
3. **Set credentials as environment variables.** Never put them in code or commit them.
   ```bash
   export KITE_API_KEY=your_api_key
   export KITE_API_SECRET=your_api_secret
   ```
   To keep them across sessions, add these lines to `~/.zshrc`.

## How to run

**Every trading morning.** Kite access tokens expire daily:
```bash
uv run expiry_collector.py --login
```
Open the printed URL, log in, then paste the URL you're redirected to (or just the `request_token`). The token is saved to `.kite_token.json`.

**After 15:30 on any trading day:**
```bash
uv run expiry_collector.py
```
On a non-expiry day it prints the next expiry and exits.

### Options
| Flag | Purpose |
|---|---|
| `--indices SENSEX` | Check only one index (default `SENSEX,NIFTY`) |
| `--force` | Collect even on a non-expiry day (uses the nearest expiry) |
| `--out DIR` | Output folder (default: the script's folder) |

### Run automatically (macOS cron)
```bash
crontab -e
# weekdays at 15:45 IST
45 15 * * 1-5 cd /path/to/repo && /path/to/uv run expiry_collector.py >> collector.log 2>&1
```
The laptop must be awake, and the morning `--login` is still a manual step.

## Notes
- **Run on the expiry day itself.** Expired contracts are removed from Kite the next day, and their data can't be fetched afterwards.
- If an option didn't trade at 09:15, its first trade of the day is used and `first_candle_is_0915` is `False`. Options with no trades are kept with `traded = False`.
- The script stays within the historical API's rate limit (about 3 requests per second). Each expiry takes about 20–40 seconds.

## Disclaimer
For personal research only. Not investment advice.
