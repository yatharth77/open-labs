"""Build the report's data files from the research pipeline outputs.
Pipeline: build.py -> scan_index.py -> scan_stocks.py -> eval_stocks.py -> strategies.py -> idx_strats.py -> intraday.py -> export.py
Writes out/<name>.csv and out/<name>.json ({rows, meta}) where meta carries the code that made it."""
import pandas as pd, numpy as np, common as c, json, os
O = 'out/'
N = pd.read_parquet('idx_NIFTY_50.parquet'); B = pd.read_parquet('idx_NIFTY_BANK.parquet')
KINDLAB = {'normal': 'No major event', 'boom': 'Post-crash boom', 'correction': 'Correction (FPI selling, rate hikes)',
           'crisis': 'Crisis (COVID crash, China 2015, IL&FS)', 'shock': 'One-off shock (demonetisation, Adani, tariffs)',
           'war': 'War (Russia-Ukraine, Op Sindoor, US-Iran)', 'event': 'Election / budget day'}
OUT = {}
def save(name, df, scripts):
    df.to_csv(O + name + '.csv', index=False); OUT[name] = (df, scripts)

# 1 strategies
ss = pd.read_parquet('summary_stocks.parquet'); si = pd.read_parquet('summary_index.parquet')
keep = {'A2. same, hold 21d': ('Buy the dip in an uptrend', 'Stock still above its 200-DMA falls >12% in 5 days. Buy next open, sell after 21 sessions.'),
 'A5. A + target 5% / stop 10% / max 21d': ('Dip in uptrend, high-win version', 'Same entry. Exit at +5% target, -10% stop, or 21 sessions.'),
 'A3. Dip in uptrend: day fall >5%, stock >200DMA, hold 10d': ('One-day drop in uptrend', 'Stock above 200-DMA falls >5% in a day. Buy next open, sell after 10 sessions.'),
 'B4. B + only when NIFTY >50-DMA': ('52-week-high breakout on volume', 'Close above prior 52-week high on >2x average volume, NIFTY above its 200- and 50-DMA. Hold 63 sessions.'),
 'B3. 52w-high breakout + vol >2x, NIFTY <200DMA, hold 63d': ('Breakout while market is weak', 'Same breakout, but NIFTY below its 200-DMA. Few trades.'),
 'C. 20d breakout from tight base (<8%) + vol>1.5x, hold 63d': ('Tight-base 20-day breakout', '20-day breakout after a quiet range (<8%) on 1.5x volume. Hold 63 sessions.')}
S = pd.read_parquet('stocks_feat.parquet')
b = {10: S.f5.mean() * 0 + ((S.close.groupby([S.sym, S.seg]).shift(-10) / S.close - 1).mean() * 100 - 0.25), 21: S.f21.mean() * 100 - 0.25, 63: S.f63.mean() * 100 - 0.25}
rows = []
for k, (nm, rule) in keep.items():
    r = ss.loc[k]; h = 10 if '10d' in k else (63 if '63' in k or k.startswith('B4') else 21)
    rows.append(dict(id=k.split('.')[0], strategy=nm, rule=rule, market='Stocks (Nifty 100)', trades=int(r.trades), win_pct=round(r.win, 1), avg_trade_pct=round(r.avg, 2),
        random_entry_pct=round(b[h], 2), is_avg_pct=round(r.is_avg, 2), oos_avg_pct=round(r.oos_avg, 2), oos_trades=int(r.oos_n), oos_win_pct=round(r.oos_win, 1),
        cagr_pct=round(r.port_cagr, 1), max_dd_pct=round(r.port_maxdd, 1)))
for _, r in si.iterrows():
    if any(w in r['name'] for w in ['buy & hold', 'risk-off', 'overnight', 'prior vol', 'trend filter', 'buy fear']):
        rows.append(dict(id=r['name'][:40], strategy=r['name'], rule='', market='Index', trades=None if pd.isna(r.trades) else int(r.trades),
          win_pct=None if pd.isna(r.win) else round(r.win, 1), avg_trade_pct=None if pd.isna(r.avg_trade) else round(r.avg_trade, 3), random_entry_pct=None,
          is_avg_pct=None, oos_avg_pct=None, oos_trades=None, oos_win_pct=None, cagr_pct=round(r.cagr, 1), max_dd_pct=round(r.maxdd, 1), is_cagr_pct=round(r.is_cagr, 1), oos_cagr_pct=round(r.oos_cagr, 1)))
st = pd.DataFrame(rows)
st.loc[st.market == 'Index', 'id'] = ['IDX' + str(i) for i in range(1, (st.market == 'Index').sum() + 1)]
save('strategies', st, ['strategies.py', 'idx_strats.py', 'intraday.py'])

# 2 NIFTY weekly + events
w = N[['close', 'dma200']].resample('W-FRI').last().dropna(subset=['close']).reset_index().rename(columns={'day': 'date'})
w['date'] = w.date.dt.strftime('%Y-%m-%d'); w = w.round(1); save('nifty_weekly', w, ['build.py', 'scan_index.py'])
ev = pd.DataFrame([dict(start=s, end=e, event=l, kind=k) for s, e, l, k in c.EVENTS])
def mv(s, e):
    x = N.close[s:e]
    if len(x) < 1: return None
    pc = N.close[:s].iloc[-2] if len(N.close[:s]) > 1 else x.iloc[0]
    return round((x.iloc[-1] / pc - 1) * 100, 1), round((x.min() / pc - 1) * 100, 1)
ev[['nifty_change_pct', 'worst_point_pct']] = [mv(s, e) for s, e in zip(ev.start, ev.end)]
save('events', ev, ['scan_index.py'])

# 3 drawdowns
cl = N.close; dd = cl / cl.cummax() - 1; eps = []; inside = False
for d, v in dd.items():
    if not inside and v <= -0.08:
        inside = True; pk = cl[:d][cl[:d] == cl[:d].max()].index[-1]; tr = d
    if inside:
        if cl[d] < cl[tr]: tr = d
        if v == 0: eps.append((pk, tr, d)); inside = False
if inside: eps.append((pk, tr, None))
dr = []
for pk, tr, rec in eps:
    r = N.loc[pk]
    dr.append(dict(peak=pk.strftime('%Y-%m-%d'), trough=tr.strftime('%Y-%m-%d'), recovered=rec.strftime('%Y-%m-%d') if rec is not None else 'not yet (data ends 8 Apr 2026)',
      fall_pct=round((cl[tr] / cl[pk] - 1) * 100, 1), days_to_bottom=(tr - pk).days, days_to_recover=(rec - tr).days if rec is not None else None,
      vol_at_peak=round(r.rv20, 1), pct_above_200dma_at_peak=round(r.dist200 * 100, 1) if pd.notna(r.dist200) else None, breadth_at_peak=round(r.br50) if pd.notna(r.br50) else None,
      cause=c.event_of(tr)[0] if c.event_of(tr)[0] != 'normal' else 'Global slowdown / US-China trade war (2019)'))
save('drawdowns', pd.DataFrame(dr), ['scan_index.py'])

# 4 fall risk
n = N.copy(); n['fall5'] = n.mdd5 <= -0.03; n['fall21'] = n.mdd21 <= -0.05; n['fall63'] = n.mdd63 <= -0.08
x = n.dropna(subset=['f63', 'dma200']).copy()
x['breadth'] = pd.cut(x.br50, [0, 20, 40, 60, 80, 101], labels=['<20%', '20-40%', '40-60%', '60-80%', '>80%'])
x['above200'] = pd.cut(x.dist200, [-1, -0.05, 0, 0.05, 0.10, 1], labels=['>5% below', '0-5% below', '0-5% above', '5-10% above', '>10% above'])
x['kindl'] = x.event_kind.map(KINDLAB)
fr = []
for dim, col in [('Volatility (20-day)', 'vol'), ('Trend', 'trend'), ('Breadth (% of stocks above 50-DMA)', 'breadth'), ('Distance from 200-DMA', 'above200'), ('Market situation', 'kindl')]:
    for k, g in x.groupby(col, observed=True):
        if len(g) < 20: continue
        fr.append(dict(dimension=dim, bucket=str(k), days=len(g), fall_1w_pct=round(g.fall5.mean() * 100, 1), fall_1m_pct=round(g.fall21.mean() * 100, 1), fall_3m_pct=round(g.fall63.mean() * 100, 1), avg_next_1m_pct=round(g.f21.mean() * 100, 2)))
fr.append(dict(dimension='All days', bucket='All days', days=len(x), fall_1w_pct=round(x.fall5.mean() * 100, 1), fall_1m_pct=round(x.fall21.mean() * 100, 1), fall_3m_pct=round(x.fall63.mean() * 100, 1), avg_next_1m_pct=round(x.f21.mean() * 100, 2)))
for (t, v), g in x.groupby(['trend', 'vol'], observed=True):
    fr.append(dict(dimension='Trend x volatility', bucket=f'{t.split(" ")[0]} market, {v} vol', days=len(g), fall_1w_pct=round(g.fall5.mean() * 100, 1), fall_1m_pct=round(g.fall21.mean() * 100, 1), fall_3m_pct=round(g.fall63.mean() * 100, 1), avg_next_1m_pct=round(g.f21.mean() * 100, 2)))
save('fall_risk', pd.DataFrame(fr), ['scan_index.py'])

# 5 index signals
R = pd.read_parquet('scan_index.parquet'); R = R[R.sym == 'NIFTY 50']; out = []
for sig, g in R.groupby('signal'):
    d = dict(signal=sig)
    for h, lab in [(5, '1w'), (21, '1m'), (63, '3m')]:
        a = g[(g.h == h) & (g.per == 'all')]; i = g[(g.h == h) & (g.per == 'IS 2015-21')]; o = g[(g.h == h) & (g.per == 'OOS 2022-26')]
        if len(a): d.update({f'n_{lab}': int(a.n.iloc[0]), f'win_{lab}': round(a.win.iloc[0], 1), f'avg_{lab}': round(a.avg.iloc[0], 2), f'fall_{lab}': round(a.fall.iloc[0], 1)})
        if len(i): d[f'is_avg_{lab}'] = round(i.avg.iloc[0], 2)
        if len(o): d[f'oos_avg_{lab}'] = round(o.avg.iloc[0], 2)
    out.append(d)
save('index_signals', pd.DataFrame(out), ['scan_index.py'])

# 6 stock signals
R = pd.read_parquet('scan_stocks.parquet'); out = []
for sig, g in R.groupby('signal'):
    d = dict(signal=sig)
    for h, lab in [(5, '1w'), (21, '1m'), (63, '3m'), (126, '6m')]:
        a = g[(g.h == h) & (g.per == 'all')]; i = g[(g.h == h) & (g.per == 'IS')]; o = g[(g.h == h) & (g.per == 'OOS')]
        if len(a): d.update({f'n_{lab}': int(a.n.iloc[0]), f'win_{lab}': round(a.win.iloc[0], 1), f'avg_{lab}': round(a.avg.iloc[0], 2), f'excess_{lab}': round(a.xavg.iloc[0], 2)})
        if len(i): d[f'is_excess_{lab}'] = round(i.xavg.iloc[0], 2)
        if len(o): d[f'oos_excess_{lab}'] = round(o.xavg.iloc[0], 2)
    out.append(d)
save('stock_signals', pd.DataFrame(out), ['scan_stocks.py', 'eval_stocks.py'])

# 7 equity curves weekly
E = pd.DataFrame({'nifty_buy_hold': (1 + N.close.pct_change().fillna(0)).cumprod()})
for k, f in [('dip_buy_21d', 'eq_A2.parquet'), ('dip_buy_target_stop', 'eq_A5.parquet'), ('breakout_52w', 'eq_B4.parquet')]:
    E[k] = pd.read_parquet(f)['eq'].reindex(E.index).ffill()
ei = pd.read_parquet('eq_index.parquet')
E['bank_risk_off'] = ei['BANKNIFTY risk-off overlay (vol up + breadth <45% + below 50-DMA)'].reindex(E.index)
E['banknifty_buy_hold'] = ei['BANKNIFTY buy & hold'].reindex(E.index)
E = E.resample('W-FRI').last().mul(100).round(1).reset_index().rename(columns={'day': 'date'}); E['date'] = E.date.dt.strftime('%Y-%m-%d')
save('equity_weekly', E, ['strategies.py', 'idx_strats.py'])

# 8 strategy by situation / year
T = pd.read_parquet('trades_stocks.parquet'); T['situation'] = T.entry.map(lambda d: c.event_of(d)[1]).map(KINDLAB)
names = {'A2. same, hold 21d': 'Buy the dip in an uptrend', 'B4. B + only when NIFTY >50-DMA': '52-week-high breakout on volume', 'A5. A + target 5% / stop 10% / max 21d': 'Dip in uptrend, high-win version'}
TT = T[T.strategy.isin(names)].assign(strategy=lambda d: d.strategy.map(names))
agg = dict(trades=('ret', 'size'), win_pct=('ret', lambda s: round((s > 0).mean() * 100, 1)), avg_pct=('ret', lambda s: round(s.mean() * 100, 2)))
save('strategy_by_situation', TT.groupby(['strategy', 'situation']).agg(**agg).reset_index(), ['strategies.py'])
save('strategy_by_year', TT.assign(year=TT.entry.dt.year).groupby(['strategy', 'year']).agg(**agg).reset_index(), ['strategies.py'])

# 9 overnight vs intraday
D = c.load_daily(); ov = []
for sym, lab in [('NIFTY 50', 'NIFTY'), ('NIFTY BANK', 'BANKNIFTY')]:
    n2 = D[D.sym == sym].set_index('day'); on = np.log(n2.open / n2.pc); intr = np.log(n2.close / n2.open)
    for y in sorted(set(n2.index.year)):
        ov.append(dict(index=lab, year=y, overnight_pct=round(on[str(y)].sum() * 100, 1), intraday_pct=round(intr[str(y)].sum() * 100, 1)))
save('overnight_intraday', pd.DataFrame(ov), ['build.py'])

# 10 intraday fall predictors (NIFTY)
n2 = D[D.sym == 'NIFTY 50'].set_index('day').copy(); n2['gap'] = n2.open / n2.pc - 1; n2['oc'] = n2.close / n2.open - 1
n2['fill'] = np.where(n2.gap > 0, n2.low <= n2.pc, n2.high >= n2.pc); n2['rv'] = N.rv20.shift(1).reindex(n2.index); n2['prev'] = N.r.shift(1).reindex(n2.index); n2['trend'] = N.trend.shift(1).reindex(n2.index)
n2['bf'] = n2.oc <= -0.01; g = []
gb = pd.cut(n2.gap * 100, [-10, -1, -0.5, -0.2, 0.2, 0.5, 1, 10], labels=['Gap down >1%', 'Gap down 0.5-1%', 'Gap down 0.2-0.5%', 'Flat open (±0.2%)', 'Gap up 0.2-0.5%', 'Gap up 0.5-1%', 'Gap up >1%'])
for k, x in n2.groupby(gb, observed=True):
    g.append(dict(factor='Opening gap', bucket=str(k), days=len(x), gap_filled_pct=round(x.fill.mean() * 100, 1), closed_above_open_pct=round((x.oc > 0).mean() * 100, 1), intraday_fall_1pct_chance=round(x.bf.mean() * 100, 1)))
for lab, col, bins, labels in [('Volatility (20-day, prior day)', 'rv', [0, 12, 18, 25, 200], ['<12%', '12-18%', '18-25%', '>25%']), ('Previous day move', 'prev', [-1, -0.015, -0.005, 0.005, 0.015, 1], ['Fell >1.5%', 'Fell 0.5-1.5%', 'Flat', 'Rose 0.5-1.5%', 'Rose >1.5%'])]:
    for k, x in n2.groupby(pd.cut(n2[col], bins, labels=labels), observed=True):
        g.append(dict(factor=lab, bucket=str(k), days=len(x), gap_filled_pct=None, closed_above_open_pct=round((x.oc > 0).mean() * 100, 1), intraday_fall_1pct_chance=round(x.bf.mean() * 100, 1)))
for k, x in n2.groupby('trend'):
    if k == 'n/a': continue
    g.append(dict(factor='Trend', bucket=k, days=len(x), gap_filled_pct=None, closed_above_open_pct=round((x.oc > 0).mean() * 100, 1), intraday_fall_1pct_chance=round(x.bf.mean() * 100, 1)))
wk = n2[n2.index.dayofweek < 5]
for k, x in wk.groupby(wk.index.dayofweek):
    g.append(dict(factor='Weekday', bucket=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][k], days=len(x), gap_filled_pct=None, closed_above_open_pct=round((x.oc > 0).mean() * 100, 1), intraday_fall_1pct_chance=round(x.bf.mean() * 100, 1)))
g.append(dict(factor='All days', bucket='All days', days=len(n2), gap_filled_pct=round(n2.fill.mean() * 100, 1), closed_above_open_pct=round((n2.oc > 0).mean() * 100, 1), intraday_fall_1pct_chance=round(n2.bf.mean() * 100, 1)))
save('intraday_nifty', pd.DataFrame(g), ['build.py'])

# 11 ORB by volatility
orb = []
for sym, f in [('NIFTY', 'orb_NIFTY_50.parquet'), ('BANKNIFTY', 'orb_NIFTY_BANK.parquet')]:
    O2 = pd.read_parquet(f); O2 = O2[(O2.M == 30) & O2['first'].notna()].copy(); idxN = N if sym == 'NIFTY' else B
    O2['rvp'] = idxN.rv20.shift(1).reindex(O2.day).values; O2['per'] = np.where(O2.day < '2022', '2015-21', '2022-26')
    O2['vb'] = pd.cut(O2.rvp, [0, 12, 18, 25, 200], labels=['<12%', '12-18%', '18-25%', '>25%'])
    for (p, v), x in O2.groupby(['per', 'vb'], observed=True):
        orb.append(dict(index=sym, period=p, volatility=str(v), days=len(x), win_pct=round((x.pnl > 0).mean() * 100, 1), avg_pct=round(x.pnl.mean(), 3), avg_after_cost_pct=round(x.pnl.mean() - 0.03, 3)))
save('orb', pd.DataFrame(orb), ['intraday.py'])

# JSON with the code that made each file
me = open(__file__).read(); common = open('common.py').read()
for name, (df, scripts) in OUT.items():
    recs = json.loads(df.to_json(orient='records'))
    meta = [{'label': 'export.py', 'value': me, 'type': 'code', 'lang': 'python'}, {'label': 'common.py', 'value': common, 'type': 'code', 'lang': 'python'}]
    meta += [{'label': s, 'value': open(s).read(), 'type': 'code', 'lang': 'python'} for s in scripts]
    meta += [{'label': 'Input', 'value': 'github.com/yatharth77/open-labs nifty-data/ (1-minute CSVs, Kaggle debashis74017), Jan 2015 - 8 Apr 2026'}]
    json.dump({'rows': recs, 'meta': meta}, open(O + name + '.json', 'w'), allow_nan=False)
print('ok', list(OUT))
