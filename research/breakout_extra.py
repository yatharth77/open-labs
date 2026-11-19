import pandas as pd, numpy as np, json, breakout_strats as B
S = B.S; T = pd.read_parquet('bo_trades.parquet')
# excess vs a random stock bought the same day and held the same 63 sessions
f63 = S.set_index(['sym', 'day']).f63; u63 = S.set_index(['sym', 'day']).u63
def exc(T, entry_shift=0):
    keys = list(zip(T.sym, T.signal)); return u63.reindex(keys).values
out = {}
# 1. volume buckets: breakouts (any tier, base>=20, day vol>=2x) by 2-week volume ratio
base = (B.base & B.heavy)
x = S[base & S.f63.notna()].copy()
# one per stock per 63 sessions
keep = []; last = {}
for i, r in x.iterrows():
    k = (r.sym, r.seg)
    if k not in last or (i - last[k]) >= 63: keep.append(i); last[k] = i
x = x.loc[keep]
x['vb'] = pd.cut(x.v10, [0, 1.0, 1.25, 1.5, 2.0, 3.0, 99], labels=['<1.0x', '1.0-1.25x', '1.25-1.5x', '1.5-2x', '2-3x', '>3x'])
vb = x.groupby('vb', observed=True).agg(breakouts=('f63', 'size'), win_pct=('f63', lambda s: round((s > 0).mean() * 100, 1)), avg_3m_pct=('f63', lambda s: round(s.mean() * 100, 2)),
     excess_pct=('u63', lambda s: round(s.mean() * 100, 2)), is_excess_pct=('u63', lambda s: np.nan), ).reset_index().rename(columns={'vb': 'volume_2wk'})
vb['is_excess_pct'] = [round(x[(x.vb == b) & (x.day < '2022')].u63.mean() * 100, 2) for b in vb.volume_2wk]
vb['oos_excess_pct'] = [round(x[(x.vb == b) & (x.day >= '2022')].u63.mean() * 100, 2) for b in vb.volume_2wk]
vb['volume_2wk'] = vb.volume_2wk.astype(str); print(vb)
vb.to_csv('out/bo_volume_buckets.csv', index=False)
# breakout-day volume buckets (2-week >=1.5)
y = S[B.base & B.wk2 & S.f63.notna()].copy(); keep = []; last = {}
for i, r in y.iterrows():
    k = (r.sym, r.seg)
    if k not in last or (i - last[k]) >= 63: keep.append(i); last[k] = i
y = y.loc[keep]; y['db'] = pd.cut(y.vr, [0, 1, 1.5, 2, 3, 5, 99], labels=['<1x', '1-1.5x', '1.5-2x', '2-3x', '3-5x', '>5x'])
db = y.groupby('db', observed=True).agg(breakouts=('f63', 'size'), win_pct=('f63', lambda s: round((s > 0).mean() * 100, 1)), avg_3m_pct=('f63', lambda s: round(s.mean() * 100, 2)), excess_pct=('u63', lambda s: round(s.mean() * 100, 2))).reset_index().rename(columns={'db': 'breakout_day_volume'})
db['breakout_day_volume'] = db.breakout_day_volume.astype(str); print(db); db.to_csv('out/bo_dayvol_buckets.csv', index=False)
# tier table (main rule)
tt = []
for tier in ['6-month (multi-month)', '1-year', '2-year+ (multi-year)']:
    for nm, m in [('Breakout only', B.base), ('Main rule', B.base & B.heavy & B.wk2)]:
        z = S[m & (S.tier == tier) & S.f63.notna()]; keep = []; last = {}
        for i, r in z.iterrows():
            k = (r.sym, r.seg)
            if k not in last or (i - last[k]) >= 63: keep.append(i); last[k] = i
        z = z.loc[keep]
        tt.append(dict(level_broken=tier, setup=nm, breakouts=len(z), win_pct=round((z.f63 > 0).mean() * 100, 1), avg_3m_pct=round(z.f63.mean() * 100, 2), excess_pct=round(z.u63.mean() * 100, 2), avg_6m_pct=round(z.f126.mean() * 100, 2)))
tt = pd.DataFrame(tt); print(tt); tt.to_csv('out/bo_tiers.csv', index=False)
# situation & year for S2 and S6
agg = dict(trades=('ret', 'size'), win_pct=('ret', lambda s: round((s > 0).mean() * 100, 1)), avg_pct=('ret', lambda s: round(s.mean() * 100, 2)))
name = {'S2': 'Volume Breakout (main rule)', 'S6': 'Confirm over the next 2 weeks', 'S0': 'Breakout only'}
TT = T[T.strategy.isin(name)].copy(); TT['setup'] = TT.strategy.map(name)
TT.groupby(['setup', 'situation']).agg(**agg).reset_index().to_csv('out/bo_by_situation.csv', index=False)
TT.assign(year=TT.entry.dt.year).groupby(['setup', 'year']).agg(**agg).reset_index().to_csv('out/bo_by_year.csv', index=False)
TT.assign(nifty=np.where(TT.nifty_ok, 'NIFTY above 50 & 200-DMA', 'NIFTY weak')).groupby(['setup', 'nifty']).agg(**agg).reset_index().to_csv('out/bo_by_market.csv', index=False)
print(pd.read_csv('out/bo_by_situation.csv')); print(pd.read_csv('out/bo_by_market.csv'))
# excess for strategies
st = pd.read_csv('out/bo_strategies.csv')
ex = []
for k in st.id:
    t = T[T.strategy == k]
    if k == 'S6':  # entry 10 sessions after the breakout: compare with random stock bought on the same entry day held 63
        ks = list(zip(t.sym, t.signal)); e = u63.reindex(ks)
    else:
        e = u63.reindex(list(zip(t.sym, t.signal))) if k in ('S0', 'S1', 'S2', 'S3', 'S4', 'S5') else pd.Series(dtype=float)
    ex.append(round(e.mean() * 100, 2) if len(e) else None)
st['excess_vs_random_pct'] = ex
st['random_entry_pct'] = round(S.f63.mean() * 100, 2)
st.to_csv('out/bo_strategies.csv', index=False); print(st[['id', 'trades', 'win_pct', 'avg_pct', 'excess_vs_random_pct', 'oos_avg_pct', 'cagr', 'max_dd']])
# trade list for the main and confirm strategies
TL = T[T.strategy.isin(['S2', 'S6'])].copy(); TL['setup'] = TL.strategy.map(name)
TL['ret_pct'] = (TL.ret * 100).round(2)
for col in ('signal', 'entry', 'exit'): TL[col] = TL[col].dt.strftime('%Y-%m-%d')
TL[['setup', 'sym', 'signal', 'entry', 'exit', 'tier', 'vr', 'v10', 'level', 'entry_px', 'exit_px', 'ret_pct', 'situation']].sort_values(['setup', 'entry']).to_csv('out/bo_trades.csv', index=False)
# equity weekly
E = pd.read_parquet('bo_equity.parquet')[['nifty', 'S0', 'S2', 'S2-time126', 'S6']].ffill()
E = E.resample('W-FRI').last().mul(100).round(1).reset_index().rename(columns={'day': 'date'}); E['date'] = E.date.dt.strftime('%Y-%m-%d')
E.columns = ['date', 'nifty', 'breakout_only', 'main_63d', 'main_126d', 'confirm_2wk']; E.to_csv('out/bo_equity.csv', index=False)
print(E.tail(1))
