import pandas as pd, numpy as np, json
O = 'out/'
st = pd.read_csv(O + 'bo_strategies.csv'); cb = pd.read_csv(O + 'bo_combo.csv')
cb['rule'] = ['Same entry as "Confirm over the next 2 weeks", held 126 sessions.',
              'Confirmation entry, only when the level broken is a 2-year (or older) high.',
              'Breakout on 2x day volume, 2 weeks before it at 1.5x+ volume, then 2 weeks after it still above the level on 1.3x+ volume. Enter day 11.']
cb['exit'] = ['Hold 126 sessions (6 months)', 'Hold 63 sessions (3 months)', 'Hold 63 sessions (3 months)']
cb['random_entry_pct'] = st.random_entry_pct.iloc[0]
cb.loc[cb.id == 'S9', 'setup'] = 'Volume Breakout (final: 2 weeks before + 2 weeks after)'
st = pd.concat([st, cb], ignore_index=True)
order = ['S9', 'S6', 'S7', 'S8', 'S2', 'S2-time126', 'S2-trail50', 'S2-stop10', 'S2-fail', 'S3', 'S4', 'S5', 'S1', 'S0']
st = st.set_index('id').loc[order].reset_index()
cols = ['id', 'setup', 'rule', 'exit', 'trades', 'win_pct', 'avg_pct', 'median_pct', 'random_entry_pct', 'excess_vs_random_pct', 'avg_win_pct', 'avg_loss_pct', 'worst_pct', 'best_pct',
        'is_trades', 'is_win_pct', 'is_avg_pct', 'oos_trades', 'oos_win_pct', 'oos_avg_pct', 'cagr', 'max_dd', 'avg_slots_used']
st = st[cols]; st.to_csv(O + 'bo_strategies_all.csv', index=False)
# equity
E = pd.read_csv(O + 'bo_equity.csv')
s9 = pd.read_parquet('bo_eq_S9.parquet')['eq']; s9 = s9.resample('W-FRI').last().mul(100).round(1)
E['final_s9'] = s9.reindex(pd.to_datetime(E.date)).values
E.to_csv(O + 'bo_equity.csv', index=False)
# trades: S9 + S2
T9 = pd.read_parquet('bo_trades_S9.parquet'); T9['setup'] = 'Volume Breakout (final)'
for col in ('signal', 'entry', 'exit'): T9[col] = T9[col].dt.strftime('%Y-%m-%d')
T9['ret_pct'] = (T9.ret * 100).round(2)
TL = pd.read_csv(O + 'bo_trades.csv')
TL = pd.concat([T9[TL.columns], TL[TL.setup == 'Volume Breakout (main rule)']], ignore_index=True)
TL.to_csv(O + 'bo_trades_all.csv', index=False)
# situation/year/market incl S9
agg = dict(trades=('ret', 'size'), win_pct=('ret', lambda s: round((s > 0).mean() * 100, 1)), avg_pct=('ret', lambda s: round(s.mean() * 100, 2)))
T9r = pd.read_parquet('bo_trades_S9.parquet'); T9r['setup'] = 'Volume Breakout (final)'
for f, key in [('bo_by_situation.csv', 'situation'), ('bo_by_year.csv', 'year'), ('bo_by_market.csv', 'nifty')]:
    old = pd.read_csv(O + f)
    x = T9r.assign(year=T9r.entry.dt.year, nifty=np.where(T9r.nifty_ok, 'NIFTY above 50 & 200-DMA', 'NIFTY weak'))
    new = x.groupby(['setup', key]).agg(**agg).reset_index()
    pd.concat([new, old], ignore_index=True).to_csv(O + f, index=False)
data = {k: json.loads(pd.read_csv(O + f).to_json(orient='records')) for k, f in [
    ('strategies', 'bo_strategies_all.csv'), ('equity', 'bo_equity.csv'), ('vol2wk', 'bo_volume_buckets.csv'), ('dayvol', 'bo_dayvol_buckets.csv'),
    ('tiers', 'bo_tiers.csv'), ('situation', 'bo_by_situation.csv'), ('year', 'bo_by_year.csv'), ('market', 'bo_by_market.csv'), ('trades', 'bo_trades_all.csv')]}
json.dump(data, open(O + 'bo_data.json', 'w'), allow_nan=False)
print({k: len(v) for k, v in data.items()}, round(len(json.dumps(data)) / 1024), 'KB')
print(st[['id', 'trades', 'win_pct', 'avg_pct', 'oos_win_pct', 'oos_avg_pct', 'cagr', 'max_dd']].to_string())
