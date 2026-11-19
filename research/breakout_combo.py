import pandas as pd, numpy as np, breakout_strats as B
S = B.S
def confirm_sig(extra):
    conf = (B.base & B.heavy & extra & (S.c_next10 > S.lvl) & (S.v_next10 >= 1.3)).values
    sig = pd.Series(conf, index=S.index).groupby([S.sym, S.seg]).shift(10).fillna(False).astype(bool).values
    return sig
rows = []
lvl0, tier0 = S.lvl.copy(), S.tier.copy()
S['tier'] = np.where(True, S.groupby(['sym', 'seg']).tier.shift(10), S.tier)
for k, nm, extra, H in [('S7', 'Confirm 2 weeks, hold 6 months', pd.Series(True, index=S.index), 126),
                        ('S8', 'Confirm 2 weeks, multi-year breakouts only', B.my, 63),
                        ('S9', 'Confirm 2 weeks + 2-week pre-volume 1.5x', B.wk2, 63)]:
    T = B.sim(confirm_sig(extra), 'time', H=H); eq, p = B.portfolio(T)
    r = dict(id=k, setup=nm, **B.summ(T), **p); rows.append(r); print(r)
    T.assign(strategy=k).to_parquet(f'bo_trades_{k}.parquet'); eq.to_frame('eq').to_parquet(f'bo_eq_{k}.parquet')
S['tier'] = tier0
pd.DataFrame(rows).to_csv('out/bo_combo.csv', index=False)
