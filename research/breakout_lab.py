"""Heavy-volume breakout strategy with two-week volume tracking.
Signals on day t (close); entry next open. Uses cleaned daily data from common.load_daily()."""
import pandas as pd, numpy as np, common as c
D = c.load_daily(); N = pd.read_parquet('idx_NIFTY_50.parquet')
S = D[~D.sym.isin(c.IDX)].sort_values(['sym', 'day']).copy()
LB = {'6m': 126, '1y': 252, '2y': 504, '3y': 756}
def per_group(x):
    h = x.high.values; cl = x.close.values; v = x.volume.values.astype(float); o = x.open.values; lo = x.low.values; n = len(x)
    out = {}
    hs = pd.Series(h)
    for k, L in LB.items():
        out[f'hh{k}'] = hs.shift(1).rolling(L, min_periods=L).max().values
        # age (sessions) of that prior high
        idx = hs.shift(1).rolling(L, min_periods=L).apply(lambda a: len(a) - 1 - np.argmax(a[::-1]) if False else (len(a) - 1 - int(np.argmax(a))), raw=True).values
        out[f'age{k}'] = idx + 1  # sessions between that high and t
    vs = pd.Series(v)
    v50 = vs.shift(1).rolling(50, min_periods=50).mean()
    base = vs.shift(10).rolling(50, min_periods=50).mean()          # normal volume before the 2-week window
    out['vr'] = (vs / v50).values
    out['v10'] = (vs.rolling(10).mean() / base).values              # 2 weeks up to & incl. breakout day
    out['vpre'] = (vs.shift(1).rolling(9).mean() / base).values     # the 9 days before breakout
    r = pd.Series(cl).pct_change()
    up = (vs.where(r > 0, 0)).rolling(10).sum(); dn = (vs.where(r < 0, 0)).rolling(10).sum()
    out['updn10'] = (up / dn.replace(0, np.nan)).values
    out['clpos'] = ((cl - lo) / np.where(h - lo > 0, h - lo, np.nan))
    out['d50'] = pd.Series(cl).rolling(50).mean().values
    out['d200'] = pd.Series(cl).rolling(200).mean().values
    out['r5'] = (pd.Series(cl) / pd.Series(cl).shift(5) - 1).values
    # forward window after 10 more sessions (post-breakout confirmation)
    out['v_next10'] = (vs.shift(-10).rolling(10).mean() / base).values  # avg vol t+1..t+10 vs base
    out['c_next10'] = pd.Series(cl).shift(-10).values
    out['minlow_next10'] = pd.Series(lo)[::-1].rolling(10, min_periods=10).min()[::-1].shift(-1).values
    return pd.DataFrame(out, index=x.index)
F = S.groupby(['sym', 'seg'], group_keys=False).apply(per_group)
S = S.join(F)
S = S.join(N[['trend', 'close', 'dma50', 'dma200', 'rv20']].rename(columns={'close': 'nclose', 'dma50': 'nd50', 'dma200': 'nd200'}), on='day')
S['n_ok'] = (S.nclose > S.nd200) & (S.nclose > S.nd50)
S['situation'] = [c.event_of(d)[1] for d in S.day]
S.to_parquet('breakout_feat.parquet')
print(S.shape, S[['vr', 'v10', 'vpre', 'updn10', 'age6m', 'age2y']].describe().T[['count', 'mean', '50%']])
