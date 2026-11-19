import pandas as pd, numpy as np, common as c
D=c.load_daily(); n=pd.read_parquet('idx_NIFTY_50.parquet')
S=D[~D.sym.isin(c.IDX)].copy()
g=S.groupby(['sym','seg'])
S['r']=S.close/S.pc-1; S['gap']=S.open/S.pc-1
for h in (1,5,21,63,126):
    S[f'f{h}']=g.close.shift(-h)/S.close-1
for k in (20,50,252):
    S[f'phi{k}']=g.high.transform(lambda x:x.shift(1).rolling(k,min_periods=k).max())
    S[f'plo{k}']=g.low.transform(lambda x:x.shift(1).rolling(k,min_periods=k).min())
S['d50']=g.close.transform(lambda x:x.rolling(50).mean()); S['d200']=g.close.transform(lambda x:x.rolling(200).mean())
S['v50']=g.volume.transform(lambda x:x.shift(1).rolling(50).mean()); S['vr']=S.volume/S.v50
S['rng20']=g.high.transform(lambda x:x.shift(1).rolling(20).max())/g.low.transform(lambda x:x.shift(1).rolling(20).min())-1
S['r5']=g.close.transform(lambda x:x/x.shift(5)-1); S['r126']=g.close.transform(lambda x:x/x.shift(126)-1)
S['r252']=g.close.transform(lambda x:x/x.shift(252)-1)
S['dd252']=S.close/S.phi252-1
# universe average forward return per day (same survivorship) -> excess
for h in (1,5,21,63,126):
    S[f'x{h}']=S[f'f{h}']-S.groupby('day')[f'f{h}'].transform('mean')
S=S.join(n[['trend','vol','event','event_kind','dma50','close']].rename(columns={'close':'nclose'}),on='day')
S['n_above50']=S.nclose>S.dma50
S.to_parquet('stocks_feat.parquet')
