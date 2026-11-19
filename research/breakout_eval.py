import pandas as pd, numpy as np
S = pd.read_parquet('breakout_feat.parquet').sort_values(['sym', 'day']).reset_index(drop=True)
g = S.groupby(['sym', 'seg'])
S['o1'] = g.open.shift(-1)
for h in (21, 63, 126):
    S[f'f{h}'] = g.close.shift(-h) / S.o1 - 1 - 0.0025
    S[f'u{h}'] = S[f'f{h}'] - S.groupby('day')[f'f{h}'].transform('mean')   # vs random stock bought same day
S['tier'] = np.select([S.close > S.hh2y, S.close > S.hh1y, S.close > S.hh6m], ['2-year+ (multi-year)', '1-year', '6-month (multi-month)'], '')
S['age'] = np.select([S.close > S.hh2y, S.close > S.hh1y, S.close > S.hh6m], [S.age2y, S.age1y, S.age6m], np.nan)
S['lvl'] = np.select([S.close > S.hh2y, S.close > S.hh1y, S.close > S.hh6m], [S.hh2y, S.hh1y, S.hh6m], np.nan)
S.to_parquet('breakout_eval.parquet')
def dedupe(m, gap=63):
    keep = np.zeros(len(S), bool); last = {}
    for i in np.where(m)[0]:
        k = (S.sym.iat[i], S.seg.iat[i]); d = i
        if k not in last or d - last[k] >= gap: keep[i] = True; last[k] = d
    return keep
def stat(x, h=63):
    x = x.dropna(subset=[f'f{h}'])
    if len(x) < 8: return None
    f = x[f'f{h}']; u = x[f'u{h}']
    return dict(n=len(x), win=round((f > 0).mean() * 100, 1), avg=round(f.mean() * 100, 2), med=round(f.median() * 100, 2), excess=round(u.mean() * 100, 2),
                beat=round((u > 0).mean() * 100, 1), is_ex=round(x[x.day < '2022'][f'u{h}'].mean() * 100, 2), oos_ex=round(x[x.day >= '2022'][f'u{h}'].mean() * 100, 2),
                oos_n=int((x.day >= '2022').sum()), worst=round(f.min() * 100, 1))
if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    rows = []
    base = S.f63.notna()
    print('random stock, 63d:', round(S.f63.mean() * 100, 2), 'win', round((S.f63 > 0).mean() * 100, 1))
    for tier in ['6-month (multi-month)', '1-year', '2-year+ (multi-year)', 'any']:
        T = (S.tier != '') if tier == 'any' else (S.tier == tier)
        for agemin in (1, 20):
            for vth in (0, 1.5, 2, 3):
                for v10 in (0, 1.3, 1.6, 2.0):
                    m = T & (S.age >= agemin) & (S.vr >= vth) & (S.v10 >= v10)
                    k = dedupe(m.values)
                    s = stat(S[k])
                    if s: rows.append(dict(tier=tier, base_age=agemin, bo_vol=vth, vol_2wk=v10, **s))
    R = pd.DataFrame(rows); R.to_parquet('bo_grid.parquet')
    print(R[R.base_age == 20].to_string())
