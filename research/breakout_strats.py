"""Backtest the heavy-volume breakout strategy and variants. Entry next open after the signal day; 0.25% round-trip cost."""
import pandas as pd, numpy as np, common as c, json
S = pd.read_parquet('breakout_eval.parquet').sort_values(['sym', 'day']).reset_index(drop=True)
N = pd.read_parquet('idx_NIFTY_50.parquet'); DAYS = N.index
COST = 0.0025
KIND = {'normal': 'No major event', 'boom': 'Post-crash boom', 'correction': 'Correction (FPI selling, rate hikes)',
        'crisis': 'Crisis (COVID crash, China 2015, IL&FS)', 'shock': 'One-off shock (demonetisation, Adani, tariffs)',
        'war': 'War (Russia-Ukraine, Op Sindoor, US-Iran)', 'event': 'Election / budget day'}
mm = S.tier.isin(['6-month (multi-month)', '1-year']); my = S.tier.eq('2-year+ (multi-year)'); anyb = S.tier.ne('')
base = anyb & (S.age >= 20)
heavy = S.vr >= 2; wk2 = S.v10 >= 1.5; healthy = S.n_ok
SETUPS = {
 'S0': ('Breakout only', 'Close above the 6-month, 1-year or 2-year high (high set 20+ sessions earlier). No volume rule.', base),
 'S1': ('+ heavy breakout-day volume', 'Breakout day volume at least 2x its 50-day average.', base & heavy),
 'S2': ('Volume Breakout (main rule)', 'Breakout on 2x day volume, and the 2 weeks up to the breakout average at least 1.5x normal volume.', base & heavy & wk2),
 'S3': ('Main rule, multi-year breakouts only', 'S2 where the level broken is a 2-year (or older) high.', base & heavy & wk2 & my),
 'S4': ('Main rule, multi-month breakouts only', 'S2 where the level broken is a 6- to 12-month high.', base & heavy & wk2 & mm),
 'S5': ('Main rule + NIFTY healthy', 'S2, only when NIFTY is above its 50- and 200-day averages.', base & heavy & wk2 & healthy),
}
def sim(sig, exit_rule, H=63, gap=63):
    """sig: boolean array on S rows. exit_rule: 'time' | 'stop10' | 'fail' | 'trail50'. Returns trades."""
    rows = []
    S['_m'] = sig
    for (sym, seg), x in S.groupby(['sym', 'seg'], sort=False):
        x = x.reset_index(drop=True); idx = np.where(x._m.values)[0]
        o = x.open.values; cl = x.close.values; lo = x.low.values; d = x.day.values; d50 = x.d50.values; lvl = x.lvl.values
        busy = -1; last = -10**9
        for i in idx:
            if i <= busy or i - last < gap or i + 1 >= len(x): continue
            e = o[i + 1]; maxh = H if exit_rule in ('time', 'stop10', 'fail') else 252
            j_end = min(i + maxh, len(x) - 1)
            if j_end - i < min(maxh, 21): continue
            ex = cl[j_end]; je = j_end; why = 'time'
            for j in range(i + 1, j_end + 1):
                if exit_rule == 'stop10' and lo[j] <= e * 0.90:
                    ex = min(o[j], e * 0.90) if j > i + 1 else e * 0.90; je = j; why = 'stop'; break
                if exit_rule == 'fail' and cl[j] < lvl[i] * 0.97:
                    if j + 1 < len(x): ex = o[j + 1]; je = j + 1
                    else: ex = cl[j]; je = j
                    why = 'failed breakout'; break
                if exit_rule == 'trail50' and j - i >= 10 and cl[j] < d50[j]:
                    if j + 1 < len(x): ex = o[j + 1]; je = j + 1
                    else: ex = cl[j]; je = j
                    why = 'below 50-DMA'; break
            if why == 'time' and j_end - i < maxh and exit_rule != 'trail50': continue
            rows.append(dict(sym=sym, signal=pd.Timestamp(d[i]), entry=pd.Timestamp(d[i + 1]), exit=pd.Timestamp(d[je]), ret=ex / e - 1 - COST,
                             why=why, hold=int(je - i), tier=x.tier.iat[i], vr=round(x.vr.iat[i], 2), v10=round(x.v10.iat[i], 2),
                             level=round(lvl[i], 2), entry_px=round(e, 2), exit_px=round(ex, 2), nifty_ok=bool(x.n_ok.iat[i]),
                             situation=KIND[x.situation.iat[i]]))
            busy = je; last = i
    return pd.DataFrame(rows)
def portfolio(T, slots=10):
    T = T.sort_values(['entry', 'sym']).reset_index(drop=True); open_ = []; keep = []
    for _, t in T.iterrows():
        open_ = [x for x in open_ if x >= t.entry]
        if len(open_) < slots: open_.append(t.exit); keep.append(True)
        else: keep.append(False)
    P = T[keep]; pnl = P.groupby('exit').ret.sum() / slots
    eq = (1 + pnl.reindex(DAYS).fillna(0)).cumprod()
    yrs = (DAYS[-1] - DAYS[0]).days / 365.25
    # time invested
    inv = pd.Series(0, index=DAYS)
    for _, t in P.iterrows(): inv[t.entry:t.exit] += 1
    return eq, dict(port_trades=len(P), cagr=round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1), max_dd=round((eq / eq.cummax() - 1).min() * 100, 1),
                    avg_slots_used=round(inv.mean(), 1))
def summ(T):
    is_ = T[T.entry < '2022']; oo = T[T.entry >= '2022']
    return dict(trades=len(T), win_pct=round((T.ret > 0).mean() * 100, 1), avg_pct=round(T.ret.mean() * 100, 2), median_pct=round(T.ret.median() * 100, 2),
                avg_win_pct=round(T[T.ret > 0].ret.mean() * 100, 1), avg_loss_pct=round(T[T.ret <= 0].ret.mean() * 100, 1),
                worst_pct=round(T.ret.min() * 100, 1), best_pct=round(T.ret.max() * 100, 1), avg_hold=round(T.hold.mean()),
                is_trades=len(is_), is_win_pct=round((is_.ret > 0).mean() * 100, 1), is_avg_pct=round(is_.ret.mean() * 100, 2),
                oos_trades=len(oo), oos_win_pct=round((oo.ret > 0).mean() * 100, 1), oos_avg_pct=round(oo.ret.mean() * 100, 2))
if __name__ == '__main__':
    out = []; trades = {}; eqs = {}
    # random-entry benchmark for 63 sessions (same cost)
    rnd = S.f63.dropna(); print('random 63d', round(rnd.mean() * 100, 2), round((rnd > 0).mean() * 100, 1))
    for k, (nm, rule, m) in SETUPS.items():
        T = sim(m.values, 'time'); eq, p = portfolio(T)
        out.append(dict(id=k, setup=nm, rule=rule, exit='Hold 63 sessions (3 months)', **summ(T), **p)); trades[k] = T; eqs[k] = eq
        print(k, out[-1])
    m = SETUPS['S2'][2].values
    for ek, ename in [('time126', 'Hold 126 sessions (6 months)'), ('stop10', '10% stop, else hold 63'), ('fail', 'Exit if it closes 3% back below the breakout level, else hold 63'), ('trail50', 'Exit on a close below the 50-day average (max 1 year)')]:
        rule = 'time' if ek == 'time126' else ek
        T = sim(m, rule, H=126 if ek == 'time126' else 63); eq, p = portfolio(T)
        out.append(dict(id='S2-' + ek, setup='Volume Breakout (main rule)', rule=SETUPS['S2'][1], exit=ename, **summ(T), **p)); trades['S2-' + ek] = T; eqs['S2-' + ek] = eq
        print('S2', ek, out[-1])
    # post-breakout confirmation reading: watch 2 weeks after the breakout, enter on day 11 if price held and volume stayed >=1.3x
    conf = (base & heavy & (S.c_next10 > S.lvl) & (S.v_next10 >= 1.3)).values
    S2 = S.copy()
    # shift the signal 10 sessions later within each stock so entry is at open of t+11
    sig = pd.Series(conf, index=S.index).groupby([S.sym, S.seg]).shift(10).fillna(False).astype(bool).values
    lv = S.groupby(['sym', 'seg']).lvl.shift(10); S['lvl_save'] = S.lvl; S['lvl'] = np.where(sig, lv, S.lvl)
    tier_save = S.tier.copy(); S['tier'] = np.where(sig, S.groupby(['sym', 'seg']).tier.shift(10), S.tier)
    T = sim(sig, 'time'); eq, p = portfolio(T)
    S['lvl'] = S.lvl_save; S['tier'] = tier_save
    out.append(dict(id='S6', setup='Confirm over the next 2 weeks', rule='Breakout on 2x volume; enter 10 sessions later only if price is still above the breakout level and volume in those 2 weeks averaged 1.3x normal.', exit='Hold 63 sessions (3 months)', **summ(T), **p))
    trades['S6'] = T; eqs['S6'] = eq; print('S6', out[-1])
    R = pd.DataFrame(out); R.to_csv('out/bo_strategies.csv', index=False)
    allT = pd.concat([t.assign(strategy=k) for k, t in trades.items()]); allT.to_parquet('bo_trades.parquet')
    E = pd.DataFrame(eqs); E['nifty'] = (1 + N.close.pct_change().fillna(0)).cumprod()
    E.to_parquet('bo_equity.parquet')
