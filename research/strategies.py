import pandas as pd, numpy as np, common as c, json
S=pd.read_parquet('stocks_feat.parquet').sort_values(['sym','day'])
N=pd.read_parquet('idx_NIFTY_50.parquet'); B=pd.read_parquet('idx_NIFTY_BANK.parquet')
days=N.index
COST_DELIV=0.0025   # round trip, delivery equity (STT 0.1%x2 + exch/stamp/GST/slippage)
def ev(day):
    return c.event_of(day)[0]
# ---------------- event-level stock strategy: entry next open, exit after H sessions at close
def stock_trades(mask,H,tp=None,sl=None,name=''):
    rows=[]
    S['_m']=mask.values
    for (sym,seg),x in S.groupby(['sym','seg'],sort=False):
        x=x.reset_index(drop=True); idx=np.where(x._m.values)[0]; busy=-1
        o=x.open.values; cl=x.close.values; hi=x.high.values; lo=x.low.values; dd=x.day.values
        for i in idx:
            if i<=busy or i+1>=len(x): continue
            e=o[i+1]; j_end=min(i+H,len(x)-1); ex=cl[j_end]; je=j_end; why='time'
            for j in range(i+1,j_end+1):
                if sl is not None and lo[j]<=e*(1-sl): ex=min(o[j],e*(1-sl)) if j>i+1 else e*(1-sl); je=j; why='stop'; break
                if tp is not None and hi[j]>=e*(1+tp): ex=max(o[j],e*(1+tp)) if j>i+1 else e*(1+tp); je=j; why='target'; break
            if j_end-i<H and why=='time': continue  # incomplete
            rows.append(dict(sym=sym,sig=dd[i],entry=dd[i+1],exit=dd[je],ret=ex/e-1-COST_DELIV,why=why,hold=je-i))
            busy=je
    T=pd.DataFrame(rows); T['strategy']=name; return T
def portfolio(T,slots=10,cap=1.0):
    """equal slots; take trades in signal order if a slot is free; daily mark-to-market approximated by realising at exit."""
    T=T.sort_values(['entry','sym']).reset_index(drop=True)
    open_=[]; taken=[]
    for _,t in T.iterrows():
        open_=[x for x in open_ if x>=t.entry]
        if len(open_)<slots: open_.append(t.exit); taken.append(True)
        else: taken.append(False)
    P=T[taken].copy()
    # equity: each trade uses 1/slots of equity at entry (compounding approx by exit-date ordering)
    eq=pd.Series(0.0,index=days); 
    pnl=P.groupby('exit').ret.sum()/slots
    eqc=(1+pnl.reindex(days).fillna(0)).cumprod()
    return P,eqc
def stats_eq(eq):
    r=eq.pct_change().fillna(0); yrs=(eq.index[-1]-eq.index[0]).days/365.25
    cagr=eq.iloc[-1]**(1/yrs)-1; dd=(eq/eq.cummax()-1).min()
    return dict(cagr=cagr*100,maxdd=dd*100)
def summ(T):
    return dict(trades=len(T),win=(T.ret>0).mean()*100,avg=T.ret.mean()*100,med=T.ret.median()*100,
                worst=T.ret.min()*100,best=T.ret.max()*100,
                is_avg=T[T.entry<'2022'].ret.mean()*100,is_win=(T[T.entry<'2022'].ret>0).mean()*100,
                oos_avg=T[T.entry>='2022'].ret.mean()*100,oos_win=(T[T.entry>='2022'].ret>0).mean()*100,
                oos_n=int((T.entry>='2022').sum()))
OUT={}
bull=S.trend.eq('bull (>200DMA)')
cands={
 'A. Dip in uptrend: 5-day fall >12%, stock >200DMA, hold 10d':(lambda: (S.r5<-0.12)&(S.close>S.d200),10,None,None),
 'A2. same, hold 21d':(lambda: (S.r5<-0.12)&(S.close>S.d200),21,None,None),
 'A3. Dip in uptrend: day fall >5%, stock >200DMA, hold 10d':(lambda: (S.r<-0.05)&(S.close>S.d200),10,None,None),
 'A4. A3 + target 6% / stop 8% / max 15d':(lambda: (S.r<-0.05)&(S.close>S.d200),15,0.06,0.08),
 'A5. A + target 5% / stop 10% / max 21d':(lambda: (S.r5<-0.12)&(S.close>S.d200),21,0.05,0.10),
 'B. 52w-high breakout + vol >2x, NIFTY >200DMA, hold 63d':(lambda: (S.close>S.phi252)&(S.vr>2)&bull,63,None,None),
 'B2. B with 10% stop, hold 63d':(lambda: (S.close>S.phi252)&(S.vr>2)&bull,63,None,0.10),
 'B3. 52w-high breakout + vol >2x, NIFTY <200DMA, hold 63d':(lambda: (S.close>S.phi252)&(S.vr>2)&~bull&S.trend.ne('n/a'),63,None,None),
 'A6. A2 + only when NIFTY >50-DMA':(lambda: (S.r5<-0.12)&(S.close>S.d200)&S.n_above50,21,None,None),
 'A7. A5 + only when NIFTY >50-DMA':(lambda: (S.r5<-0.12)&(S.close>S.d200)&S.n_above50,21,0.05,0.10),
 'B4. B + only when NIFTY >50-DMA':(lambda: (S.close>S.phi252)&(S.vr>2)&bull&S.n_above50,63,None,None),
 'C. 20d breakout from tight base (<8%) + vol>1.5x, hold 63d':(lambda: (S.close>S.phi20)&(S.rng20<0.08)&(S.vr>1.5),63,None,None),
}
allT=[]; summary={}
for name,(fn,H,tp,sl) in cands.items():
    T=stock_trades(fn().fillna(False),H,tp,sl,name); allT.append(T)
    P,eq=portfolio(T)
    s=summ(T); s.update({f'port_{k}':v for k,v in stats_eq(eq).items()}); s['port_trades']=len(P)
    summary[name]=s; eq.to_frame('eq').to_parquet(f'eq_{name.split(".")[0]}.parquet')
    print(name, {k:round(v,1) if isinstance(v,float) else v for k,v in s.items()})
T=pd.concat(allT); T['event']=T.entry.map(ev); T.to_parquet('trades_stocks.parquet')
pd.DataFrame(summary).T.to_parquet('summary_stocks.parquet')
