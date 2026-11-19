import pandas as pd, numpy as np, common as c
N=pd.read_parquet('idx_NIFTY_50.parquet'); B=pd.read_parquet('idx_NIFTY_BANK.parquet')
D=c.load_daily()
FUT=0.0003  # index futures round trip incl. slippage (approx)
def eqstats(r,label):
    eq=(1+r.fillna(0)).cumprod(); yrs=(eq.index[-1]-eq.index[0]).days/365.25
    dd=(eq/eq.cummax()-1)
    return dict(name=label,cagr=(eq.iloc[-1]**(1/yrs)-1)*100,maxdd=dd.min()*100,expo=(r!=0).mean()*100,
                is_cagr=((1+r[:'2021'].fillna(0)).prod()**(1/7)-1)*100,oos_cagr=((1+r['2022':].fillna(0)).prod()**(1/((r.index[-1]-pd.Timestamp('2022-01-01')).days/365.25))-1)*100), eq
res=[];EQ={}
for sym,n in [('NIFTY',N),('BANKNIFTY',B)]:
    r=n.close.pct_change()
    s,eq=eqstats(r,f'{sym} buy & hold'); res.append(s); EQ[s['name']]=eq
    # trend filter: hold if yesterday close > 200DMA (signal at close, act next day close-to-close)
    sig=(n.close>n.dma200).shift(1).fillna(False)
    rr=r.where(sig,0); chg=sig.astype(int).diff().abs().fillna(0); rr=rr-chg*0.0005
    s,eq=eqstats(rr,f'{sym} trend filter (hold only above 200-DMA)'); res.append(s); EQ[s['name']]=eq
    # fall-risk overlay: exit when (calm->rising vol) AND breadth weakening: rv20>15 & br50<50 & close<dma50; re-enter when close>dma50 and br50>50
    state=[];inpos=True
    for d,row in n.iterrows():
        if inpos and row.rv20>15 and (row.br50<45) and row.close<row.dma50: inpos=False
        elif not inpos and row.close>row.dma50 and row.br50>55: inpos=True
        state.append(inpos)
    sig=pd.Series(state,index=n.index).shift(1).fillna(True)
    rr=r.where(sig,0)-sig.astype(int).diff().abs().fillna(0)*0.0005
    s,eq=eqstats(rr,f'{sym} risk-off overlay (vol up + breadth <45% + below 50-DMA)'); res.append(s); EQ[s['name']]=eq
    # buy fear: enter when rv20>25 or week<-4% or br50<15; hold 63 sessions
    fear=((n.rv20>25)|(n.wk<-0.04)|(n.br50<15))
    hold=pd.Series(False,index=n.index); i=0; idx=np.where(fear.values)[0]; until=-1
    for k in idx:
        if k>until: until=k+63
        hold.iloc[k+1:min(until+1,len(n))]=True
    rr=r.where(hold,0)
    s,eq=eqstats(rr,f'{sym} buy fear only (vol>25%, week<-4% or breadth<15%), hold 3 months'); res.append(s); EQ[s['name']]=eq
    # overnight: buy close, sell next open
    dd=D[D.sym==('NIFTY 50' if sym=='NIFTY' else 'NIFTY BANK')].set_index('day')
    on=(dd.open/dd.pc-1-FUT)
    s,eq=eqstats(on.reindex(n.index),f'{sym} overnight only (buy 15:29, sell 9:15), futures cost'); res.append(s); EQ[s['name']]=eq
    # ORB30 with prior-day vol filter
    O=pd.read_parquet(f'orb_{"NIFTY_50" if sym=="NIFTY" else "NIFTY_BANK"}.parquet'); O=O[O.M==30].set_index('day')
    rvp=n.rv20.shift(1).reindex(O.index)
    for thr in (0,18):
        x=O.pnl.where(O['first'].notna(),0)/100; x=x.where(rvp>thr,0); x=x-(x!=0)*FUT
        s,eq=eqstats(x.reindex(n.index),f'{sym} 30-min opening-range breakout'+(f' (only if prior vol >{thr}%)' if thr else ' (every day)'))
        t=x[x!=0]; s['trades']=len(t); s['win']=(t>0).mean()*100; s['avg_trade']=t.mean()*100
        res.append(s); EQ[s['name']]=eq
R=pd.DataFrame(res); pd.set_option('display.width',250); print(R.round(1).to_string())
R.to_parquet('summary_index.parquet'); pd.DataFrame(EQ).to_parquet('eq_index.parquet')
