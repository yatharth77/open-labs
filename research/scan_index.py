import pandas as pd, numpy as np, common as c
D=c.load_daily()
def rsi(s,n):
    d=s.diff(); u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); v=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+u/v)
# breadth from stocks
S=D[~D.sym.isin(c.IDX)].copy()
S['d50']=S.groupby(['sym','seg']).close.transform(lambda x:x.rolling(50).mean())
S['d200']=S.groupby(['sym','seg']).close.transform(lambda x:x.rolling(200).mean())
br=S.dropna(subset=['d50']).assign(a50=lambda x:x.close>x.d50).groupby('day').a50.mean()*100
br200=S.dropna(subset=['d200']).assign(a=lambda x:x.close>x.d200).groupby('day').a.mean()*100
cnt=S.dropna(subset=['d50']).groupby('day').size()
br[cnt<40]=np.nan; br200[S.dropna(subset=['d200']).groupby('day').size()<40]=np.nan
def build(sym):
    n=D[D.sym==sym].set_index('day')[['open','high','low','close','pc']].copy()
    n=c.label_regimes(n)
    cl=n.close; r=cl.pct_change()
    for h in (1,5,21,63,126,252):
        n[f'f{h}']=cl.shift(-h)/cl-1
        n[f'mdd{h}']=cl[::-1].rolling(h,min_periods=1).min()[::-1].shift(-1)/cl-1  # worst close within next h days
    n['r']=r; n['gap']=n.open/n.pc-1
    n['rsi2']=rsi(cl,2); n['rsi14']=rsi(cl,14)
    m20=cl.rolling(20).mean(); sd=cl.rolling(20).std()
    n['bb_lo']=cl<m20-2*sd; n['bb_hi']=cl>m20+2*sd
    for k in (20,55,252):
        n[f'hi{k}']=cl>=cl.rolling(k).max(); n[f'lo{k}']=cl<=cl.rolling(k).min()
    n['rng20']=(n.high.rolling(20).max()/n.low.rolling(20).min()-1)
    n['ath']=cl.cummax(); n['dd_ath']=cl/n.ath-1
    n['dist200']=cl/n.dma200-1
    dn=(r<0).astype(int); up=(r>0).astype(int)
    n['dnrun']=dn.groupby((dn==0).cumsum()).cumsum(); n['uprun']=up.groupby((up==0).cumsum()).cumsum()
    n['wk']=cl/cl.shift(5)-1; n['mo']=cl/cl.shift(21)-1
    n['br50']=br.reindex(n.index); n['br200']=br200.reindex(n.index)
    n['br50_chg']=n.br50-n.br50.shift(10)
    above=(cl>n.dma200).astype(int)
    n['x_below200']=(above==0)&(above.shift(1).rolling(20).min()==1)
    n['x_above200']=(above==1)&(above.shift(1).rolling(20).max()==0)
    g=(n.dma50>n.dma200).astype(int); n['death']=(g==0)&(g.shift()==1); n['golden']=(g==1)&(g.shift()==0)
    n['engulf_top']=n.hi20.shift(1)&(n.close<n.open)&(n.open>=n.close.shift())&(n.close<n.open.shift().where(n.close.shift()>n.open.shift(),n.close.shift()))
    n['dow']=n.index.dayofweek
    return n
SIG={
 'Down 3 days in a row':lambda n:n.dnrun==3,
 'Down 5+ days in a row':lambda n:n.dnrun>=5,
 'Up 5+ days in a row':lambda n:n.uprun>=5,
 'RSI(2) < 10 (short-term oversold)':lambda n:n.rsi2<10,
 'RSI(2) > 95 (short-term overbought)':lambda n:n.rsi2>95,
 'RSI(14) < 30':lambda n:n.rsi14<30,
 'RSI(14) > 75':lambda n:n.rsi14>75,
 'Close below lower Bollinger band':lambda n:n.bb_lo,
 'Close above upper Bollinger band':lambda n:n.bb_hi,
 'New 20-day closing high':lambda n:n.hi20,
 'New 55-day closing high':lambda n:n.hi55,
 'New 52-week closing high':lambda n:n.hi252,
 'Breakout from tight 20-day range (<4%)':lambda n:n.hi20&(n.rng20.shift()<0.04),
 'New 20-day closing low':lambda n:n.lo20,
 'New 52-week closing low':lambda n:n.lo252,
 'Day fall > 2%':lambda n:n.r<-0.02,
 'Day rise > 2%':lambda n:n.r>0.02,
 'Gap down > 1%':lambda n:n.gap<-0.01,
 'Gap up > 1%':lambda n:n.gap>0.01,
 'Week down > 4%':lambda n:n.wk<-0.04,
 'Month down > 8%':lambda n:n.mo<-0.08,
 '>10% above 200-DMA (stretched)':lambda n:n.dist200>0.10,
 '>8% below 200-DMA':lambda n:n.dist200<-0.08,
 'First close below 200-DMA':lambda n:n.x_below200,
 'First close above 200-DMA':lambda n:n.x_above200,
 'Death cross (50 below 200 DMA)':lambda n:n.death,
 'Golden cross (50 above 200 DMA)':lambda n:n.golden,
 'Bearish engulfing at 20-day high':lambda n:n.engulf_top,
 '10%+ below all-time high':lambda n:(n.dd_ath<-0.10),
 '20%+ below all-time high':lambda n:(n.dd_ath<-0.20),
 'Breadth washout: <15% stocks above 50-DMA':lambda n:n.br50<15,
 'Breadth euphoria: >85% stocks above 50-DMA':lambda n:n.br50>85,
 'Divergence: index at 55d high but <55% stocks above 50-DMA':lambda n:n.hi55&(n.br50<55),
 'Breadth collapsing: -30 pts in 10 days':lambda n:n.br50_chg<-30,
 'Realised vol > 25% (panic)':lambda n:n.rv20>25,
 'Realised vol < 10% (complacency)':lambda n:n.rv20<10,
}
H=[1,5,21,63,252]
FALL={1:-0.01,5:-0.03,21:-0.05,63:-0.08,252:-0.10}
def episodes(mask,h):
    idx=np.where(mask.values)[0]; keep=[]; last=-10**9
    for i in idx:
        if i-last>=h: keep.append(i); last=i
    return mask.index[keep]
def stats(n,days,h):
    x=n.loc[days].dropna(subset=[f'f{h}'])
    if len(x)<5: return None
    f=x[f'f{h}']; fall=(x[f'mdd{h}']<=FALL[h]) if h>1 else (f<=FALL[1])
    return dict(n=len(x),win=(f>0).mean()*100,avg=f.mean()*100,med=f.median()*100,fall=fall.mean()*100)
if __name__=='__main__':
    rows=[]
    for sym in c.IDX:
        n=build(sym); n.to_parquet(f'idx_{sym.replace(" ","_")}.parquet')
        for h in H:
            base_all=n.dropna(subset=[f'f{h}'])
            for per,(a,b) in {'all':('2015','2027'),'IS 2015-21':('2015','2021-12-31'),'OOS 2022-26':('2022','2027')}.items():
                bb=base_all.loc[a:b]; f=bb[f'f{h}']; fall=(bb[f'mdd{h}']<=FALL[h]) if h>1 else (f<=FALL[1])
                rows.append(dict(sym=sym,signal='(baseline: any day)',h=h,per=per,n=len(bb),win=(f>0).mean()*100,avg=f.mean()*100,med=f.median()*100,fall=fall.mean()*100))
            for name,fn in SIG.items():
                m=fn(n).fillna(False).astype(bool)
                ep=episodes(m,h)
                for per,(a,b) in {'all':('2015','2027'),'IS 2015-21':('2015','2021-12-31'),'OOS 2022-26':('2022','2027')}.items():
                    s=stats(n,[d for d in ep if pd.Timestamp(a)<=d<=pd.Timestamp(b)],h)
                    if s: rows.append(dict(sym=sym,signal=name,h=h,per=per,**s))
    R=pd.DataFrame(rows); R.to_parquet('scan_index.parquet')
    pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
    for h in H:
        t=R[(R.sym=='NIFTY 50')&(R.h==h)].pivot_table(index='signal',columns='per',values=['n','win','avg','fall']).round(1)
        print(f'\n===== NIFTY horizon {h}d'); print(t)
