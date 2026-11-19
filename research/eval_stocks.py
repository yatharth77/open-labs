import pandas as pd, numpy as np
S=pd.read_parquet('stocks_feat.parquet')
print(S.shape, S.f21.notna().sum())
SIG={
 '52w-high breakout (close > prior 52w high)':lambda s:s.close>s.phi252,
 '52w-high breakout + volume >2x':lambda s:(s.close>s.phi252)&(s.vr>2),
 '52w-high breakout + volume >2x, market >200DMA':lambda s:(s.close>s.phi252)&(s.vr>2)&(s.trend=='bull (>200DMA)'),
 '52w-high breakout, market <200DMA':lambda s:(s.close>s.phi252)&(s.trend=='bear (<200DMA)'),
 '20d breakout after tight base (20d range <8%) + vol >1.5x':lambda s:(s.close>s.phi20)&(s.rng20<0.08)&(s.vr>1.5),
 '20d breakout after tight base, stock >200DMA':lambda s:(s.close>s.phi20)&(s.rng20<0.08)&(s.close>s.d200),
 '50d breakout, stock >200DMA, 6m return >20%':lambda s:(s.close>s.phi50)&(s.close>s.d200)&(s.r126>0.2),
 'Gap up >3% + volume >3x (news gap), holds gap':lambda s:(s.gap>0.03)&(s.vr>3)&(s.close>s.open),
 'Gap up >3%, fades (closes below open)':lambda s:(s.gap>0.03)&(s.close<s.open),
 '52w-low breakdown (close < prior 52w low)':lambda s:s.close<s.plo252,
 '52w-low breakdown + volume >2x':lambda s:(s.close<s.plo252)&(s.vr>2),
 '20d breakdown, stock <200DMA':lambda s:(s.close<s.plo20)&(s.close<s.d200),
 'Day fall >5% on volume >2x':lambda s:(s.r<-0.05)&(s.vr>2),
 'Day fall >5%, stock >200DMA (dip in uptrend)':lambda s:(s.r<-0.05)&(s.close>s.d200),
 '5-day fall >12%':lambda s:s.r5<-0.12,
 '5-day fall >12%, stock >200DMA':lambda s:(s.r5<-0.12)&(s.close>s.d200),
 '30%+ below 52w high':lambda s:s.dd252<-0.30,
 'Day rise >5% on volume >2x':lambda s:(s.r>0.05)&(s.vr>2),
 'Stock <200DMA and 50DMA<200DMA (downtrend)':lambda s:(s.close<s.d200)&(s.d50<s.d200),
 '5-day rise >12% (short-term blow-off)':lambda s:s.r5>0.12,
}
H=[5,21,63,126]
def episodes(df,mask,h):
    out=[]
    for sym,x in df[mask.values].groupby(df[mask.values].sym.values):
        last=None
        for d in x.day:
            if last is None or (d-last).days>=h*1.45: out.append((sym,d)); last=d
    return out
S=S.sort_values(['sym','day']); S.index=pd.MultiIndex.from_arrays([S.sym.values,S.day.values],names=['s','d'])
rows=[]
base={}
for h in H:
    b=S.dropna(subset=[f'f{h}'])
    for per,(a,z) in {'all':('2015','2027'),'IS':('2015','2021-12-31'),'OOS':('2022','2027')}.items():
        bb=b[(b.day>=a)&(b.day<=z)]
        rows.append(dict(signal='(baseline: any stock-day)',h=h,per=per,n=len(bb),win=(bb[f'f{h}']>0).mean()*100,avg=bb[f'f{h}'].mean()*100,xavg=0,xwin=50,p_down10=(bb[f'f{h}']<-0.10).mean()*100))
for name,fn in SIG.items():
    m=fn(S).fillna(False).astype(bool)
    for h in H:
        ep=episodes(S,m,h)
        x=S.loc[ep].dropna(subset=[f'f{h}'])
        for per,(a,z) in {'all':('2015','2027'),'IS':('2015','2021-12-31'),'OOS':('2022','2027')}.items():
            y=x[(x.day>=a)&(x.day<=z)]
            if len(y)<10: continue
            rows.append(dict(signal=name,h=h,per=per,n=len(y),win=(y[f'f{h}']>0).mean()*100,avg=y[f'f{h}'].mean()*100,
                xavg=y[f'x{h}'].mean()*100,xwin=(y[f'x{h}']>0).mean()*100,p_down10=(y[f'f{h}']<-0.10).mean()*100))
R=pd.DataFrame(rows); R.to_parquet('scan_stocks.parquet')
pd.set_option('display.width',260); pd.set_option('display.max_rows',500)
for h in H:
    t=R[R.h==h]
    a=t[t.per=='all'].set_index('signal')[['n','win','avg','xavg','xwin','p_down10']]
    i=t[t.per=='IS'].set_index('signal')[['xavg','xwin']].add_prefix('IS_')
    o=t[t.per=='OOS'].set_index('signal')[['n','win','xavg','xwin']].add_prefix('OOS_')
    print(f'\n===== stocks {h}d  (xavg = avg return minus universe avg same day; p_down10 = % ending >10% lower)')
    print(a.join(i).join(o).round(1).sort_values('xavg').to_string())
