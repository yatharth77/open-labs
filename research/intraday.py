import pandas as pd, numpy as np, pyarrow.csv as pc
P='/home/claude/yatharth77/open-labs/nifty-data/'
res={}
def load(name):
    t=pc.read_csv(P+name+'_minute.csv').to_pandas(); t['date']=pd.to_datetime(t.date)
    t['day']=t.date.dt.normalize(); t['tod']=t.date.dt.hour*60+t.date.dt.minute
    t=t[(t.tod>=555)&(t.tod<=929)]
    ok=t.groupby('day').size(); t=t[t.day.isin(ok[ok>=300].index)]
    return t
pd.set_option('display.width',250)
out=[]
for name in ['NIFTY 50','NIFTY BANK']:
    t=load(name); n=pd.read_parquet(f'idx_{name.replace(" ","_")}.parquet')
    rows=[]
    for d,g in t.groupby("day"):
        if g.tod.iloc[0]>556: continue
        o=g.open.iloc[0]; cl=g.close.iloc[-1]; tod=g.tod.values; hi=g.high.values; lo=g.low.values; c=g.close.values
        for M,lab in [(15,'15'),(30,'30')]:
            m=tod<555+M; rh=hi[m].max(); rl=lo[m].min(); after=~m
            ih=np.where(after&(hi>rh))[0]; il=np.where(after&(lo<rl))[0]
            first=None
            if len(ih) and (not len(il) or ih[0]<il[0]): first='up'; i0=ih[0]
            elif len(il): first='dn'; i0=il[0]
            if first=='up':
                ent=rh; stop=rl; seg=lo[i0:]; hitstop=(seg<=stop).any()
                pnl=(stop-ent) if hitstop else (cl-ent)
                pnl_nostop=cl-ent
                # failed breakout: closes back inside / below range
            elif first=='dn':
                ent=rl; stop=rh; seg=hi[i0:]; hitstop=(seg>=stop).any()
                pnl=(ent-stop) if hitstop else (ent-cl); pnl_nostop=ent-cl
            else: pnl=pnl_nostop=0; hitstop=False
            rows.append(dict(day=d,M=M,first=first,rng=(rh-rl)/o*100,pnl=pnl/o*100,pnl_nostop=pnl_nostop/o*100,stopped=hitstop,
                             time=(tod[i0] if first else None)))
    O=pd.DataFrame(rows).join(n[['trend','vol','event','event_kind','gap','r','rv20']],on='day')
    O['year']=O.day.dt.year
    print(f'\n===== {name}: opening-range breakout, enter at range edge, stop = other side, exit 15:29 (pnl % of open, no costs)')
    for M in (15,30):
        x=O[(O.M==M)&O['first'].notna()]
        print(f'ORB{M}: days {len(x)} win {(x.pnl>0).mean()*100:.1f}% avg {x.pnl.mean():+.3f}% | no-stop win {(x.pnl_nostop>0).mean()*100:.1f}% avg {x.pnl_nostop.mean():+.3f}% | stopped {x.stopped.mean()*100:.0f}%')
        print('  by side', x.groupby('first').pnl.agg(['size','mean',lambda s:(s>0).mean()]).round(3).to_dict('index'))
        print('  by year', x.groupby('year').pnl.mean().round(3).to_dict())
        print('  by vol', x.groupby('vol',observed=True).pnl.agg(['size','mean']).round(3).to_dict('index'))
        x['rq']=pd.qcut(x.rng,4); print('  by opening range size', x.groupby('rq',observed=True).pnl.agg(['size','mean',lambda s:(s>0).mean()]).round(3).to_dict('index'))
    O.to_parquet(f'orb_{name.replace(" ","_")}.parquet')
