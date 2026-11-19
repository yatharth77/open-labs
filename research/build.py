import pandas as pd, numpy as np, pyarrow.csv as pc, sys, os, glob
from concurrent.futures import ProcessPoolExecutor
SRC='/home/claude/yatharth77/open-labs/nifty-data'
def load(f):
    t=pc.read_csv(f).to_pandas()
    t['date']=pd.to_datetime(t['date'])
    t=t[(t.date.dt.hour*60+t.date.dt.minute>=555)&(t.date.dt.hour*60+t.date.dt.minute<=930)]
    return t
def feat(f):
    name=os.path.basename(f).replace('_minute_new.csv','').replace('_minute.csv','')
    t=load(f)
    if name=='MM':
        t2=load(f.replace('MM_minute','MM_minute_new')) if 'new' not in f else None
    t['day']=t.date.dt.normalize(); t['tod']=t.date.dt.hour*60+t.date.dt.minute
    g=t.groupby('day')
    d=pd.DataFrame({'open':g.open.first(),'high':g.high.max(),'low':g.low.min(),'close':g.close.last(),'volume':g.volume.sum(),'n':g.size()})
    def at(m,col='close'):
        x=t[t.tod<=m].groupby('day')[col].last(); return x
    for m,lab in [(569,'c0929'),(584,'c0944'),(614,'c1014'),(659,'c1059'),(779,'c1259'),(899,'c1459')]:
        d[lab]=at(m)
    f30=t[t.tod<585].groupby('day'); d['h30']=f30.high.max(); d['l30']=f30.low.min()
    f15=t[t.tod<570].groupby('day'); d['h15']=f15.high.max(); d['l15']=f15.low.min()
    # after 9:45 extremes
    a=t[t.tod>=585].groupby('day'); d['hA']=a.high.max(); d['lA']=a.low.min()
    d['sym']=name
    return d.reset_index()
if __name__=='__main__':
    fs=sorted(glob.glob(SRC+'/*.csv'))
    fs=[f for f in fs if not f.endswith('MM_minute_new.csv')]
    out=[]
    with ProcessPoolExecutor(2) as ex:
        for i,d in enumerate(ex.map(feat,fs)):
            out.append(d); print(i,d.sym.iloc[0],len(d),flush=True)
    D=pd.concat(out); D.to_parquet('daily.parquet'); print(D.shape)
