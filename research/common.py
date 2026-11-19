import pandas as pd, numpy as np
IDX=['NIFTY 50','NIFTY BANK']
EVENTS=[ # (start, end, label, kind)
 ('2015-08-11','2016-02-29','China devaluation & oil crash selloff','crisis'),
 ('2016-06-24','2016-06-24','Brexit vote','shock'),
 ('2016-11-09','2016-12-31','Demonetisation + Trump election','shock'),
 ('2018-02-01','2018-03-31','LTCG tax budget + global VIX spike','correction'),
 ('2018-09-21','2018-10-31','IL&FS / NBFC crisis','crisis'),
 ('2019-05-23','2019-05-23','2019 election result','event'),
 ('2019-09-20','2019-09-20','Corporate tax cut','event'),
 ('2020-02-20','2020-03-23','COVID crash','crisis'),
 ('2020-03-24','2021-01-31','COVID recovery / liquidity rally','boom'),
 ('2021-04-01','2021-05-31','COVID second wave','shock'),
 ('2021-10-19','2022-02-23','FPI selling from Oct-2021 peak','correction'),
 ('2022-02-24','2022-03-31','Russia-Ukraine war','war'),
 ('2022-04-01','2022-06-30','Fed/RBI rate hikes, inflation','correction'),
 ('2023-01-25','2023-03-31','Adani-Hindenburg','shock'),
 ('2024-06-04','2024-06-04','2024 election result shock','event'),
 ('2024-09-27','2025-03-31','FPI selloff / earnings slowdown','correction'),
 ('2025-04-02','2025-04-11','US tariff shock (Liberation Day)','shock'),
 ('2025-05-07','2025-05-12','Operation Sindoor (India-Pakistan)','war'),
 ('2026-02-28','2026-04-08','US-Iran war (data ends 8 Apr)','war'),
]
def event_of(day):
    for s,e,l,k in EVENTS:
        if pd.Timestamp(s)<=day<=pd.Timestamp(e): return l,k
    return 'normal','normal'
def load_daily():
    D=pd.read_parquet('/home/claude/research/daily.parquet')
    D=D[D.sym!='LTIM']
    D=D[(D[['open','high','low','close']]>0).all(axis=1)&(D.n>=200)].sort_values(['sym','day']).copy()
    D['pc']=D.groupby('sym').close.shift()
    r=D.close/D.pc-1; o=D.open/D.pc-1
    bad=(r.abs()>0.35)|(o.abs()>0.35)
    D['seg']=bad.groupby(D.sym).cumsum()
    D.loc[bad,'pc']=np.nan
    return D.reset_index(drop=True)
def label_regimes(n):
    """n: NIFTY daily frame indexed by day with close."""
    n=n.copy()
    n['dma200']=n.close.rolling(200).mean(); n['dma50']=n.close.rolling(50).mean()
    n['trend']=np.where(n.close>n.dma200,'bull (>200DMA)','bear (<200DMA)')
    n.loc[n.dma200.isna(),'trend']='n/a'
    rv=np.log(n.close).diff().rolling(20).std()*np.sqrt(252)*100
    n['rv20']=rv
    n['vol']=pd.cut(rv,[0,12,18,25,200],labels=['calm <12%','normal 12-18%','high 18-25%','extreme >25%'])
    ev=[event_of(d) for d in n.index]; n['event']=[e[0] for e in ev]; n['event_kind']=[e[1] for e in ev]
    return n
