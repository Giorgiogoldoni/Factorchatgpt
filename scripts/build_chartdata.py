import json,csv
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf

MAP={"SPEQ":"SPEQ.MI","MWEQ":"MWEQ.MI","MEEQ":"MEEQ.MI","IEWQ":"IEWQ.MI","WELE":"WELE.MI","IWMO":"IWMO.MI","XDEM":"XDEM.MI","QDVA":"QDVA.DE","IEMO":"IEMO.MI","MCEU":"MCEU.MI","IWVL":"IWVL.MI","XDEV":"XDEV.MI","IEVL":"IEVL.MI","QDVI":"QDVI.DE","VCEU":"VCEU.MI","IWQU":"IWQU.MI","IEQU":"IEQU.MI","QDVB":"QDVB.DE","XDEQ":"XDEQ.MI","QCEU":"QCEU.MI","IAPD":"IAPD.MI","VHYA":"VHYA.MI"}

def kama(s,n=10,fast=2,slow=30):
    change=s.diff(n).abs()
    volatility=s.diff().abs().rolling(n).sum()
    er=(change/volatility.replace(0,np.nan)).fillna(0)
    sc=(er*(2/(fast+1)-2/(slow+1))+2/(slow+1))**2
    out=pd.Series(index=s.index,dtype=float)
    if len(s): out.iloc[0]=s.iloc[0]
    for i in range(1,len(s)):
        out.iloc[i]=out.iloc[i-1]+(sc.iloc[i] if pd.notna(sc.iloc[i]) else 2/(slow+1))*(s.iloc[i]-out.iloc[i-1])
    return out

def smma(s,n):
    # Wilder-style smoothing, seeded with an n-period SMA
    out=pd.Series(index=s.index,dtype=float)
    if len(s)>=n:
        out.iloc[n-1]=s.iloc[:n].mean()
        for i in range(n,len(s)):out.iloc[i]=(out.iloc[i-1]*(n-1)+s.iloc[i])/n
    return out

def calc_rsi(close,n=14):
    d=close.diff();gain=d.clip(lower=0);loss=-d.clip(upper=0)
    ag=gain.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    al=loss.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    rsi=100-100/(1+rs)
    rsi=rsi.mask((al==0)&(ag>0),100).mask((ag==0)&(al>0),0)
    return rsi

def add_indicators(z):
    o,h,l,c,v=z.open,z.high,z.low,z.close,z.volume
    z["kama10"]=kama(c,10);z["kama30"]=kama(c,30)
    for n in (20,55):
        z[f"d{n}h"]=h.shift(1).rolling(n).max();z[f"d{n}l"]=l.shift(1).rolling(n).min()
    for n in (20,50,200):z[f"ema{n}"]=c.ewm(span=n,adjust=False).mean()
    mid=(h+l)/2;z["jaw"]=smma(mid,13);z["teeth"]=smma(mid,8);z["lips"]=smma(mid,5)
    # Parabolic SAR approximation: trend state, acceleration factor, prior two-bar constraints
    sar=[];bull=True;ep=h.iloc[0];cur=l.iloc[0];af=.02
    for i in range(len(z)):
        if i:
            cur=cur+af*(ep-cur)
            if bull:cur=min(cur,l.iloc[i-1],l.iloc[i-2] if i>1 else l.iloc[i-1])
            else:cur=max(cur,h.iloc[i-1],h.iloc[i-2] if i>1 else h.iloc[i-1])
            if bull and l.iloc[i]<cur:
                bull=False;cur=ep;ep=l.iloc[i];af=.02
            elif not bull and h.iloc[i]>cur:
                bull=True;cur=ep;ep=h.iloc[i];af=.02
            elif bull and h.iloc[i]>ep:
                ep=h.iloc[i];af=min(.2,af+.02)
            elif not bull and l.iloc[i]<ep:
                ep=l.iloc[i];af=min(.2,af+.02)
        sar.append(cur)
    z["sar"]=sar
    pc=c.shift()
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    z["atr14"]=smma(tr,14)
    z["viPlus"]=(h-l.shift()).abs().rolling(14).sum()/tr.rolling(14).sum()
    z["viMinus"]=(l-h.shift()).abs().rolling(14).sum()/tr.rolling(14).sum()
    z["ao"]=mid.rolling(5).mean()-mid.rolling(34).mean()
    # Relative Vigor Index: smoothed weighted close-open / high-low, then signal SMA(4)
    num=((c-o)*2+(c.shift(1)-o.shift(1))+(c.shift(2)-o.shift(2))+(c.shift(3)-o.shift(3))/2
    den=((h-l)*2+(h.shift(1)-l.shift(1))+(h.shift(2)-l.shift(2))+(h.shift(3)-l.shift(3))/2
    z["rvgi"]=num.rolling(10).sum()/den.rolling(10).sum().replace(0,np.nan)
    z["rvgiSignal"]=z.rvgi.rolling(4).mean()
    prevhi=h.shift().rolling(5).max();prevlo=l.shift().rolling(5).min();state=0;ss=[]
    for i in range(len(z)):
        if pd.notna(prevhi.iloc[i]) and c.iloc[i]>prevhi.iloc[i]:state=1
        elif pd.notna(prevlo.iloc[i]) and c.iloc[i]<prevlo.iloc[i]:state=0
        ss.append(state)
    z["k5"]=ss
    z["rsi14"]=calc_rsi(c,14)
    e12=c.ewm(span=12,adjust=False).mean();e26=c.ewm(span=26,adjust=False).mean()
    z["macd"]=e12-e26;z["macdSignal"]=z.macd.ewm(span=9,adjust=False).mean();z["macdHist"]=z.macd-z.macdSignal
    z["bbMid"]=c.rolling(20).mean();sd=c.rolling(20).std(ddof=0);z["bbUpper"]=z.bbMid+2*sd;z["bbLower"]=z.bbMid-2*sd
    z["roc20"]=c.pct_change(20)*100
    z["volSma20"]=v.rolling(20).mean()
    return z

def main():
    universe=list(csv.DictReader(open("data/factor_universe.csv",encoding="utf-8-sig")))
    symbols={MAP.get(x["ticker"].split(".")[0],x["ticker"]):x for x in universe}
    raw=yf.download(list(symbols),period="5y",interval="1d",auto_adjust=True,group_by="ticker",progress=False,threads=True)
    fields=["open","high","low","close","volume","d20h","d20l","d55h","d55l","kama10","kama30","ema20","ema50","ema200","jaw","teeth","lips","sar","ao","viPlus","viMinus","rvgi","rvgiSignal","k5","rsi14","macd","macdSignal","macdHist","atr14","bbMid","bbUpper","bbLower","roc20","volSma20"]
    assets=[];errors=[]
    for t,meta in symbols.items():
        try:
            if isinstance(raw.columns,pd.MultiIndex):
                if t not in raw.columns.get_level_values(0):raise ValueError("Ticker assente nel download")
                z=raw[t].copy()
            else:z=raw.copy()
            z=z.dropna(subset=["Open","High","Low","Close"])
            z.columns=[str(x).lower() for x in z.columns]
            z=z[z.close>0].copy()
            if len(z)<200:raise ValueError(f"Storico insufficiente: {len(z)} righe")
            # Non filtrare rendimenti estremi: eliminare barre di mercato altera il grafico.
            z=add_indicators(z)
            rows=[]
            for date,r in z.tail(504).iterrows():
                row=[date.strftime("%Y-%m-%d")]
                for f in fields:row.append(None if pd.isna(r.get(f)) else round(float(r.get(f)),6))
                rows.append(row)
            assets.append({"ticker":t,"name":meta["name"],"category":meta["category"],"rows":rows})
        except Exception as e:
            errors.append(f"{t}: {e}")
            print("SKIP",t,str(e))
    if not assets:raise RuntimeError("Nessuna serie OHLC generata")
    output={"updated":pd.Timestamp.now("UTC").isoformat(),"columns":["date"]+fields,"assets":assets,"errors":errors}
    Path("data").mkdir(exist_ok=True)
    Path("data/chartdata.json").write_text(json.dumps(output,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    print(f"Chart feed: {len(assets)}/{len(symbols)} ETF | {sum(len(a['rows']) for a in assets)} righe")
    if errors:print("Ticker skipped:", "; ".join(errors))

if __name__=="__main__":main()
