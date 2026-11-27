import json,csv
from pathlib import Path
import numpy as np,pandas as pd,yfinance as yf
MAP={"SPEQ":"SPEQ.MI","MWEQ":"MWEQ.MI","MEEQ":"MEEQ.MI","IEWQ":"IEWQ.MI","WELE":"WELE.MI","IWMO":"IWMO.MI","XDEM":"XDEM.MI","QDVA":"QDVA.DE","IEMO":"IEMO.MI","MCEU":"MCEU.MI","IWVL":"IWVL.MI","XDEV":"XDEV.MI","IEVL":"IEVL.MI","QDVI":"QDVI.DE","VCEU":"VCEU.MI","IWQU":"IWQU.MI","IEQU":"IEQU.MI","QDVB":"QDVB.DE","XDEQ":"XDEQ.MI","QCEU":"QCEU.MI","IAPD":"IAPD.MI","VHYA":"VHYA.MI"}
def kama(s,n=10,fast=2,slow=30):
 ch=s.diff(n).abs();vol=s.diff().abs().rolling(n).sum();er=(ch/vol.replace(0,np.nan)).fillna(0);sc=(er*(2/(fast+1)-2/(slow+1))+2/(slow+1))**2;out=pd.Series(index=s.index,dtype=float);out.iloc[0]=s.iloc[0]
 for i in range(1,len(s)):out.iloc[i]=out.iloc[i-1]+(sc.iloc[i] if pd.notna(sc.iloc[i]) else 2/(slow+1))*(s.iloc[i]-out.iloc[i-1])
 return out
def smma(s,n):return s.ewm(alpha=1/n,adjust=False).mean()
def main():
 uni=list(csv.DictReader(open("data/factor_universe.csv",encoding="utf-8-sig")));symbols={MAP.get(x["ticker"].split(".")[0],x["ticker"]):x for x in uni}
 raw=yf.download(list(symbols),period="5y",interval="1d",auto_adjust=True,group_by="ticker",progress=False,threads=True);assets=[]
 for t,row in symbols.items():
  try:
   z=raw[t].dropna().copy() if isinstance(raw.columns,pd.MultiIndex) else raw.dropna();z.columns=[c.lower() for c in z.columns];z=z[z.close>0];z=z[z.close.pct_change().abs()<.25]
   if len(z)<200:continue
   o,h,l,c=z.open,z.high,z.low,z.close;z["kama10"]=kama(c,10);z["kama30"]=kama(c,30)
   for n in (20,55):z[f"d{n}h"]=h.shift(1).rolling(n).max();z[f"d{n}l"]=l.shift(1).rolling(n).min()
   for n in (20,50,200):z[f"ema{n}"]=c.ewm(span=n,adjust=False).mean()
   mid=(h+l)/2;z["jaw"]=smma(mid,13);z["teeth"]=smma(mid,8);z["lips"]=smma(mid,5)
   # Parabolic SAR approximation
   sar=[];bull=True;ep=h.iloc[0];cur=l.iloc[0];af=.02
   for i in range(len(z)):
    if i:cur+=af*(ep-cur)
    if bull and i:cur=min(cur,l.iloc[i-1],l.iloc[i-2] if i>1 else l.iloc[i-1])
    if not bull and i:cur=max(cur,h.iloc[i-1],h.iloc[i-2] if i>1 else h.iloc[i-1])
    if bull and h.iloc[i]>ep:ep=h.iloc[i];af=min(.2,af+.02)
    elif not bull and l.iloc[i]<ep:ep=l.iloc[i];af=min(.2,af+.02)
    if bull and l.iloc[i]<cur:bull=False;cur=ep;ep=l.iloc[i];af=.02
    elif not bull and h.iloc[i]>cur:bull=True;cur=ep;ep=h.iloc[i];af=.02
    sar.append(cur)
   z["sar"]=sar;pc=c.shift();tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1);z["viPlus"]=(h-l.shift()).abs().rolling(14).sum()/tr.rolling(14).sum();z["viMinus"]=(l-h.shift()).abs().rolling(14).sum()/tr.rolling(14).sum();z["ao"]=mid.rolling(5).mean()-mid.rolling(34).mean();z["rvgi"]=(c-o)/(h-l).replace(0,np.nan);z["rvgi"]=z.rvgi.rolling(10).mean();z["rvgiSignal"]=z.rvgi.rolling(4).mean()
   prevhi=h.shift().rolling(5).max();prevlo=l.shift().rolling(5).min();state=0;ss=[]
   for i in range(len(z)):
    if c.iloc[i]>prevhi.iloc[i]:state=1
    elif c.iloc[i]<prevlo.iloc[i]:state=0
    ss.append(state)
   z["k5"]=ss;fields=["open","high","low","close","volume","d20h","d20l","d55h","d55l","kama10","kama30","ema20","ema50","ema200","jaw","teeth","lips","sar","ao","viPlus","viMinus","rvgi","rvgiSignal","k5"];rows=[]
   for date,r in z.tail(1250).iterrows():
    d={"date":date.strftime("%Y-%m-%d")}
    for f in fields:d[f]=None if pd.isna(r.get(f)) else round(float(r.get(f)),5)
    rows.append(d)
   assets.append({"ticker":t,"name":row["name"],"category":row["category"],"rows":rows})
  except Exception as e:print("SKIP",t,str(e))
 if not assets:raise RuntimeError("Nessuna serie OHLC generata")
 Path("data").mkdir(exist_ok=True);Path("data/chartdata.json").write_text(json.dumps({"updated":pd.Timestamp.now("UTC").isoformat(),"assets":assets},ensure_ascii=False),encoding="utf-8");print(f"Chart feed: {len(assets)} ETF OHLC")
if __name__=="__main__":main()
