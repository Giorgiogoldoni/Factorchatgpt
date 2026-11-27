import json, csv
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

MAP={"SPEQ":"SPEQ.MI","MWEQ":"MWEQ.MI","MEEQ":"MEEQ.MI","IEWQ":"IEWQ.MI","WELE":"WELE.MI",
"IWMO":"IWMO.MI","XDEM":"XDEM.MI","QDVA":"QDVA.DE","IEMO":"IEMO.MI","MCEU":"MCEU.MI",
"IWVL":"IWVL.MI","XDEV":"XDEV.MI","IEVL":"IEVL.MI","QDVI":"QDVI.DE","VCEU":"VCEU.MI",
"IWQU":"IWQU.MI","IEQU":"IEQU.MI","QDVB":"QDVB.DE","XDEQ":"XDEQ.MI","QCEU":"QCEU.MI",
"IAPD":"IAPD.MI","VHYA":"VHYA.MI"}
BENCH="SWDA.MI"; COST=0.001

def kama(s, n=10, fast=2, slow=30):
    change=s.diff(n).abs()
    volatility=s.diff().abs().rolling(n).sum()
    er=(change/volatility.replace(0,np.nan)).fillna(0)
    sc=(er*(2/(fast+1)-2/(slow+1))+2/(slow+1))**2
    out=pd.Series(index=s.index,dtype=float); out.iloc[0]=s.iloc[0]
    for i in range(1,len(s)):
        c=sc.iloc[i]
        out.iloc[i]=out.iloc[i-1]+(c if pd.notna(c) else 2/(slow+1))*(s.iloc[i]-out.iloc[i-1])
    return out

def clean_prices(price, max_daily_move=0.25):
    """Rimuove singole osservazioni anomale provider (ETF non leveraged)."""
    p=price.dropna().astype(float)
    p=p[p>0]
    moves=p.pct_change()
    bad=moves.index[moves.abs()>max_daily_move]
    # Elimina la seduta anomala: evita di propagare split/quote errate nei rendimenti.
    p=p.drop(bad)
    return p, [d.strftime("%Y-%m-%d") for d in bad]

def metrics(price, signal):
    p=price.dropna(); sig=signal.shift(1).reindex(p.index).fillna(0).astype(float)
    ret=p.pct_change().fillna(0); turnover=sig.diff().abs().fillna(sig.abs())
    strat=ret*sig-turnover*COST
    rows={}
    for label,days in [("6m",126),("1y",252),("3y",756)]:
        if len(p)<days+2: rows[label]=None; continue
        r=strat.iloc[-days:]; bh=ret.iloc[-days:]
        equity=(1+r).cumprod(); b=(1+bh).cumprod()
        dd=(equity/equity.cummax()-1).min()
        rows[label]={"strategy_return":round(float(equity.iloc[-1]-1),5),
        "buy_hold_return":round(float(b.iloc[-1]-1),5),
        "excess":round(float(equity.iloc[-1]-b.iloc[-1]),5),
        "strategy_maxdd":round(float(dd),5),
        "buy_hold_maxdd":round(float((b/b.cummax()-1).min()),5),
        "exposure":round(float(sig.mean()),4)}
    return rows

def calc(close):
    # Breakout five sessions: close exceeds previous five-session high; exit below previous low.
    prevhigh=close.shift(1).rolling(5).max(); prevlow=close.shift(1).rolling(5).min()
    k5=pd.Series(0,index=close.index,dtype=int); state=0
    for i in range(len(close)):
        if close.iloc[i]>prevhigh.iloc[i]: state=1
        elif close.iloc[i]<prevlow.iloc[i]: state=0
        k5.iloc[i]=state
    fast=kama(close,10); slow=kama(close,30)
    cloud=((close>fast)&(fast>slow)).astype(int)
    hybrid=(k5.astype(bool)&cloud.astype(bool)).astype(int)
    return {"Kamal5":k5,"KAMA Cloud":cloud,"Ibrida":hybrid}, {"fast":fast,"slow":slow}

def main():
    universe=list(csv.DictReader(open("data/factor_universe.csv",encoding="utf-8-sig")))
    symbols={MAP.get(row["ticker"].split(".")[0],row["ticker"]):row for row in universe}
    tickers=list(symbols)+[BENCH]
    raw=yf.download(tickers,period="5y",interval="1d",auto_adjust=True,progress=False,threads=True,group_by="ticker")
    data={"updated":pd.Timestamp.now("UTC").isoformat(),"benchmark":BENCH,"transaction_cost":COST,
    "methodology":{"Kamal5":"Close > massimo dei 5 giorni precedenti entra; close < minimo dei 5 giorni precedenti esce, long/cash.","KAMA Cloud":"KAMA(10) e KAMA(30); long quando prezzo > KAMA10 > KAMA30, altrimenti cash.","Ibrida":"Long solo quando Kamal5 e KAMA Cloud sono entrambi long.","costs":"0.10% per cambio esposizione; prezzi adjusted Yahoo Finance, proxy Buy & Hold."},"assets":[]}
    def series(t):
        try:
            s=raw[t]["Close"] if isinstance(raw.columns,pd.MultiIndex) and t in raw.columns.get_level_values(0) else (raw["Close"][t] if isinstance(raw.columns,pd.MultiIndex) and t in raw.columns.get_level_values(1) else raw["Close"])
            return s.dropna()
        except Exception:return pd.Series(dtype=float)
    bench=series(BENCH)
    for t,row in symbols.items():
        p0=series(t)
        p,bad_days=clean_prices(p0)
        if len(p)<300 or len(bench)<300: continue
        sig,lines=calc(p)
        item={"ticker":t,"name":row["name"],"category":row["category"],"last_price":round(float(p.iloc[-1]),4),
        "data_quality":{"removed_outlier_sessions":len(bad_days),"dates":bad_days},
        "signals":{k:int(v.iloc[-1]) for k,v in sig.items()},"kama_fast":round(float(lines["fast"].iloc[-1]),4),
        "kama_slow":round(float(lines["slow"].iloc[-1]),4),"metrics":{}}
        for k,v in sig.items():item["metrics"][k]=metrics(p,v)
        item["metrics"]["Buy & Hold"]={period:{"strategy_return":m["buy_hold_return"],"buy_hold_return":m["buy_hold_return"],"strategy_maxdd":m["buy_hold_maxdd"],"exposure":1} if m else None for period,m in item["metrics"]["Kamal5"].items()}
        data["assets"].append(item)
    if not data["assets"]:
        raise RuntimeError("Zero ETF calcolati: interrompo il workflow per non pubblicare un feed vuoto.")
    Path("data").mkdir(exist_ok=True)
    Path("data/strategies.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    flagged=[a for a in data["assets"] if a["data_quality"]["removed_outlier_sessions"]]
    print(f"Strategy feed: {len(data['assets'])} ETF; anomalie prezzo rimosse su {len(flagged)} ETF")
    for a in flagged: print(a["ticker"],a["data_quality"]["dates"])
if __name__=="__main__":main()
