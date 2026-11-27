import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# RRG pubblico approssimato: non replica gli indicatori proprietari JdK.
# Universo: solo strumenti con ticker Yahoo verificato nel catalogo.
SYMBOLS = {
    "SPEQ.MI": "Equal Weight · USA",
    "MWEQ.MI": "Equal Weight · World",
    "MEEQ.MI": "Equal Weight · Europe",
    "IEWQ.MI": "Equal Weight · Nasdaq 100",
    "WELE.MI": "Equal Weight · USA ESG",
    "IWMO.MI": "Momentum · World",
    "XDEM.MI": "Momentum · World",
    "QDVA.DE": "Momentum · USA",
    "IEMO.MI": "Momentum · Europe",
    "MCEU.MI": "Momentum · Europe",
    "IWVL.MI": "Value · World",
    "XDEV.MI": "Value · World",
    "IEVL.MI": "Value · Europe",
    "QDVI.DE": "Value · USA",
    "VCEU.MI": "Value · Europe",
    "QDVB.DE": "Quality · USA",
    "XDEQ.MI": "Quality · World",
    "QCEU.MI": "Quality · Europe",
    "IAPD.MI": "High Dividend · Asia Pacific",
    "VHYA.MI": "High Dividend · Global",
}
BENCHMARK = "IWDA.MI"  # iShares Core MSCI World UCITS ETF, benchmark globale
WINDOW = 26             # settimane per normalizzazione
TAIL = 8                # punti settimanali nella coda

def zscore(series, window):
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std

def wma(series, window=3):
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def main():
    tickers = list(SYMBOLS) + [BENCHMARK]
    raw = yf.download(tickers, period="5y", interval="1d", auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    closes = {}
    for ticker in tickers:
        try:
            s = raw[ticker]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            closes[ticker] = s.dropna()
        except Exception:
            continue
    if BENCHMARK not in closes:
        raise RuntimeError(f"Benchmark {BENCHMARK} non scaricato")
    benchmark = closes[BENCHMARK].resample("W-FRI").last().dropna()
    result = {"benchmark": BENCHMARK, "frequency": "weekly", "method": "RRG public approximation",
              "window_weeks": WINDOW, "tail_weeks": TAIL, "updated": pd.Timestamp.utcnow().isoformat(),
              "disclaimer": "Approssimazione pubblica: non coincide con gli indicatori proprietari JdK. Dati adjusted close; nessun valore se la serie è insufficiente.",
              "assets": []}
    for ticker, label in SYMBOLS.items():
        if ticker not in closes:
            continue
        price = closes[ticker].resample("W-FRI").last().dropna()
        frame = pd.concat([price.rename("price"), benchmark.rename("benchmark")], axis=1).dropna()
        if len(frame) < WINDOW + TAIL + 5:
            continue
        relative = frame["price"] / frame["benchmark"] * 100
        smoothed = wma(relative, 3)
        ratio = 100 + zscore(smoothed, WINDOW)
        momentum = 100 + zscore(ratio.diff(4), WINDOW)
        points = []
        values = pd.concat([ratio.rename("x"), momentum.rename("y")], axis=1).dropna().tail(TAIL)
        for date, row in values.iterrows():
            points.append({"date": date.strftime("%Y-%m-%d"), "x": round(float(row.x), 3), "y": round(float(row.y), 3)})
        if not points:
            continue
        x, y = points[-1]["x"], points[-1]["y"]
        quadrant = ("Leading" if x >= 100 and y >= 100 else
                    "Weakening" if x >= 100 and y < 100 else
                    "Lagging" if x < 100 and y < 100 else "Improving")
        result["assets"].append({"ticker": ticker, "label": label, "quadrant": quadrant, "trail": points})
    result["assets"].sort(key=lambda a: (a["quadrant"], a["ticker"]))
    Path("data").mkdir(exist_ok=True)
    Path("data/rrg.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RRG: {len(result['assets'])} strumenti pubblicati; benchmark {BENCHMARK}")

if __name__ == "__main__":
    main()
