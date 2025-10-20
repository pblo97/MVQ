# qvm_trend/data_io.py
import os, time, requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from .cache_io import save_df, load_df, save_panel, load_panel

FMP_API_KEY = os.getenv("FMP_API_KEY", "")
DEFAULT_START = "2020-01-01"
DEFAULT_END = datetime.today().strftime("%Y-%m-%d")

EXCHANGES_OK = {
    "NASDAQ","Nasdaq","NasdaqGS","NasdaqGM",
    "NYSE","NYSE ARCA","NYSE Arca","NYSE American",
    "AMEX","BATS"
}
IPO_MIN_DAYS_DEFAULT = 365

def _http_get(url: str, params: dict | None = None, sleep: float = 0.0):
    if sleep > 0: time.sleep(sleep)
    params = params or {}
    key = os.getenv("FMP_API_KEY", "")
    if not key:
        raise RuntimeError('FMP_API_KEY no configurada (secrets.toml o env).')
    params["apikey"] = key
    r = requests.get(url, params=params, timeout=30)
    if not r.ok:
        raise RuntimeError(f"FMP HTTP {r.status_code} {r.reason}: {url} | {r.text[:200]}")
    return r.json()

def clean_symbol(sym: str) -> str:
    return (sym or "").strip().upper()

# --- SCREENER & PERFILES -----------------------------------------------------
def run_fmp_screener(limit=200) -> pd.DataFrame:
    url = "https://financialmodelingprep.com/api/v3/stock-screener"
    params = {
        "epsGrowthMoreThan": 15,
        "returnOnEquityMoreThan": 10,
        "volumeMoreThan": 500000,
        "marketCapMoreThan": 1e7,
        "limit": int(limit),
    }
    base = _http_get(url, params=params)
    df = pd.DataFrame(base)
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()
    # perfiles
    profiles = []
    for sym in df["symbol"].dropna().unique():
        try:
            prof = _http_get(f"https://financialmodelingprep.com/api/v3/profile/{sym}")
            if isinstance(prof, list) and prof:
                p0 = prof[0]
                profiles.append({
                    "symbol": clean_symbol(sym),
                    "sector": p0.get("sector"),
                    "industry": p0.get("industry"),
                    "marketCap_profile": p0.get("mktCap") or p0.get("marketCap"),
                    "price_profile": p0.get("price"),
                    "exchange": p0.get("exchangeShortName") or p0.get("exchange"),
                    "ipoDate": p0.get("ipoDate"),
                })
        except Exception:
            continue
    dfp = pd.DataFrame(profiles)
    out = df.merge(dfp, on="symbol", how="left")
    # normaliza
    for c in ["marketCap","marketCap_profile","price","price_profile"]:
        if c not in out.columns: out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["marketCap"] = out["marketCap"].fillna(out["marketCap_profile"])
    for c in ["sector","industry"]:
        if c not in out.columns: out[c] = "Unknown"
        out[c] = out[c].fillna("Unknown").astype(str)
    for c in ["exchange"]:
        if c not in out.columns: out[c] = ""
        out[c] = out[c].fillna("").astype(str)
    out["ipoDate"] = pd.to_datetime(out.get("ipoDate"), errors="coerce")
    out["symbol"] = out["symbol"].astype(str).apply(clean_symbol)
    return out.drop_duplicates("symbol")
    
def filter_universe(df: pd.DataFrame, min_mcap=5e8, ipo_min_days=IPO_MIN_DAYS_DEFAULT) -> pd.DataFrame:
    if df.empty: return df
    today = pd.Timestamp.today().normalize()
    typ_ok = True  # ya filtrado por screener
    exch = df["exchange"].fillna("")
    df2 = df[(exch.isin(EXCHANGES_OK)) | (exch=="")].copy()
    df2 = df2[df2["marketCap"] >= float(min_mcap)]
    df2 = df2[(df2["ipoDate"].isna()) | (df2["ipoDate"] <= today - pd.Timedelta(days=int(ipo_min_days)))]
    return df2.sort_values("marketCap", ascending=False)

# --- PRECIOS (OHLCV) ---------------------------------------------------------
def get_prices_fmp(symbol: str, start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.DataFrame | None:
    sym = clean_symbol(symbol)
    base = f"https://financialmodelingprep.com/api/v3/historical-price-full/{sym}"

    # 1) intento con rango
    try:
        j = _http_get(base, params={"from": start, "to": end})
        hist = j.get("historical", [])
    except Exception:
        hist = []

    # 2) fallback sin rango
    if not isinstance(hist, list) or len(hist) == 0:
        try:
            j2 = _http_get(base)
            hist = j2.get("historical", [])
        except Exception:
            hist = []

    if not isinstance(hist, list) or len(hist) == 0:
        return None

    dfp = pd.DataFrame(hist)
    if dfp.empty:
        return None

    # Normaliza nombres (lower / sin guiones) y mapea sinónimos
    rename_map = {c: c.lower().replace("-", "_") for c in dfp.columns}
    dfp = dfp.rename(columns=rename_map)

    # sinónimos → estándar
    synonyms = {
        "close": ["close", "adjclose", "adj_close", "adjusted_close", "price"],
        "open":  ["open", "adjopen", "adj_open", "adjusted_open"],
        "high":  ["high", "adjhigh", "adj_high", "adjusted_high"],
        "low":   ["low", "adjlow", "adj_low", "adjusted_low"],
        "volume":["volume", "volumeto", "volumes", "qty"]
    }

    def first_present(cols: list[str]) -> str | None:
        for c in cols:
            if c in dfp.columns:
                return c
        return None

    cols = {}
    for std, cands in synonyms.items():
        src = first_present(cands)
        if src is not None:
            cols[std] = src

    # Debe existir al menos 'close'
    if "close" not in cols:
        return None

    # Crea DataFrame base con 'close'
    out = pd.DataFrame()
    out["close"] = pd.to_numeric(dfp[cols["close"]], errors="coerce")

    # Reconstruye O/H/L si faltan usando 'close' (fallback tolerante)
    for k in ["open", "high", "low"]:
        if k in cols:
            out[k] = pd.to_numeric(dfp[cols[k]], errors="coerce")
        else:
            out[k] = out["close"]  # fallback: igual a close

    # Volume: si no hay, crea 0
    if "volume" in cols:
        out["volume"] = pd.to_numeric(dfp[cols["volume"]], errors="coerce").fillna(0)
    else:
        out["volume"] = 0.0

    # Fecha
    date_col = "date" if "date" in dfp.columns else None
    if date_col is None:
        # algunos endpoints traen 'label'
        if "label" in dfp.columns:
            dfp["date"] = pd.to_datetime(dfp["label"], errors="coerce")
        else:
            return None
    else:
        dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")

    out["date"] = dfp["date"]
    out = out.dropna(subset=["date", "close"])
    if out.empty:
        return None

    out = out.sort_values("date").set_index("date")

    # Orden y tipos finales
    out = out[["open", "high", "low", "close", "volume"]].astype(
        {"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"}
    )
    return out

def load_prices_panel(symbols: List[str], start: str, end: str, cache_key: str | None = None, force: bool=False) -> Dict[str, pd.DataFrame]:
    # lee panel cacheado si existe
    if cache_key and not force:
        cached = load_panel(f"prices_{cache_key}")
        if cached: return cached
    panel = {}
    for s in symbols:
        dfp = get_prices_fmp(s, start, end)
        if dfp is not None and not dfp.empty:
            panel[s] = dfp
    if cache_key:
        save_panel(panel, f"prices_{cache_key}")
    return panel

def load_benchmark(symbol: str, start: str, end: str, cache_key: str | None = None, force: bool=False) -> pd.Series | None:
    key = f"bench_{symbol}_{cache_key}" if cache_key else None
    if key and not force:
        dfc = load_df(key)
        if dfc is not None:
            return dfc["close"]
    df = get_prices_fmp(symbol, start, end)
    if df is None or df.empty: return None
    out = df[["close"]]
    if key: save_df(out, key)
    return out["close"]

