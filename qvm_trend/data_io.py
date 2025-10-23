# qvm_trend/data_io.py
from __future__ import annotations

import os
import time
import math
import json
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

__all__ = [
    "_http_get",
    "DEFAULT_START",
    "DEFAULT_END",
    "run_fmp_screener",
    "filter_universe",
    "get_prices_fmp",
    "load_prices_panel",
    "load_benchmark",
]

# ===================== FECHAS POR DEFECTO (usadas por pipeline) =====================

DEFAULT_START: str = os.environ.get("DEFAULT_START", "2020-01-01")
DEFAULT_END: str = os.environ.get("DEFAULT_END", datetime.today().date().isoformat())

# =========================== CONFIGURACIÓN FMP ============================

# API key desde entorno
FMP_API_KEY = os.environ.get("FMP_API_KEY") or os.environ.get("FMP_APIKEY") or ""

# Objetivo de llamadas: 4 req/s ≈ 240/min (por debajo del límite de 300/min)
TARGET_RPS: float = float(os.environ.get("FMP_TARGET_RPS", 4.0))
MAX_RETRIES: int = int(os.environ.get("FMP_MAX_RETRIES", 5))
TIMEOUT_SECS: int = int(os.environ.get("FMP_TIMEOUT", 30))

# Sesión HTTP con retries/backoff para 429/5xx
_session = requests.Session()
_retry = Retry(
    total=MAX_RETRIES,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=32, pool_maxsize=32)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

_last_call_ts = 0.0

def _respect_rate_limit():
    """Limiter global simple para no superar TARGET_RPS."""
    global _last_call_ts
    rps = max(TARGET_RPS, 0.1)
    min_interval = 1.0 / rps
    now = time.monotonic()
    sleep_needed = _last_call_ts + min_interval - now
    if sleep_needed > 0:
        time.sleep(sleep_needed)
    _last_call_ts = time.monotonic()

def _http_get(url: str, params: dict | None = None, *, timeout: int = TIMEOUT_SECS):
    """
    GET robusto con:
      - apikey inyectada,
      - rate limit,
      - retries + backoff con jitter para 429/5xx,
      - errores detallados.
    """
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY no configurada (variable de entorno 'FMP_API_KEY').")

    params = dict(params or {})
    params["apikey"] = FMP_API_KEY

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        _respect_rate_limit()
        try:
            r = _session.get(url, params=params, timeout=timeout)
        except Exception as e:
            last_exc = e
            time.sleep(0.25 * attempt)
            continue

        code = r.status_code
        if code == 200:
            try:
                return r.json()
            except Exception:
                snippet = r.text[:200].replace("\n", " ")
                raise RuntimeError(f"FMP JSON decode error: {url} | {snippet}")

        if code in (429, 500, 502, 503, 504):
            base = 0.75 * (2 ** (attempt - 1))
            jitter = 0.25 * (2 ** (attempt - 1)) * ((int.from_bytes(os.urandom(1), "big") / 255.0) - 0.5) * 2
            time.sleep(max(0.5, base + jitter))
            continue

        snippet = r.text[:200].replace("\n", " ")
        raise RuntimeError(f"FMP HTTP {code} {r.reason}: {url} | {snippet}")

    if last_exc:
        raise RuntimeError(f"FMP: sin respuesta estable tras {MAX_RETRIES} intentos: {url}") from last_exc
    raise RuntimeError(f"FMP: agotados retries: {url}")

# ============================ UTILIDADES VARIAS ============================

EXCHANGES_OK = {
    "NASDAQ", "Nasdaq", "NasdaqGS", "NasdaqGM",
    "NYSE", "NYSE ARCA", "NYSE Arca", "NYSE American",
    "AMEX", "BATS"
}

def _clean_symbol(sym: str) -> str:
    return (sym or "").strip().upper()

def _to_num(s, default=np.nan):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return default

# ============================ SCREENER (UNIVERSO) ============================

def run_fmp_screener(limit: int = 300,
                     eps_growth_min: float = 15,
                     roe_min: float = 10,
                     volume_min: int = 500_000,
                     mcap_min: float = 1e7,
                     *,
                     fetch_profiles: bool = True,
                     cache_key: str | None = None,
                     force: bool = False) -> pd.DataFrame:
    """
    Descarga universo base desde /stock-screener de FMP y (opcionalmente) enriquece con /profile/{sym}.
    Los kwargs cache_key/force están por compatibilidad con la app (no se usan aquí).
    """
    url = "https://financialmodelingprep.com/api/v3/stock-screener"
    params = {
        "epsGrowthMoreThan": eps_growth_min,
        "returnOnEquityMoreThan": roe_min,
        "volumeMoreThan": int(volume_min),
        "marketCapMoreThan": mcap_min,
        "limit": int(limit),
    }
    base = _http_get(url, params=params)
    df = pd.DataFrame(base)
    if df.empty or "symbol" not in df.columns:
        raise RuntimeError("Screener FMP devolvió vacío o sin 'symbol'.")

    df["symbol"] = df["symbol"].astype(str).apply(_clean_symbol)

    if fetch_profiles:
        symbols = df["symbol"].dropna().unique().tolist()
        profiles = []
        batch = 40
        pause = 1.0
        for i in range(0, len(symbols), batch):
            blk = symbols[i:i+batch]
            for sym in blk:
                try:
                    prof = _http_get(f"https://financialmodelingprep.com/api/v3/profile/{sym}")
                    if isinstance(prof, list) and prof:
                        p0 = prof[0]
                        profiles.append({
                            "symbol": sym,
                            "sector": p0.get("sector"),
                            "industry": p0.get("industry"),
                            "marketCap_profile": p0.get("mktCap") or p0.get("marketCap"),
                            "beta_profile": p0.get("beta"),
                            "price_profile": p0.get("price"),
                            "currency": p0.get("currency"),
                            "isEtf": bool(p0.get("isEtf")),
                            "isFund": bool(p0.get("isFund")),
                            "isAdr":  bool(p0.get("isAdr")),
                            "exchange": p0.get("exchangeShortName") or p0.get("exchange"),
                            "type": p0.get("type"),
                            "country": p0.get("country"),
                            "ipoDate": p0.get("ipoDate"),
                        })
                except Exception:
                    profiles.append({"symbol": sym})
            time.sleep(pause)

        dfp = pd.DataFrame(profiles)
        if not dfp.empty:
            df = df.merge(dfp, on="symbol", how="left")

    for col in ["marketCap", "marketCap_profile", "price", "price_profile", "beta", "beta_profile"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["marketCap"] = df["marketCap"].fillna(df.get("marketCap_profile"))
    df["price"]     = df["price"].fillna(df.get("price_profile"))
    df["beta"]      = df["beta"].fillna(df.get("beta_profile"))

    for c in ["sector", "industry", "exchange", "type", "country"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    for c in ["isEtf", "isFund", "isAdr"]:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].fillna(False).astype(bool)

    if "ipoDate" not in df.columns:
        df["ipoDate"] = pd.NaT
    else:
        df["ipoDate"] = pd.to_datetime(df["ipoDate"], errors="coerce")

    df = df.sort_values("marketCap", ascending=False).drop_duplicates(subset=["symbol"], keep="first")

    keep = [
        "symbol", "sector", "industry", "exchange", "type", "country",
        "marketCap", "price", "beta", "isEtf", "isFund", "isAdr", "ipoDate", "volume"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

def filter_universe(df: pd.DataFrame,
                    min_mcap: float = 5e8,
                    exchanges_ok: set[str] = EXCHANGES_OK,
                    ipo_min_days: int = 365) -> pd.DataFrame:
    """
    Limpia el universo:
      - excluye ETFs/fondos/ADRs
      - limita a exchanges esperados
      - exige IPO ≥ ipo_min_days
      - exige market cap mínimo
      - tipo stock/equity (si existe)
    """
    d = df.copy()
    for c in ["isEtf", "isFund", "isAdr"]:
        if c not in d.columns:
            d[c] = False
    d = d[(~d["isEtf"]) & (~d["isFund"]) & (~d["isAdr"])]

    if "type" in d.columns:
        typ = d["type"].fillna("").str.lower()
        ok_type = typ.str.contains("stock") | typ.str.contains("equity") | (typ == "")
        d = d[ok_type]

    if "exchange" in d.columns:
        exch = d["exchange"].fillna("")
        d = d[(exch.isin(exchanges_ok)) | (exch == "")]

    if "ipoDate" in d.columns:
        today = pd.Timestamp.today().normalize()
        d = d[(d["ipoDate"].isna()) | (d["ipoDate"] <= today - pd.Timedelta(days=int(ipo_min_days)))]

    if "marketCap" in d.columns:
        d = d[pd.to_numeric(d["marketCap"], errors="coerce") >= float(min_mcap)]

    if "marketCap" in d.columns:
        d = d.sort_values("marketCap", ascending=False)

    d["symbol"] = d["symbol"].astype(str).apply(_clean_symbol)
    d = d.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"], keep="first")

    return d.copy()

# ============================= PRECIOS (HISTÓRICO) =============================

def _hist_full(symbol: str, start: str | None, end: str | None):
    base = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    params = {}
    if start:
        params["from"] = start
    if end:
        params["to"] = end
    return _http_get(base, params=params)

def _hist_compact(symbol: str):
    base = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    return _http_get(base, params={})  # fallback sin rangos

def get_prices_fmp(symbol: str,
                   start: str | None = None,
                   end: str | None = None) -> pd.DataFrame | None:
    """
    Devuelve DataFrame con columnas: ['open','high','low','close','volume'] indexado por fecha.
    """
    sym = _clean_symbol(symbol)
    try:
        j = _hist_full(sym, start, end)
        hist = j.get("historical", [])
        if not isinstance(hist, list) or len(hist) == 0:
            j2 = _hist_compact(sym)
            hist = j2.get("historical", [])
        if not isinstance(hist, list) or len(hist) == 0:
            return None

        dfp = pd.DataFrame(hist)
        mapping = {
            "adjClose": "close",
            "adj_open": "open",
            "adj_high": "high",
            "adj_low":  "low",
        }
        for k, v in mapping.items():
            if k in dfp.columns and v not in dfp.columns:
                dfp[v] = dfp[k]

        need = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in need if c not in dfp.columns]
        if missing:
            base_ok = all(c in dfp.columns for c in ["date", "close", "volume"])
            if base_ok:
                for c in ["open", "high", "low"]:
                    if c not in dfp.columns:
                        dfp[c] = dfp["close"]
                need = ["date", "open", "high", "low", "close", "volume"]
            else:
                return None

        dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
        dfp = dfp.dropna(subset=["date"]).sort_values("date").set_index("date")

        for c in ["open", "high", "low", "close", "volume"]:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

        dfp = dfp[["open", "high", "low", "close", "volume"]].dropna(how="all")
        if dfp.empty:
            return None
        return dfp
    except Exception:
        return None

# =========================== PANEL DE PRECIOS (BULK) ===========================

def _process_symbols(symbols: List[str], fn_fetch, batch: int = 40, pause: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    Procesa símbolos en lotes con pausa entre lotes para respetar cuota.
    fn_fetch: callable(symbol) -> DataFrame | None
    """
    out: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), batch):
        blk = symbols[i:i + batch]
        for s in blk:
            sym = _clean_symbol(s)
            try:
                dfp = fn_fetch(sym)
                if dfp is not None and not dfp.empty:
                    out[sym] = dfp
            except Exception:
                continue
        time.sleep(pause)
    return out

def load_prices_panel(symbols: List[str],
                      start: str | None = None,
                      end: str | None = None,
                      *,
                      cache_key: str | None = None,
                      force: bool = False,
                      batch: int = 40,
                      pause: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    Descarga precios para lista de símbolos en un dict {symbol: DataFrame}.
    Parám. cache_key/force están para compatibilidad con tu app (cache externo).
    """
    fetch = lambda s: get_prices_fmp(s, start, end)
    panel = _process_symbols(symbols, fetch, batch=batch, pause=pause)
    return panel

def _normalize_close(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza columna 'close' a partir de variantes comunes."""
    if df is None or df.empty:
        return df
    if 'close' in df.columns:
        return df[['close']].sort_index()
    df = df.copy()
    for cand in ('adjClose', 'Close', 'close_price', 'c'):
        if cand in df.columns:
            df['close'] = df[cand]
            return df[['close']].sort_index()
    # último recurso: si hay OHLC tipo 'o','h','l','c'
    if 'c' in df.columns:
        df['close'] = df['c']
        return df[['close']].sort_index()
    # si no conseguimos, devolvemos vacío (el backtest lo ignorará)
    return pd.DataFrame()

def load_price_panel(symbols: List[str],
                     start: str | None = None,
                     end: str | None = None,
                     *,
                     cache_key: str | None = None,
                     force: bool = False,
                     batch: int = 40,
                     pause: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    Wrapper de compatibilidad para backtesting/gestión.
    Llama a tu load_prices_panel(...) y normaliza 'close'.
    """
    raw = load_prices_panel(symbols, start, end,
                            cache_key=cache_key,
                            force=force,
                            batch=batch,
                            pause=pause)
    panel: Dict[str, pd.DataFrame] = {}
    for s, df in (raw or {}).items():
        try:
            norm = _normalize_close(df)
            if norm is not None and not norm.empty:
                # quitar tz si viene con zona horaria
                if getattr(norm.index, "tz", None) is not None:
                    norm.index = norm.index.tz_localize(None)
                panel[s] = norm
        except Exception:
            # continúa con los demás símbolos
            continue
    return panel

def load_benchmark(symbol: str,
                   start: str | None = None,
                   end: str | None = None) -> pd.DataFrame | None:
    """
    Descarga la serie de un benchmark (p.ej. 'SPY', 'QQQ', 'IPSA').
    """
    return get_prices_fmp(symbol, start, end)

def fmp_probe(symbol: str = "AAPL") -> dict:
    """
    Sonda rápida: verifica que la API FMP responde y el API key está entrando.
    Devuelve info mínima + códigos.
    """
    info = {"symbol": symbol}
    try:
        j = _http_get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}")
        obj = j[0] if isinstance(j, list) and j else (j if isinstance(j, dict) else {})
        info["key_metrics_ttm_ok"] = bool(obj)
        info["has_ev_ebitda_ttm"] = obj.get("enterpriseValueOverEBITDATTM") is not None
    except Exception as e:
        info["key_metrics_ttm_ok"] = False
        info["key_metrics_ttm_err"] = str(e)[:180]

    try:
        j2 = _http_get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}")
        obj2 = j2[0] if isinstance(j2, list) and j2 else (j2 if isinstance(j2, dict) else {})
        info["ratios_ttm_ok"] = bool(obj2)
        info["has_net_margin_ttm"] = obj2.get("netProfitMarginTTM") is not None
    except Exception as e:
        info["ratios_ttm_ok"] = False
        info["ratios_ttm_err"] = str(e)[:180]

    return info