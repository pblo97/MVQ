# qvm_trend/pipeline.py
import numpy as np
import pandas as pd
from typing import Dict, List
from .data_io import load_prices_panel, load_benchmark, get_prices_fmp, DEFAULT_START, DEFAULT_END
from .factors import compute_breakout_features
from .scoring import breakout_score, entry_signal, DEFAULT_TH

def _ma(s: pd.Series, n=200): return s.rolling(n, min_periods=n).mean()
def _mom_12_1(s: pd.Series):  return s.shift(21) / s.shift(252) - 1

def apply_trend_filter(panel: Dict[str, pd.DataFrame], use_and=False) -> List[str]:
    import numpy as np
    elig = []
    for sym, df in panel.items():
        if df is None or df.empty: 
            continue
        close = df['close']
        ma200 = _ma(close, 200)
        mom   = _mom_12_1(close)

        last_close = float(close.dropna().iloc[-1]) if not close.dropna().empty else np.nan
        last_ma200 = float(ma200.dropna().iloc[-1]) if not ma200.dropna().empty else np.nan
        last_mom   = float(mom.dropna().iloc[-1])   if not mom.dropna().empty   else np.nan

        cond_ma = (not np.isnan(last_ma200)) and (last_close > last_ma200)
        cond_mo = (not np.isnan(last_mom))   and (last_mom > 0)

        pass_sig = (cond_ma and cond_mo) if use_and else (cond_ma or cond_mo)
        if pass_sig: 
            elig.append(sym)
    return elig

def enrich_with_breakout(
    panel: dict[str, pd.DataFrame],
    rvol_lookback: int = 20,
    rvol_th: float = 1.5,
    closepos_th: float = 0.6,
    p52_th: float = 0.95,
    updown_vol_th: float = 1.2,
    bench_series: pd.Series | None = None,
    min_hits: int = 4,             # <-- nuevo: mínimo de checks que deben cumplirse (0..4)
    use_rs_slope: bool = False,    # opcional: añadir RS slope>0 como check extra
    rs_min_slope: float = 0.0
) -> pd.DataFrame:
    rows = []
    if panel is None:
        return pd.DataFrame(columns=["symbol","signal_breakout"])

    bench_close = None
    if isinstance(bench_series, pd.Series):
        bench_close = bench_series.astype(float).copy()
        bench_close.index = pd.to_datetime(bench_close.index)

    for sym, df in panel.items():
        if df is None or df.empty:
            rows.append({"symbol": sym, "signal_breakout": False}); continue
        dfx = df.copy().sort_index()

        rvol = _rvol_last(dfx, rvol_lookback)
        cpos = _closepos_last(dfx)
        p52  = _p52_last(dfx, 252)
        udr  = _updown_vol_ratio(dfx, 20)
        rs_sl = _rs_ma20_slope(dfx, bench_close) if bench_close is not None else np.nan

        c_rvol = bool(np.isfinite(rvol) and (rvol >= rvol_th))
        c_cpos = bool(np.isfinite(cpos) and (cpos >= closepos_th))
        c_p52  = bool(np.isfinite(p52)  and (p52  >= p52_th))
        c_udr  = bool(np.isfinite(udr)  and (udr  >= updown_vol_th))

        hits = int(c_rvol) + int(c_cpos) + int(c_p52) + int(c_udr)

        # opcional: RS slope como check extra
        c_rss = False
        if use_rs_slope and np.isfinite(rs_sl):
            c_rss = (rs_sl > rs_min_slope)
            hits += int(c_rss)

        sig = (hits >= int(min_hits))

        rows.append({
            "symbol": sym,
            "RVOL20": rvol,
            "ClosePos": cpos,
            "P52": p52,
            "UDVol20": udr,
            "rs_ma20_slope": rs_sl,
            "c_RVOL": c_rvol,
            "c_ClosePos": c_cpos,
            "c_P52": c_p52,
            "c_UDVol": c_udr,
            "c_RSslope": c_rss,
            "hits": hits,
            "signal_breakout": bool(sig)
        })

    return pd.DataFrame(rows).drop_duplicates("symbol", keep="last")



def _ma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _mom_12_1(close: pd.Series) -> pd.Series:
    # ~ 12–1 momentum: precio t-21 vs t-252
    return close.shift(21) / close.shift(252) - 1.0

def _safe_last(x: pd.Series, default=np.nan):
    try:
        return x.iloc[-1]
    except Exception:
        return default

# ----------------- Tendencia -----------------
def apply_trend_filter(
    panel: dict[str, pd.DataFrame],
    use_and_condition: bool = False
) -> pd.DataFrame:
    """
    Calcula MA200 y Mom 12–1 por símbolo y entrega señal de tendencia.
    - OR (por defecto): signal_trend = (close > MA200) OR (Mom12–1 > 0)
    - AND: signal_trend = (close > MA200) AND (Mom12–1 > 0)

    Retorna DataFrame:
      ['symbol','last_date','last_close','ma200','mom_12_1','cond_ma','cond_mom','signal_trend']
    """
    rows = []
    if panel is None:
        return pd.DataFrame(columns=["symbol","signal_trend"])

    for sym, df in panel.items():
        if df is None or df.empty or "close" not in df.columns:
            rows.append({"symbol": sym, "signal_trend": False})
            continue

        dfx = df.copy()
        # asegurar orden
        dfx = dfx.sort_index()
        # MA200
        dfx["ma200"] = _ma(dfx["close"].astype(float), 200)
        # Mom 12–1
        dfx["mom_12_1"] = _mom_12_1(dfx["close"].astype(float))

        last = dfx.tail(1)
        if last.empty:
            rows.append({"symbol": sym, "signal_trend": False})
            continue

        last_close = float(_safe_last(last["close"]))
        ma200      = float(_safe_last(last["ma200"]))
        mom12_1    = float(_safe_last(last["mom_12_1"]))
        last_date  = _safe_last(last.index)

        cond_ma  = (last_close > ma200) if np.isfinite(ma200) else False
        cond_mom = (mom12_1 > 0)       if np.isfinite(mom12_1) else False
        signal   = (cond_ma and cond_mom) if use_and_condition else (cond_ma or cond_mom)

        rows.append({
            "symbol": sym,
            "last_date": last_date,
            "last_close": last_close,
            "ma200": ma200,
            "mom_12_1": mom12_1,
            "cond_ma": bool(cond_ma),
            "cond_mom": bool(cond_mom),
            "signal_trend": bool(signal),
        })

    return pd.DataFrame(rows).drop_duplicates("symbol", keep="last")


# ----------------- Breakout (mínimo potente) -----------------
def _rvol_last(d: pd.DataFrame, lookback: int = 20) -> float:
    if "volume" not in d.columns: return np.nan
    v = d["volume"].astype(float)
    med = v.shift(1).rolling(lookback, min_periods=lookback).median()
    return float((v / med).iloc[-1]) if med.notna().iloc[-1] else np.nan

def _closepos_last(d: pd.DataFrame) -> float:
    # (close - low) / (high - low)
    if not set(["high","low","close"]).issubset(d.columns): return np.nan
    h = float(d["high"].iloc[-1]); l = float(d["low"].iloc[-1]); c = float(d["close"].iloc[-1])
    rng = (h - l)
    return float((c - l) / rng) if rng and np.isfinite(rng) and rng != 0 else np.nan

def _p52_last(d: pd.DataFrame, lookback: int = 252) -> float:
    if "high" not in d.columns or "close" not in d.columns: return np.nan
    hh = float(d["high"].tail(lookback).max())
    c  = float(d["close"].iloc[-1])
    return float(c / hh) if hh and np.isfinite(hh) and hh != 0 else np.nan

def _updown_vol_ratio(d: pd.DataFrame, lookback: int = 20) -> float:
    if not set(["close","volume"]).issubset(d.columns): return np.nan
    dd = d.tail(lookback+1).copy()
    dd["ret"] = dd["close"].pct_change()
    up_vol   = dd.loc[dd["ret"] > 0, "volume"].sum()
    down_vol = dd.loc[dd["ret"] < 0, "volume"].sum()
    if down_vol == 0:
        return np.inf if up_vol > 0 else np.nan
    return float(up_vol / down_vol)

def _rs_ma20_slope(d_asset: pd.DataFrame, bench_close: pd.Series | None) -> float:
    """Pendiente de MA20 del RS (asset/bench) usando regresión simple sobre los últimos 20 puntos."""
    if bench_close is None or "close" not in d_asset.columns: return np.nan
    s = d_asset["close"].reindex_like(bench_close).fillna(method="ffill")
    rs = (s / bench_close).dropna()
    if rs.shape[0] < 25:
        return np.nan
    ma20 = rs.rolling(20, min_periods=20).mean().dropna().tail(20)
    if ma20.empty: return np.nan
    y = ma20.values
    x = np.arange(len(y), dtype=float)
    # pendiente normalizada
    num = ((x - x.mean()) * (y - y.mean())).sum()
    den = ((x - x.mean())**2).sum()
    return float(num / den) if den != 0 else np.nan

def enrich_with_breakout(
    panel: dict[str, pd.DataFrame],
    rvol_lookback: int = 20,
    rvol_th: float = 1.5,
    closepos_th: float = 0.6,
    p52_th: float = 0.95,
    updown_vol_th: float = 1.2,
    bench_series: pd.Series | None = None,
    min_hits: int = 4,             # NUEVO: K-de-4
    use_rs_slope: bool = False,    # opcional: check extra
    rs_min_slope: float = 0.0
) -> pd.DataFrame:
    rows = []
    if panel is None:
        return pd.DataFrame(columns=["symbol","signal_breakout"])

    bench_close = None
    if isinstance(bench_series, pd.Series):
        bench_close = bench_series.astype(float).copy()
        bench_close.index = pd.to_datetime(bench_close.index)

    for sym, df in panel.items():
        if df is None or df.empty:
            rows.append({"symbol": sym, "signal_breakout": False}); continue
        dfx = df.copy().sort_index()

        # --- métricas últimas ---
        rvol = _rvol_last(dfx, rvol_lookback)
        cpos = _closepos_last(dfx)
        p52  = _p52_last(dfx, 252)
        udr  = _updown_vol_ratio(dfx, 20)
        rs_sl = _rs_ma20_slope(dfx, bench_close) if bench_close is not None else np.nan

        # --- checks individuales ---
        c_rvol = bool(np.isfinite(rvol) and (rvol >= rvol_th))
        c_cpos = bool(np.isfinite(cpos) and (cpos >= closepos_th))
        c_p52  = bool(np.isfinite(p52)  and (p52  >= p52_th))
        c_udr  = bool(np.isfinite(udr)  and (udr  >= updown_vol_th))

        hits = int(c_rvol) + int(c_cpos) + int(c_p52) + int(c_udr)

        c_rss = False
        if use_rs_slope and np.isfinite(rs_sl):
            c_rss = (rs_sl > rs_min_slope)
            hits += int(c_rss)

        sig = (hits >= int(min_hits))

        rows.append({
            "symbol": sym,
            "RVOL20": rvol,
            "ClosePos": cpos,
            "P52": p52,
            "UDVol20": udr,
            "rs_ma20_slope": rs_sl,
            "c_RVOL": c_rvol,
            "c_ClosePos": c_cpos,
            "c_P52": c_p52,
            "c_UDVol": c_udr,
            "c_RSslope": c_rss,
            "hits": hits,
            "signal_breakout": bool(sig)
        })

    return pd.DataFrame(rows).drop_duplicates("symbol", keep="last")
