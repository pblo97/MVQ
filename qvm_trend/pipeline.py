# qvm_trend/pipeline.py
import numpy as np
import pandas as pd

# -------------------- Helpers técnicos --------------------
def _ma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _mom_12_1(close: pd.Series) -> pd.Series:
    # (t-21) / (t-252) - 1
    return close.shift(21) / close.shift(252) - 1

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _rvol(volume: pd.Series, lookback: int = 20) -> pd.Series:
    # Vol / mediana(Vol últimos 'lookback' días, excluyendo hoy)
    med = volume.shift(1).rolling(lookback, min_periods=max(5, lookback//2)).median()
    return volume / med

def _close_pos(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["low"]) / rng

def _p52(close: pd.Series, high: pd.Series, lookback: int = 252) -> pd.Series:
    hh = high.rolling(lookback, min_periods=int(lookback*0.6)).max()
    return close / hh

def _updown_vol_ratio(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    close = df["close"]
    vol = df["volume"].astype(float)
    up = (close.diff() > 0).astype(int)
    down = (close.diff() < 0).astype(int)
    up_vol = (vol * up).rolling(lookback, min_periods=max(5, lookback//2)).sum()
    down_vol = (vol * down).rolling(lookback, min_periods=max(5, lookback//2)).sum()
    return up_vol / down_vol.replace(0, np.nan)

def _rs_slope(asset_close: pd.Series, bench_close: pd.Series, ma: int = 20) -> float:
    # pendiente de la MA20 de RS (asset/bench) vía regresión simple (x = 0..n-1)
    if bench_close is None or bench_close.empty: 
        return np.nan
    rs = (asset_close / bench_close.reindex(asset_close.index).ffill()).dropna()
    if rs.empty: 
        return np.nan
    rs_ma = rs.rolling(ma, min_periods=ma).mean().dropna()
    if len(rs_ma) < ma: 
        return np.nan
    y = rs_ma.tail(ma).values
    x = np.arange(len(y))
    # pendiente normalizada (por nivel) para hacerlo adimensional
    denom = (x - x.mean())
    if denom.std() == 0: 
        return np.nan
    slope = np.polyfit(x, y, 1)[0]
    return float(slope / np.nanmean(y)) if np.nanmean(y) not in (0, np.nan) else float(slope)

def _atr_percentile(atr_series: pd.Series, window_days: int = 252) -> pd.Series:
    # percentil rolling del ATR(14) en los últimos ~12m
    def _pct_rank(a):
        if len(a) == 0 or np.all(np.isnan(a)): 
            return np.nan
        last = a[-1]
        valid = a[~np.isnan(a)]
        if len(valid) == 0: 
            return np.nan
        return (valid <= last).mean()
    return atr_series.rolling(window_days, min_periods=int(window_days*0.5)).apply(_pct_rank, raw=False)

# -------------------- 1) Filtro de tendencia --------------------
def apply_trend_filter(panel: dict[str, pd.DataFrame], use_and_condition: bool = False) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas:
      - symbol
      - signal_trend: bool (close > MA200) OR (Mom12–1 > 0)  [si use_and_condition=True -> AND]
      - close, ma200, mom_12_1  (último valor disponible)
    """
    rows = []
    for sym, df in (panel or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df = df.sort_index()
        c = df["close"].astype(float)
        ma200 = _ma(c, 200)
        mom = _mom_12_1(c)
        last = df.index.max()
        close_v = float(c.loc[last]) if last in c.index else np.nan
        ma_v = float(ma200.loc[last]) if last in ma200.index else np.nan
        mom_v = float(mom.loc[last]) if last in mom.index else np.nan

        cond_ma = (close_v > ma_v) if np.isfinite(ma_v) else False
        cond_mo = (mom_v > 0) if np.isfinite(mom_v) else False
        sig = (cond_ma and cond_mo) if use_and_condition else (cond_ma or cond_mo)
        rows.append({
            "symbol": sym,
            "signal_trend": bool(sig),
            "last_close": close_v,
            "ma200": ma_v,
            "mom_12_1": mom_v
        })
    if not rows:
        return pd.DataFrame(columns=["symbol","signal_trend","last_close","ma200","mom_12_1"])
    return pd.DataFrame(rows).drop_duplicates("symbol")

# -------------------- 2) Breakout enrich --------------------
def enrich_with_breakout(
    panel: dict[str, pd.DataFrame],
    rvol_lookback: int = 20,
    rvol_th: float = 1.2,
    closepos_th: float = 0.60,
    p52_th: float = 0.95,
    updown_vol_th: float = 1.2,
    bench_series: pd.Series | None = None,
    min_hits: int = 3,
    use_rs_slope: bool = False,
    rs_min_slope: float = 0.0
) -> pd.DataFrame:
    """
    Calcula métricas de breakout por símbolo y devuelve:
      - symbol
      - RVOL20, ClosePos, P52, UDVol20, ATR_pct, rs_ma20_slope
      - c_RVOL, c_ClosePos, c_P52, c_UDVol, c_RSslope (checks booleanos)
      - hits (suma de checks activos)
      - signal_breakout: True si hits >= min_hits
      - BreakoutScore: 0..100 (promedio ponderado simple de métricas normalizadas)
    """
    out = []
    for sym, df in (panel or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        d = df.sort_index().copy()
        # métricas
        rvol = _rvol(d["volume"].astype(float), rvol_lookback)
        cpos = _close_pos(d)
        p52 = _p52(d["close"].astype(float), d["high"].astype(float), 252)
        udr = _updown_vol_ratio(d, 20)
        atr14 = _atr(d, 14)
        atr_pct = _atr_percentile(atr14, 252)

        # RS slope (opcional)
        rs_slope = np.nan
        if use_rs_slope and bench_series is not None and isinstance(bench_series, pd.Series):
            rs_slope = _rs_slope(d["close"].astype(float), bench_series.astype(float), ma=20)

        last = d.index.max()
        RVOL20 = float(rvol.loc[last]) if last in rvol.index else np.nan
        ClosePos = float(cpos.loc[last]) if last in cpos.index else np.nan
        P52 = float(p52.loc[last]) if last in p52.index else np.nan
        UDVol20 = float(udr.loc[last]) if last in udr.index else np.nan
        ATRpct = float(atr_pct.loc[last]) if last in atr_pct.index else np.nan

        # checks
        c_RVOL = bool(np.isfinite(RVOL20) and RVOL20 >= rvol_th)
        c_ClosePos = bool(np.isfinite(ClosePos) and ClosePos >= closepos_th)
        c_P52 = bool(np.isfinite(P52) and P52 >= p52_th)
        c_UDVol = bool(np.isfinite(UDVol20) and UDVol20 >= updown_vol_th)
        if use_rs_slope:
            c_RSslope = bool(np.isfinite(rs_slope) and rs_slope > rs_min_slope)
        else:
            c_RSslope = False  # no cuenta si no se usa

        # hits (solo los checks activados por parámetros)
        checks = [c_RVOL, c_ClosePos, c_P52, c_UDVol] + ([c_RSslope] if use_rs_slope else [])
        hits = int(sum(checks))
        signal_breakout = (hits >= int(min_hits))

        # BreakoutScore simple (0..100): normalizar métricas "a ojo"
        sc = 0.0; w = 0.0
        if np.isfinite(RVOL20):
            sc += min(RVOL20/2.0, 1.0) * 25; w += 25
        if np.isfinite(ClosePos):
            sc += ClosePos * 25; w += 25
        if np.isfinite(P52):
            sc += max((P52-0.90)/(1.00-0.90), 0.0) * 25; w += 25
        if np.isfinite(UDVol20):
            sc += min(UDVol20/2.0, 1.0) * 15; w += 15
        if np.isfinite(ATRpct):
            sc += ATRpct * 10; w += 10
        if use_rs_slope and np.isfinite(rs_slope):
            sc += min(max(rs_slope, 0.0), 0.02) / 0.02 * 10; w += 10
        score = (sc / w * 100.0) if w > 0 else np.nan

        out.append({
            "symbol": sym,
            "RVOL20": RVOL20,
            "ClosePos": ClosePos,
            "P52": P52,
            "UDVol20": UDVol20,
            "ATR_pct": ATRpct,
            "rs_ma20_slope": rs_slope,
            "c_RVOL": c_RVOL,
            "c_ClosePos": c_ClosePos,
            "c_P52": c_P52,
            "c_UDVol": c_UDVol,
            "c_RSslope": c_RSslope if use_rs_slope else np.nan,
            "hits": hits,
            "signal_breakout": bool(signal_breakout),
            "BreakoutScore": score
        })

    if not out:
        return pd.DataFrame(columns=[
            "symbol","RVOL20","ClosePos","P52","UDVol20","ATR_pct","rs_ma20_slope",
            "c_RVOL","c_ClosePos","c_P52","c_UDVol","c_RSslope","hits","signal_breakout","BreakoutScore"
        ])
    return pd.DataFrame(out).drop_duplicates("symbol")

# -------------------- 3) Régimen de mercado --------------------
def market_regime_on(
    bench_px: pd.DataFrame | pd.Series | None,
    panel: dict[str, pd.DataFrame] | None,
    ma_bench: int = 200,
    breadth_ma: int = 50,
    breadth_min: float = 0.5
) -> bool:
    """
    Risk-ON si:
      - Benchmark (close) > MA(ma_bench)
      - Breadth: % de símbolos con close > MA(breadth_ma) >= breadth_min
    """
    # benchmark
    if bench_px is None:
        bench_ok = True  # si no hay benchmark, no bloqueamos
    else:
        if isinstance(bench_px, pd.DataFrame) and "close" in bench_px.columns:
            bclose = bench_px["close"].astype(float).sort_index()
        elif isinstance(bench_px, pd.Series):
            bclose = bench_px.astype(float).sort_index()
        else:
            bclose = pd.Series(dtype=float)
        if len(bclose) < ma_bench:
            bench_ok = True
        else:
            ma_b = _ma(bclose, ma_bench)
            last = bclose.index.max()
            bench_ok = bool(bclose.loc[last] > ma_b.loc[last])

    # breadth
    breadth_ok = True
    if panel and len(panel) > 0:
        flags = []
        for _, df in panel.items():
            if not isinstance(df, pd.DataFrame) or df.empty: 
                continue
            c = df["close"].astype(float).sort_index()
            if len(c) < breadth_ma: 
                continue
            ma = _ma(c, breadth_ma)
            last = c.index.max()
            flags.append(bool(c.loc[last] > ma.loc[last]))
        if len(flags) >= max(20, int(0.3*len(panel))):  # necesita un mínimo de símbolos
            breadth = np.mean(flags)
            breadth_ok = bool(breadth >= float(breadth_min))

    return bool(bench_ok and breadth_ok)
