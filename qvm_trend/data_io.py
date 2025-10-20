
# ===============================
# qvm_trend/data_io.py
# ===============================
import os
import time
import requests
import pandas as pd
import numpy as np
from functools import lru_cache
from datetime import datetime

FMP_API_KEY = os.getenv("FMP_API_KEY", "YOUR_FMP_KEY_HERE")
DEFAULT_START = "2020-01-01"
DEFAULT_END = datetime.today().strftime("%Y-%m-%d")

EXCHANGES_OK = {
    "NASDAQ", "Nasdaq", "NasdaqGS", "NasdaqGM",
    "NYSE", "NYSE ARCA", "NYSE Arca", "NYSE American",
    "AMEX", "BATS"
}

IPO_MIN_DAYS_DEFAULT = 365


def _http_get(url: str, params: dict | None = None, sleep: float = 0.0):
    if sleep > 0:
        time.sleep(sleep)
    params = params or {}
    params["apikey"] = FMP_API_KEY
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def clean_symbol(sym: str) -> str:
    return (sym or "").strip().upper()


def run_fmp_screener(limit=200, min_mcap=5e8, ipo_min_days=IPO_MIN_DAYS_DEFAULT):
    """Screener base y perfiles mínimos (sector, exchange, ipoDate)."""
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

    symbols = df["symbol"].dropna().unique().tolist()
    profiles = []
    for sym in symbols:
        try:
            prof = _http_get(f"https://financialmodelingprep.com/api/v3/profile/{sym}")
            if isinstance(prof, list) and prof:
                p0 = prof[0]
                profiles.append({
                    "symbol": sym,
                    "sector": p0.get("sector"),
                    "industry": p0.get("industry"),
                    "marketCap_profile": p0.get("mktCap") or p0.get("marketCap"),
                    "price_profile": p0.get("price"),
                    "exchange": p0.get("exchangeShortName") or p0.get("exchange"),
                    "type": p0.get("type"),
                    "country": p0.get("country"),
                    "ipoDate": p0.get("ipoDate"),
                })
        except Exception:
            continue

    dfp = pd.DataFrame(profiles)
    df = df.merge(dfp, on="symbol", how="left")

    # Completa y limpia
    for col in ["marketCap", "marketCap_profile", "price", "price_profile"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["marketCap"] = df["marketCap"].fillna(df["marketCap_profile"]) \
                                    .fillna(df["price_profile"] * np.nan)  # noop si falta shares

    for c in ["sector", "industry"]:
        if c not in df.columns:
            df[c] = "Unknown"
        df[c] = df[c].fillna("Unknown").astype(str)

    for c in ["type", "exchange", "country"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    # filtros básicos de universo
    before = len(df)
    typ = df["type"].fillna("").str.lower()
    ok_type = (typ.str.contains("stock") | typ.str.contains("equity") | (typ == ""))
    df = df[ok_type].copy()
    exch = df["exchange"].fillna("")
    df = df[(exch.isin(EXCHANGES_OK)) | (exch == "")].copy()

    df["ipoDate"] = pd.to_datetime(df.get("ipoDate"), errors="coerce")
    today = pd.Timestamp.today().normalize()
    df = df[(df["ipoDate"].isna()) | (df["ipoDate"] <= today - pd.Timedelta(days=ipo_min_days))].copy()

    df["symbol"] = df["symbol"].astype(str).apply(clean_symbol)
    df = df.dropna(subset=["symbol"]).drop_duplicates("symbol")

    return df


def get_prices_fmp(symbol: str, start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.DataFrame | None:
    sym = clean_symbol(symbol)
    try:
        base = f"https://financialmodelingprep.com/api/v3/historical-price-full/{sym}"
        j = _http_get(base, params={"from": start, "to": end})
        hist = j.get("historical", [])
        if not isinstance(hist, list) or len(hist) == 0:
            j2 = _http_get(base)
            hist = j2.get("historical", [])
        if not hist:
            return None
        dfp = pd.DataFrame(hist)
        needed = ["date", "open", "high", "low", "close", "volume"]
        for c in needed:
            if c not in dfp.columns:
                return None
        dfp["date"] = pd.to_datetime(dfp["date"])
        dfp = dfp.sort_values("date").set_index("date")
        return dfp[needed]
    except Exception:
        return None


def load_prices_panel(symbols, start=DEFAULT_START, end=DEFAULT_END):
    panel = {}
    for s in symbols:
        dfp = get_prices_fmp(s, start, end)
        if dfp is None or dfp.empty:
            continue
        panel[s] = dfp
    return panel


def load_benchmark(symbol: str, start=DEFAULT_START, end=DEFAULT_END) -> pd.Series | None:
    df = get_prices_fmp(symbol, start, end)
    if df is None or df.empty:
        return None
    return df["close"]


def load_float_map(symbols):
    out = {}
    for sym in symbols:
        try:
            j = _http_get(f"https://financialmodelingprep.com/api/v3/shares_float/{sym}")
            if isinstance(j, dict) and j.get("symbol"):
                out[sym] = j.get("floatShares") or j.get("freeFloat")
            elif isinstance(j, list) and j:
                out[sym] = j[0].get("floatShares") or j[0].get("freeFloat")
        except Exception:
            out[sym] = None
    return out


# ===============================
# qvm_trend/factors.py
# ===============================
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class BreakoutFeatures:
    rvol20: float
    closepos: float
    p52: float
    tsmom20: float
    tsmom63: float
    ma20_slope: float
    obv_slope20: float
    adl_slope20: float
    updown_vol_ratio20: float
    rs_ma20_slope: Optional[float]
    atr_pct_rank: float
    gap_hold: bool
    float_velocity: Optional[float]


def _slope_linear(y: pd.Series) -> float:
    y = y.dropna()
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float(np.polyfit(x, y, 1)[0])


def rolling_slope(y: pd.Series, win=20) -> pd.Series:
    return y.rolling(win).apply(lambda s: _slope_linear(pd.Series(s)), raw=False)


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['close'].diff().fillna(0.0))
    return (direction * df['volume']).fillna(0.0).cumsum()


def accumulation_distribution_line(df: pd.DataFrame) -> pd.Series:
    clv = ( (df['close'] - df['low']) - (df['high'] - df['close']) ) / ((df['high'] - df['low']).replace(0, np.nan))
    clv = clv.fillna(0.0)
    return (clv * df['volume']).cumsum()


def atr(df: pd.DataFrame, n=14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def percent_rank(s: pd.Series, lookback: int) -> pd.Series:
    return s.rolling(lookback).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])


def compute_breakout_features(
    df: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    shares_float: Optional[float] = None
) -> Tuple[BreakoutFeatures, Dict[str, pd.Series]]:
    d = df.copy()
    rvol20_series = d['volume'] / d['volume'].shift(1).rolling(20).median()
    rvol20 = float(rvol20_series.iloc[-1])

    rng = (d['high'] - d['low']).replace(0, np.nan)
    closepos_series = (d['close'] - d['low']) / rng
    closepos = float(closepos_series.iloc[-1])

    p52_series = d['close'] / d['high'].rolling(252).max()
    p52 = float(p52_series.iloc[-1])

    tsmom20_series = d['close'] / d['close'].shift(20) - 1
    tsmom63_series = d['close'] / d['close'].shift(63) - 1
    tsmom20 = float(tsmom20_series.iloc[-1])
    tsmom63 = float(tsmom63_series.iloc[-1])

    ma20 = d['close'].rolling(20).mean()
    ma20_slope_series = rolling_slope(ma20, win=20)
    ma20_slope = float(ma20_slope_series.iloc[-1])

    obv = on_balance_volume(d)
    adl = accumulation_distribution_line(d)
    obv_slope20_series = rolling_slope(obv, win=20)
    adl_slope20_series = rolling_slope(adl, win=20)
    obv_slope20 = float(obv_slope20_series.iloc[-1])
    adl_slope20 = float(adl_slope20_series.iloc[-1])

    up = d['volume'].where(d['close'] > d['close'].shift(), 0.0)
    dn = d['volume'].where(d['close'] < d['close'].shift(), 0.0)
    updown_vol_ratio20_series = up.rolling(20).sum() / (dn.rolling(20).sum().replace(0, np.nan))
    updown_vol_ratio20 = float(updown_vol_ratio20_series.iloc[-1])

    rs_ma20_slope_val = None
    rs_ma20_slope_series = None
    if benchmark is not None:
        rs = (d['close'] / benchmark).dropna()
        rs_ma20 = rs.rolling(20).mean()
        rs_ma20_slope_series = rolling_slope(rs_ma20, win=20)
        rs_ma20_slope_val = float(rs_ma20_slope_series.iloc[-1])

    atr14 = atr(d, 14)
    atr_pct_rank_series = percent_rank(atr14, lookback=252)
    atr_pct_rank_val = float(atr_pct_rank_series.iloc[-1])

    prev_high = d['high'].shift()
    gap = d['open'] > prev_high
    gap_hold_series = gap & (closepos_series >= 0.6) & (rvol20_series >= 1.5)
    gap_hold = bool(gap_hold_series.iloc[-1])

    float_velocity_val = None
    med_dollar_vol_60 = None
    if shares_float is not None and shares_float > 0:
        med_dollar_vol_60 = (d['close']*d['volume']).rolling(60).median()
        float_velocity_series = med_dollar_vol_60 / (shares_float * d['close'])
        float_velocity_val = float(float_velocity_series.iloc[-1])

    features = BreakoutFeatures(
        rvol20=rvol20,
        closepos=closepos,
        p52=p52,
        tsmom20=tsmom20,
        tsmom63=tsmom63,
        ma20_slope=ma20_slope,
        obv_slope20=obv_slope20,
        adl_slope20=adl_slope20,
        updown_vol_ratio20=updown_vol_ratio20,
        rs_ma20_slope=rs_ma20_slope_val,
        atr_pct_rank=atr_pct_rank_val,
        gap_hold=gap_hold,
        float_velocity=float_velocity_val
    )
    series_map = {
        "rvol20": rvol20_series,
        "closepos": closepos_series,
        "p52": p52_series,
        "tsmom20": tsmom20_series,
        "tsmom63": tsmom63_series,
        "ma20": ma20,
        "ma20_slope": ma20_slope_series,
        "obv": obv,
        "adl": adl,
        "obv_slope20": obv_slope20_series,
        "adl_slope20": adl_slope20_series,
        "updown_vol_ratio20": updown_vol_ratio20_series,
        "atr14": atr14,
        "atr_pct_rank": atr_pct_rank_series,
        "gap_hold": gap_hold_series
    }
    if rs_ma20_slope_series is not None:
        series_map["rs_ma20_slope"] = rs_ma20_slope_series
    if shares_float is not None and shares_float > 0 and med_dollar_vol_60 is not None:
        series_map["float_velocity"] = med_dollar_vol_60 / (shares_float * d['close'])
    return features, series_map


# ===============================
# qvm_trend/scoring.py
# ===============================
from typing import Dict, Tuple
from .factors import BreakoutFeatures

DEFAULT_TH = {
    "rvol_min": 1.5,
    "closepos_min": 0.60,
    "p52_min": 0.95,
    "ud_vol_min": 1.2,
    "rs_slope_min": 0.0,
    "atr_pct_min": 0.6,
    "float_vel_min": 0.01
}

WEIGHTS = {
    # ponderaciones (puedes ajustarlas desde la UI si quieres)
    "RVOL": 2.0,
    "ClosePos": 2.0,
    "P52": 1.5,
    "TSMOM20": 1.0,
    "TSMOM63": 1.0,
    "MA20_slope": 1.0,
    "OBV_slope20": 1.0,
    "ADL_slope20": 1.0,
    "UDVolRatio20": 1.0,
    "RS_MA20_slope": 1.0,
    "ATR_pct": 1.0,
    "GapHold": 1.0,
    "FloatVelocity": 1.0,
}


def breakout_score(feat: BreakoutFeatures, th: Dict, weights: Dict = WEIGHTS) -> Tuple[float, Dict[str, bool]]:
    f = feat
    tests = {
        "RVOL": f.rvol20 >= th["rvol_min"],
        "ClosePos": f.closepos >= th["closepos_min"],
        "P52": f.p52 >= th["p52_min"],
        "TSMOM20": f.tsmom20 > 0,
        "TSMOM63": f.tsmom63 > 0,
        "MA20_slope": (f.ma20_slope if f.ma20_slope is not None else -1) > 0,
        "OBV_slope20": (f.obv_slope20 if f.obv_slope20 is not None else -1) > 0,
        "ADL_slope20": (f.adl_slope20 if f.adl_slope20 is not None else -1) > 0,
        "UDVolRatio20": f.updown_vol_ratio20 >= th["ud_vol_min"],
        "RS_MA20_slope": (f.rs_ma20_slope if f.rs_ma20_slope is not None else -1) > th["rs_slope_min"],
        "ATR_pct": f.atr_pct_rank >= th["atr_pct_min"],
        "GapHold": bool(f.gap_hold)
    }
    if f.float_velocity is not None:
        tests["FloatVelocity"] = f.float_velocity >= th["float_vel_min"]

    # score ponderado
    w_sum = 0.0
    s_sum = 0.0
    for k, ok in tests.items():
        w = float(weights.get(k, 1.0))
        w_sum += w
        s_sum += (w if ok else 0.0)
    score = s_sum / w_sum if w_sum > 0 else 0.0
    return float(score), tests


def entry_signal(score: float, tests: Dict[str, bool], min_score=0.6) -> bool:
    core_ok = tests.get("RVOL", False) and tests.get("ClosePos", False) and tests.get("P52", False)
    return (score >= min_score) and core_ok


# ===============================

# ===============================
# app_streamlit.py
# ===============================
import os
import streamlit as st
import pandas as pd
from datetime import date
from qvm_trend.data_io import run_fmp_screener, load_prices_panel, load_benchmark, load_float_map, get_prices_fmp
from qvm_trend.pipeline import run_pipeline_vfq, enrich_with_breakout
from qvm_trend.backtests import backtest_vfq_trend_v2
from qvm_trend.scoring import DEFAULT_TH
from qvm_trend.mc import gbm_paths

st.set_page_config(page_title="QVM + Liquidez + Breakout", layout="wide")

st.sidebar.header("⚙️ Parámetros base")
limit = st.sidebar.number_input("Tamaño screener", 50, 1000, 200, step=50)
min_mcap = st.sidebar.number_input("Min MarketCap", 1e7, 1e11, 5e8, step=1e8, format="%.0f")
ipo_days = st.sidebar.number_input("IPO ≥ días", 0, 2000, 365, step=30)
start = st.sidebar.date_input("Start", value=date(2020,1,1))
end = st.sidebar.date_input("End", value=date.today())
bench_ticker = st.sidebar.selectbox("Benchmark", ["SPY","QQQ","^IPSA"], index=0)

@st.cache_data(show_spinner=False)
def _universe(limit, min_mcap, ipo_days):
    df = run_fmp_screener(limit=limit, min_mcap=min_mcap, ipo_min_days=ipo_days)
    if df.empty:
        st.warning("Screener vacío.")
    return df


def _thresholds_from_ui():
    col1, col2, col3 = st.columns(3)
    with col1:
        rvol_min = st.slider("RVOL mínimo", 1.0, 5.0, DEFAULT_TH["rvol_min"], 0.1)
        closepos_min = st.slider("ClosePos mínimo", 0.0, 1.0, DEFAULT_TH["closepos_min"], 0.05)
        p52_min = st.slider("Cercanía 52W (P52)", 0.80, 1.05, DEFAULT_TH["p52_min"], 0.01)
    with col2:
        ud_min = st.slider("Up/Down Vol Ratio", 0.5, 3.0, DEFAULT_TH["ud_vol_min"], 0.05)
        atr_pct_min = st.slider("ATR pct rank (12m)", 0.0, 1.0, DEFAULT_TH["atr_pct_min"], 0.05)
        rs_slope_min = st.slider("Pendiente RS(MA20) mín.", -1.0, 1.0, DEFAULT_TH["rs_slope_min"], 0.05)
    with col3:
        float_vel_min = st.slider("FloatVelocity (%/día)", 0.0, 0.05, DEFAULT_TH["float_vel_min"], 0.001)
        min_score = st.slider("Score de entrada mínimo", 0.0, 1.0, 0.6, 0.05)
    th = dict(DEFAULT_TH)
    th.update(dict(rvol_min=rvol_min, closepos_min=closepos_min, p52_min=p52_min,
                   ud_vol_min=ud_min, atr_pct_min=atr_pct_min, rs_slope_min=rs_slope_min,
                   float_vel_min=float_vel_min))
    return th, min_score


tab1, tab2, tab3, tab4 = st.tabs(["Universo", "Filtros", "Señales & Gráficos", "Backtest & Estadística"]) 

with tab1:
    st.subheader("1) Universo por Screener (FMP)")
    uni = _universe(limit, min_mcap, ipo_days)
    st.dataframe(uni[["symbol","sector","marketCap","exchange"]].head(500), use_container_width=True)

with tab2:
    st.subheader("2) Ajusta los umbrales del breakout")
    th, min_score = _thresholds_from_ui()
    st.info("Los umbrales se aplican en la pestaña 3 para calcular BreakoutScore y EntrySignal.")

with tab3:
    st.subheader("3) Señales de entrada y calidad del breakout")
    if uni.empty:
        st.stop()
    df_vfq, df_vfq_sel, cartera = run_pipeline_vfq(uni, start=start.isoformat(), end=end.isoformat())
    if cartera is None or cartera.empty:
        st.info("Sin finalistas tras filtro de tendencia. Ajusta parámetros o periodo.")
        st.stop()
    symbols = cartera["symbol"].unique().tolist()
    panel = load_prices_panel(symbols, start.isoformat(), end.isoformat())
    bench = load_benchmark(bench_ticker, start.isoformat(), end.isoformat())
    float_map = load_float_map(symbols)

    enriched = enrich_with_breakout(cartera, panel, benchmark_series=bench, float_map=float_map, th=th, min_score=min_score)
    st.dataframe(enriched.sort_values(["EntrySignal","BreakoutScore"], ascending=[False, False]), use_container_width=True)

    sym = st.selectbox("Ver símbolo", symbols)
    if sym:
        dfp = panel.get(sym)
        st.line_chart(dfp[["close"]])
        # Monte Carlo (GBM) opcional
        st.markdown("### Monte Carlo (20 días, GBM)")
        paths = gbm_paths(dfp["close"], horizon_days=20, n_sims=1000)
        st.line_chart(paths)

with tab4:
    st.subheader("4) Backtest (tendencia) & Métricas")
    if uni.empty:
        st.stop()
    df_vfq, df_vfq_sel, cartera = run_pipeline_vfq(uni, start=start.isoformat(), end=end.isoformat())
    if cartera is None or cartera.empty:
        st.info("No hay símbolos elegibles.")
        st.stop()
    universo = cartera[["symbol"]].drop_duplicates()
    eq, rets, summary = backtest_vfq_trend_v2(
        df_symbols=universo,
        price_loader=get_prices_fmp,
        start=start.isoformat(),
        end=end.isoformat(),
        hold_top_k=None,
        rebalance_freq="M",
        cost_bps=15,
        use_and_condition=False,
        lag_days=60,
        plot=False
    )
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("CAGR", f"{summary['CAGR']:.2%}")
    c2.metric("Sharpe (anual)", f"{summary['Sharpe_anual']:.2f}")
    c3.metric("MaxDD", f"{summary['MaxDD']:.2%}")
    c4.metric("N° medio posiciones", f"{summary['N_posiciones_medio']:.1f}")
    st.line_chart(eq.rename("Equity"))

st.caption("Tip: exporta las tablas desde los menús de Streamlit (⋮)")
