import pandas as pd
import numpy as np
from .data_io import run_fmp_screener, load_prices_panel, load_benchmark, load_float_map, DEFAULT_START, DEFAULT_END
from .factors import compute_breakout_features
from .scoring import breakout_score, entry_signal, DEFAULT_TH


def _ma(s: pd.Series, n=200):
    return s.rolling(n, min_periods=n).mean()


def _mom_12_1(s: pd.Series):
    return s.shift(21) / s.shift(252) - 1


def apply_trend_filter(panel: dict, use_and=False):
    elig = []
    for sym, df in panel.items():
        if df is None or df.empty:
            continue
        close = df['close']
        ma200 = _ma(close, 200)
        mom = _mom_12_1(close)
        row = pd.DataFrame({"close": close, "ma200": ma200, "mom": mom}).dropna().tail(1)
        if row.empty:
            continue
        cond_ma = row['close'].iloc[-1] > row['ma200'].iloc[-1]
        cond_mo = row['mom'].iloc[-1] > 0
        pass_sig = (cond_ma and cond_mo) if use_and else (cond_ma or cond_mo)
        if pass_sig:
            elig.append(sym)
    return elig


def run_pipeline_vfq(df_universe: pd.DataFrame, start=DEFAULT_START, end=DEFAULT_END,
                     use_and_condition=False):
    # Paso 1: precios de universo
    symbols = df_universe["symbol"].dropna().astype(str).unique().tolist()
    panel = load_prices_panel(symbols, start, end)

    # Paso 2: filtro tendencia
    eligibles = apply_trend_filter(panel, use_and=use_and_condition)
    cartera = pd.DataFrame({"symbol": eligibles})

    # (en esta versión mínima no re-computamos VFQ: conservamos placeholder)
    cartera["VFQ"] = np.nan
    df_vfq = df_universe.copy()
    df_vfq_sel = df_universe[df_universe["symbol"].isin(eligibles)].copy()
    return df_vfq, df_vfq_sel, cartera


def enrich_with_breakout(df_sel: pd.DataFrame, price_map: dict, benchmark_series=None, float_map=None, th=DEFAULT_TH,
                          min_score=0.6):
    rows = []
    for sym in df_sel["symbol"].tolist():
        dfp = price_map.get(sym)
        if dfp is None or dfp.empty:
            continue
        bench = None
        if benchmark_series is not None:
            bench = benchmark_series.reindex(dfp.index).ffill()
        shares_float = None
        if float_map is not None:
            shares_float = float_map.get(sym)
        feat, _ = compute_breakout_features(dfp, benchmark=bench, shares_float=shares_float)
        score, tests = breakout_score(feat, th)
        signal = entry_signal(score, tests, min_score=min_score)
        row = {"symbol": sym, "BreakoutScore": score, "EntrySignal": signal}
        row.update({
            "RVOL20": feat.rvol20, "ClosePos": feat.closepos, "P52": feat.p52,
            "TSMOM20": feat.tsmom20, "TSMOM63": feat.tsmom63, "MA20_slope": feat.ma20_slope,
            "OBV_slope20": feat.obv_slope20, "ADL_slope20": feat.adl_slope20,
            "UDVolRatio20": feat.updown_vol_ratio20, "RS_MA20_slope": feat.rs_ma20_slope,
            "ATR_pct_rank": feat.atr_pct_rank, "GapHold": feat.gap_hold,
            "FloatVelocity": feat.float_velocity
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    return df_sel.merge(out, on="symbol", how="left")


# ===============================
# qvm_trend/backtests.py
# ===============================
import numpy as np
import pandas as pd
from .pipeline import _ma, _mom_12_1


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    df = pd.DataFrame(index=idx)
    return df.resample("M").last().index


def build_price_panel(symbols, loader_fn, start, end):
    panel = {}
    for s in symbols:
        df = loader_fn(s, start, end)
        if df is not None and not df.empty:
            x = pd.DataFrame({"close": df['close']})
            x["ma200"] = _ma(x['close'], 200)
            x["mom_12_1"] = _mom_12_1(x['close'])
            panel[s] = x
    return panel


def backtest_vfq_trend_v2(
    df_symbols: pd.DataFrame,
    price_loader,
    start="2020-01-01",
    end=None,
    hold_top_k=None,
    rebalance_freq="M",
    cost_bps=15,
    use_and_condition=False,
    lag_days=60,
    plot=False
):
    symbols = df_symbols["symbol"].dropna().astype(str).unique().tolist()
    panel = build_price_panel(symbols, price_loader, start, end)

    all_idx = pd.DatetimeIndex(sorted(set().union(*[df.index for df in panel.values()])))
    mes_ends = _month_ends(all_idx)
    if len(mes_ends) < 2:
        raise ValueError("Muy pocos puntos mensuales para backtest.")

    equity = [1.0]
    weights_prev = {s: 0.0 for s in symbols}
    turnover_hist, npos_hist, rets_hist = [], [], []

    for t0, t1 in zip(mes_ends[:-1], mes_ends[1:]):
        eligibles = []
        for sym, df in panel.items():
            if df.index[0] > t0:
                continue
            row = df.loc[:t0].iloc[-1]
            cond_ma = (row['close'] > row['ma200']) if pd.notna(row['ma200']) else False
            cond_mom = (row['mom_12_1'] > 0) if pd.notna(row['mom_12_1']) else False
            pass_sig = (cond_ma and cond_mom) if use_and_condition else (cond_ma or cond_mom)
            if pass_sig:
                eligibles.append(sym)
        chosen = eligibles if hold_top_k is None else eligibles[:hold_top_k]
        npos = len(chosen)
        weights_new = {s: (1.0/npos if s in chosen and npos>0 else 0.0) for s in symbols}

        tw = 0.5 * sum(abs(weights_new[s] - weights_prev.get(s, 0.0)) for s in symbols)
        turnover_hist.append(tw)
        npos_hist.append(npos)

        r_month = 0.0
        if npos > 0:
            ret_sum = 0.0
            for sym in symbols:
                wgt = weights_new[sym]
                if wgt == 0.0:
                    continue
                df = panel[sym]
                if df.index[0] > t0:
                    continue
                p0 = df.loc[:t0]['close'].iloc[-1]
                p1 = df.loc[:t1]['close'].iloc[-1]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    ret_sum += wgt * (p1/p0 - 1)
            cost = tw * (cost_bps / 1e4)
            r_month = ret_sum - cost
        rets_hist.append(r_month)
        equity.append(equity[-1] * (1 + r_month))
        weights_prev = weights_new

    eq = pd.Series(equity, index=pd.Index(mes_ends, name="date")).iloc[1:]
    rets = pd.Series(rets_hist, index=mes_ends[:-1])

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1/years) - 1 if years > 0 else np.nan
    mu, sd = rets.mean(), rets.std()
    sharpe = (mu*12)/sd if sd and sd>0 else np.nan
    dd = rets[rets<0].std()
    sortino = (mu*12)/dd if dd and dd>0 else np.nan
    rollmax = eq.cummax(); maxdd = (eq/rollmax - 1).min()

    summary = {
        "Inicio": eq.index[0].date().isoformat(),
        "Fin": eq.index[-1].date().isoformat(),
        "CAGR": float(cagr),
        "Sharpe_anual": float(sharpe),
        "Sortino_anual": float(sortino),
        "MaxDD": float(maxdd),
        "Turnover_medio": float(np.mean(turnover_hist)),
        "N_posiciones_medio": float(np.mean(npos_hist)),
        "Periodos": int(len(rets)),
        "Coste_bps": int(cost_bps),
        "Regla_tendencia": "MA200 AND 12-1>0" if use_and_condition else "MA200 OR 12-1>0",
        "Rebalance": "Mensual",
        "TopK": hold_top_k if hold_top_k is not None else "all",
        "Lag_dias": int(lag_days)
    }
    return eq, rets, summary
