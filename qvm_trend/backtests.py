# === backtests.py (ADD) ===
from __future__ import annotations
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass

# --------- Métricas básicas (si no las tienes en stats.py) ----------
def _cagr(returns: pd.Series, freq_per_year=252) -> float:
    if len(returns) == 0:
        return 0.0
    equity = (1 + returns.fillna(0)).cumprod()
    n_years = len(returns) / freq_per_year
    if n_years <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return equity.iloc[-1] ** (1.0 / n_years) - 1.0

def _sharpe(returns: pd.Series, freq_per_year=252) -> float:
    mu = returns.mean() * freq_per_year
    sd = returns.std(ddof=0) * math.sqrt(freq_per_year)
    return 0.0 if sd == 0 or np.isnan(sd) else mu / sd

def _sortino(returns: pd.Series, freq_per_year=252) -> float:
    downside = returns[returns < 0]
    dd = downside.std(ddof=0) * math.sqrt(freq_per_year)
    mu = returns.mean() * freq_per_year
    return 0.0 if dd == 0 or np.isnan(dd) else mu / dd

def _maxdd(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min() if len(dd) else 0.0

# --------------- Señales: MA200 y Momentum 12-1 ----------------------
def _ensure_sig_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'ma200' not in out.columns:
        out['ma200'] = out['close'].rolling(200).mean()
    if 'mom_12_1' not in out.columns:
        # momentum 12-1: precio actual vs precio de hace 252-21 días aprox
        lag_12 = out['close'].shift(252)
        lag_1  = out['close'].shift(21)
        out['mom_12_1'] = (lag_1 / lag_12) - 1.0
    return out

def _passes_signal(row, use_and_condition: bool):
    cond_ma  = pd.notna(row.get('ma200')) and row['close'] > row['ma200']
    cond_mom = pd.notna(row.get('mom_12_1')) and row['mom_12_1'] > 0
    return (cond_ma and cond_mom) if use_and_condition else (cond_ma or cond_mom)

@dataclass
class BTResult:
    symbol: str
    cagr: float
    sharpe: float
    sortino: float
    maxdd: float
    turnover: float
    n_trades: int
    equity: pd.Series
    rets: pd.Series

def backtest_single_symbol(df_price: pd.DataFrame,
                           symbol: str,
                           cost_bps: int = 10,
                           lag_days: int = 0,
                           use_and_condition: bool = False,
                           rebalance_freq: str = 'M') -> BTResult:
    """
    Backtest long-only con regla de tendencia (MA200 OR/AND Mom 12-1>0).
    Rebalancea con frecuencia 'M' por simplicidad; aplica lag sobre precios.
    """
    if df_price is None or df_price.empty:
        return BTResult(symbol, 0,0,0,0,0,0, pd.Series(dtype=float), pd.Series(dtype=float))

    df = df_price[['close']].copy().sort_index()
    df = _ensure_sig_cols(df)

    lag = pd.Timedelta(days=int(lag_days)) if lag_days else pd.Timedelta(0)

    # Rebalanceo mensual
    month_ends = df.resample(rebalance_freq).last().index

    equity = [1.0]
    rets = []
    weights_prev = 0.0  # 0 o 1 (in/out)
    turnover_hist = []
    n_trades = 0

    for t0, t1 in zip(month_ends[:-1], month_ends[1:]):
        row_t0 = df.loc[:t0].iloc[-1]
        in_signal = _passes_signal(row_t0, use_and_condition)

        weight_new = 1.0 if in_signal else 0.0

        # turnover (long-only binario)
        tw = abs(weight_new - weights_prev) * 0.5  # convención
        turnover_hist.append(tw)
        if weight_new != weights_prev:
            n_trades += 1

        # precios con lag
        p0 = df.loc[:t0+lag]['close'].iloc[-1]
        p1 = df.loc[:t1+lag]['close'].iloc[-1]
        r = 0.0
        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
            gross = (p1/p0 - 1.0) * weight_new
            cost = tw * (cost_bps / 1e4)
            r = gross - cost

        rets.append(r)
        equity.append(equity[-1] * (1.0 + r))
        weights_prev = weight_new

    rets = pd.Series(rets, index=month_ends[1:], name=symbol)
    equity = pd.Series(equity, index=[month_ends[0]] + list(month_ends[1:]), name=symbol)
    cagr = _cagr(rets, freq_per_year=12)
    sharpe = _sharpe(rets, freq_per_year=12)
    sortino = _sortino(rets, freq_per_year=12)
    maxdd = _maxdd(equity)
    turnover = float(np.mean(turnover_hist)) if turnover_hist else 0.0

    return BTResult(symbol, cagr, sharpe, sortino, maxdd, turnover, n_trades, equity, rets)

def backtest_many(panel: dict[str, pd.DataFrame],
                  symbols: list[str],
                  cost_bps: int = 10,
                  lag_days: int = 0,
                  use_and_condition: bool = False,
                  rebalance_freq: str = 'M') -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """
    Corre backtest por símbolo y devuelve tabla de métricas + curvas de equity.
    """
    rows = []
    curves = {}
    for s in symbols:
        df = panel.get(s)
        if df is None or df.empty:
            continue
        res = backtest_single_symbol(df, s, cost_bps, lag_days, use_and_condition, rebalance_freq)
        rows.append({
            'symbol': s,
            'CAGR': res.cagr,
            'Sharpe': res.sharpe,
            'Sortino': res.sortino,
            'MaxDD': res.maxdd,
            'Turnover': res.turnover,
            'Trades': res.n_trades
        })
        curves[s] = res.equity
    metrics = pd.DataFrame(rows).sort_values('CAGR', ascending=False)
    return metrics, curves
