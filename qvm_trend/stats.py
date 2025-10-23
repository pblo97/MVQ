import numpy as np
import pandas as pd
from scipy import stats


def future_return(close: pd.Series, horizon=20):
    return close.shift(-horizon) / close - 1


def information_coefficient(df: pd.DataFrame, score_col="BreakoutScore", ret_col="ret_20"):
    s = df[[score_col, ret_col]].dropna()
    if len(s) < 10:
        return np.nan
    return stats.spearmanr(s[score_col], s[ret_col]).correlation

def beta_vs_bench(asset_returns: pd.Series, bench_returns: pd.Series) -> float:
    a = asset_returns.align(bench_returns, join='inner')[0]
    b = bench_returns.align(asset_returns, join='inner')[0]
    if len(a) < 10:
        return np.nan
    cov = np.cov(a, b)[0,1]
    var = np.var(b)
    return np.nan if var == 0 else cov / var

def expectancy(hit_rate: float, avg_win: float, avg_loss: float) -> float:
    # E = p*avg_win + (1-p)*avg_loss   (avg_loss es negativo)
    p = float(hit_rate)
    return p * avg_win + (1.0 - p) * avg_loss

def win_loss_stats(returns: pd.Series) -> tuple[float, float, float]:
    if returns is None or returns.empty:
        return 0.5, 0.02, -0.01
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    p = len(wins) / len(returns)
    avg_win = wins.mean() if len(wins) else 0.01
    avg_loss = losses.mean() if len(losses) else -0.01
    return p, avg_win, avg_loss