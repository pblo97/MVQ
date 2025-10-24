import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf


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

def _winsor(s: pd.Series, p=0.02):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

def _ewm_mean_std(x: pd.Series, span=12):
    m = x.ewm(span=span, min_periods=max(3, span//3)).mean()
    v = (x - m).pow(2).ewm(span=span, min_periods=max(3, span//3)).mean()
    return m, np.sqrt(v)

def kelly_metrics_single(asset_ret_excess: pd.Series, costs_per_period=0.001, 
                         winsor_p=0.02, shrink_kappa=20, ewm_span=12):
    """
    asset_ret_excess: retornos del activo - benchmark (o RF) en misma frecuencia (mensual recomendado)
    costs_per_period: coste/derrape a restar a la media (p.ej. 0.001 = 0.1%)
    """
    r = asset_ret_excess.dropna()
    if r.size < 24:
        return dict(n=r.size, p=np.nan, payoff=np.nan, mu=np.nan, sigma=np.nan, 
                    k_bin=0.0, k_cont=0.0, k_raw=0.0)

    r = _winsor(r, p=winsor_p)
    # Hit rate / payoff en exceso
    gains = r[r > 0]
    losses = -r[r < 0]
    hits, misses = gains.size, losses.size
    p_emp = hits / (hits + misses) if hits + misses > 0 else 0.5
    payoff_emp = (gains.mean() / losses.mean()) if (hits > 0 and losses.size > 0 and losses.mean() > 0) else 1.0

    # Shrinkage bayesiano hacia 0.5 en p y 1.0 en payoff
    n = r.size
    p_hat = (p_emp * n + 0.5 * shrink_kappa) / (n + shrink_kappa)
    payoff_hat = (payoff_emp * n + 1.0 * shrink_kappa) / (n + shrink_kappa)

    # Kelly binomial (cap [0,1])
    k_bin = p_hat - (1 - p_hat) / max(payoff_hat, 1e-6)
    k_bin = float(np.clip(k_bin, 0.0, 1.0))

    # Media/vol robustas en exceso
    mu_ewm, sigma_ewm = _ewm_mean_std(r, span=ewm_span)
    mu = float(mu_ewm.iloc[-1] - costs_per_period)  # restamos costos
    sigma = float(sigma_ewm.iloc[-1])
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 1e-8:
        k_cont = 0.0
    else:
        k_cont = float(np.clip(mu / (sigma ** 2), 0.0, 1.0))  # cap prudente

    # Mezcla
    k_raw = 0.5 * k_bin + 0.5 * k_cont

    return dict(n=n, p=p_hat, payoff=payoff_hat, mu=mu, sigma=sigma, k_bin=k_bin, k_cont=k_cont, k_raw=k_raw)

def penalize_by_corr(k_series: pd.Series, ret_df: pd.DataFrame, lambda_corr=0.5):
    """
    Penaliza cada k por correlación con el 'proto-portfolio' (media de activos k>0).
    """
    k = k_series.clip(lower=0.0).copy()
    keep = k[k > 0].index
    if len(keep) < 2:
        return k
    proto = ret_df[keep].mean(axis=1).dropna()
    pen = {}
    for s in k.index:
        a = ret_df.get(s)
        if a is None or a.dropna().empty:
            pen[s] = 1.0
            continue
        c = pd.concat([proto, a], axis=1).dropna()
        if c.shape[0] < 12:
            pen[s] = 1.0
        else:
            rho = float(c.corr().iloc[0,1])
            pen[s] = 1.0 / (1.0 + lambda_corr * max(0.0, rho))
    return (k * pd.Series(pen)).fillna(0.0)

def kelly_vector_multi(mu_vec: np.ndarray, ret_mat: np.ndarray, f_kelly=0.25):
    """
    Kelly vectorial: w* ∝ Σ^{-1} μ  (excess returns). Devuelve pesos no negativos normalizados.
    """
    if ret_mat.shape[0] < 12 or ret_mat.shape[1] < 1:
        return np.zeros_like(mu_vec)
    lw = LedoitWolf().fit(ret_mat)
    Sigma_inv_mu = lw.precision_.dot(mu_vec.reshape(-1,1)).flatten()
    w = np.maximum(0.0, Sigma_inv_mu)
    if w.sum() == 0:
        return w
    w = w / w.sum()
    # Fracción de Kelly como “intensidad” (el resto puede ser cash)
    return f_kelly * w