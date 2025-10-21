import numpy as np
import pandas as pd

def mc_bootstrap_trades(trade_R: pd.Series | list, n_sims: int = 5000, block: int = 5, seed: int | None = 42, start_equity: float = 1.0) -> dict:
    """
    trade_R: serie de retornos por trade en notación 'R' (p.ej. +1.0, -1.0, +2.3, ...)
    block: tamaño de bloque para conservar rachas (block bootstrap)
    Devuelve dict con resúmenes + equity paths (opcional).
    """
    rng = np.random.default_rng(seed)
    R = np.asarray(trade_R, dtype=float)
    R = R[np.isfinite(R)]
    n = len(R)
    if n == 0:
        raise ValueError("No hay trades para Monte Carlo")
    n_blocks = int(np.ceil(n / block))
    cagr_list, maxdd_list, eq_last = [], [], []

    for _ in range(n_sims):
        path = []
        for _b in range(n_blocks):
            i = rng.integers(0, max(1, n - block + 1))
            path.extend(R[i:i+block])
        path = np.array(path[:n])  # recorta exacto
        equity = np.empty(n+1); equity[0] = start_equity
        for t in range(n):
            equity[t+1] = equity[t] * (1.0 + path[t])
        eq = pd.Series(equity)
        rollmax = eq.cummax()
        dd = eq/rollmax - 1.0
        maxdd = float(dd.min())
        years = n / 200.0  # aprox si 200 trades ~ 1y; ajusta a tu frecuencia real
        cagr = float((eq.iloc[-1] / eq.iloc[0])**(1/years) - 1) if years > 0 else np.nan
        cagr_list.append(cagr); maxdd_list.append(maxdd); eq_last.append(eq.iloc[-1])

    return {
        "n": int(n),
        "n_sims": int(n_sims),
        "block": int(block),
        "CAGR_p50": float(np.nanpercentile(cagr_list, 50)),
        "CAGR_p10": float(np.nanpercentile(cagr_list, 10)),
        "CAGR_p90": float(np.nanpercentile(cagr_list, 90)),
        "MaxDD_p50": float(np.nanpercentile(maxdd_list, 50)),
        "MaxDD_p10": float(np.nanpercentile(maxdd_list, 10)),
        "MaxDD_p90": float(np.nanpercentile(maxdd_list, 90)),
        "Terminal_p50": float(np.nanpercentile(eq_last, 50)),
        "Terminal_p10": float(np.nanpercentile(eq_last, 10)),
        "Terminal_p90": float(np.nanpercentile(eq_last, 90)),
    }