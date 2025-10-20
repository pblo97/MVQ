import numpy as np
import pandas as pd


def gbm_paths(close: pd.Series, horizon_days=20, n_sims=2000):
    r = close.pct_change().dropna()
    mu = r.mean() * 252
    sigma = r.std() * np.sqrt(252)
    dt = 1/252
    s0 = float(close.iloc[-1])
    paths = np.zeros((horizon_days+1, n_sims))
    paths[0,:] = s0
    for t in range(1, horizon_days+1):
        z = np.random.normal(size=n_sims)
        paths[t,:] = paths[t-1,:]*np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return pd.DataFrame(paths, index=range(horizon_days+1))


def block_bootstrap_paths(close: pd.Series, horizon_days=20, block=5, n_sims=2000):
    r = close.pct_change().dropna().values
    T = horizon_days
    k = int(np.ceil(T/block))
    sims = np.zeros((T, n_sims))
    for j in range(n_sims):
        seq = []
        for _ in range(k):
            i = np.random.randint(0, len(r)-block)
            seq.extend(r[i:i+block])
        sims[:,j] = np.array(seq[:T])
    s0 = float(close.iloc[-1])
    paths = s0 * (1 + pd.DataFrame(sims)).cumprod()
    paths = pd.concat([pd.DataFrame([np.repeat(s0, n_sims)]), paths], axis=0).reset_index(drop=True)
    return paths