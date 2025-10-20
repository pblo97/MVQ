# qvm_trend/cache_io.py
import os
import pandas as pd
from typing import Dict

CACHE_DIR = os.getenv("QVM_CACHE_DIR", ".cache_qvm")

def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(name: str) -> str:
    _ensure_dir()
    return os.path.join(CACHE_DIR, name)

def save_df(df: pd.DataFrame, name: str):
    path = cache_path(name)
    df.to_parquet(path, index=True)

def load_df(name: str) -> pd.DataFrame | None:
    path = cache_path(name)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

def save_panel(panel: Dict[str, pd.DataFrame], name: str):
    # guarda cada símbolo en un subdirectorio
    base = cache_path(name)
    os.makedirs(base, exist_ok=True)
    for sym, df in panel.items():
        df.to_parquet(os.path.join(base, f"{sym}.parquet"), index=True)

def load_panel(name: str) -> Dict[str, pd.DataFrame] | None:
    base = cache_path(name)
    if not os.path.isdir(base):
        return None
    out = {}
    for fn in os.listdir(base):
        if not fn.endswith(".parquet"): 
            continue
        sym = fn.replace(".parquet","")
        out[sym] = pd.read_parquet(os.path.join(base, fn))
    return out if out else None
