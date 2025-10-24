# qvm_trend/macro/macro_score.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class MacroRegime:
    z: float              # z-score macro ≈ [-3..+3]
    label: str            # 'OFF' | 'NEUTRAL' | 'ON'
    m_multiplier: float   # multiplicador de riesgo global (0.5..1.0)
    beta_cap: float       # sugerencia de tope ∑(beta·w)
    vol_cap: float        # sugerencia de cap por posición (proxy de vol)

def _m_mult_from_z(z: float) -> float:
    """
    Multiplicador suave y acotado 0.5..1.0 según el z macro.
    Usamos tanh para transiciones continuas (evita saltos).
    """
    return float(np.clip(0.75 + 0.25 * np.tanh(z / 1.0), 0.5, 1.0))

def z_to_regime(z: float) -> MacroRegime:
    """
    Mappea un z-score macro a un 'régimen' y knobs sugeridos.
    - OFF: entorno adverso → menor riesgo
    - NEUTRAL: intermedio
    - ON: entorno favorable → mayor tolerancia
    """
    z = float(z)
    if z <= -1.0:
        return MacroRegime(z=z, label="OFF",
                           m_multiplier=_m_mult_from_z(z),
                           beta_cap=0.80, vol_cap=0.03)
    if z >=  1.0:
        return MacroRegime(z=z, label="ON",
                           m_multiplier=_m_mult_from_z(z),
                           beta_cap=1.10, vol_cap=0.05)
    return MacroRegime(z=z, label="NEUTRAL",
                       m_multiplier=_m_mult_from_z(z),
                       beta_cap=0.90, vol_cap=0.04)

def macro_z_from_series(s: pd.Series, winsor_p: float = 0.02) -> float:
    """
    Convierte tu serie de 'Macro Monitor' (alto=mejor) en un z-score robusto.
    Aplica winsorización y estandariza. Devuelve el z más reciente.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return 0.0
    lo, hi = s.quantile(winsor_p), s.quantile(1 - winsor_p)
    sw = s.clip(lo, hi)
    mu = float(sw.mean())
    sd = float(sw.std(ddof=1) or 1.0)
    z = (float(sw.iloc[-1]) - mu) / sd
    return float(np.clip(z, -3.0, 3.0))
