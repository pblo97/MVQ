import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class MacroOverlayParams:
    # Suavizado y confianza
    ewm_span: int = 12            # meses
    conf_lambda: float = 50.0     # ↑ => más castigo a var alta
    var_win: int = 6              # ventana para var de Δz_suavizado

    # Histéresis y control de cambios
    enter_th: float =  +0.50      # umbral ON
    exit_th: float  =  -0.25      # umbral OFF/NEU (exit de ON)
    gate_z_th: float = -0.50      # bajo esto bloquearnuevas entradas
    cooldown_bars: int = 3        # barras tras cambio de régimen
    step_max: float = 0.15        # |ΔM_macro| máximo por barra

    # Sigmoide -> riesgo
    k_sig: float = 1.25           # pendiente sigmoide
    w_min: float = 0.60           # riesgo mínimo
    w_max: float = 1.25           # riesgo máximo

    # Caps por régimen base (se pueden refinar)
    beta_cap_off: float = 0.60
    beta_cap_neu: float = 1.00
    beta_cap_on:  float = 1.25
    pos_cap_off:  float = 0.03
    pos_cap_neu:  float = 0.05
    pos_cap_on:   float = 0.07

def _ewm(x: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").dropna().ewm(span=span, min_periods=max(3, span//3)).mean()

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _regime_from_hysteresis(z: float, prev_regime: str, enter_th: float, exit_th: float) -> str:
    # Estados: OFF, NEUTRAL, ON
    if prev_regime == "ON":
        if z <= exit_th:
            return "NEUTRAL" if z > 0 else "OFF"
        return "ON"
    elif prev_regime == "OFF":
        if z >= -exit_th:
            return "NEUTRAL" if z < 0 else "ON"
        return "OFF"
    else:  # NEUTRAL
        if z >= enter_th: 
            return "ON"
        if z <= -enter_th:
            return "OFF"
        return "NEUTRAL"

def _caps_for_regime(reg: str, p: MacroOverlayParams) -> tuple[float, float]:
    if reg == "ON":
        return (p.beta_cap_on, p.pos_cap_on)
    if reg == "OFF":
        return (p.beta_cap_off, p.pos_cap_off)
    return (p.beta_cap_neu, p.pos_cap_neu)

def compute_macro_overlay(series_z: pd.Series, params: MacroOverlayParams | None = None) -> pd.DataFrame:
    """
    Entrada: serie de macro_z con índice de tiempo (mensual o semanal).
    Salida (DataFrame):
        ['z','z_smooth','conf','regime','M_macro','beta_cap','pos_cap','gate_new']
    """
    if params is None:
        params = MacroOverlayParams()

    z = pd.to_numeric(series_z, errors="coerce")
    z_s = _ewm(z, span=params.ewm_span).reindex(z.index)

    dz = z_s.diff()
    var = dz.rolling(params.var_win).var()
    conf = 1.0 / (1.0 + params.conf_lambda * (var.clip(lower=1e-8)))
    conf = conf.fillna(conf.mean() if conf.notna().any() else 1.0).clip(0.1, 1.0)

    df = pd.DataFrame({"z": z, "z_smooth": z_s, "conf": conf})
    reg_list, M_list, beta_list, pos_list, gate_list = [], [], [], [], []

    prev_reg, prev_M = "NEUTRAL", 1.0
    cooldown = 0

    for t, row in df.iterrows():
        zt = float(row["z_smooth"])
        # régimen con histéresis
        reg = _regime_from_hysteresis(zt, prev_reg, params.enter_th, params.exit_th)
        if reg != prev_reg:
            cooldown = params.cooldown_bars

        # sigmoide -> riesgo base
        w = params.w_min + (params.w_max - params.w_min) * _sigmoid(params.k_sig * zt)
        M_target = 1.0 + float(row["conf"]) * (w - 1.0)

        # límite de cambio por barra
        M_low, M_high = prev_M - params.step_max, prev_M + params.step_max
        M = float(np.clip(M_target, M_low, M_high))

        beta_cap, pos_cap = _caps_for_regime(reg, params)

        # gate de nuevas entradas
        gate_new = 0 if (zt < params.gate_z_th or cooldown > 0) else 1
        cooldown = max(0, cooldown - 1)

        reg_list.append(reg)
        M_list.append(M)
        beta_list.append(beta_cap)
        pos_list.append(pos_cap)
        gate_list.append(gate_new)

        prev_reg, prev_M = reg, M

    out = df.copy()
    out["regime"] = reg_list
    out["M_macro"] = M_list
    out["beta_cap"] = beta_list
    out["pos_cap"] = pos_list
    out["gate_new"] = gate_list
    return out

@dataclass(frozen=True)
class Regime:
    label: str
    z: float
    m_multiplier: float
    beta_cap: float
    vol_cap: float

def z_to_regime(z: float) -> Regime:
    z = float(z)
    if z <= -0.5:
        return Regime("OFF", z, 0.70, 0.60, 0.03)
    if z >= 0.5:
        return Regime("ON",  z, 1.25, 1.25, 0.07)
    return Regime("NEUTRAL", z, 0.95, 1.00, 0.05)

def macro_z_from_series(s: pd.Series, window: int | None = None) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty: return 0.0
    if window and len(x) >= max(24, window):
        mu = float(x.rolling(window, min_periods=max(12, window//3)).mean().iloc[-1])
        sd = float(x.rolling(window, min_periods=max(12, window//3)).std().iloc[-1])
    else:
        mu = float(x.mean()); sd = float(x.std(ddof=1))
    return 0.0 if not np.isfinite(sd) or sd <= 1e-12 else float((x.iloc[-1]-mu)/sd)

__all__ = ["Regime", "z_to_regime", "macro_z_from_series"]