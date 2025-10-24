# MArimax.py — Macro Monitor con compuesto Z, PCA, Markov, Overlay estable y panel ON/OFF
# Parche completo: BAA–AAA, USD Broad, min-dwell overlay, bundle enriquecido, sin import loops.

import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Volatilidad opcional
try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False

# ==========================
# Configuración de página
# ==========================
st.set_page_config(page_title="Macro Monitor", layout="wide")
st.title("Macro Monitor — Z-Score compuesto, Regímenes y Overlay")
st.caption("Inversiones - Macro | Composite (Term, Crédito, Liquidez, USD) | Markov 2 regímenes | Overlay OOS | GARCH opcional")

# ==========================
# Sidebar de parámetros
# ==========================
with st.sidebar:
    st.header("Parámetros")
    mode = st.radio("Modo de uso", ["Generar desde FRED", "Subir CSV ya generado"], index=0)
    freq = st.selectbox("Frecuencia", ["Semanal (W)", "Mensual (M)"], index=0)
    freq_key = "W" if freq.startswith("Semanal") else "M"
    start_date = st.date_input("Fecha inicio", value=pd.to_datetime("2010-01-01"))
    roll_z_w = st.slider("Ventana Z-score semanal (semanas)", 26, 78, 52)
    roll_z_m = st.slider("Ventana Z-score mensual (meses)", 18, 60, 36)

    st.markdown("---")
    st.markdown("Umbrales iniciales (se optimizan OOS con grid)")
    thr_comp_init = st.number_input("Umbral inicial COMPOSITE (z)", value=0.0, step=0.1, format="%.2f")
    thr_prob_init = st.number_input("Umbral inicial Prob. Estrés", value=0.40, step=0.05, format="%.2f")

    st.markdown("---")
    min_dwell = st.slider("Min-dwell overlay (barras)", 1, 8, 3, 1)
    use_garch = st.checkbox("Usar GARCH si está disponible", value=False and HAVE_ARCH)
    annual_target_vol = st.number_input("Target de volatilidad anual (opcional)", value=0.15, step=0.01, format="%.2f")
    st.caption("El target de vol no se aplica por defecto; solo informativo.")

    st.markdown("---")
    st.info("Coloca tu FRED API key en st.secrets['FRED_API_KEY'].")

# ==========================
# Utilidades
# ==========================
def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def zscore_roll(s: pd.Series, window: int) -> pd.Series:
    r = s.rolling(window, min_periods=window)
    return (s - r.mean()) / r.std()

def z_last(s: pd.Series, window: int) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.size < window:
        return float("nan")
    z = zscore_roll(s, window)
    return float(z.iloc[-1])

def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return (x.mean() / x.std()) if x.std() not in (0, None, np.nan) else np.nan

def sortino(x: pd.Series) -> float:
    x = x.dropna()
    d = x[x < 0].std()
    return x.mean() / d if d and d > 0 else np.nan

def max_drawdown(ret: pd.Series) -> float:
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return dd.min() if not dd.empty else np.nan

def drawdown_curve(ret: pd.Series) -> pd.Series:
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    peak = eq.cummax()
    return (eq / peak) - 1.0

def share_on(signal: pd.Series) -> float:
    s = signal.dropna()
    return float(s.mean()) if len(s) else np.nan

def enforce_min_dwell(sig: pd.Series, min_len: int = 3) -> pd.Series:
    """Evita flips rápidos: obliga a mantener ON/OFF al menos min_len barras."""
    if sig.dropna().empty:
        return sig
    s = sig.astype(int).copy()
    last = s.iloc[0]; run = 1
    for i in range(1, len(s)):
        if s.iloc[i] == last:
            run += 1
        else:
            if run < min_len:
                s.iloc[i] = last  # cancela flip prematuro
                run += 1
            else:
                last = s.iloc[i]; run = 1
    return s

# ==========================
# Mapeo a régimen local (sin import externo)
# ==========================
class _Regime:
    __slots__ = ("label", "z", "m_multiplier", "beta_cap", "vol_cap")
    def __init__(self, label, z, m_multiplier, beta_cap, vol_cap):
        self.label = label; self.z = z
        self.m_multiplier = m_multiplier
        self.beta_cap = beta_cap
        self.vol_cap = vol_cap

def z_to_regime_local(z: float) -> _Regime:
    # Thresholds conservadores con 3 niveles
    if z >= 0.5:
        return _Regime("ON", z, 1.25, 1.25, 0.07)
    elif z <= -0.5:
        return _Regime("OFF", z, 0.70, 0.60, 0.03)
    else:
        return _Regime("NEUTRAL", z, 0.95, 1.00, 0.05)

def macro_z_from_series_local(s: pd.Series, window: int = 36) -> float:
    """z-score rolling y devuelve el último valor (estable)."""
    return z_last(s, window)

# ==========================
# Carga FRED
# ==========================
@st.cache_data(show_spinner=True)
def fetch_fred_series(start_dt: pd.Timestamp) -> pd.DataFrame:
    from fredapi import Fred
    fred_key = st.secrets.get("FRED_API_KEY", "")
    fred = Fred(api_key=fred_key)

    series = {
        "SOFR": "SOFR",
        "RRP": "RRPONTSYD",
        "NFCI": "NFCI",
        "EFFR": "EFFR",
        "OBFR": "OBFR",
        "SP500": "SP500",
        "TGA": "WTREGEN",
        "STLFSI4": "STLFSI4",
        "TB3MS": "TB3MS",
        "DGS3MO": "DGS3MO",
        "BAMLH0A0HYM2": "BAMLH0A0HYM2",
        "T10Y2Y": "T10Y2Y",
        "DGS10": "DGS10",           # 10y para Term 10y-3m
        # === NUEVOS pilares ===
        "BAA_Yield": "BAA",         # Moody's BAA
        "AAA_Yield": "AAA",         # Moody's AAA
        "USD_BROAD": "DTWEXBGS",    # Broad Dollar Index
    }
    df = pd.DataFrame()
    for name, code in series.items():
        try:
            df[name] = fred.get_series(code)
        except Exception as e:
            st.warning(f"FRED {name}: {e}")

    df.index.name = "Date"
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.rename_axis("Date").copy()
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.loc[df.index >= pd.to_datetime(start_dt)]
    dfd = df.resample("D").last().ffill()

    # Derivaciones
    dfd["diferencial_colateral"] = dfd["EFFR"] - dfd["SOFR"]
    dfd["sofr_spread"] = dfd["OBFR"] - dfd["SOFR"]
    dfd = dfd.rename(columns={"RRP": "Reverse_Repo_Volume", "TGA": "WTREGEN"})

    # TERM spreads
    if {"DGS10", "TB3MS"}.issubset(dfd.columns):
        dfd["T10Y3M"] = dfd["DGS10"] - dfd["TB3MS"]
    # Crédito calidad
    if {"BAA_Yield","AAA_Yield"}.issubset(dfd.columns):
        dfd["BAA_AAA"] = dfd["BAA_Yield"] - dfd["AAA_Yield"]

    return dfd

# ==========================
# Compuestos y pilares
# ==========================
def _pillar(df, cols, signs, win, diff_flags=None):
    zs = []
    if diff_flags is None:
        diff_flags = [False]*len(cols)
    for c,sg,diff in zip(cols, signs, diff_flags):
        if c not in df.columns: 
            continue
        x = df[c].astype(float)
        if diff:
            x = x.diff()
        x = winsorize(x)
        z = zscore_roll(x, win)
        zs.append(sg * z)
    if not zs:
        return pd.Series(index=df.index, dtype=float)
    Z = pd.concat(zs, axis=1)
    return Z.mean(axis=1)

def build_pillars_and_composite(dfd: pd.DataFrame, freq_key: str, roll_z_w: int, roll_z_m: int):
    if freq_key == "W":
        df = dfd.resample("W").last().ffill(); win = roll_z_w
    else:
        df = dfd.resample("M").last().ffill(); win = roll_z_m

    # Pilares (ligeramente ortogonales)
    TERM   = _pillar(df, ["T10Y3M", "T10Y2Y"],          [+1, +1], win)
    CREDIT = _pillar(df, ["BAMLH0A0HYM2", "BAA_AAA"],   [+1, +1], win)
    LIQ    = _pillar(df, ["Reverse_Repo_Volume", "WTREGEN", "sofr_spread"], [+1, +1, +1], win, diff_flags=[True, True, False])
    USD    = _pillar(df, ["USD_BROAD"],                 [-1],     win)  # USD fuerte = peor

    pillars = pd.concat({"TERM":TERM, "CREDIT":CREDIT, "LIQ":LIQ, "USD":USD}, axis=1)
    composite = pillars.mean(axis=1).rename("COMPOSITE_Z")  # equal-weight para robustez
    return pillars, composite

def composite_pca(dfd: pd.DataFrame, freq_key: str, roll_z_w: int, roll_z_m: int) -> pd.Series:
    if freq_key == "W":
        df = dfd.resample("W").last().ffill()
        window = roll_z_w
    else:
        df = dfd.resample("M").last().ffill()
        window = roll_z_m

    FACTORES = [
        "NFCI", "STLFSI4", "BAMLH0A0HYM2", "T10Y2Y", "T10Y3M",
        "diferencial_colateral", "Reverse_Repo_Volume", "WTREGEN", "sofr_spread",
        "BAA_AAA", "USD_BROAD"
    ]
    Zs, names = [], []
    for fac in FACTORES:
        if fac not in df.columns:
            continue
        s = df[fac].astype(float)
        if fac in ["Reverse_Repo_Volume", "WTREGEN"]:
            s = s.diff()
        s = winsorize(s)
        Zs.append(zscore_roll(s, window))
        names.append(fac)
    Z = pd.concat(Zs, axis=1).dropna()
    if Z.empty:
        return pd.Series(index=df.index, dtype=float, name="COMPOSITE_PCA")
    Z.columns = names
    pc1 = pd.Series(PCA(n_components=1).fit_transform(Z).ravel(), index=Z.index, name="COMPOSITE_PCA")
    pc1 = zscore_roll(pc1, window)
    return pc1

# ==========================
# Equity premium
# ==========================
def equity_premium(dfd: pd.DataFrame, freq_key: str) -> pd.Series:
    if "SP500" not in dfd.columns:
        return pd.Series(dtype=float, name="Excess_Ret")
    if freq_key == "M":
        sp = dfd["SP500"].resample("M").last().dropna()
        ret = np.log(sp).diff()
        rf = (dfd["TB3MS"] / 100.0 / 12.0).resample("M").last().reindex(sp.index).ffill()
    else:
        sp = dfd["SP500"].resample("W").last().dropna()
        ret = np.log(sp).diff()
        rf = (dfd["DGS3MO"] / 100.0 / 52.0).resample("W").last().reindex(sp.index).ffill()
    y = (ret - rf).dropna()
    y.name = "Excess_Ret"
    return y

# ==========================
# Markov 2 regímenes
# ==========================
def markov_two_regimes(y: pd.Series, comp_l1: pd.Series):
    df = pd.concat([y, comp_l1], axis=1).dropna()
    if df.empty:
        return None, None
    sc = StandardScaler()
    arr = sc.fit_transform(df.values)
    dfs = pd.DataFrame(arr, index=df.index, columns=[y.name, comp_l1.name])
    try:
        mod = MarkovRegression(dfs[y.name], exog=dfs[[comp_l1.name]], k_regimes=2, trend='c', switching_variance=True)
        res = mod.fit(method='lbfgs', maxiter=1000, disp=False)
        prob_reg0 = res.smoothed_marginal_probabilities[0]
        return prob_reg0.rename("P_reg0"), res
    except Exception as e:
        st.warning(f"Markov error: {e}")
        return None, None

# ==========================
# Overlay grid-search
# ==========================
def overlay_gridsearch(y, composite, prob_stress=None,
                       comp_grid=np.arange(-0.5, 1.01, 0.05),
                       pst_grid=np.arange(0.4, 0.91, 0.05),
                       split=0.7):
    idx = y.index.intersection(composite.index)
    y = y.loc[idx].dropna()
    comp = composite.reindex(y.index).ffill()

    n = len(y)
    cut = int(n * split) if n > 10 else max(2, int(n * 0.7))
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]
    c_te = comp.iloc[cut:]

    pst_te = None
    if prob_stress is not None:
        pst = prob_stress.reindex(y.index).ffill()
        pst_te = pst.iloc[cut:]

    best = {"sh_te": -np.inf, "thr_comp": None, "thr_prob": None}
    for tc in comp_grid:
        for tp in (pst_grid if prob_stress is not None else [1.1]):
            mask_te = ~((c_te > tc) | ((pst_te > tp) if pst_te is not None else False))
            sh_te = sharpe(y_te * mask_te.astype(int))
            if sh_te > best["sh_te"]:
                best = {"sh_te": sh_te, "thr_comp": tc, "thr_prob": None if pst_te is None else tp}

    tc = best["thr_comp"] if best["thr_comp"] is not None else 0.0
    tp = best["thr_prob"] if best["thr_prob"] is not None else 1.1
    pst_all = (prob_stress.reindex(y.index).ffill() if prob_stress is not None else pd.Series(0, index=y.index))
    sig_all = ~((comp > tc) | (pst_all > tp))
    ret_filt = y * sig_all.astype(int)

    return best, sig_all.rename("Overlay_Signal"), ret_filt.rename("Ret_Filtered")

# ==========================
# GARCH opcional
# ==========================
def garch_vol_forecast(y: pd.Series, comp_l1: pd.Series):
    if not HAVE_ARCH:
        return None
    df = pd.concat([y, comp_l1], axis=1).dropna()
    if df.empty:
        return None
    X = sm.add_constant(df[['COMP_L1']])
    try:
        am = arch_model(df[y.name] * 100, mean='ARX', lags=0, x=X[['COMP_L1']],
                        vol='GARCH', p=1, o=0, q=1, dist='t')
        res = am.fit(disp='off')
        vol = (res.conditional_volatility / 100.0).rename("VolForecast")
        return vol.reindex(y.index)
    except Exception as e:
        st.warning(f"GARCH error: {e}")
        return None

# ==========================
# Modo CSV ya generado
# ==========================
if mode == "Subir CSV ya generado":
    st.subheader("Cargar bundle CSV")
    up = st.file_uploader("Selecciona tu macro_monitor_bundle.csv", type=["csv"])
    if up:
        df_in = pd.read_csv(up, parse_dates=True, index_col=0)
        st.success(f"Cargado: {df_in.shape[0]} filas, {df_in.shape[1]} columnas")
        st.dataframe(df_in.tail(10))

        c1, c2 = st.columns(2)
        with c1:
            cols = [c for c in ["COMPOSITE_Z", "COMPOSITE_PCA"] if c in df_in.columns]
            if cols:
                dfp = df_in[cols].rename_axis("Date").reset_index()
                fig = px.line(dfp, x="Date", y=cols, title="Composite (Weighted vs PCA)")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "P_reg0" in df_in.columns:
                fig = px.line(df_in.rename_axis("Date").reset_index(), x="Date", y="P_reg0", title="Probabilidad Régimen 0 (calma)")
                st.plotly_chart(fig, use_container_width=True)

        if {"Ret_Filtered", "Excess_Ret"}.issubset(df_in.columns):
            k1, k2, k3 = st.columns(3)
            k1.metric("Sharpe naive", f"{sharpe(df_in['Excess_Ret']):.3f}")
            k2.metric("Sharpe filtrado", f"{sharpe(df_in['Ret_Filtered']):.3f}")
            k3.metric("Mejora", f"{(sharpe(df_in['Ret_Filtered']) - sharpe(df_in['Excess_Ret'])):.3f}")
        st.stop()
    else:
        st.info("Sube un CSV para visualizar.")
        st.stop()

# ==========================
# Pipeline FRED
# ==========================
st.subheader("Descarga FRED y cálculo")
if st.button("Ejecutar pipeline"):
    with st.spinner("Obteniendo series de FRED..."):
        dfd = fetch_fred_series(pd.to_datetime(start_date))

    with st.spinner("Construyendo pilares, composite y equity premium..."):
        pillars, comp_w = build_pillars_and_composite(dfd, freq_key, roll_z_w, roll_z_m)
        comp_p = composite_pca(dfd, freq_key, roll_z_w, roll_z_m)
        y = equity_premium(dfd, freq_key)

    # Lag típico: 3 en semanal; 2–3 en mensual. Tomamos 3 por defecto
    comp_l = comp_w.shift(3).rename("COMP_L").reindex(y.index)

    with st.spinner("Markov (2 regímenes)..."):
        prob_reg0, ms_res = markov_two_regimes(y, comp_l.rename("COMP_L1"))
        prob_stress = (1 - prob_reg0) if prob_reg0 is not None else None

    with st.spinner("Overlay OOS (grid)..."):
        best, signal, ret_filt = overlay_gridsearch(
            y=y,
            composite=comp_l,              # z-score compuesto con lag
            prob_stress=prob_stress,
            comp_grid=np.arange(-0.5, 1.01, 0.05),
            pst_grid=np.arange(0.4, 0.91, 0.05),
            split=0.7
        )

    # Estabiliza overlay con min-dwell
    signal_stable = enforce_min_dwell(signal, min_len=int(min_dwell)).rename("Overlay_Signal_Stable")

    st.success(f"Overlay óptimo OOS → Sharpe={best['sh_te']:.3f} | thr_comp={best['thr_comp']} | thr_prob={best['thr_prob']}")

    # ==========================
    # Estado actual ON/OFF y explicación
    # ==========================
    prob_stress_series = (1 - prob_reg0) if prob_reg0 is not None else None
    last_idx = signal_stable.dropna().index[-1]
    last_sig = int(signal_stable.loc[last_idx])
    last_comp = float(comp_l.reindex([last_idx]).iloc[0])
    last_prob = float(prob_stress_series.reindex([last_idx]).iloc[0]) if prob_stress_series is not None else float("nan")

    thr_comp = float(best["thr_comp"]) if best["thr_comp"] is not None else 0.0
    thr_prob = float(best["thr_prob"]) if best["thr_prob"] is not None else np.inf
    state_txt = "ON" if last_sig == 1 else "OFF"

    st.subheader("Estado actual")
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Estado (overlay estab.)", state_txt)
    cB.metric("COMP_L (últ.)", f"{last_comp:.2f}")
    cC.metric("ProbEstrés (últ.)", f"{last_prob:.2f}")
    cD.metric("Umbrales", f"COMP ≤ {thr_comp:.2f} | Prob ≤ {thr_prob if np.isfinite(thr_prob) else float('nan'):.2f}")

    if np.isfinite(thr_prob):
        fired = []
        if last_comp > thr_comp: fired.append("COMP_L > umbral")
        if last_prob > thr_prob: fired.append("ProbEstrés > umbral")
        reason = " y ".join(fired) if fired else "ambas por debajo"
    else:
        reason = "solo regla de COMP_L; por debajo" if last_sig == 1 else "COMP_L superó su umbral"

    st.caption(f"Regla: ON si (COMP_L ≤ {thr_comp:.2f}) y (ProbEstrés ≤ {thr_prob if np.isfinite(thr_prob) else float('nan'):.2f}). Motivo: {reason}.")

    st.dataframe(
        pd.concat(
            [
                pillars,                            # muestra pilares
                comp_l.rename("COMP_L"),
                (prob_stress_series.rename("ProbEstrés") if prob_stress_series is not None else pd.Series(index=signal.index, dtype=float)),
                signal.rename("Overlay_Signal"),
                signal_stable
            ],
            axis=1
        ).dropna().tail(12)
    )

    # ==========================
    # Métricas
    # ==========================
    volf = None
    if use_garch:
        with st.spinner("Pronóstico de volatilidad (GARCH)..."):
            volf = garch_vol_forecast(y, comp_l.rename("COMP_L1"))

    sharpe_naive = sharpe(y)
    sharpe_filtered = sharpe(ret_filt)
    sortino_naive = sortino(y)
    sortino_filtered = sortino(ret_filt)
    mdd_naive = max_drawdown(y)
    mdd_filtered = max_drawdown(ret_filt)
    pct_on = share_on(signal_stable)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sharpe naive", f"{sharpe_naive:.3f}")
    k2.metric("Sharpe filtrado", f"{sharpe_filtered:.3f}", f"{(sharpe_filtered - sharpe_naive):+.3f}")
    k3.metric("Sortino naive", f"{sortino_naive:.3f}")
    k4.metric("Sortino filtrado", f"{sortino_filtered:.3f}")
    k5.metric("% tiempo ON (est.)", f"{100*pct_on:.1f}%")

    # ==========================
    # Gráficos
    # ==========================
    c1, c2 = st.columns(2)
    with c1:
        # Composite y PCA
        df_comp = pd.concat([comp_w.rename("COMPOSITE_Z"), comp_p.rename("COMPOSITE_PCA")], axis=1)
        fig = px.line(df_comp.rename_axis("Date").reset_index(), x="Date", y=["COMPOSITE_Z", "COMPOSITE_PCA"],
                      title="Composite (Equal-weight) vs PCA(1)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # Pilares y contribuciones
        if not pillars.dropna(how="all").empty:
            dfp = pillars.rename_axis("Date").reset_index().melt(id_vars="Date", var_name="Pilar", value_name="Z")
            fig = px.line(dfp, x="Date", y="Z", color="Pilar", title="Pilares (Z)")
            st.plotly_chart(fig, use_container_width=True)

            # Últimas contribuciones (barra)
            last = pillars.dropna().iloc[-1]
            contrib = (last / len(pillars.columns)).rename("Contrib")
            figc = px.bar(contrib.reset_index().rename(columns={"index":"Pilar"}), x="Pilar", y="Contrib",
                          title="Contribución por pilar (último)")
            st.plotly_chart(figc, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.step(signal.rename_axis("Date").reset_index(), x="Date", y="Overlay_Signal", title="Señal Overlay (0/1)")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.step(signal_stable.rename_axis("Date").reset_index(), x="Date", y="Overlay_Signal_Stable", title=f"Overlay Estabilizado (min-dwell={min_dwell})")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Curvas de capital")
    df_ret = pd.concat([y.rename("Excess_Ret"), ret_filt], axis=1).dropna()
    if not df_ret.empty:
        df_ret["EQ_naive"] = (1 + df_ret["Excess_Ret"]).cumprod()
        df_ret["EQ_filtered"] = (1 + df_ret["Ret_Filtered"]).cumprod()
        fig = px.line(df_ret.rename_axis("Date").reset_index(), x="Date", y=["EQ_naive", "EQ_filtered"], title="Curva de capital")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Distribución de retornos (hist)")
    df_hist = pd.concat(
        [
            y.rename("Ret").to_frame().assign(Serie="Excess_Ret"),
            ret_filt.rename("Ret").to_frame().assign(Serie="Ret_Filtered"),
        ],
        axis=0
    )
    if not df_hist.dropna().empty:
        fig = px.histogram(df_hist.reset_index(drop=True), x="Ret", color="Serie", barmode="overlay", nbins=60, title="Distribución de retornos")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Drawdowns")
    dd_naive = drawdown_curve(y).rename("DD_naive")
    dd_filt = drawdown_curve(ret_filt).rename("DD_filtered")
    fig = px.line(pd.concat([dd_naive, dd_filt], axis=1).rename_axis("Date").reset_index(),
                  x="Date", y=["DD_naive", "DD_filtered"], title="Curva de drawdown (pico a valle)")
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Bundle y descarga
    # ==========================
    # Macro-z (último de z-roll del composite) y sugerencias de caps por régimen
    macro_z_val = macro_z_from_series_local(comp_w, window=(roll_z_m if freq_key == "M" else roll_z_w))
    reg = z_to_regime_local(macro_z_val)

    bundle = pd.concat({
        "COMPOSITE_Z": comp_w,
        "COMPOSITE_PCA": comp_p,
        "COMP_L": comp_l,
        "P_reg0": prob_reg0 if prob_reg0 is not None else pd.Series(index=y.index, dtype=float),
        "Overlay_Signal": signal.astype(int),
        "Overlay_Signal_Stable": signal_stable.astype(int),
        "VolForecast": volf if volf is not None else pd.Series(index=y.index, dtype=float),
        "Ret_Filtered": ret_filt,
        "Excess_Ret": y
    }, axis=1)

    # Constantes (útiles para PM)
    bundle["macro_z"] = float(macro_z_val)
    bundle["beta_cap_sug"] = float(reg.beta_cap)
    bundle["pos_cap_sug"]  = float(reg.vol_cap)

    st.subheader("Descargar macro_monitor_bundle.csv")
    st.download_button(
        label="Descargar CSV",
        data=bundle.to_csv(index=True).encode("utf-8"),
        file_name="macro_monitor_bundle.csv",
        mime="text/csv"
    )

    st.dataframe(bundle.tail(12))

else:
    st.info("Configura la API key de FRED, ajusta parámetros y pulsa Ejecutar pipeline.")
