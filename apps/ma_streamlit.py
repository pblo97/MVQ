# MArimax.py — Macro Monitor con compuesto Z, PCA, Markov, Overlay y panel ON/OFF

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
st.caption("Inversiones - Macro | Compuesto con HY OAS y Term Spread | Markov 2 regímenes | Overlay OOS | GARCH opcional")


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

def zscore(s: pd.Series, window: int) -> pd.Series:
    roll = s.rolling(window, min_periods=window)
    return (s - roll.mean()) / roll.std()

def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return (x.mean() / x.std()) if x.std() != 0 else np.nan

def sortino(x: pd.Series) -> float:
    x = x.dropna()
    neg = x[x < 0]
    d = neg.std()
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


# ==========================
# Carga FRED
# ==========================
@st.cache_data(show_spinner=True)
def fetch_fred_series(start_dt: pd.Timestamp) -> pd.DataFrame:
    from fredapi import Fred

    # Validación y feedback de secret
    fred_key = st.secrets.get("FRED_API_KEY", "")
    st.write("Secret presente:", bool(fred_key))
    st.write("Formato válido:", bool(isinstance(fred_key, str) and len(fred_key) == 32 and fred_key.isalnum() and fred_key.islower()))

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
    }
    df = pd.DataFrame()
    for name, code in series.items():
        try:
            df[name] = fred.get_series(code)
        except Exception as e:
            st.warning(f"FRED {name}: {e}")

    df.index.name = "Date"
    # Recorta por fecha de inicio solo si hay índice de fechas
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.loc[df.index >= pd.to_datetime(start_dt)]
    else:
        # Si FRED devolviera un índice no temporal, lo convertimos
        df = df.rename_axis("Date").copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.loc[df.index >= pd.to_datetime(start_dt)]

    dfd = df.resample("D").last().ffill()
    dfd["diferencial_colateral"] = dfd["EFFR"] - dfd["SOFR"]
    dfd["sofr_spread"] = dfd["OBFR"] - dfd["SOFR"]
    dfd = dfd.rename(columns={"RRP": "Reverse_Repo_Volume", "TGA": "WTREGEN"})
    return dfd


# ==========================
# Compuestos
# ==========================
def build_composite(dfd: pd.DataFrame, freq_key: str, roll_z_w: int, roll_z_m: int) -> pd.Series:
    FACTORES = {
        "NFCI":                  {"w": 0.20, "sign": +1, "diff": False, "winsor": True},
        "STLFSI4":               {"w": 0.15, "sign": +1, "diff": False, "winsor": True},
        "BAMLH0A0HYM2":          {"w": 0.20, "sign": +1, "diff": False, "winsor": True},
        "T10Y2Y":                {"w": 0.10, "sign": -1, "diff": False, "winsor": True},
        "diferencial_colateral": {"w": 0.10, "sign": +1, "diff": False, "winsor": True},
        "Reverse_Repo_Volume":   {"w": 0.15, "sign": +1, "diff": True,  "winsor": True},
        "WTREGEN":               {"w": 0.10, "sign": +1, "diff": True,  "winsor": True},
        "sofr_spread":           {"w": 0.10, "sign": +1, "diff": False, "winsor": True},
    }
    if freq_key == "W":
        df = dfd.resample("W").last().ffill()
        window = roll_z_w
    else:
        df = dfd.resample("M").last().ffill()
        window = roll_z_m

    cols = []
    for fac, cfg in FACTORES.items():
        if fac not in df.columns:
            continue
        s = df[fac].astype(float)
        if cfg.get("diff", False):
            s = s.diff()
        if cfg.get("winsor", True):
            s = winsorize(s)
        z = zscore(s, window)
        cols.append(cfg["sign"] * cfg["w"] * z)

    comp = pd.concat(cols, axis=1).sum(axis=1)
    comp.name = "COMPOSITE_Z"
    return comp

def composite_pca(dfd: pd.DataFrame, freq_key: str, roll_z_w: int, roll_z_m: int) -> pd.Series:
    if freq_key == "W":
        df = dfd.resample("W").last().ffill()
        window = roll_z_w
    else:
        df = dfd.resample("M").last().ffill()
        window = roll_z_m

    FACTORES = [
        "NFCI", "STLFSI4", "BAMLH0A0HYM2", "T10Y2Y",
        "diferencial_colateral", "Reverse_Repo_Volume", "WTREGEN", "sofr_spread"
    ]
    Zs, names = [], []
    for fac in FACTORES:
        if fac not in df.columns:
            continue
        s = df[fac].astype(float)
        if fac in ["Reverse_Repo_Volume", "WTREGEN"]:
            s = s.diff()
        s = winsorize(s)
        Zs.append(zscore(s, window))
        names.append(fac)
    Z = pd.concat(Zs, axis=1).dropna()
    Z.columns = names
    pc1 = pd.Series(PCA(n_components=1).fit_transform(Z).ravel(), index=Z.index, name="COMPOSITE_PCA")
    pc1 = zscore(pc1, window)
    return pc1


# ==========================
# Equity premium
# ==========================
def equity_premium(dfd: pd.DataFrame, freq_key: str) -> pd.Series:
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
    sc = StandardScaler()
    arr = sc.fit_transform(df.values)
    dfs = pd.DataFrame(arr, index=df.index, columns=[y.name, comp_l1.name])
    try:
        mod = MarkovRegression(
            dfs[y.name], exog=dfs[[comp_l1.name]],
            k_regimes=2, trend='c', switching_variance=True
        )
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
    cut = int(n * split)
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

    tc = best["thr_comp"]
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
            dfp = df_in.rename_axis("Date").reset_index()
            fig = px.line(dfp, x="Date", y=["COMPOSITE_Z", "COMPOSITE_PCA"], title="Composite (Weighted vs PCA)")
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

    with st.spinner("Construyendo compuestos y equity premium..."):
        comp_w = build_composite(dfd, freq_key, roll_z_w, roll_z_m)
        comp_p = composite_pca(dfd, freq_key, roll_z_w, roll_z_m)
        y = equity_premium(dfd, freq_key)

    # Lag típico: 3 en semanal; 2–3 en mensual. Tomamos 3 para ejemplo
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
    st.success(f"Overlay óptimo OOS → Sharpe={best['sh_te']:.3f} | thr_comp={best['thr_comp']} | thr_prob={best['thr_prob']}")

    # ==========================
    # BLOQUE NUEVO: Estado actual ON/OFF y explicación
    # ==========================
    prob_stress_series = (1 - prob_reg0) if prob_reg0 is not None else None
    last_idx = signal.dropna().index[-1]
    last_sig = int(signal.loc[last_idx])
    last_comp = float(comp_l.reindex([last_idx]).iloc[0])
    last_prob = float(prob_stress_series.reindex([last_idx]).iloc[0]) if prob_stress_series is not None else float("nan")

    thr_comp = float(best["thr_comp"])
    thr_prob = float(best["thr_prob"]) if best["thr_prob"] is not None else np.inf
    state_txt = "ON" if last_sig == 1 else "OFF"

    st.subheader("Estado actual")
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Estado", state_txt)
    cB.metric("COMP_L (últ.)", f"{last_comp:.2f}")
    cC.metric("ProbEstrés (últ.)", f"{last_prob:.2f}")
    cD.metric("Umbrales", f"COMP ≤ {thr_comp:.2f} | Prob ≤ {thr_prob if np.isfinite(thr_prob) else float('nan'):.2f}")

    if np.isfinite(thr_prob):
        fired = []
        if last_comp > thr_comp:
            fired.append("COMP_L superó su umbral")
        if last_prob > thr_prob:
            fired.append("ProbEstrés superó su umbral")
        reason = " y ".join(fired) if fired else "ambas condiciones por debajo del umbral"
    else:
        reason = "solo regla de COMP_L; por debajo del umbral" if last_sig == 1 else "COMP_L superó su umbral"

    st.caption(
        f"Regla: ON si (COMP_L ≤ {thr_comp:.2f}) y (ProbEstrés ≤ {thr_prob if np.isfinite(thr_prob) else float('nan'):.2f}). "
        f"Motivo: {reason}."
    )

    st.dataframe(
        pd.concat(
            [
                comp_l.rename("COMP_L"),
                (prob_stress_series.rename("ProbEstrés") if prob_stress_series is not None else pd.Series(index=signal.index, dtype=float)),
                signal.rename("Overlay_Signal")
            ],
            axis=1
        ).dropna().tail(8)
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
    pct_on = share_on(signal)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sharpe naive", f"{sharpe_naive:.3f}")
    k2.metric("Sharpe filtrado", f"{sharpe_filtered:.3f}", f"{(sharpe_filtered - sharpe_naive):+.3f}")
    k3.metric("Sortino naive", f"{sortino_naive:.3f}")
    k4.metric("Sortino filtrado", f"{sortino_filtered:.3f}")
    k5.metric("% tiempo ON", f"{100*pct_on:.1f}%")

    # ==========================
    # Gráficos
    # ==========================
    c1, c2 = st.columns(2)
    with c1:
        df_comp = pd.concat([comp_w.rename("COMPOSITE_Z"), comp_p.rename("COMPOSITE_PCA")], axis=1)
        fig = px.line(df_comp.rename_axis("Date").reset_index(), x="Date", y=["COMPOSITE_Z", "COMPOSITE_PCA"],
                      title="Composite (Weighted vs PCA)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if prob_reg0 is not None:
            fig = px.line(prob_reg0.rename_axis("Date").reset_index(), x="Date", y="P_reg0",
                          title="Probabilidad Régimen 0 (calma)")
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.line(signal.rename_axis("Date").reset_index(), x="Date", y="Overlay_Signal",
                      title="Señal Overlay (0/1)")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        df_ret = pd.concat([y.rename("Excess_Ret"), ret_filt], axis=1).dropna()
        df_ret["EQ_naive"] = (1 + df_ret["Excess_Ret"]).cumprod()
        df_ret["EQ_filtered"] = (1 + df_ret["Ret_Filtered"]).cumprod()
        fig = px.line(df_ret.rename_axis("Date").reset_index(), x="Date", y=["EQ_naive", "EQ_filtered"],
                      title="Curva de capital")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Distribución de retornos (hist)")
    df_hist = pd.concat(
        [
            y.rename("Ret").to_frame().assign(Serie="Excess_Ret"),
            ret_filt.rename("Ret").to_frame().assign(Serie="Ret_Filtered"),
        ],
        axis=0
    )
    fig = px.histogram(df_hist.reset_index(drop=True), x="Ret", color="Serie", barmode="overlay",
                       nbins=60, title="Distribución de retornos")
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
    bundle = pd.concat({
        "COMPOSITE_Z": comp_w,
        "COMPOSITE_PCA": comp_p,
        "COMP_L": comp_l,
        "P_reg0": prob_reg0 if prob_reg0 is not None else pd.Series(index=y.index, dtype=float),
        "Overlay_Signal": signal.astype(int),
        "VolForecast": volf if volf is not None else pd.Series(index=y.index, dtype=float),
        "Ret_Filtered": ret_filt,
        "Excess_Ret": y
    }, axis=1)

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