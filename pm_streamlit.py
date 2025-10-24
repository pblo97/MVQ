# pm_streamlit.py
import os, io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# === Loaders / Helpers tuyos ===
from qvm_trend.data_io import load_prices_panel, load_benchmark, DEFAULT_START, DEFAULT_END
from qvm_trend.stats import beta_vs_bench, win_loss_stats  # asumimos existen

# ===================== CONFIG & ESTILO =====================
st.set_page_config(page_title="Gestión de Cartera", page_icon="🧮", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
[data-testid="stDataFrame"] tbody tr:hover { background: rgba(59,130,246,.08) !important; }
</style>
""", unsafe_allow_html=True)

# ===================== PERSISTENCIA REMOTA (GitHub Gist) =====================
# Secrets necesarios en Streamlit Cloud (Settings → Secrets):
# GITHUB_TOKEN  = "ghp_..."
# GIST_ID       = "abc123..."          (id del gist)
# GIST_FILENAME = "portfolio_symbols.csv"
from github import Github, InputFileContent

def _gh():
    return Github(st.secrets["GITHUB_TOKEN"])

def load_portfolio_remote() -> pd.DataFrame:
    """Lee el CSV 'portfolio_symbols.csv' desde el Gist y devuelve DataFrame ['symbol','date_added']"""
    try:
        gh = _gh()
        gist = gh.get_gist(st.secrets["GIST_ID"])
        fname = st.secrets["GIST_FILENAME"]
        if fname not in gist.files:
            return pd.DataFrame(columns=["symbol","date_added"])
        content = gist.files[fname].content
        df = pd.read_csv(io.StringIO(content))
        if df.empty:
            return pd.DataFrame(columns=["symbol","date_added"])
        df["symbol"] = df["symbol"].astype(str).str.upper()
        if "date_added" not in df.columns:
            df["date_added"] = datetime.now().strftime("%Y-%m-%d")
        return df[["symbol","date_added"]].drop_duplicates("symbol").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["symbol","date_added"])

def save_portfolio_remote(df: pd.DataFrame):
    """Escribe el CSV al Gist (sobrescribe)."""
    gh = _gh()
    gist = gh.get_gist(st.secrets["GIST_ID"])
    fname = st.secrets["GIST_FILENAME"]
    if df is None or df.empty:
        csv_txt = "symbol,date_added\n"
    else:
        out = df.copy()
        out["symbol"] = out["symbol"].astype(str).str.upper()
        if "date_added" not in out.columns:
            out["date_added"] = datetime.now().strftime("%Y-%m-%d")
        out = out[["symbol","date_added"]].drop_duplicates("symbol")
        csv_txt = out.to_csv(index=False)
    gist.edit(files={fname: InputFileContent(csv_txt)})

def init_portfolio_session():
    if "pm_portfolio" not in st.session_state:
        st.session_state["pm_portfolio"] = load_portfolio_remote()

def add_symbols(symbols: list[str]):
    df = st.session_state.get("pm_portfolio", pd.DataFrame(columns=["symbol","date_added"])).copy()
    add = pd.DataFrame({
        "symbol": [s.strip().upper() for s in symbols if s and str(s).strip()],
        "date_added": datetime.now().strftime("%Y-%m-%d")
    })
    out = pd.concat([df, add], ignore_index=True).drop_duplicates("symbol").reset_index(drop=True)
    st.session_state["pm_portfolio"] = out
    save_portfolio_remote(out)

def remove_symbols(symbols: list[str]):
    df = st.session_state.get("pm_portfolio", pd.DataFrame(columns=["symbol","date_added"])).copy()
    keep = df[~df["symbol"].isin([s.strip().upper() for s in symbols])]
    st.session_state["pm_portfolio"] = keep
    save_portfolio_remote(keep)

# ===================== CACHÉ DE I/O =====================
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_prices(symbols, start, end, cache_key=""):
    return load_prices_panel(symbols, start, end, cache_key=cache_key, force=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_bench(bench, start, end):
    return load_benchmark(bench, start, end)

# ===================== UI PRINCIPAL =====================
st.title("🧮 Gestión de Cartera (Gist + Kelly robusto)")

with st.sidebar:
    st.markdown("### Parámetros de sizing")
    method = st.selectbox(
        "Método de pesos",
        ["Kelly fraccionado", "Equal Weight", "Risk Parity (Inv-Vol)", "Mean–Variance (Σ⁻¹μ, heur.)"],
        index=0
    )
    base_kelly = st.slider("Fracción Kelly", 0.1, 1.0, 0.5, 0.1)
    vol_cap = st.number_input("Cap por posición (fracción del equity)", 0.01, 0.20, 0.05, 0.01, format="%.2f")
    beta_cap = st.number_input("Cap ∑(β·w) <=", 0.25, 2.0, 1.0, 0.05)
    enforce_sum1 = st.toggle("Forzar ∑w = 1.0 (después de beta cap)", value=True)

    st.markdown("---")
    st.markdown("### Kelly avanzado")
    winsor_p = st.slider("Winsor p (cola)", 0.0, 0.10, 0.05, 0.01)
    t0 = st.slider("Umbral t-stat (prudencia)", 1.0, 4.0, 2.0, 0.1)
    lam_blend = st.slider("λ (mezcla con μ/σ²)", 0.0, 0.5, 0.2, 0.05)
    min_months = st.number_input("Mín. meses por activo", 6, 120, 36, 6)

    st.markdown("---")
    st.markdown("### Datos")
    bench = st.text_input("Benchmark", value="SPY").strip().upper() or "SPY"
    start = st.date_input("Inicio", pd.to_datetime(DEFAULT_START).date())
    end = st.date_input("Fin", pd.to_datetime(DEFAULT_END).date())
    extend_months = st.slider("Meses extra de historial\n(para estadísticas)", 6, 24, 14)

init_portfolio_session()

# ===================== BLOQUE: GESTIÓN DE LISTA =====================
st.subheader("Cartera persistente (Gist)")
c0, c1 = st.columns([0.65, 0.35], vertical_alignment="bottom")

with c0:
    st.dataframe(
        st.session_state["pm_portfolio"] if not st.session_state["pm_portfolio"].empty else pd.DataFrame(columns=["symbol","date_added"]),
        use_container_width=True, hide_index=True
    )

with c1:
    pf = st.session_state["pm_portfolio"]
    st.download_button(
        "⬇️ Descargar cartera (CSV)",
        (pf if not pf.empty else pd.DataFrame(columns=["symbol","date_added"])).to_csv(index=False).encode(),
        file_name="portfolio_symbols.csv",
        mime="text/csv",
        use_container_width=True
    )
    up = st.file_uploader("Subir cartera (CSV)", type=["csv"])
    replace = st.toggle("Reemplazar completamente al subir", value=False)
    if up is not None:
        try:
            up_df = pd.read_csv(up)
            if "symbol" not in up_df.columns:
                st.error("El CSV debe contener una columna 'symbol'.")
            else:
                up_df["symbol"] = up_df["symbol"].astype(str).str.upper()
                if "date_added" not in up_df.columns:
                    up_df["date_added"] = datetime.now().strftime("%Y-%m-%d")
                up_df = up_df[["symbol","date_added"]].drop_duplicates("symbol")
                if replace:
                    st.session_state["pm_portfolio"] = up_df
                else:
                    merged = pd.concat([st.session_state["pm_portfolio"], up_df], ignore_index=True)\
                              .drop_duplicates("symbol").reset_index(drop=True)
                    st.session_state["pm_portfolio"] = merged
                save_portfolio_remote(st.session_state["pm_portfolio"])
                st.success("Cartera actualizada desde CSV.")
        except Exception as e:
            st.error(f"Error leyendo CSV: {e}")

st.markdown("#### Alta / Baja")
a1, a2, a3 = st.columns([0.6, 0.2, 0.2])
with a1:
    add_text = st.text_input("Agregar símbolos (coma-separado)", placeholder="AAPL, MSFT, NVDA")
with a2:
    ok_add = st.button("➕ Añadir", use_container_width=True)
with a3:
    current_syms = st.session_state["pm_portfolio"]["symbol"].tolist()
    rm_sel = st.multiselect("Remover", options=current_syms, default=[])
    ok_rm = st.button("🗑️ Remover", use_container_width=True)

if ok_add:
    syms = [s.strip().upper() for s in add_text.split(",") if s.strip()]
    if syms:
        add_symbols(syms)
        st.success(f"Agregados: {', '.join(syms)}")
    else:
        st.info("No hay símbolos para agregar.")

if ok_rm:
    if rm_sel:
        remove_symbols(rm_sel)
        st.success(f"Removidos: {', '.join(rm_sel)}")
    else:
        st.info("No seleccionaste símbolos.")

st.markdown("---")

# ===================== KELLY ROBUSTO & MÉTRICAS =====================
def _winsor_series(x: pd.Series, p: float) -> pd.Series:
    if p <= 0 or x.empty: return x
    lo, hi = x.quantile(p), x.quantile(1-p)
    return x.clip(lower=lo, upper=hi)

def _kelly_p_b(p: float, payoff: float) -> float:
    if payoff is None or payoff <= 0: return 0.0
    k = p - (1-p)/payoff
    return float(max(0.0, min(1.0, k)))

def _kelly_merton(mu: float, sigma: float) -> float:
    if sigma is None or sigma <= 1e-9: return 0.0
    k = mu / (sigma**2)
    return float(max(0.0, min(1.0, k)))

def _monthly(series_daily: pd.Series) -> pd.Series:
    series_daily = series_daily.dropna()
    if series_daily.empty: return series_daily
    return series_daily.resample("M").apply(lambda x: (1 + x).prod() - 1).dropna()

def _stats_per_symbol(ser_m: pd.Series, bench_m: pd.Series,
                      winsor_p: float, t0: float, min_months: int, lam_blend: float):
    ser_m = ser_m.dropna()
    n = len(ser_m)
    if n < min_months:
        return dict(valid=False)

    rw = _winsor_series(ser_m, winsor_p)
    wins = (rw > 0).sum()
    p_hat = (wins + 0.5) / (n + 1)  # Jeffreys prior

    avg_win = rw[rw > 0].mean() if (rw > 0).any() else np.nan
    avg_loss = rw[rw < 0].mean() if (rw < 0).any() else np.nan
    payoff = abs(avg_win / avg_loss) if (avg_loss not in (0, None)) else np.nan

    mu = rw.mean()
    sigma = rw.std(ddof=1)
    se = sigma / np.sqrt(n) if sigma and sigma>0 else np.nan
    t_stat = float(mu / se) if (se and se>0) else 0.0

    beta = beta_vs_bench(rw, bench_m.reindex(rw.index).dropna())

    k1 = _kelly_p_b(p_hat, payoff) if not np.isnan(payoff) else 0.0
    k2 = _kelly_merton(mu, sigma)
    u = float(np.clip(t_stat / max(t0, 1e-9), 0.0, 1.0))
    k_blend = u * ((1 - lam_blend) * k1 + lam_blend * k2)

    return dict(valid=True, HitRate=p_hat, AvgWin=avg_win, AvgLoss=avg_loss,
                Payoff=payoff, Kelly_raw=k1, Kelly_blend=k_blend,
                Beta=beta, Mu=mu, Sigma=sigma, Sharpe_m=(mu/sigma if sigma and sigma>0 else np.nan),
                n_months=n, t_stat=t_stat)

def _risk_parity_weights(sigmas: np.ndarray, cap: float):
    inv = np.where(sigmas>0, 1.0/np.maximum(sigmas, 1e-8), 0.0)
    if inv.sum() == 0: return np.zeros_like(inv)
    w = inv / inv.sum()
    w = np.minimum(w, cap)
    s = w.sum()
    return w / s if s>0 else w

def _mean_variance_weights(mu: np.ndarray, cov: np.ndarray, cap: float):
    try:
        inv = np.linalg.pinv(cov + 1e-8*np.eye(len(cov)))
        raw = inv @ mu
        raw = np.maximum(raw, 0.0)        # no short
        w = raw / raw.sum() if raw.sum() > 0 else np.ones_like(raw)/len(raw)
        w = np.minimum(w, cap)
        s = w.sum()
        return w / s if s>0 else w
    except Exception:
        return np.ones_like(mu) / len(mu)

def _apply_beta_cap(weights: np.ndarray, betas: np.ndarray, beta_cap: float):
    total_beta_w = float(np.nansum(weights * np.nan_to_num(betas, nan=1.0)))
    scale = 1.0
    if total_beta_w > beta_cap and total_beta_w > 0:
        scale = beta_cap / total_beta_w
    return weights * scale

# ===================== CÁLCULO DE PESOS + ANÁLISIS =====================
st.subheader("Cálculo de pesos y análisis")
run_btn = st.button("🧮 Calcular", type="primary", use_container_width=True)

if run_btn:
    pf = st.session_state["pm_portfolio"]
    syms = pf["symbol"].tolist() if not pf.empty else []
    if not syms:
        st.warning("Cartera vacía. Agrega símbolos arriba.")
    else:
        extend_days = int(extend_months * 30)
        start_ext = (pd.to_datetime(start) - pd.Timedelta(days=extend_days)).date().isoformat()
        end_iso = end.isoformat()

        try:
            pnl = _cached_prices(syms + [bench], start_ext, end_iso, cache_key="pm_panel")
            if bench not in pnl:
                st.error(f"No pude cargar el benchmark '{bench}'.")
            else:
                bench_daily = pnl[bench]["close"].pct_change().dropna()
                bench_m = _monthly(bench_daily)

                # --- Métricas por símbolo ---
                rows, used = [], []
                sym_series_m = {}
                for s in syms:
                    if s not in pnl or "close" not in pnl[s].columns:
                        continue
                    ser_d = pnl[s]["close"].pct_change().dropna()
                    ser_m = _monthly(ser_d)
                    common = ser_m.index.intersection(bench_m.index)
                    if len(common) < min_months:
                        continue
                    ser_m = ser_m.loc[common]
                    bench_mc = bench_m.loc[common]

                    stt = _stats_per_symbol(ser_m, bench_mc, winsor_p, t0, min_months, lam_blend)
                    if not stt.get("valid", False):
                        continue
                    rows.append(dict(symbol=s, **stt))
                    used.append(s)
                    sym_series_m[s] = ser_m

                dfm = pd.DataFrame(rows)
                if dfm.empty:
                    st.warning("No hay datos suficientes (historial mensual insuficiente).")
                    st.stop()

                # --- Pesos según método ---
                n = len(used)
                betas = dfm["Beta"].values
                caps = float(vol_cap)

                if method == "Equal Weight":
                    w = np.ones(n) / n
                    w = np.minimum(w, caps); w = w / w.sum()
                elif method == "Risk Parity (Inv-Vol)":
                    sigmas = dfm["Sigma"].values
                    w = _risk_parity_weights(sigmas, caps)
                elif method == "Mean–Variance (Σ⁻¹μ, heur.)":
                    mu = dfm["Mu"].values
                    mret = pd.DataFrame({s: sym_series_m[s] for s in used}).dropna()
                    cov = np.cov(mret.values.T)
                    w = _mean_variance_weights(mu, cov, caps)
                else:  # Kelly fraccionado ROBUSTO
                    kelly_vec = dfm["Kelly_blend"].fillna(0).values
                    w = np.minimum(base_kelly * kelly_vec, caps)
                    if w.sum() == 0:
                        w = np.ones(n) / n
                    else:
                        w = w / w.sum()

                # --- Cap de beta ---
                w_beta = _apply_beta_cap(w, betas, float(beta_cap))

                # --- Forzar suma 1 (opcional) ---
                w_final = w_beta.copy()
                if enforce_sum1 and w_final.sum() > 0:
                    w_final = w_final / w_final.sum()

                dfm["w_prop"] = w
                dfm["w_final"] = w_final
                dfm["beta_contrib"] = dfm["Beta"].fillna(1.0) * dfm["w_final"]

                st.subheader("Métricas por símbolo & Pesos")
                st.dataframe(
                    dfm[["symbol","n_months","HitRate","AvgWin","AvgLoss","Payoff",
                         "Kelly_raw","Kelly_blend","t_stat","Beta","Mu","Sigma","Sharpe_m",
                         "w_final","beta_contrib"]]\
                       .rename(columns={"w_final":"weight","beta_contrib":"beta_w","Mu":"Mu_m","Sigma":"Sigma_m"}),
                    use_container_width=True, hide_index=True
                )

                # --- Gráficos de pesos ---
                st.markdown("### Pesos y Exposición β")
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots()
                    ax.bar(dfm["symbol"], dfm["w_final"])
                    ax.set_title("Pesos finales (w)")
                    ax.set_xticklabels(dfm["symbol"], rotation=45, ha="right")
                    st.pyplot(fig, use_container_width=True)
                with c2:
                    fig, ax = plt.subplots()
                    ax.bar(dfm["symbol"], dfm["beta_contrib"])
                    ax.set_title("Contribución beta (beta_w)")
                    ax.set_xticklabels(dfm["symbol"], rotation=45, ha="right")
                    st.pyplot(fig, use_container_width=True)

                st.info(f"sum(beta_w) = {dfm['beta_contrib'].sum():.3f} (cap={beta_cap})   |   sum(w) = {dfm['w_final'].sum():.3f}")

                # --- Correlaciones (mensual) ---
                st.markdown("### Correlaciones (mensuales)")
                mret = pd.DataFrame({s: sym_series_m[s] for s in used}).dropna()
                corr = mret.corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=False, cmap="vlag", center=0, ax=ax)
                ax.set_title("Matriz de correlaciones mensuales")
                st.pyplot(fig, use_container_width=True)

                # --- Cartera vs Benchmark (w constantes) ---
                st.markdown("### Cartera vs Benchmark (estática con w constantes)")
                daily = pd.DataFrame({
                    s: pnl[s]["close"].pct_change() for s in used
                    if s in pnl and "close" in pnl[s].columns
                }).dropna(how="all")
                common_days = daily.dropna(axis=1, how="all").columns.intersection(used)
                w_map = dict(zip(used, w_final))
                r_p = (daily[common_days].fillna(0) * pd.Series({k: w_map.get(k, 0.0) for k in common_days})).sum(axis=1)
                eq_p = (1 + r_p.fillna(0)).cumprod()

                bench_d = pnl[bench]["close"].pct_change().reindex(eq_p.index).fillna(0)
                eq_b = (1 + bench_d).cumprod()

                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots()
                    ax.plot(eq_p.index, eq_p.values, label="Portfolio")
                    ax.plot(eq_b.index, eq_b.values, label=bench)
                    ax.set_title("Cartera vs Benchmark (w constantes)")
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)
                with c2:
                    dd = eq_p/eq_p.cummax() - 1.0
                    fig, ax = plt.subplots()
                    ax.plot(dd.index, dd.values)
                    ax.set_title("Drawdown (cartera)")
                    ax.axhline(0, color="black", lw=0.5)
                    st.pyplot(fig, use_container_width=True)

                # --- Histograma de retornos mensuales (cartera) ---
                st.markdown("### Distribución de retornos mensuales (cartera)")
                r_p_m = _monthly(r_p)
                fig, ax = plt.subplots()
                ax.hist(r_p_m.dropna(), bins=20)
                ax.set_title("Histograma retornos mensuales")
                st.pyplot(fig, use_container_width=True)

                # --- Descarga pesos (CSV, ASCII) ---
                out = dfm[["symbol","w_final","beta_contrib"]].rename(
                    columns={"w_final":"weight", "beta_contrib":"beta_w"}
                )
                st.download_button(
                    "⬇️ Descargar pesos (CSV)",
                    out.to_csv(index=False).encode(),
                    file_name="pesos_portafolio.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error calculando: {e}")
