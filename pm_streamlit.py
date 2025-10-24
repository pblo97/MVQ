import os, sys
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
import plotly.graph_objects as go
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False
# --- RUTA DEL PROYECTO ---
ROOT = os.path.abspath(os.path.dirname(__file__))          # .../mvq
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# --- Inicialización de paquetes (por si faltan) ---
for d in ["qvm_trend", "qvm_trend/macro", "qvm_trend/pm"]:
    p = os.path.join(ROOT, d, "__init__.py")
    if not os.path.exists(p):
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write("# package\n")
        except Exception:
            pass

# ==== imports del paquete (con diagnóstico) ====
from qvm_trend.data_io import load_prices_panel, load_benchmark, DEFAULT_START, DEFAULT_END
from qvm_trend.pm.exits import build_exit_table

# Intento robusto de import del macro_score:
try:
    from qvm_trend.macro.macro_score import z_to_regime, macro_z_from_series
except Exception as e:
    # Muestra el error real (Streamlit suele redacted)
    st.error(f"Error importando qvm_trend.macro.macro_score: {type(e).__name__}: {e}")
    # Fallback mínimo para no romper la app mientras depuras:
    from dataclasses import dataclass
    @dataclass
    class _Reg:
        label: str
        z: float
        m_multiplier: float
        beta_cap: float
        vol_cap: float
    def z_to_regime(z: float) -> _Reg:
        # reglas suaves por z-score
        if z <= -0.5:  # OFF
            return _Reg("OFF", z, 0.70, 0.60, 0.03)
        if z >= 0.5:   # ON
            return _Reg("ON",  z, 1.25, 1.25, 0.07)
        return _Reg("NEUTRAL", z, 0.95, 1.00, 0.05)
    def macro_z_from_series(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        return float((s - s.mean())/(s.std(ddof=1)+1e-12)).iloc[-1] if len(s) else 0.0

# Importa orchestrator DESPUÉS de tener z_to_regime disponible
from qvm_trend.pm.orchestrator import build_portfolio

st.set_page_config(page_title="Gestión de Cartera", page_icon="🧭", layout="wide")

st.title("Gestión de Cartera")
st.caption("Pesos por Kelly robusto, tilt por calidad, caps por régimen macro, y reglas de salida.")

# ------------------ TABS ------------------
tab_in, tab_macro, tab_port, tab_exits, tab_diag = st.tabs(
    ["Entradas", "Macro", "Cartera", "Salidas", "Diagnóstico"]
)

# ------------------ ENTRADAS ------------------
with tab_in:
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        symbols_txt = st.text_area("Símbolos (coma-separados)", "CHDN,CNR,GOOGL,MMM,PYPL,PZZA,UBER", height=80)
    with c2:
        bench = st.text_input("Benchmark", value="SPY").strip().upper()
        start = st.date_input("Inicio", value=pd.to_datetime(DEFAULT_START).date())
        end   = st.date_input("Fin", value=pd.to_datetime(DEFAULT_END).date())
    with c3:
        base_kelly = st.slider("Fracción Kelly base", 0.05, 0.50, 0.25, 0.01)
        pos_cap = st.number_input("Cap por posición", 0.01, 0.10, 0.05, 0.01, format="%.2f")
        beta_cap = st.number_input("Cap ∑(β·w)", 0.25, 2.00, 1.00, 0.05)

    st.markdown("### Ajustes Kelly avanzados")
    ck1, ck2, ck3 = st.columns(3)
    with ck1:
        winsor_p = st.slider("Winsor p (%)", 0.0, 5.0, 2.0, 0.25) / 100.0
    with ck2:
        costs_per_period = st.number_input("Costos mensuales (bps)", 0, 100, 10, 1) / 10_000.0
    with ck3:
        lambda_corr = st.slider("Penalización correlación λ", 0.0, 1.0, 0.50, 0.05)

    ck4, ck5 = st.columns(2)
    with ck4:
        ewm_span = st.slider("EWMA span (meses)", 6, 24, 12, 1)
    with ck5:
        shrink_kappa = st.slider("Shrink κ (p/payoff)", 0, 50, 20, 1)

    st.markdown("**(Opcional) CSV de calidad (VFQ o QualityScore)**")
    up_vfq = st.file_uploader("vfq.csv (de tu screener)", type=["csv"])

    quality_df = None
    if up_vfq is not None:
        try:
            qdf = pd.read_csv(up_vfq)
            if "symbol" in qdf.columns:
                keep_cols = [c for c in ["symbol","VFQ","QualityScore"] if c in qdf.columns]
                quality_df = qdf[keep_cols].copy()
                st.success(f"Cargado VFQ/Quality: {quality_df.shape}")
            else:
                st.warning("El CSV no tiene columna 'symbol'.")
        except Exception as e:
            st.error(f"No pude leer VFQ: {e}")

    st.markdown("**(Opcional) CSV de calidad (VFQ o QualityScore)**")
    up_vfq = st.file_uploader("vfq.csv (de tu screener)", type=["csv"])

    quality_df = None
    if up_vfq is not None:
        try:
            qdf = pd.read_csv(up_vfq)
            if "symbol" in qdf.columns:
                keep_cols = [c for c in ["symbol","VFQ","QualityScore"] if c in qdf.columns]
                quality_df = qdf[keep_cols].copy()
                st.success(f"Cargado VFQ/Quality: {quality_df.shape}")
            else:
                st.warning("El CSV no tiene columna 'symbol'.")
        except Exception as e:
            st.error(f"No pude leer VFQ: {e}")

    # >>> NUEVO: si NO subiste CSV, intenta FMP <<<
    if quality_df is None:
        fmp_key = st.secrets.get("FMP_API_KEY", "")
        symbols_for_fmp = [s.strip().upper() for s in symbols_txt.split(",") if s.strip()]
        if fmp_key and symbols_for_fmp:
            try:
                with st.spinner("Descargando fundamentals (FMP) para QualityScore..."):
                    from qvm_trend.fundamentals.fmp_quality import compute_quality_from_fmp
                    qdf = compute_quality_from_fmp(symbols_for_fmp, fmp_key)
                    if not qdf.empty:
                        quality_df = qdf[["symbol","QualityScore"]].copy()
                        st.info("QualityScore obtenido automáticamente desde FMP.")
            except Exception as e:
                st.warning(f"No pude calcular QualityScore con FMP: {e}")

# ------------------ MACRO ------------------
# ------------------ MACRO ------------------
with tab_macro:
    st.subheader("Macro Monitor: control interno (sin CSV) + bundle opcional ↪")

    # --- 0) Cargar bundle automático o por upload ---
    macro_bundle = None
    auto_paths = [
        "./macro_monitor_bundle.csv",
        "./exports/macro_monitor_bundle.csv",
        "./data/macro_monitor_bundle.csv",
        "/mnt/data/macro_monitor_bundle.csv",
    ]
    for p in auto_paths:
        try:
            if os.path.exists(p):
                macro_bundle = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
                st.info(f"Bundle macro cargado automáticamente: `{p}`")
                break
        except Exception as e:
            st.warning(f"No pude leer {p}: {e}")

    up_macro = st.file_uploader("Subir macro_monitor_bundle.csv (opcional)", type=["csv"])
    if up_macro is not None:
        try:
            macro_bundle = pd.read_csv(up_macro, index_col=0, parse_dates=True).sort_index()
            st.success("Bundle macro cargado desde upload.")
        except Exception as e:
            st.error(f"Error leyendo bundle: {e}")
            macro_bundle = None

    # --- 1) Controles manuales (slider/EMA/umbrales) ---
    c_top = st.columns([1,1,1,1])
    with c_top[0]:
        macro_z_manual = st.slider("macro_z (manual)", -2.5, 2.5, 0.0, 0.05)
        alpha = st.slider("Suavizado EMA (α)", 0.05, 0.50, 0.20, 0.01,
                          help="Qué tan rápido reacciona el macro_z suavizado.")
        # EMA simple local (visual). Si quieres EMA real histórica, cámbialo por una serie.
        macro_z_ema = (1 - alpha) * 0.0 + alpha * macro_z_manual
        st.metric("macro_z (EMA)", f"{macro_z_ema:.2f}")
    with c_top[1]:
        thr_on  = st.number_input("Umbral ON (≥)", value=0.50,  step=0.05, format="%.2f")
        thr_off = st.number_input("Umbral OFF (≤)", value=-0.50, step=0.05, format="%.2f")

    # --- 2) Si hay bundle, tomar macro_z del bundle y sugerencias ---
    macro_z_from_bundle = None
    beta_cap_sug = None
    pos_cap_sug  = None
    overlay_gate_series = None

    if macro_bundle is not None:
        try:
            if "macro_z" in macro_bundle.columns and pd.notna(macro_bundle["macro_z"]).any():
                macro_z_from_bundle = float(macro_bundle["macro_z"].dropna().iloc[-1])
            elif "COMPOSITE_Z" in macro_bundle.columns and pd.notna(macro_bundle["COMPOSITE_Z"]).any():
                macro_z_from_bundle = float(macro_z_from_series(macro_bundle["COMPOSITE_Z"]))
            elif "COMPOSITE_PCA" in macro_bundle.columns and pd.notna(macro_bundle["COMPOSITE_PCA"]).any():
                macro_z_from_bundle = float(macro_z_from_series(macro_bundle["COMPOSITE_PCA"]))

            if "beta_cap_sug" in macro_bundle.columns:
                beta_cap_sug = float(macro_bundle["beta_cap_sug"].dropna().iloc[-1])
            if "pos_cap_sug" in macro_bundle.columns:
                pos_cap_sug  = float(macro_bundle["pos_cap_sug"].dropna().iloc[-1])

            overlay_gate_series = macro_bundle.get("Overlay_Signal")
        except Exception as e:
            st.warning(f"No pude derivar macro_z del bundle: {e}")

    # --- 3) Decidir macro_z EFECTIVO y mapear a régimen ---
    macro_z_eff = float(macro_z_from_bundle) if macro_z_from_bundle is not None else float(macro_z_ema)
    reg_eff = z_to_regime(macro_z_eff)

    # KPIs (usar SIEMPRE el efectivo)
    with c_top[2]:
        st.metric("Régimen", reg_eff.label)
        st.metric("M_macro", f"{reg_eff.m_multiplier:.2f}")
    with c_top[3]:
        st.metric("β cap / pos cap", f"{reg_eff.beta_cap:.2f} / {reg_eff.vol_cap:.2f}")

    # Fuente del z efectivo
    st.caption(f"Fuente de macro_z: **{'bundle' if macro_z_from_bundle is not None else 'manual/EMA'}** — macro_z (efectivo) = {macro_z_eff:.2f}")

    # --- 4) Gauge y barras de caps (con el efectivo) ---
    g1, g2 = st.columns(2)
    with g1:
        try:
            import plotly.graph_objects as go
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=reg_eff.m_multiplier,
                gauge={"axis": {"range": [0.7, 1.3]}},
                title={"text": "M_macro (×)"}
            ))
            st.plotly_chart(gauge, use_container_width=True)
        except Exception:
            st.write("M_macro:", reg_eff.m_multiplier)
    with g2:
        st.bar_chart(pd.Series({"Posición": reg_eff.vol_cap, "β·w total": reg_eff.beta_cap}).to_frame("caps"))

    # --- 5) Gráficos históricos del bundle (si existe) ---
    try:
        import plotly.express as px
        HAVE_PX = True
    except Exception:
        HAVE_PX = False

    if macro_bundle is not None and HAVE_PX:
        st.markdown("### Históricos (del bundle)")
        cols_plot = [c for c in ["COMPOSITE_Z", "COMPOSITE_PCA"] if c in macro_bundle.columns]
        if cols_plot:
            st.plotly_chart(
                px.line(macro_bundle[cols_plot].rename_axis("Date").reset_index(),
                        x="Date", y=cols_plot, title="Composite (Weighted/PCA)"),
                use_container_width=True
            )
        if "Overlay_Signal" in macro_bundle.columns:
            st.plotly_chart(
                px.step(macro_bundle["Overlay_Signal"].rename_axis("Date").reset_index(),
                        x="Date", y="Overlay_Signal", title="Overlay (0/1)"),
                use_container_width=True
            )
        if {"Excess_Ret","Ret_Filtered"}.issubset(macro_bundle.columns):
            df_ret = macro_bundle[["Excess_Ret","Ret_Filtered"]].dropna().copy()
            df_ret["EQ_naive"] = (1 + df_ret["Excess_Ret"]).cumprod()
            df_ret["EQ_filtered"] = (1 + df_ret["Ret_Filtered"]).cumprod()
            st.plotly_chart(
                px.line(df_ret.rename_axis("Date").reset_index(),
                        x="Date", y=["EQ_naive","EQ_filtered"], title="Curva de capital"),
                use_container_width=True
            )

    # --- 6) Propagar a otras pestañas (EFECTIVO + sugerencias) ---
    st.session_state["macro_z_eff"] = macro_z_eff
    st.session_state["beta_cap_sug"] = beta_cap_sug if beta_cap_sug is not None else reg_eff.beta_cap
    st.session_state["pos_cap_sug"]  = pos_cap_sug  if pos_cap_sug  is not None else reg_eff.vol_cap
    st.session_state["overlay_gate_series"] = overlay_gate_series

    st.caption("Sube **macro_monitor_bundle.csv** o déjalo vacío para usar el control interno.")


# ------------------ CARTERA ------------------
with tab_port:
    st.subheader("Pesos y métricas (Kelly + Macro + Quality)")
    symbols = [s.strip().upper() for s in symbols_txt.split(",") if s.strip()]
    if not symbols:
        st.warning("Ingrese símbolos en la pestaña Entradas.")
        st.stop()

    # Caps efectivos = min(usuario, sugerido por régimen si viene del bundle)
    beta_cap_eff = beta_cap
    pos_cap_eff  = pos_cap
    if st.session_state.get("beta_cap_sug") is not None:
        beta_cap_eff = min(beta_cap_eff, float(st.session_state["beta_cap_sug"]))
    if st.session_state.get("pos_cap_sug") is not None:
        pos_cap_eff = min(pos_cap_eff, float(st.session_state["pos_cap_sug"]))

    # Gate táctico con overlay (si el último valor es 0 → bloquear nuevas)
    allow_new_when_z_below = -0.5
    ov = st.session_state.get("overlay_gate_series")
    if ov is not None:
        try:
            if int(pd.Series(ov).astype(int).iloc[-1]) == 0:
                allow_new_when_z_below = 10.0  # bloquea nuevas totalmente
        except Exception:
            pass

    dfw = build_portfolio(
        symbols=symbols,
        bench=bench,
        start=start.isoformat(),
        end=end.isoformat(),
        # === Kelly pro ===
        base_kelly=base_kelly,
        winsor_p=winsor_p,
        costs_per_period=costs_per_period,
        ewm_span=ewm_span,
        shrink_kappa=shrink_kappa,
        lambda_corr=lambda_corr,
        # === Macro / Quality / Caps ===
        macro_z=float(st.session_state.get("macro_z_eff", 0.0)),
        quality_df=quality_df,
        pos_cap=pos_cap_eff,
        beta_cap_user=beta_cap_eff,
        allow_new_when_z_below=allow_new_when_z_below,
        current_holdings=None  # o lista de tus posiciones ya abiertas
    )

    if dfw.empty:
        st.warning("No se pudieron calcular pesos (verifica precios/fechas).")
        st.stop()

    st.dataframe(dfw, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.bar_chart(dfw.set_index("symbol")["weight"])
        st.caption("Pesos finales (w)")
    with c2:
        st.bar_chart(dfw.set_index("symbol")["beta_w"])
        st.caption("Contribución β·w")

    st.markdown("---")
    st.subheader("Sizing con capital ficticio")

    col_cap1, col_cap2, col_cap3 = st.columns([1,1,1])
    with col_cap1:
        capital_usd = st.number_input("Capital (USD)", value=380000, min_value=0, step=1000, format="%.0f")
    with col_cap2:
        cash_pct = st.slider("Cash %", 0.0, 1.0, 0.00, 0.01)
    with col_cap3:
        use_macro_mult = st.toggle("Multiplicar por M_macro del régimen", value=True)

    # multiplicador macro desde pestaña Macro
    M_macro = 1.0
    if use_macro_mult:
        reg_here = z_to_regime(float(st.session_state.get("macro_z_eff", 0.0)))
        M_macro = reg_here.m_multiplier

    alloc_capital = capital_usd * (1.0 - cash_pct)
    w = dfw.set_index("symbol")["weight"].astype(float)
    alloc = (w * alloc_capital * M_macro).rename("usd_alloc")

    # (Opcional) cantidades aproximadas: usa último precio disponible
    last_prices = {}
    panel_tmp = load_prices_panel(symbols, start.isoformat(), end.isoformat(), cache_key="pm_panel")
    for s in w.index:
        try:
            last_p = float(panel_tmp.get(s, {}).get("close", pd.Series(dtype=float)).dropna().iloc[-1])
        except Exception:
            last_p = float("nan")
        last_prices[s] = last_p
    px = pd.Series(last_prices, name="last_price")

    qty = (alloc / px).fillna(0.0).apply(lambda x: int(max(0, np.floor(x))))  # enteros
    alloc_eff = (qty * px).rename("usd_used")
    cash_left = float(capital_usd - cash_pct*capital_usd - alloc_eff.sum())

    sizing_tbl = pd.concat([w.rename("weight"), px, alloc, qty.rename("qty"), alloc_eff], axis=1).reset_index().rename(columns={"index":"symbol"})
    st.dataframe(sizing_tbl, use_container_width=True)
    st.caption(f"Cash libre aprox.: **${cash_left:,.0f}** (M_macro={M_macro:.2f})")

    # Export rápido a CSV
    st.download_button(
        "Descargar sizing (CSV)",
        sizing_tbl.to_csv(index=False).encode(),
        file_name="sizing.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Export de calidad/Kelly de la tabla principal
    st.download_button(
        "Descargar métricas Kelly/β (CSV)",
        dfw.to_csv(index=False).encode(),
        file_name="kelly_metrics.csv",
        mime="text/csv",
        use_container_width=True
    )

# ------------------ SALIDAS ------------------
# ------------------ SALIDAS ------------------
with tab_exits:
    st.subheader("Reglas de salida por símbolo (MA200, Mom 12-1, Calidad 1Q)")

    # Parámetros
    colE1, colE2, colE3 = st.columns(3)
    with colE1:
        ma_window = st.number_input("MA (días)", 100, 400, 200, 10)
    with colE2:
        mom_lookback = st.number_input("Lookback Momentum (días)", 120, 400, 252, 5)
    with colE3:
        vfq_delta_thr = st.number_input("Umbral ΔVFQ 1Q (degradación)", 0.00, 1.00, 0.10, 0.01, format="%.2f")

    st.caption("Salida si: rompe MA200 y/o Momentum 12-1 < 0; se refuerza con degradación de calidad (ΔVFQ 1Q < -umbral). Revisión trimestral.")

    # Carga opcional de histórico de calidad
    vfq_hist = None
    up_vfq_hist = st.file_uploader("Histórico de calidad (opcional) — columnas: symbol,date,VFQ", type=["csv"])
    if up_vfq_hist is not None:
        try:
            vfq_hist = pd.read_csv(up_vfq_hist)
            # normalizamos nombres
            vfq_hist.columns = [c.strip() for c in vfq_hist.columns]
            st.success(f"Histórico de calidad cargado: {vfq_hist.shape}")
        except Exception as e:
            st.error(f"No pude leer histórico de calidad: {e}")

    try:
        panel = load_prices_panel(symbols + [bench], start.isoformat(), end.isoformat(), cache_key="pm_panel")
        bench_px = load_benchmark(bench, start.isoformat(), end.isoformat())

        from qvm_trend.pm.exits import build_exit_table  # usa el parche nuevo

        table = build_exit_table(
            panel=panel,
            bench_close=None if bench_px is None else bench_px.get("close"),
            ma_window=int(ma_window),
            mom_lookback=int(mom_lookback),
            review_freq="Q",
            vfq_hist=vfq_hist,
            vfq_col="VFQ",
            vfq_delta_thr=float(vfq_delta_thr),
        )

        if table.empty:
            st.warning("No se pudo generar la tabla de salidas.")
        else:
            # Filtro por acción
            act_sel = st.multiselect("Filtrar por acción", options=["EXIT","TRIM","HOLD"], default=["EXIT","TRIM"])
            if act_sel:
                table = table[table["action"].isin(act_sel)]

            # Quita símbolos manualmente si quieres
            sym_del = st.multiselect("Eliminar símbolos de la tabla", options=sorted(table["symbol"].unique().tolist()))
            if sym_del:
                table = table[~table["symbol"].isin(sym_del)]

            st.dataframe(table, use_container_width=True)

            st.download_button(
                "Descargar salidas (CSV)",
                table.to_csv(index=False).encode(),
                file_name="exits.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error generando salidas: {e}")


# ------------------ DIAGNÓSTICO ------------------
with tab_diag:
    st.subheader("Equity (pesos estáticos) y diagnóstico")
    try:
        pnl = load_prices_panel(symbols + [bench], start.isoformat(), end.isoformat(), cache_key="pm_panel")

        rets_cols = []
        for s in symbols:
            if s in pnl and "close" in pnl[s].columns:
                r = pnl[s]["close"].pct_change().rename(s)
                rets_cols.append(r)

        if not rets_cols:
            st.warning("Sin retornos para diagnóstico.")
            st.stop()

        # Asegura DataFrame multicolumna
        merged = pd.concat(rets_cols, axis=1).sort_index()
        merged = merged.dropna(how="all").fillna(0.0)

        weights = dfw.set_index("symbol")["weight"].reindex(merged.columns).fillna(0.0).values
        port_ret = (merged * weights).sum(axis=1)

        bench_ret = None
        bdf = load_benchmark(bench, start.isoformat(), end.isoformat())
        if bdf is not None and "close" in bdf.columns:
            bench_ret = bdf["close"].pct_change().reindex(port_ret.index).fillna(0.0)

        eq = (1 + port_ret).cumprod().rename("Portfolio")
        if bench_ret is not None:
            eq_b = (1 + bench_ret).cumprod().rename(bench)
            st.line_chart(pd.concat([eq, eq_b], axis=1))
        else:
            st.line_chart(eq)
    except Exception as e:
        st.error(f"Diag error: {e}")

# ======= GUIA / INSTRUCCIONES (debajo de las pestañas) =======
st.markdown(
    """
    <style>
      .guide-card {
        padding: 18px 20px;
        border-radius: 14px;
        background: linear-gradient(180deg, #0f172a0a, #0f172a08);
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-top: 8px; margin-bottom: 16px;
      }
      .guide-title {
        font-size: 1.05rem; font-weight: 700;
        margin-bottom: 6px;
      }
      .guide-sub {
        color: #64748b; margin-bottom: 10px;
      }
      .guide-badge {
        display:inline-block; padding:3px 10px; border-radius:999px;
        background: #0ea5e91a; color:#0369a1; font-weight:600; font-size:12px;
        border: 1px solid #0ea5e955; margin-right:6px;
      }
      .pill {
        display:inline-block; padding:2px 8px; border-radius:999px;
        background:#22c55e1a; color:#15803d; font-size:12px; border:1px solid #22c55e55;
        margin-left:6px;
      }
      .small-muted { color:#718096; font-size:12px; }
      ul.compact li { margin-bottom: 4px; }
      code.k { background:#111827; color:#e5e7eb; padding:2px 6px; border-radius:6px; }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown("<div class='guide-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='guide-title'>Guía rápida de uso</div>
        <div class='guide-sub'>Resumen de flujo: Entradas → Macro → Cartera → Salidas → Diagnóstico</div>
        """,
        unsafe_allow_html=True
    )
    cA, cB, cC = st.columns(3)
    with cA:
        st.markdown(
            """
            **1) Entradas**  
            - Carga *símbolos*, *benchmark* y *fechas*  
            - Ajusta **Kelly base** (0.15–0.30 recomendado)  
            - Define *caps* del usuario  
            - (Opcional) sube `VFQ/QualityScore`  
            """)
    with cB:
        st.markdown(
            """
            **2) Macro**  
            - Ajusta `macro_z` (manual)  
            - Define **EMA α** y **umbrales ON/OFF**  
            - Edita **M_macro**, **β cap**, **pos cap** por régimen  
            - (Opcional) Exporta/Importa JSON
            """)
    with cC:
        st.markdown(
            """
            **3) Cartera / Sizing**  
            - Calcula **Kelly robusto** por activo  
            - (Opcional) Multiplicar por **M_macro**  
            - Respeta **caps** y **gate** (overlay=0)  
            - Sizing con capital ficticio + export CSV
            """)
    st.markdown("<span class='small-muted'>Tip: usa presets (Conservador / Balanceado / Agresivo) como atajos y luego ajusta según tu bibliografía.</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("📘 Detalle técnico: Kelly robusto + macro (clic para abrir)"):
    st.markdown(
        """
        **Kelly por activo (mensual, en exceso del benchmark)**  
        - **Binomial**: estima _hit rate_ **p** y _payoff_ **b**, con *shrinkage* a 0.5/1.0.  
          Fórmula: \\( f^* = p - \\frac{1-p}{b} \\) → `k_bin` (cap 0..1).  
        - **Continuo**: \\( \\mu/\\sigma^2 \\) con **EWMA** y **winsorize**, restando **costos** → `k_cont`.  
        - **Mezcla**: `k_raw = 0.5·k_bin + 0.5·k_cont`  
        - **Penalización por correlación**:  
          \\( k' = k_{raw} / (1 + \\lambda·\\max(0,\\rho_{i,proto})) \\) → `k_pen`.

        **Kelly fraccionado y normalización**  
        - Peso base:  
          \\[
          w^{(0)}_i = \\text{base\\_kelly}\\cdot
          \\frac{k_{pen,i}}{\\sum_j k_{pen,j}}
          \\]
        - (Opcional) **Macro**: multiplicar por **M_macro** del régimen.  
        - **Caps** efectivos (mínimo entre usuario y régimen):  
          - **Cap por posición**: \\( w_i \\leq \\text{pos\\_cap} \\)  
          - **Cap de beta**: \\( \\sum_i \\beta_i w_i \\leq \\beta\\_cap \\)  
        - **Gate** (overlay=0): impide **nuevas** entradas, mantiene existentes.

        **Macro (sin CSV, configurable)**  
        - Usamos `macro_z` **suavizado** con **EMA (α)**.  
        - **Histéresis** con dos umbrales:  
          - ON si \\( z_{EMA} \\geq \\text{thr\\_on} \\)  
          - OFF si \\( z_{EMA} \\leq \\text{thr\\_off} \\)  
          - Intermedio = NEU  
        - Cada régimen mapea a tus parámetros: **M_macro**, **β cap**, **pos cap** (100% editables).  

        **Quality tilt (opcional)**  
        - Multiplicador suave: \\( \\exp(\\alpha · z(q)) \\) (acotado), con **α** menor en OFF.

        ---
        **Export/Import de parámetros**  
        - Guarda tus *umbrales, α de EMA y parámetros por régimen* como JSON.  
        - Carga el JSON para trabajar con tus presets de bibliografía.
        """,
        unsafe_allow_html=False
    )