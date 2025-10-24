import os, sys
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
import plotly.graph_objects as go

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

# ------------------ MACRO ------------------
with tab_macro:
    st.subheader("Macro Monitor: control interno (sin CSV) + bundle opcional")

    # ---------- Parámetros anti-ruido / histéresis ----------
    ctop1, ctop2, ctop3, ctop4 = st.columns(4)
    with ctop1:
        macro_z_slider = st.slider("macro_z (manual)", -2.5, 2.5, 0.0, 0.1)
    with ctop2:
        ema_alpha = st.slider("Suavizado EMA (α)", 0.05, 0.50, 0.20, 0.01,
                              help="Más alto = reacciona más rápido (menos suavizado).")
    with ctop3:
        thr_on  = st.number_input("Umbral ON (≥)", value=0.50, step=0.05, format="%.2f",
                                  help="Si z_suav ≥ este nivel → estado ON.")
    with ctop4:
        thr_off = st.number_input("Umbral OFF (≤)", value=-0.50, step=0.05, format="%.2f",
                                  help="Si z_suav ≤ este nivel → estado OFF. Entre ambos = NEUTRAL.")

    # ---------- Estado y buffers en sesión ----------
    hist = st.session_state.get("macro_hist", pd.DataFrame(columns=["ts","z_raw","z_ema","overlay"]))
    last_state = st.session_state.get("overlay_state", 1)  # 1=ON por defecto
    last_z_ema = float(hist["z_ema"].iloc[-1]) if not hist.empty else macro_z_slider

    # EMA
    z_ema = (1-ema_alpha)*last_z_ema + ema_alpha*float(macro_z_slider) if not hist.empty else float(macro_z_slider)

    # Histéresis de overlay
    state = last_state
    if z_ema >= float(thr_on):
        state = 1
    elif z_ema <= float(thr_off):
        state = 0

    # Append punto actual (tipo “streaming”)
    new_row = pd.DataFrame([{"ts": pd.Timestamp.utcnow(), "z_raw": float(macro_z_slider),
                             "z_ema": z_ema, "overlay": int(state)}])
    hist = pd.concat([hist, new_row], ignore_index=True)
    # Mantén sólo los últimos N puntos (evitar crecer sin límite)
    N = 600
    if len(hist) > N:
        hist = hist.iloc[-N:].reset_index(drop=True)

    # Guarda sesión
    st.session_state["macro_hist"] = hist
    st.session_state["overlay_state"] = state

    # ---------- Métricas del régimen usando z_ema ----------
    reg = z_to_regime(float(z_ema))
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("macro_z (EMA)", f"{reg.z:.2f}")
    k2.metric("Régimen", reg.label)
    k3.metric("M_macro", f"{reg.m_multiplier:.2f}")
    k4.metric("β cap / pos cap", f"{reg.beta_cap:.2f} / {reg.vol_cap:.2f}")

    # ---------- Visual mínimo siempre disponible ----------
    import plotly.graph_objects as go
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(reg.m_multiplier),
            title={"text": "M_macro (×)"},
            gauge={"axis": {"range": [0.5, 1.5]}}
        ))
        st.plotly_chart(fig, use_container_width=True)
    with c_g2:
        caps_df = pd.DataFrame({
            "Cap": ["β·w total", "Posición"],
            "Valor": [float(reg.beta_cap), float(reg.vol_cap)]
        })
        st.bar_chart(caps_df.set_index("Cap"))
        st.caption("Caps sugeridos por régimen")

    # ---------- Series internas (línea y step) ----------
    st.markdown("### Serie interna (z & overlay)")
    if not hist.empty:
        h = hist.copy()
        h["ts"] = pd.to_datetime(h["ts"])
        h = h.set_index("ts").tail(400)  # muestra última ventana

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            import plotly.express as px
            dfz = h[["z_raw","z_ema"]].rename(columns={"z_raw":"z", "z_ema":"z_ema"})
            fig = px.line(dfz.reset_index(), x="ts", y=["z","z_ema"], title="macro_z crudo vs EMA")
            # bandas de histéresis
            fig.add_hline(y=float(thr_on), line_dash="dot", line_color="green")
            fig.add_hline(y=float(thr_off), line_dash="dot", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        with c_s2:
            dfov = h[["overlay"]].astype(int).reset_index()
            import plotly.express as px
            fig = px.step(dfov, x="ts", y="overlay", title="Overlay (0/1)")
            st.plotly_chart(fig, use_container_width=True)

        # Banda de régimen
        st.caption("Timeline de régimen (OFF/NEU/ON) según z_ema + histéresis")
        reg_series = h["z_ema"].apply(lambda z: z_to_regime(float(z)).label)
        dfreg = pd.DataFrame({"ts": h.index, "Regime": reg_series.values})
        import plotly.express as px
        fig = px.area(dfreg, x="ts",
                      y=dfreg["Regime"].apply(lambda r: 1 if r=="ON" else (0 if r=="OFF" else 0.5)),
                      title="Régimen", range_y=[-0.1, 1.1])
        fig.update_yaxes(tickvals=[0,0.5,1.0], ticktext=["OFF","NEU","ON"])
        st.plotly_chart(fig, use_container_width=True)

        # KPI derivados de overlay interno
        pct_on = float(h["overlay"].mean()) if len(h) else 0.0
        st.metric("% tiempo ON (ventana)", f"{100*pct_on:.1f}%")

    # ---------- Bundle OPCIONAL (si quieres ver compuestos y equity) ----------
    with st.expander("Cargar bundle macro (opcional)"):
        up_macro = st.file_uploader("macro_monitor_bundle.csv", type=["csv"], key="macro_bundle_file")
        if up_macro is not None:
            try:
                mb = pd.read_csv(up_macro, index_col=0, parse_dates=True).sort_index()
                st.success(f"Bundle macro cargado: {mb.shape[0]} filas, {mb.shape[1]} columnas")
                # z opcional desde bundle (no pisa el control interno, solo visual)
                cols_plot = [c for c in ["COMPOSITE_Z","COMPOSITE_PCA"] if c in mb.columns]
                if cols_plot:
                    import plotly.express as px
                    fig = px.line(mb[cols_plot].rename_axis("Date").reset_index(),
                                  x="Date", y=cols_plot, title="Composites macro (bundle)")
                    st.plotly_chart(fig, use_container_width=True)
                if "Overlay_Signal" in mb.columns:
                    import plotly.express as px
                    fig = px.step(mb["Overlay_Signal"].rename_axis("Date").reset_index(),
                                  x="Date", y="Overlay_Signal", title="Overlay bundle (0/1)")
                    st.plotly_chart(fig, use_container_width=True)
                if {"Excess_Ret","Ret_Filtered"}.issubset(mb.columns):
                    dfr = mb[["Excess_Ret","Ret_Filtered"]].dropna().copy()
                    dfr["EQ_naive"] = (1 + dfr["Excess_Ret"]).cumprod()
                    dfr["EQ_filtered"] = (1 + dfr["Ret_Filtered"]).cumprod()
                    import plotly.express as px
                    fig = px.line(dfr.rename_axis("Date").reset_index(),
                                  x="Date", y=["EQ_naive","EQ_filtered"], title="Curva de capital (bundle)")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error leyendo bundle: {e}")

    # ---------- Variables compartidas con otras pestañas ----------
    st.session_state["macro_z_eff"] = float(z_ema)  # usamos z suavizado como “efectivo”
    st.session_state["beta_cap_sug"] = reg.beta_cap
    st.session_state["pos_cap_sug"]  = reg.vol_cap
    # Gate táctico (usa el último de nuestra serie interna)
    st.session_state["overlay_gate_series"] = st.session_state["macro_hist"]["overlay"] if "macro_hist" in st.session_state else None

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
with tab_exits:
    st.subheader("Reglas de salida por símbolo")
    try:
        panel = load_prices_panel(symbols + [bench], start.isoformat(), end.isoformat(), cache_key="pm_panel")
        bench_px = load_benchmark(bench, start.isoformat(), end.isoformat())
        table = build_exit_table(
            panel=panel,
            bench_close=None if bench_px is None else bench_px["close"],
            ma_window=200,
            mom_lookback=252,
            review_freq="Q"   # revisión trimestral
        )
        # Filtro para eliminar símbolos
        sym_del = st.multiselect("Eliminar símbolos de la tabla", options=sorted(table["symbol"].unique().tolist()))
        if sym_del:
            table = table[~table["symbol"].isin(sym_del)]

        st.dataframe(table, use_container_width=True)
        st.caption("Salida si: rompe MA200 y/o Mom 12-1 < 0 en revisión trimestral; motivos y fecha estimada.")

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
