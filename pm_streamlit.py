import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

# ==== imports del paquete ====
from qvm_trend.data_io import load_prices_panel, load_benchmark, DEFAULT_START, DEFAULT_END
from qvm_trend.pm.orchestrator import build_portfolio
from qvm_trend.pm.exits import build_exit_table
from qvm_trend.macro.macro_score import z_to_regime, macro_z_from_series

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
    st.subheader("Macro Monitor: bundle o slider")
    c1, c2 = st.columns(2)

    macro_z_val = None
    beta_cap_sug = None
    pos_cap_sug  = None
    overlay_gate_series = None
    macro_bundle = None

    with c1:
        up_macro = st.file_uploader("macro_monitor_bundle.csv", type=["csv"])
        if up_macro is not None:
            try:
                mb = pd.read_csv(up_macro, index_col=0, parse_dates=True)
                macro_bundle = mb.sort_index()
                st.success(f"Bundle macro cargado: {macro_bundle.shape[0]} filas, {macro_bundle.shape[1]} columnas")

                # 1) macro_z directo si viene; si no, intenta COMPOSITE_Z / COMPOSITE_PCA usando macro_z_from_series
                if "macro_z" in macro_bundle.columns and pd.notna(macro_bundle["macro_z"]).any():
                    macro_z_val = float(macro_bundle["macro_z"].dropna().iloc[-1])
                else:
                    if "COMPOSITE_Z" in macro_bundle.columns and pd.notna(macro_bundle["COMPOSITE_Z"]).any():
                        macro_z_val = float(macro_z_from_series(macro_bundle["COMPOSITE_Z"]))
                    elif "COMPOSITE_PCA" in macro_bundle.columns and pd.notna(macro_bundle["COMPOSITE_PCA"]).any():
                        macro_z_val = float(macro_z_from_series(macro_bundle["COMPOSITE_PCA"]))
                    else:
                        macro_z_val = 0.0  # fallback

                if "beta_cap_sug" in macro_bundle.columns:
                    beta_cap_sug = float(macro_bundle["beta_cap_sug"].dropna().iloc[-1])
                if "pos_cap_sug" in macro_bundle.columns:
                    pos_cap_sug  = float(macro_bundle["pos_cap_sug"].dropna().iloc[-1])

                overlay_gate_series = macro_bundle.get("Overlay_Signal")
                st.success(f"Macro listo (z={macro_z_val:.2f})")

                # ==== Visualizaciones interactivas ====
                st.markdown("### Gráficos del bundle")
                g1, g2 = st.columns(2)
                with g1:
                    cols_plot = [c for c in ["COMPOSITE_Z","COMPOSITE_PCA"] if c in macro_bundle.columns]
                    if cols_plot:
                        dfp = macro_bundle[cols_plot].rename_axis("Date").reset_index()
                        import plotly.express as px
                        fig = px.line(dfp, x="Date", y=cols_plot, title="Composites macro")
                        st.plotly_chart(fig, use_container_width=True)
                with g2:
                    if "Overlay_Signal" in macro_bundle.columns:
                        dfov = macro_bundle["Overlay_Signal"].rename_axis("Date").reset_index()
                        import plotly.express as px
                        fig = px.step(dfov, x="Date", y="Overlay_Signal", title="Overlay (0/1)", markers=False)
                        st.plotly_chart(fig, use_container_width=True)

                g3, g4 = st.columns(2)
                with g3:
                    # Régimen derivado desde macro_z (o desde composite si no hay)
                    reg_series = None
                    try:
                        mz_series = None
                        if "macro_z" in macro_bundle.columns:
                            mz_series = macro_bundle["macro_z"]
                        elif "COMPOSITE_Z" in macro_bundle.columns:
                            mz_series = macro_bundle["COMPOSITE_Z"]
                        if mz_series is not None:
                            # mapa simple a régimen usando z_to_regime por punto
                            reg_series = mz_series.dropna().apply(lambda z: z_to_regime(float(z)).label)
                            dfreg = reg_series.rename("Regime").rename_axis("Date").reset_index()
                            import plotly.express as px
                            fig = px.area(dfreg, x="Date", y=dfreg["Regime"].apply(lambda r: 1 if r=="ON" else (0 if r=="OFF" else 0.5)),
                                          title="Régimen (OFF/NEU/ON)", range_y=[-0.1, 1.1])
                            fig.update_yaxes(tickvals=[0,0.5,1.0], ticktext=["OFF","NEU","ON"])
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as _:
                        pass
                with g4:
                    # Equity naive vs filtered si están
                    if {"Excess_Ret","Ret_Filtered"}.issubset(macro_bundle.columns):
                        df_ret = macro_bundle[["Excess_Ret","Ret_Filtered"]].dropna().copy()
                        df_ret["EQ_naive"] = (1 + df_ret["Excess_Ret"]).cumprod()
                        df_ret["EQ_filtered"] = (1 + df_ret["Ret_Filtered"]).cumprod()
                        import plotly.express as px
                        fig = px.line(df_ret.rename_axis("Date").reset_index(),
                                      x="Date", y=["EQ_naive","EQ_filtered"],
                                      title="Curva de capital: naive vs filtrado")
                        st.plotly_chart(fig, use_container_width=True)

                # KPIs rápidos si hay retornos
                if {"Excess_Ret","Ret_Filtered"}.issubset(macro_bundle.columns):
                    ret = macro_bundle["Excess_Ret"].dropna()
                    ref = macro_bundle["Ret_Filtered"].dropna()
                    def _safe_sharpe(x):
                        x = x.dropna()
                        return float(x.mean()/x.std()) if x.std() not in (None,0,np.nan) and np.isfinite(x.std()) else np.nan
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Sharpe naive", f"{_safe_sharpe(ret):.3f}")
                    k2.metric("Sharpe filtrado", f"{_safe_sharpe(ref):.3f}")
                    k3.metric("% tiempo ON", f"{100.0*float(macro_bundle.get('Overlay_Signal', pd.Series()).mean()):.1f}%" if "Overlay_Signal" in macro_bundle.columns else "—")

            except Exception as e:
                st.error(f"Error leyendo macro bundle: {e}")
                macro_z_val = None

    with c2:
        st.caption("Si no subes bundle, usa el slider:")
        macro_z_slider = st.slider("macro_z (manual)", -2.5, 2.5, 0.0, 0.1)

    macro_z_eff = macro_z_val if macro_z_val is not None else macro_z_slider

    # Régimen y multiplicadores (desde z_to_regime)
    reg = z_to_regime(float(macro_z_eff))
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("macro_z", f"{reg.z:.2f}")
    k2.metric("Régimen", reg.label)
    k3.metric("M_macro", f"{reg.m_multiplier:.2f}")
    k4.metric("β cap / pos cap", f"{reg.beta_cap:.2f} / {reg.vol_cap:.2f}")

    # Guardar en sesión para otras pestañas
    st.session_state["macro_z_eff"] = macro_z_eff
    # Si bundle trae sugerencias, priorízalas; si no, usa las del régimen base
    st.session_state["beta_cap_sug"] = beta_cap_sug if beta_cap_sug is not None else reg.beta_cap
    st.session_state["pos_cap_sug"]  = pos_cap_sug  if pos_cap_sug  is not None else reg.vol_cap
    st.session_state["overlay_gate_series"] = overlay_gate_series
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
