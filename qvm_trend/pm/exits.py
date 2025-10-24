# apps/pm_streamlit.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from qvm_trend.data_io import load_prices_panel, load_benchmark, DEFAULT_START, DEFAULT_END
from qvm_trend.pm.orchestrator import build_portfolio
from qvm_trend.pm.exits import build_exit_table
from qvm_trend.macro.macro_score import z_to_regime

st.set_page_config(page_title="Gestión de Cartera QVM + Macro", page_icon="🧭", layout="wide")

st.title("Gestión de Cartera — QVM + Kelly + Macro")

# ============== Sidebar =================
with st.sidebar:
    st.markdown("### 📅 Fechas & Benchmark")
    start = st.date_input("Inicio", pd.to_datetime(DEFAULT_START).date())
    end   = st.date_input("Fin",    pd.to_datetime(DEFAULT_END).date())
    bench = st.text_input("Benchmark", "SPY").strip().upper() or "SPY"

    st.markdown("---")
    st.markdown("### ⚖️ Kelly (robusto)")
    base_kelly = st.slider("Fracción base de Kelly", 0.1, 1.0, 0.5, 0.05)
    winsor_p   = st.slider("Winsor p", 0.0, 0.10, 0.02, 0.01)
    t0         = st.slider("t₀ (prudencia)", 1.0, 3.0, 2.0, 0.1)
    lam_blend  = st.slider("λ (μ/σ²)", 0.0, 0.6, 0.2, 0.05)
    min_months = st.slider("Mín. meses", 12, 180, 60, 6)

    st.markdown("---")
    st.markdown("### 🧱 Caps")
    pos_cap = st.slider("Cap por posición", 0.01, 0.10, 0.05, 0.005)
    beta_cap= st.slider("Cap ∑(β·w)", 0.5, 1.5, 1.0, 0.05)

    st.markdown("---")
    st.markdown("### 🧭 Macro Monitor")
    macro_z = st.slider("Macro z-score", -2.0, 2.0, 0.0, 0.1)
    allow_new_when_z_below = st.slider("Bloquear nuevas si z <", -1.5, 1.0, -0.5, 0.1)

    st.markdown("---")
    st.markdown("### 💼 Capital & órdenes")
    capital = st.number_input("Capital total [USD]", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    cash_buf= st.slider("Buffer de caja", 0.0, 0.20, 0.05, 0.01)
    lot     = st.number_input("Lote mínimo (acciones)", min_value=1, value=1, step=1)

# ====== Tabs ======
tabM, tabP, tabE = st.tabs(["🧭 Macro", "📊 Cartera", "📤 Salidas"])

# ====== MACRO (visual & knobs) ======
with tabM:
    reg = z_to_regime(float(macro_z))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro z", f"{reg.z:.2f}", reg.label)
    c2.metric("M_macro", f"{reg.m_multiplier:.2f}")
    c3.metric("β_cap sugerido", f"{reg.beta_cap:.2f}")
    c4.metric("pos_cap sugerido", f"{reg.vol_cap:.2f}")
    st.caption("Estos valores se combinan con tus caps manuales: se usa el mínimo de ambos.")

# ====== CARTERA ======
with tabP:
    st.subheader("Símbolos y pesos (Kelly + Macro + Quality tilt + Gate)")

    # 1) Entrada de símbolos (p.ej. watchlist/holdings)
    syms_text = st.text_area("Símbolos (coma-separados)", "AAPL,MSFT,GOOGL,AMZN,META,UBER,CHDN,CNR,BARN.SW")
    symbols = [s.strip().upper() for s in syms_text.split(",") if s.strip()]
    holdings_text = st.text_input("Holdings actuales (opcional, coma-separados)", "AAPL,MSFT")
    holdings = [s.strip().upper() for s in holdings_text.split(",") if s.strip()]

    # (Opcional) Calidad desde CSV exportado de QVM
    st.caption("Opcional: sube CSV con ['symbol','QualityScore' o 'VFQ']")
    up = st.file_uploader("CSV de Calidad/VFQ", type=["csv"], accept_multiple_files=False)
    df_quality = None
    if up is not None:
        try:
            tmp = pd.read_csv(up)
            col_q = "QualityScore" if "QualityScore" in tmp.columns else ("VFQ" if "VFQ" in tmp.columns else None)
            if col_q:
                df_quality = tmp[["symbol", col_q]].drop_duplicates("symbol")
        except Exception:
            st.warning("No pude leer el CSV; continuo sin calidad.")

    if st.button("Calcular pesos", use_container_width=True, type="primary"):
        if not symbols:
            st.warning("Ingresa al menos 1 símbolo."); st.stop()

        dfw = build_portfolio(
            symbols, bench, start.isoformat(), end.isoformat(),
            base_kelly=base_kelly, winsor_p=winsor_p, t0=t0, min_months=min_months, lam_blend=lam_blend,
            macro_z=macro_z, quality_df=df_quality,
            beta_cap_user=beta_cap, pos_cap=pos_cap,
            allow_new_when_z_below=allow_new_when_z_below,
            current_holdings=holdings
        )
        if dfw.empty:
            st.error("No se pudieron calcular pesos (datos insuficientes)."); st.stop()

        st.dataframe(dfw, use_container_width=True)
        st.session_state["weights_df"] = dfw

        # 2) Precios y órdenes (capital → acciones)
        panel = load_prices_panel(dfw["symbol"].tolist(), start.isoformat(), end.isoformat(), cache_key="pm_panel")
        prices = {s: (panel.get(s)["close"].iloc[-1] if (panel.get(s) is not None and not panel.get(s).empty) else np.nan)
                  for s in dfw["symbol"]}

        def build_orders(weights_df, prices_map, capital: float, cash_buffer: float=0.05, lot: int=1):
            cash = capital * (1 - float(cash_buffer))
            rows = []
            for _, r in weights_df.iterrows():
                s = r["symbol"]; w = float(r["weight"])
                px = float(prices_map.get(s, np.nan))
                if not np.isfinite(px) or px <= 0 or w <= 0:
                    continue
                target = cash * w
                # lotes
                shares = int(max(lot, (target // (px * lot)) * lot)) if target >= px else 0
                alloc = shares * px
                rows.append({"symbol": s, "price": px, "weight": w, "shares": shares, "alloc": alloc})
            out = pd.DataFrame(rows)
            # Reajuste simple si sobró o faltó mucho
            if not out.empty and out["alloc"].sum() > 0:
                out["alloc_pct"] = out["alloc"] / out["alloc"].sum()
            return out.sort_values("weight", ascending=False).reset_index(drop=True)

        orders = build_orders(dfw, prices, capital, cash_buf, lot)
        st.subheader("Órdenes sugeridas (post lotes)")
        st.dataframe(orders, use_container_width=True)

        # 3) Gráfico de torta
        if not orders.empty and orders["alloc"].sum() > 0:
            fig, ax = plt.subplots()
            ax.pie(orders["alloc"], labels=orders["symbol"], autopct="%1.1f%%")
            ax.set_title("Distribución por valor")
            st.pyplot(fig, use_container_width=True)

        # 4) Descarga CSV
        if not orders.empty:
            st.download_button("Descargar órdenes (CSV)", orders.to_csv(index=False).encode(), "ordenes.csv", "text/csv")

# ====== SALIDAS ======
with tabE:
    st.subheader("Señales de salida (técnicas y trimestrales)")
    st.caption("Reglas: fallo técnico persistente (MA200/momentum), revisión trimestral con degradación de fundamentos, y stops opcionales (ATR/DD).")

    ma200 = st.number_input("MA días", 50, 300, 200, 5)
    k_days = st.number_input("K días bajo MA para 'fallo persistente'", 1, 20, 5, 1)
    use_mom = st.toggle("Usar Mom 12–1 ≤ 0", value=True)
    qwin   = st.number_input("Ventana revisión trimestral (± días)", 0, 20, 5, 1)
    use_atr= st.toggle("Usar ATR stop", value=False)
    atr_m  = st.slider("ATR múltiplo", 1.0, 6.0, 3.0, 0.5)
    use_dd = st.toggle("Usar MaxDD stop", value=False)
    dd_p   = st.slider("MaxDD %", 0.05, 0.5, 0.20, 0.05)

    # Meta fundamentos (opcional): carga CSV con ['symbol','VFQ_pct_sector','coverage_count','guard_ok']
    up_meta = st.file_uploader("CSV fundamentos (opcional)", type=["csv"], key="meta_up")
    meta_df = None
    if up_meta is not None:
        try:
            m = pd.read_csv(up_meta)
            keep = [c for c in ["symbol","VFQ_pct_sector","coverage_count","guard_ok"] if c in m.columns]
            if "symbol" in keep:
                meta_df = m[keep].drop_duplicates("symbol")
        except Exception:
            st.warning("No pude leer el CSV de fundamentos; continuo sin metas.")

    # Usa símbolos actuales o los de la pestaña Cartera
    syms_src = st.text_input("Símbolos a evaluar (coma-separados, vacío = usar los de Cartera)",
                             value="")
    if syms_src.strip():
        eval_syms = [s.strip().upper() for s in syms_src.split(",") if s.strip()]
    else:
        eval_syms = st.session_state.get("weights_df", pd.DataFrame()).get("symbol", pd.Series([], dtype=str)).tolist()

    if st.button("Evaluar salidas", use_container_width=True):
        if not eval_syms:
            st.warning("No hay símbolos para evaluar."); st.stop()
        panelE = load_prices_panel(eval_syms, start.isoformat(), end.isoformat(), cache_key="exit_panel")
        exits = build_exit_table(panelE, meta_df,
                                 ma=int(ma200), k_days=int(k_days), use_mom=bool(use_mom),
                                 quarter_win=int(qwin),
                                 use_atr_stop=bool(use_atr), atr_mult=float(atr_m),
                                 use_dd_stop=bool(use_dd), dd_pct=float(dd_p))
        if exits.empty:
            st.info("Sin señales de salida.")
        else:
            st.dataframe(exits.sort_values(["exit_flag","symbol"], ascending=[False, True]), use_container_width=True)
            st.download_button("Descargar señales de salida (CSV)",
                               exits.to_csv(index=False).encode(), "salidas.csv", "text/csv")
