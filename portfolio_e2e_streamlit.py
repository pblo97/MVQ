# portfolio_e2e_streamlit.py
"""
Portfolio Optimization End-to-End
Integración completa: Macro Monitor + Kelly + Quality 3D + Exit Signals + Persistencia CSV
"""
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime
from pathlib import Path

# Setup paths
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Imports del proyecto existente
from qvm_trend.data_io import load_prices_panel, load_benchmark, DEFAULT_START, DEFAULT_END
from qvm_trend.macro.macro_score import z_to_regime, macro_z_from_series
from qvm_trend.pm.exits import build_exit_table
from qvm_trend.fquality.fmp_quality import compute_quality_from_fmp

# Imports nuevos módulos
from portfolio_manager.data.orchestrator_enhanced import (
    build_portfolio_with_quality_caps,
    estimate_rebalance_costs
)
from portfolio_manager.monitor.persistence import PortfolioStatePersistence
from portfolio_manager.quality.composite import compute_quality_batch

# Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Portfolio E2E | Kelly + Macro + Quality",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Portfolio Optimization End-to-End")
st.caption("Kelly Fraccional + Macro Régimen + Quality Score 3D + Exit Monitoring + Persistencia CSV")

# Persistencia
persist = PortfolioStatePersistence(snapshots_dir="snapshots/")

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio Overview",
    "🌐 Macro Monitor",
    "⚠️ Risk Analytics",
    "🚪 Asset Quality & Exits",
    "📈 Backtest & Reports"
])

# ==================== SIDEBAR: INPUTS ====================
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Universe")
    symbols_txt = st.text_area(
        "Symbols (comma-separated)",
        "AAPL,GOOGL,MSFT,NVDA,JPM,BAC,UNH,LLY,XOM,CVX",
        height=80
    )
    symbols = [s.strip().upper() for s in symbols_txt.split(",") if s.strip()]

    bench = st.text_input("Benchmark", value="SPY").strip().upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01").date())
    end_date = st.date_input("End Date", value=pd.to_datetime(DEFAULT_END).date())

    st.markdown("---")
    st.subheader("Kelly Parameters")
    base_kelly = st.slider("Base Kelly Fraction", 0.05, 0.50, 0.25, 0.01)
    winsor_p = st.slider("Winsorize p (%)", 0.0, 5.0, 1.0, 0.25) / 100
    costs_bps = st.number_input("Costs per month (bps)", 0, 100, 5, 1)
    costs_per_period = costs_bps / 10_000
    lambda_corr = st.slider("Correlation Penalty λ", 0.0, 1.0, 0.25, 0.05)

    st.markdown("---")
    st.subheader("Caps & Constraints")
    beta_cap_user = st.number_input("Beta Cap (Σβ·w)", 0.25, 2.0, 1.2, 0.05)
    use_quality_caps = st.checkbox("Use Quality 3D Caps", value=True)

    st.markdown("---")
    if st.button("🚀 Run Portfolio Optimization", type="primary"):
        st.session_state['run_optimization'] = True

# ==================== TAB 1: PORTFOLIO OVERVIEW ====================
with tab1:
    st.subheader("Portfolio Allocation & Metrics")

    if not st.session_state.get('run_optimization', False):
        st.info("👈 Configure parameters in sidebar and click **Run Portfolio Optimization**")
        st.stop()

    if not symbols:
        st.warning("Please enter symbols in the sidebar")
        st.stop()

    # ========== MACRO Z-SCORE (FRED AUTO-FETCH) ==========
    st.markdown("---")
    st.subheader("🌐 Macro Indicators (Auto-fetch from FRED)")

    from portfolio_manager.macro_fred_compatible import (
        calculate_macro_zscore_auto_fred,
        get_macroarimax_default_weights
    )

    # FRED API Key input
    col_key, col_window = st.columns([2, 1])
    with col_key:
        # Try to get from secrets first
        default_fred_key = st.secrets.get("FRED_API_KEY", "")
        fred_api_key = st.text_input(
            "FRED API Key",
            value=default_fred_key,
            type="password",
            help="Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    with col_window:
        window_days = st.selectbox(
            "Rolling window",
            options=[252, 126],
            index=0,
            help="252d = annual, 126d = semi-annual"
        )

    macro_z_eff = 0.0
    macro_bundle = None
    result_df = None

    if fred_api_key and fred_api_key.strip():
        try:
            with st.spinner("🔄 Fetching FRED data and calculating macro z-score..."):
                # Auto-fetch FRED and calculate z-score
                result_df, macro_z_eff, messages = calculate_macro_zscore_auto_fred(
                    fred_api_key=fred_api_key.strip(),
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    window=window_days,
                    weights=get_macroarimax_default_weights(),
                    clip_z=3.5,
                    verbose=False
                )

            # Show diagnostic messages
            with st.expander("🔍 FRED Fetch Diagnostics", expanded=(result_df.empty)):
                for msg in messages:
                    if "❌" in msg or "Failed" in msg:
                        st.error(msg)
                    elif "⚠️" in msg:
                        st.warning(msg)
                    else:
                        st.info(msg)

            if result_df.empty:
                st.warning("⚠️ No FRED data fetched. Using default macro_z = 0.0 (NEUTRAL)")
                st.info("""
                **Troubleshooting:**
                1. **Check API Key:** Verify your FRED API key is valid
                2. **Get API Key:** https://fred.stlouisfed.org/docs/api/api_key.html
                3. **Network:** Ensure you can access FRED API (not blocked by firewall)
                4. **Check diagnostics above** for specific error messages
                """)
                macro_z_eff = 0.0
            else:
                reg = z_to_regime(macro_z_eff)

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Macro Z-Score", f"{macro_z_eff:.2f}")
                col_m2.metric("Regime", reg.label)
                col_m3.metric("M_macro", f"{reg.m_multiplier:.2f}")

                st.success(f"✓ FRED data fetched & z-score calculated: **{macro_z_eff:.2f}** (Regime: {reg.label})")

                # Charts & details
                with st.expander("📈 View composite z-score timeline & breakdown"):
                    if HAVE_PLOTLY:
                        fig = px.line(
                            result_df['composite_z'].rename_axis('Date').reset_index(),
                            x='Date',
                            y='composite_z',
                            title="Composite Z-Score (MacroArimax Method)"
                        )
                        fig.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="ON threshold")
                        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", annotation_text="OFF threshold")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.line_chart(result_df['composite_z'])

                    # Show individual z-scores
                    st.markdown("**Individual z-scores (last 10 days):**")
                    display_cols = [col for col in result_df.columns if col.endswith('_z')]
                    if display_cols:
                        st.dataframe(result_df[display_cols].tail(10), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error fetching FRED data or calculating z-score: {e}")
            st.exception(e)
            st.warning("Using default macro_z = 0.0 (NEUTRAL)")
            macro_z_eff = 0.0
    else:
        st.info("""
        **Macro z-score calculation (Auto-fetch from FRED)**

        **Required:** FRED API Key (free)
        👉 Get yours at: https://fred.stlouisfed.org/docs/api/api_key.html

        **What this does:**
        1. Automatically fetches indicators from FRED:
           - **$ family:** RRPONTSYD (RRP), WTREGEN (TGA), WRESBAL (Reserves), WALCL (Fed Balance)
           - **bp family:** SOFR, EFFR, OBFR, TGCRRATE (Repo), BAMLH0A0HYM2 (HY OAS), T10Y2Y (Curve)

        2. Calculates Net Liquidity: `NL = WRESBAL - WTREGEN - RRPONTSYD`

        3. Computes z-score using **MacroArimax method:**
           - Normalization by families ($ billions separate from bp)
           - Economic signs (drain vs inject liquidity)
           - Winsorization p1-p99 + clip |z| ≤ 3.5
           - Rolling window: 252d (annual) or 126d (semi-annual)

        4. Returns composite z-score → regime (ON/NEUTRAL/OFF) → M_macro multiplier

        **Without API key:** uses default macro_z = 0.0 (NEUTRAL regime)

        **Note:** This is the same calculation as your MacroArimax/liquidity stress program!
        """)

    st.session_state['macro_z_eff'] = macro_z_eff
    st.markdown("---")

    # Fetch fundamentals desde FMP (para quality score)
    fmp_key = st.secrets.get("FMP_API_KEY", "")
    fundamentals_df = None
    if fmp_key:
        try:
            with st.spinner("Fetching fundamentals from FMP..."):
                fundamentals_df = compute_quality_from_fmp(symbols, fmp_key)
        except Exception as e:
            st.warning(f"Could not fetch fundamentals: {e}")

    # Run portfolio optimization
    try:
        with st.spinner("Building portfolio with Quality 3D caps..."):
            portfolio_df, quality_df = build_portfolio_with_quality_caps(
                symbols=symbols,
                bench=bench,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                base_kelly=base_kelly,
                winsor_p=winsor_p,
                costs_per_period=costs_per_period,
                lambda_corr=lambda_corr,
                macro_z=macro_z_eff,
                beta_cap_user=beta_cap_user,
                use_quality_caps=use_quality_caps,
                fundamentals_df=fundamentals_df
            )
    except Exception as e:
        st.error(f"❌ Error building portfolio: {e}")
        st.exception(e)
        st.stop()

    if portfolio_df.empty:
        st.error("❌ Could not build portfolio. Possible causes:")
        st.markdown("""
        - **Insufficient data:** Need at least 36 months of price history
        - **Invalid symbols:** Check that symbols exist (e.g., AAPL, GOOGL, MSFT)
        - **Date range too short:** Start date should be >= 3 years before end date
        - **Benchmark issues:** SPY data not available for date range

        **Debug tips:**
        - Try fewer symbols (5-6 large caps: AAPL, GOOGL, MSFT, NVDA, JPM)
        - Extend start date to 2020-01-01 or earlier
        - Check symbols are valid US tickers
        """)
        st.stop()

    # Store in session
    st.session_state['portfolio_df'] = portfolio_df
    st.session_state['quality_df'] = quality_df

    # Metrics
    reg = z_to_regime(macro_z_eff)
    weights = portfolio_df['weight'].values
    betas = portfolio_df['beta'].fillna(1.0).values

    N_eff = 1.0 / np.sum(weights ** 2) if weights.sum() > 0 else 0
    n_actives = int((weights > 1e-8).sum())
    beta_total = float(np.sum(betas * weights))
    beta_util = beta_total / max(reg.beta_cap, 1e-12)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N Effective", f"{N_eff:.2f}")
    col2.metric("# Assets", f"{n_actives}")
    col3.metric("Σ(β·w)", f"{beta_total:.2f}")
    col4.metric("β-Cap Utilization", f"{beta_util:.1%}")

    # Portfolio table
    st.markdown("### Portfolio Weights")
    display_cols = ['symbol', 'weight', 'beta', 'beta_w', 'quality_score', 'quality_cap', 'lambda_quality']
    display_cols = [c for c in display_cols if c in portfolio_df.columns]
    st.dataframe(
        portfolio_df[display_cols].style.format({
            'weight': '{:.4f}',
            'beta': '{:.3f}',
            'beta_w': '{:.4f}',
            'quality_score': '{:.1f}',
            'quality_cap': '{:.4f}',
            'lambda_quality': '{:.3f}'
        }),
        use_container_width=True
    )

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.bar_chart(portfolio_df.set_index('symbol')['weight'])
        st.caption("Portfolio Weights")
    with c2:
        st.bar_chart(portfolio_df.set_index('symbol')['beta_w'])
        st.caption("Beta Contribution (β·w)")

    # Sizing section
    st.markdown("---")
    st.subheader("💰 Position Sizing")
    col_size1, col_size2, col_size3 = st.columns(3)
    with col_size1:
        capital_usd = st.number_input("Capital (USD)", value=500000, min_value=0, step=10000)
    with col_size2:
        cash_pct = st.slider("Cash Reserve %", 0.0, 0.50, 0.05, 0.01)
    with col_size3:
        use_m_macro = st.toggle("Apply M_macro Multiplier", value=True)

    M_macro = reg.m_multiplier if use_m_macro else 1.0
    investable = capital_usd * (1 - cash_pct)
    alloc_capital = investable * M_macro

    # Get latest prices
    price_panel = load_prices_panel(symbols, start_date.isoformat(), end_date.isoformat(), cache_key="e2e_prices")
    latest_prices = {}
    for sym in symbols:
        try:
            close = pd.to_numeric(price_panel.get(sym, {}).get('close', pd.Series()), errors='coerce').dropna()
            latest_prices[sym] = close.iloc[-1] if not close.empty else np.nan
        except Exception:
            latest_prices[sym] = np.nan

    px_series = pd.Series(latest_prices)
    weights_series = portfolio_df.set_index('symbol')['weight']
    alloc_usd = weights_series * alloc_capital
    qty = (alloc_usd / px_series).fillna(0).apply(lambda x: int(max(0, np.floor(x))))
    used_usd = qty * px_series

    sizing_df = pd.DataFrame({
        'symbol': weights_series.index,
        'weight': weights_series.values,
        'price': px_series.reindex(weights_series.index).values,
        'alloc_usd': alloc_usd.values,
        'qty': qty.reindex(weights_series.index).values,
        'used_usd': used_usd.reindex(weights_series.index).values
    }).sort_values('used_usd', ascending=False).reset_index(drop=True)

    cash_left = capital_usd - used_usd.sum()
    st.dataframe(sizing_df, use_container_width=True)
    st.caption(f"💵 Cash remaining: **${cash_left:,.0f}** | M_macro = {M_macro:.2f}")

    # Download buttons
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            "📥 Download Portfolio",
            portfolio_df.to_csv(index=False).encode(),
            file_name=f"portfolio_{datetime.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            "📥 Download Sizing",
            sizing_df.to_csv(index=False).encode(),
            file_name=f"sizing_{datetime.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl3:
        if st.button("💾 Save Complete State", use_container_width=True):
            try:
                today = datetime.now().date()
                paths = persist.save_complete_state(
                    date=today,
                    portfolio_df=portfolio_df,
                    macro_data={
                        'macro_z': macro_z_eff,
                        'regime': reg.label,
                        'M_macro': reg.m_multiplier,
                        'beta_cap_sug': reg.beta_cap,
                        'pos_cap_sug': reg.vol_cap
                    },
                    quality_df=quality_df if not quality_df.empty else None
                )
                st.success(f"✓ Saved {len(paths)} files to snapshots/")
            except Exception as e:
                st.error(f"Error saving state: {e}")

# ==================== TAB 2: MACRO MONITOR ====================
with tab2:
    st.subheader("🌐 Macro Monitor")

    if macro_bundle is not None and HAVE_PLOTLY:
        # Z-score gauge
        macro_z_val = st.session_state.get('macro_z_eff', 0.0)
        reg = z_to_regime(macro_z_val)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Macro Z-Score", f"{macro_z_val:.2f}")
        col_m2.metric("Regime", reg.label)
        col_m3.metric("M_macro", f"{reg.m_multiplier:.2f}")

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=reg.m_multiplier,
            gauge={"axis": {"range": [0.6, 1.3]}, "bar": {"color": "darkblue"}},
            title={"text": "M_macro Multiplier"}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Composite Z timeline
        if 'COMPOSITE_Z' in macro_bundle.columns:
            fig_comp = px.line(
                macro_bundle.rename_axis('Date').reset_index(),
                x='Date',
                y='COMPOSITE_Z',
                title="Composite Z-Score Timeline"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # Overlay signal
        if 'Overlay_Signal' in macro_bundle.columns:
            fig_overlay = px.step(
                macro_bundle.rename_axis('Date').reset_index(),
                x='Date',
                y='Overlay_Signal',
                title="Overlay Signal (0/1)"
            )
            st.plotly_chart(fig_overlay, use_container_width=True)

    else:
        st.info("Upload or auto-load `macro_monitor_bundle.csv` to view macro charts")
        st.markdown("""
        **Macro Monitor generates:**
        - COMPOSITE_Z (Term, Credit, Liquidity, USD)
        - Overlay Signal (grid-search OOS)
        - Markov regime probabilities
        - Suggested beta_cap & pos_cap per regime
        """)

# ==================== TAB 3: RISK ANALYTICS ====================
with tab3:
    st.subheader("⚠️ Risk Analytics")

    portfolio_df = st.session_state.get('portfolio_df')
    if portfolio_df is None or portfolio_df.empty:
        st.warning("Run portfolio optimization first (Tab 1)")
        st.stop()

    # Correlation heatmap
    st.markdown("### Correlation Matrix (60d rolling)")
    try:
        price_panel = load_prices_panel(symbols, start_date.isoformat(), end_date.isoformat(), cache_key="e2e_risk")
        ret_df = pd.DataFrame({
            sym: pd.to_numeric(price_panel.get(sym, {}).get('close', pd.Series()), errors='coerce').pct_change()
            for sym in symbols if sym in price_panel
        }).dropna(how='all').tail(60)

        if not ret_df.empty:
            corr = ret_df.corr()
            fig_corr = px.imshow(
                corr,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap (60d)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute correlation: {e}")

    # Risk decomposition (placeholder)
    st.markdown("### Risk Decomposition")
    st.info("**Coming soon:** Marginal CVaR contributions, VaR backtest, stress scenarios")

# ==================== TAB 4: ASSET QUALITY & EXITS ====================
with tab4:
    st.subheader("🚪 Asset Quality & Exit Signals")

    # Quality scores table
    quality_df = st.session_state.get('quality_df')
    if quality_df is not None and not quality_df.empty:
        st.markdown("### Quality Scores 3D")
        st.dataframe(quality_df, use_container_width=True)

        # Quality scatter
        if HAVE_PLOTLY:
            fig_quality = px.scatter(
                quality_df,
                x='quality_score',
                y='liq_score',
                size='ADV',
                color='position_cap',
                hover_data=['symbol'],
                title="Quality Score vs Liquidity (size = ADV)"
            )
            st.plotly_chart(fig_quality, use_container_width=True)

    # Exit signals
    st.markdown("---")
    st.markdown("### Exit Signals (MA200, Momentum 12-1, Fundamental Degradation)")

    # Configuration
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        ma_window = st.number_input("MA Window (days)", 100, 400, 200, 10)
    with col_ex2:
        mom_lookback = st.number_input("Momentum Lookback (days)", 180, 400, 252, 10)
    with col_ex3:
        use_piotroski = st.checkbox("Use Piotroski F-Score", value=True, help="Piotroski F-Score (9 signals) vs legacy VFQ")

    # Calculate Piotroski historical scores if needed
    piotroski_hist = None
    if use_piotroski and fmp_key:
        with st.spinner("📊 Calculating Piotroski F-Scores (quarterly)..."):
            try:
                from portfolio_manager.fundamentals.piotroski import calculate_piotroski_history
                piotroski_hist = calculate_piotroski_history(symbols, fmp_key)

                if not piotroski_hist.empty:
                    st.success(f"✓ Piotroski F-Scores calculated for {len(piotroski_hist['symbol'].unique())} symbols")

                    # Show Piotroski summary
                    with st.expander("📈 Piotroski F-Score Summary"):
                        # Latest scores
                        latest_scores = piotroski_hist.sort_values('date').groupby('symbol').tail(1)[['symbol', 'date', 'F_SCORE']]
                        st.markdown("**Latest F-Scores:**")
                        st.dataframe(latest_scores.sort_values('F_SCORE', ascending=False), use_container_width=True)

                        st.markdown("""
                        **Interpretation (Piotroski 2000):**
                        - **8-9:** Excellent fundamental quality
                        - **7:** Strong fundamentals
                        - **5-6:** Above average
                        - **3-4:** Below average
                        - **0-2:** Weak fundamentals

                        **9 Signals:** ROA>0, CFO>0, ΔROA>0, Accrual<0, ΔLEVER<0, ΔLIQUID>0, EQ_OFFER=0, ΔMARGIN>0, ΔTURN>0
                        """)
                else:
                    st.warning("⚠️ Could not calculate Piotroski scores (insufficient fundamental data)")
            except Exception as e:
                st.error(f"Error calculating Piotroski: {e}")
                st.warning("Falling back to technical signals only")

    try:
        price_panel = load_prices_panel(symbols + [bench], start_date.isoformat(), end_date.isoformat(), cache_key="e2e_exits")

        # Use enhanced exit table with Piotroski
        from portfolio_manager.monitor.exits_enhanced import build_exit_table_enhanced

        exit_table = build_exit_table_enhanced(
            panel=price_panel,
            bench_close=None,
            ma_window=int(ma_window),
            mom_lookback=int(mom_lookback),
            review_freq="Q",
            piotroski_hist=piotroski_hist,
            vfq_hist=None,  # Legacy fallback
            use_piotroski=use_piotroski,
            degradation_threshold=2,  # F-Score drop ≥ 2 = degradation
            vfq_col="VFQ",
            vfq_delta_thr=0.10
        )

        if not exit_table.empty:
            # Filter by action
            actions_filter = st.multiselect(
                "Filter by Action",
                options=["EXIT", "TRIM", "HOLD"],
                default=["EXIT", "TRIM"]
            )
            if actions_filter:
                exit_table = exit_table[exit_table['action'].isin(actions_filter)]

            st.dataframe(exit_table, use_container_width=True)

            st.download_button(
                "📥 Download Exit Signals",
                exit_table.to_csv(index=False).encode(),
                file_name=f"exit_signals_{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv"
            )

            # Save to persistence
            if st.button("💾 Save Exit Signals"):
                try:
                    persist.save_exit_signals(datetime.now().date(), exit_table)
                    st.success("✓ Exit signals saved to snapshots/")
                except Exception as e:
                    st.error(f"Error saving: {e}")
        else:
            st.warning("No exit signals generated")

    except Exception as e:
        st.error(f"Error building exit table: {e}")

# ==================== TAB 5: BACKTEST & REPORTS ====================
with tab5:
    st.subheader("📈 Backtest & Historical Reports")

    st.info("**Coming soon:** Walk-forward backtest engine with stress scenarios")

    # Available snapshots
    st.markdown("### Available Portfolio Snapshots")
    try:
        dates_avail = persist.list_available_dates()
        if dates_avail:
            st.write(f"Found {len(dates_avail)} snapshots:")
            st.write(", ".join(dates_avail[-10:]))  # últimos 10

            # Load specific date
            sel_date = st.selectbox("Select date to load:", options=dates_avail)
            if st.button(f"Load state from {sel_date}"):
                state = persist.load_complete_state(sel_date)
                if state['portfolio'] is not None:
                    st.dataframe(state['portfolio'], use_container_width=True)
                    st.success(f"✓ Loaded state from {sel_date}")
        else:
            st.write("No snapshots found. Save a portfolio state in Tab 1 first.")
    except Exception as e:
        st.warning(f"Could not list snapshots: {e}")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Portfolio E2E | Developed with Kelly Criterion + Macro Overlay + Quality 3D + Exit Monitoring")
