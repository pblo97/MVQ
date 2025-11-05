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
    st.subheader("🔬 Robust Covariance (Advanced)")
    with st.expander("📚 About Robust Covariance"):
        st.markdown("""
        **Why Robust Covariance?**

        Sample covariance suffers from **estimation error** when n ~ p (observations ~ assets):
        - Unstable inverse (ill-conditioned matrix)
        - Extreme weights (over-concentration)
        - Poor out-of-sample performance

        **Ledoit-Wolf Shrinkage (2004):**
        - Shrinks Σ̂ towards structured target (constant correlation or identity)
        - Optimal shrinkage intensity δ computed analytically
        - Reduces condition number → more stable weights

        **Recommended:** Ledoit-Wolf for portfolios with 5+ assets

        **Academic Reference:** Ledoit & Wolf (2004) - "Honey, I Shrunk the Sample Covariance Matrix"
        """)

    use_robust_cov = st.checkbox(
        "Use Robust Covariance Estimation",
        value=False,
        help="Applies Ledoit-Wolf shrinkage to reduce estimation error (Ledoit & Wolf 2004)"
    )

    if use_robust_cov:
        cov_method = st.selectbox(
            "Covariance Method",
            options=["ledoit_wolf", "oas", "ewm", "sample"],
            index=0,
            help="Method for covariance estimation:\n"
                 "- ledoit_wolf: Optimal shrinkage (recommended)\n"
                 "- oas: Oracle Approximating Shrinkage\n"
                 "- ewm: Exponentially weighted (RiskMetrics)\n"
                 "- sample: Standard covariance (no shrinkage)"
        )
    else:
        cov_method = "sample"

    st.markdown("---")
    st.subheader("📐 Allocation Method")
    with st.expander("📚 About Allocation Methods"):
        st.markdown("""
        **Kelly Criterion (Single-Asset):**
        - Calculates optimal fraction per asset individually
        - Robust, conservative, proven track record
        - Accounts for: p(win), payoff, μ, σ², correlation penalty
        - **Recommended** for most users

        **HRP (Hierarchical Risk Parity):**
        - López de Prado (2016) - "Building Diversified Portfolios"
        - Uses hierarchical clustering + recursive bisection
        - **No matrix inversion** (numerically stable)
        - Superior out-of-sample performance vs Markowitz
        - Good for: diversification, avoiding concentration risk
        - Does NOT account for expected returns (equal risk allocation)

        **Comparison:**
        - **Kelly:** Optimizes for growth (log utility), return-focused
        - **HRP:** Optimizes for diversification, risk-focused

        **Academic References:**
        - Kelly (1956): A New Interpretation of Information Rate
        - López de Prado (2016): Building Diversified Portfolios that Outperform Out-of-Sample
        """)

    allocation_method = st.radio(
        "Method",
        options=["Kelly (Single-Asset)", "HRP (Risk Parity)"],
        index=0,
        help="Kelly = return-optimized | HRP = risk-diversified"
    )

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

    # Run portfolio optimization based on selected method
    try:
        if allocation_method == "Kelly (Single-Asset)":
            # Kelly optimizer (existing)
            with st.spinner("Building portfolio with Kelly + Quality 3D caps..."):
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

        else:  # HRP (Risk Parity)
            with st.spinner("Building portfolio with HRP + Quality 3D caps..."):
                from portfolio_manager.allocation.hrp import compute_hrp_with_constraints
                from portfolio_manager.quality.composite import compute_quality_batch

                # Get prices and returns
                price_panel = load_prices_panel(symbols + [bench], start_date.isoformat(), end_date.isoformat(), cache_key="pm_hrp_panel")
                benchmark_df = price_panel.get(bench)

                returns_df = pd.DataFrame({
                    sym: pd.to_numeric(price_panel.get(sym, {}).get('close', pd.Series()), errors='coerce').pct_change()
                    for sym in symbols if sym in price_panel
                }).dropna(how='all')

                if returns_df.empty or len(returns_df) < 252:
                    raise ValueError("Insufficient data for HRP (need at least 252 days)")

                # Calculate quality scores (for caps)
                if use_quality_caps:
                    quality_df = compute_quality_batch(
                        symbols=symbols,
                        price_panel=price_panel,
                        fundamentals_df=fundamentals_df,
                        benchmark_df=benchmark_df,
                        weights=(0.4, 0.3, 0.3)  # liq, fund, tech
                    )
                else:
                    quality_df = pd.DataFrame()

                # Compute HRP weights (with or without quality caps)
                if use_quality_caps and not quality_df.empty:
                    # Extract position caps from quality_df
                    quality_map = dict(zip(
                        quality_df['symbol'].str.upper(),
                        quality_df['position_cap']
                    ))
                    min_weights = {sym: 0.0 for sym in returns_df.columns}
                    max_weights = {sym: quality_map.get(sym.upper(), 0.05) for sym in returns_df.columns}

                    hrp_weights = compute_hrp_with_constraints(
                        returns=returns_df,
                        min_weights=min_weights,
                        max_weights=max_weights,
                        linkage_method='single'
                    )
                else:
                    # HRP without constraints
                    from portfolio_manager.allocation.hrp import compute_hrp_weights
                    hrp_weights = compute_hrp_weights(
                        returns=returns_df,
                        linkage_method='single'
                    )

                # Apply macro multiplier
                reg = z_to_regime(macro_z_eff)
                M_macro = float(reg.m_multiplier)
                hrp_weights_scaled = hrp_weights * M_macro

                # Build portfolio DataFrame (compatible with Kelly format)
                portfolio_df = pd.DataFrame({
                    'symbol': hrp_weights_scaled.index,
                    'weight': hrp_weights_scaled.values,
                    'beta': np.nan,  # Calculate beta separately
                    'beta_w': np.nan
                })

                # Calculate betas
                if benchmark_df is not None and 'close' in benchmark_df.columns:
                    bench_ret = pd.to_numeric(benchmark_df['close'], errors='coerce').pct_change().dropna()

                    betas = []
                    for sym in portfolio_df['symbol']:
                        if sym in price_panel and 'close' in price_panel[sym].columns:
                            asset_ret = pd.to_numeric(price_panel[sym]['close'], errors='coerce').pct_change().dropna()
                            common_idx = asset_ret.index.intersection(bench_ret.index)

                            if len(common_idx) > 60:
                                asset_common = asset_ret.loc[common_idx]
                                bench_common = bench_ret.loc[common_idx]
                                cov = np.cov(asset_common, bench_common)[0, 1]
                                var_bench = np.var(bench_common)
                                beta = cov / var_bench if var_bench > 0 else 1.0
                            else:
                                beta = 1.0
                        else:
                            beta = 1.0
                        betas.append(beta)

                    portfolio_df['beta'] = betas

                # Apply beta cap
                portfolio_df['beta'] = portfolio_df['beta'].fillna(1.0)
                portfolio_df['beta_w'] = portfolio_df['beta'] * portfolio_df['weight']

                beta_cap_eff = min(beta_cap_user, reg.beta_cap)
                beta_total = portfolio_df['beta_w'].sum()

                if beta_total > beta_cap_eff and beta_total > 0:
                    scale_factor = beta_cap_eff / beta_total
                    portfolio_df['weight'] = portfolio_df['weight'] * scale_factor
                    portfolio_df['beta_w'] = portfolio_df['beta'] * portfolio_df['weight']

                # Add quality scores to portfolio_df
                if not quality_df.empty:
                    quality_score_map = dict(zip(quality_df['symbol'].str.upper(), quality_df['quality_score']))
                    quality_cap_map = dict(zip(quality_df['symbol'].str.upper(), quality_df['position_cap']))
                    portfolio_df['quality_score'] = portfolio_df['symbol'].str.upper().map(quality_score_map).fillna(50.0)
                    portfolio_df['quality_cap'] = portfolio_df['symbol'].str.upper().map(quality_cap_map).fillna(0.05)
                else:
                    portfolio_df['quality_score'] = np.nan
                    portfolio_df['quality_cap'] = np.nan

                portfolio_df['lambda_quality'] = 1.0  # HRP doesn't use quality penalty

                # Sort by weight
                portfolio_df = portfolio_df.sort_values('weight', ascending=False).reset_index(drop=True)

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

    # Show allocation method badge
    method_label = "Kelly (Single-Asset)" if allocation_method == "Kelly (Single-Asset)" else "HRP (Risk Parity)"
    method_color = "blue" if allocation_method == "Kelly (Single-Asset)" else "green"
    st.markdown(f"<p style='background-color: {method_color}; color: white; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold;'>📐 Method: {method_label}</p>", unsafe_allow_html=True)

    # Metrics
    reg = z_to_regime(macro_z_eff)
    weights = portfolio_df['weight'].values
    betas = portfolio_df['beta'].fillna(1.0).values

    N_eff = 1.0 / np.sum(weights ** 2) if weights.sum() > 0 else 0
    n_actives = int((weights > 1e-8).sum())
    beta_total = float(np.sum(betas * weights))
    beta_util = beta_total / max(reg.beta_cap, 1e-12)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("N Effective", f"{N_eff:.2f}")
    col2.metric("# Assets", f"{n_actives}")
    col3.metric("Σ(β·w)", f"{beta_total:.2f}")
    col4.metric("β-Cap Utilization", f"{beta_util:.1%}")
    col5.metric("Method", method_label[:4])

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

    # Covariance diagnostics (if robust covariance enabled)
    if use_robust_cov and cov_method != 'sample':
        st.markdown("---")
        st.markdown("### 🔬 Covariance Matrix Diagnostics")

        try:
            from portfolio_manager.allocation.kelly_vectorial import diagnose_covariance_quality

            # Get returns data for diagnostics
            price_panel = load_prices_panel(symbols, start_date.isoformat(), end_date.isoformat(), cache_key="e2e_cov_diag")
            returns_df = pd.DataFrame({
                sym: pd.to_numeric(price_panel.get(sym, {}).get('close', pd.Series()), errors='coerce').pct_change()
                for sym in symbols if sym in price_panel
            }).dropna(how='all')

            if not returns_df.empty:
                # Diagnose sample vs robust
                diag_sample = diagnose_covariance_quality(returns_df, method='sample')
                diag_robust = diagnose_covariance_quality(returns_df, method=cov_method)

                # Display comparison
                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    st.markdown("**Sample Covariance**")
                    st.metric("Condition Number", f"{diag_sample['condition_number']:.1f}")
                    st.metric("Min Eigenvalue", f"{diag_sample['min_eigenvalue']:.6f}")
                    st.metric("Rank", f"{diag_sample['rank']}/{diag_sample['dimension']}")
                    if diag_sample['is_positive_definite']:
                        st.success("✅ Positive Definite")
                    else:
                        st.error("❌ Not Positive Definite")

                with col_d2:
                    st.markdown(f"**{cov_method.upper()} (Robust)**")
                    st.metric("Condition Number", f"{diag_robust['condition_number']:.1f}")
                    st.metric("Min Eigenvalue", f"{diag_robust['min_eigenvalue']:.6f}")
                    if diag_robust['shrinkage_intensity'] is not None:
                        st.metric("Shrinkage δ", f"{diag_robust['shrinkage_intensity']:.3f}")
                    if diag_robust['is_positive_definite']:
                        st.success("✅ Positive Definite")
                    else:
                        st.error("❌ Not Positive Definite")

                # Recommendations
                st.info(f"**Sample:** {diag_sample['recommendation']}")
                st.info(f"**Robust:** {diag_robust['recommendation']}")

                st.caption("""
                **Interpretation:**
                - **Condition Number (κ):** Ratio of largest to smallest eigenvalue. High κ → unstable inverse
                  - κ < 100: Well-conditioned ✅
                  - 100 < κ < 1000: Moderate condition ⚠️
                  - κ > 1000: Ill-conditioned ❌
                - **Shrinkage δ:** Intensity of shrinkage (0 = no shrinkage, 1 = full shrinkage to target)
                - **Rank:** Full rank = dimension (no redundant assets)

                **Academic Reference:** Ledoit & Wolf (2004) - reduces condition number → more stable weights
                """)

        except Exception as e:
            st.warning(f"Could not compute covariance diagnostics: {e}")

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

    # CVaR Analysis
    st.markdown("---")
    st.markdown("### CVaR & VaR Analysis")

    try:
        from portfolio_manager.risk.cvar_analysis import (
            calculate_risk_metrics_summary,
            calculate_percentage_cvar_contribution,
            stress_test_scenarios
        )

        # Get returns data
        price_panel = load_prices_panel(symbols, start_date.isoformat(), end_date.isoformat(), cache_key="e2e_risk_cvar")
        returns_df = pd.DataFrame({
            sym: pd.to_numeric(price_panel.get(sym, {}).get('close', pd.Series()), errors='coerce').pct_change()
            for sym in symbols if sym in price_panel
        }).dropna(how='all')

        if not returns_df.empty and portfolio_df is not None:
            # Get weights
            weights_dict = portfolio_df.set_index('symbol')['weight'].to_dict()
            weights = np.array([weights_dict.get(sym, 0.0) for sym in returns_df.columns])

            # Normalize weights
            if weights.sum() > 0:
                weights = weights / weights.sum()

                # 1. Risk Metrics Summary
                st.markdown("#### VaR & CVaR (Historical vs Parametric)")
                risk_summary = calculate_risk_metrics_summary(
                    returns_df=returns_df,
                    weights=weights,
                    confidence_levels=[0.95, 0.99]
                )
                st.dataframe(risk_summary.style.format({
                    'VaR_historical': '{:.2f}%',
                    'CVaR_historical': '{:.2f}%',
                    'VaR_parametric': '{:.2f}%',
                    'CVaR_parametric': '{:.2f}%'
                }), use_container_width=True)

                st.caption("""
                **VaR (Value at Risk):** Maximum expected loss at confidence level (e.g., 95% = worst loss in 95% of cases)
                **CVaR (Conditional VaR):** Expected loss given that loss exceeds VaR (tail risk)
                """)

                # 2. Marginal CVaR Contributions
                st.markdown("---")
                st.markdown("#### Marginal CVaR Contributions (Risk Attribution)")

                pct_cvar_contrib = calculate_percentage_cvar_contribution(
                    returns_df=returns_df,
                    weights=weights,
                    confidence_level=0.95,
                    method='historical'
                )

                contrib_df = pd.DataFrame({
                    'Symbol': pct_cvar_contrib.index,
                    'Weight (%)': [weights_dict.get(sym, 0.0) * 100 for sym in pct_cvar_contrib.index],
                    'CVaR Contribution (%)': pct_cvar_contrib.values
                }).sort_values('CVaR Contribution (%)', ascending=False)

                st.dataframe(contrib_df.style.format({
                    'Weight (%)': '{:.2f}%',
                    'CVaR Contribution (%)': '{:.2f}%'
                }), use_container_width=True)

                # Chart
                if HAVE_PLOTLY:
                    fig_cvar = px.bar(
                        contrib_df,
                        x='Symbol',
                        y='CVaR Contribution (%)',
                        title="CVaR Contribution by Asset (95% confidence)",
                        color='CVaR Contribution (%)',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_cvar, use_container_width=True)

                st.caption("""
                **Marginal CVaR:** How much each asset contributes to portfolio tail risk.
                - High contribution = asset drives tail losses
                - Should sum to 100% (Euler decomposition property)
                """)

                # 3. Stress Testing
                st.markdown("---")
                st.markdown("#### Stress Testing Scenarios")

                stress_results = stress_test_scenarios(
                    returns_df=returns_df,
                    weights=weights,
                    scenarios=None  # Use default scenarios
                )

                st.dataframe(stress_results.style.format({
                    'portfolio_loss_pct': '{:.2f}%',
                    **{col: '{:.2f}%' for col in stress_results.columns if col.endswith('_shock')}
                }), use_container_width=True)

                # Worst scenario
                worst_scenario = stress_results.loc[stress_results['portfolio_loss_pct'].idxmin()]
                st.warning(f"**Worst Scenario:** {worst_scenario['scenario']} → **{worst_scenario['portfolio_loss_pct']:.2f}%** portfolio loss")

                st.caption("""
                **Stress Scenarios (Historical Events):**
                - **2008 Financial Crisis:** October 2008 market crash
                - **2020 COVID Crash:** March 2020 pandemic selloff
                - **2022 Rate Hike:** Fed tightening impact
                - **3-sigma / 5-sigma:** Statistical tail events
                - **Correlation One:** All assets down simultaneously
                """)

            else:
                st.warning("Portfolio has zero total weight - cannot calculate CVaR")
        else:
            st.warning("Insufficient returns data for CVaR analysis")

    except Exception as e:
        st.error(f"Error calculating CVaR: {e}")
        st.exception(e)

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
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    with col_ex1:
        ma_window = st.number_input("MA Window (days)", 100, 400, 200, 10)
    with col_ex2:
        mom_lookback = st.number_input("Momentum Lookback (days)", 180, 400, 252, 10)
    with col_ex3:
        use_piotroski = st.checkbox("Use Piotroski F-Score", value=True, help="Piotroski (VALUE stocks)")
    with col_ex4:
        use_mohanram = st.checkbox("Use Mohanram G-Score", value=False, help="Mohanram (GROWTH stocks)")

    # Educational note
    if use_piotroski and use_mohanram:
        st.info("💡 **Dual Mode:** Piotroski for VALUE (high B/M) | Mohanram for GROWTH (low B/M)")
    elif use_mohanram:
        st.info("💡 **Mohanram G-Score:** For GROWTH stocks. Focuses on R&D, Capex, advertising (Mohanram 2005)")

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
                    with st.expander("📈 Piotroski F-Score Summary (VALUE stocks)"):
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

                        **Best for:** VALUE stocks (high Book-to-Market)
                        """)
                else:
                    st.warning("⚠️ Could not calculate Piotroski scores (insufficient fundamental data)")
            except Exception as e:
                st.error(f"Error calculating Piotroski: {e}")
                st.warning("Falling back to technical signals only")

    # Calculate Mohanram historical scores if needed
    mohanram_hist = None
    if use_mohanram and fmp_key:
        with st.spinner("📊 Calculating Mohanram G-Scores (quarterly)..."):
            try:
                from portfolio_manager.fundamentals.mohanram import calculate_mohanram_history
                mohanram_hist = calculate_mohanram_history(symbols, fmp_key)

                if not mohanram_hist.empty:
                    st.success(f"✓ Mohanram G-Scores calculated for {len(mohanram_hist['symbol'].unique())} symbols")

                    # Show Mohanram summary
                    with st.expander("📈 Mohanram G-Score Summary (GROWTH stocks)"):
                        # Latest scores
                        latest_scores = mohanram_hist.sort_values('date').groupby('symbol').tail(1)[['symbol', 'date', 'G_SCORE']]
                        st.markdown("**Latest G-Scores:**")
                        st.dataframe(latest_scores.sort_values('G_SCORE', ascending=False), use_container_width=True)

                        st.markdown("""
                        **Interpretation (Mohanram 2005):**
                        - **7-8:** Excellent growth quality
                        - **6:** Strong growth fundamentals
                        - **4-5:** Above average
                        - **2-3:** Below average
                        - **0-1:** Weak growth fundamentals

                        **8 Signals:**
                        - Profitability: ROA, CFO, ROA > CFO
                        - Stability: ROA variability, Sales variability
                        - Investment: R&D, Capex, Advertising

                        **Best for:** GROWTH stocks (low Book-to-Market)

                        **Reference:** Mohanram (2005) - Separating Winners from Losers
                        """)
                else:
                    st.warning("⚠️ Could not calculate Mohanram scores (insufficient fundamental data)")
            except Exception as e:
                st.error(f"Error calculating Mohanram: {e}")
                st.warning("Falling back to technical signals only")

    try:
        price_panel = load_prices_panel(symbols + [bench], start_date.isoformat(), end_date.isoformat(), cache_key="e2e_exits")

        # Use enhanced exit table with Piotroski & Mohanram
        from portfolio_manager.monitor.exits_enhanced import build_exit_table_enhanced

        exit_table = build_exit_table_enhanced(
            panel=price_panel,
            bench_close=None,
            ma_window=int(ma_window),
            mom_lookback=int(mom_lookback),
            review_freq="Q",
            piotroski_hist=piotroski_hist,
            mohanram_hist=mohanram_hist,  # Mohanram G-Score (GROWTH stocks)
            vfq_hist=None,  # Legacy fallback
            use_piotroski=use_piotroski,
            use_mohanram=use_mohanram,
            degradation_threshold=2,  # F/G-Score drop ≥ 2 = degradation
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
    st.subheader("📈 Walk-Forward Backtest & Historical Reports")

    # Walk-Forward Backtest Section
    st.markdown("### Walk-Forward Backtest")
    st.markdown("""
    **Rigorous out-of-sample validation** to avoid overfitting (Bailey et al. 2014):
    - Train on rolling window → Test on next period → Step forward
    - No look-ahead bias, simulates real trading conditions
    - Comprehensive metrics: Sharpe, Sortino, Information Ratio, Calmar, Max Drawdown
    """)

    # Backtest Configuration
    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        train_window_months = st.selectbox("Training Window", [12, 24, 36], index=2, help="Months for training")
        train_window = train_window_months * 21  # Trading days
    with col_bt2:
        test_window_months = st.selectbox("Test Window", [1, 3, 6], index=1, help="Months for testing")
        test_window = test_window_months * 21
    with col_bt3:
        step_size_months = st.selectbox("Step Size", [1, 3], index=0, help="Months to step forward")
        step_size = step_size_months * 21

    col_bt4, col_bt5 = st.columns(2)
    with col_bt4:
        backtest_method = st.radio(
            "Backtest Method",
            ["Rolling Window", "Expanding Window"],
            help="Rolling = fixed training window, Expanding = cumulative training"
        )
    with col_bt5:
        benchmark_method = st.selectbox(
            "Benchmark Weights",
            ["Equal Weight", "Market Cap", "Custom"],
            help="Benchmark allocation for comparison"
        )

    if st.button("🚀 Run Backtest", type="primary"):
        try:
            from portfolio_manager.backtest.walk_forward import (
                walk_forward_backtest,
                expanding_window_backtest,
                calculate_backtest_metrics
            )

            with st.spinner("🔄 Running walk-forward backtest... (this may take 1-2 minutes)"):
                # Get returns data
                price_panel = load_prices_panel(
                    symbols + [bench],
                    start_date.isoformat(),
                    end_date.isoformat(),
                    cache_key="e2e_backtest"
                )

                returns_df = pd.DataFrame({
                    sym: pd.to_numeric(price_panel.get(sym, {}).get('close', pd.Series()), errors='coerce').pct_change()
                    for sym in symbols if sym in price_panel
                }).dropna(how='all')

                if returns_df.empty or len(returns_df) < train_window + test_window:
                    st.error(f"❌ Insufficient data for backtest. Need at least {train_window + test_window} days ({(train_window + test_window)/21:.0f} months)")
                    st.stop()

                # Benchmark weights
                n_assets = len(returns_df.columns)
                if benchmark_method == "Equal Weight":
                    benchmark_weights = np.ones(n_assets) / n_assets
                elif benchmark_method == "Market Cap":
                    # Approximate with inverse volatility (proxy for market cap)
                    vols = returns_df.std()
                    inv_vol = 1 / vols
                    benchmark_weights = (inv_vol / inv_vol.sum()).values
                else:
                    benchmark_weights = np.ones(n_assets) / n_assets

                # Strategy function (simplified Kelly)
                def simple_kelly_strategy(train_returns, base_kelly=0.25, winsor_p=0.01):
                    """Simplified Kelly strategy for backtest"""
                    # Winsorize
                    train_w = train_returns.clip(
                        lower=train_returns.quantile(winsor_p),
                        upper=train_returns.quantile(1 - winsor_p)
                    )

                    # Kelly weights: w = Σ^-1 μ / κ
                    mu = train_w.mean()
                    cov = train_w.cov()

                    try:
                        cov_inv = np.linalg.inv(cov.values + np.eye(len(cov)) * 1e-8)
                        weights_raw = base_kelly * (cov_inv @ mu.values)
                        weights_raw = np.clip(weights_raw, 0, None)  # Long-only

                        if weights_raw.sum() > 0:
                            weights = weights_raw / weights_raw.sum()
                        else:
                            weights = np.ones(len(mu)) / len(mu)
                    except np.linalg.LinAlgError:
                        # Fallback to equal weight
                        weights = np.ones(len(mu)) / len(mu)

                    return weights

                # Run backtest
                if backtest_method == "Rolling Window":
                    result = walk_forward_backtest(
                        returns_df=returns_df,
                        strategy_func=simple_kelly_strategy,
                        train_window=train_window,
                        test_window=test_window,
                        step_size=step_size,
                        min_train_obs=252,
                        benchmark_weights=benchmark_weights,
                        base_kelly=base_kelly,
                        winsor_p=winsor_p
                    )
                else:  # Expanding Window
                    result = expanding_window_backtest(
                        returns_df=returns_df,
                        strategy_func=simple_kelly_strategy,
                        initial_train_window=train_window,
                        test_window=test_window,
                        step_size=step_size,
                        min_train_obs=252,
                        benchmark_weights=benchmark_weights,
                        base_kelly=base_kelly,
                        winsor_p=winsor_p
                    )

                st.success("✓ Backtest completed!")

                # Store in session
                st.session_state['backtest_result'] = result

                # Display metrics
                st.markdown("### Backtest Performance Metrics")

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Sharpe Ratio (Strategy)", f"{result.metrics['sharpe_strategy']:.3f}")
                col_m2.metric("Sharpe Ratio (Benchmark)", f"{result.metrics['sharpe_benchmark']:.3f}")
                col_m3.metric("Information Ratio", f"{result.metrics['information_ratio']:.3f}")
                col_m4.metric("Calmar Ratio", f"{result.metrics['calmar_ratio']:.3f}")

                col_m5, col_m6, col_m7, col_m8 = st.columns(4)
                col_m5.metric("Sortino Ratio", f"{result.metrics['sortino_ratio']:.3f}")
                col_m6.metric("Win Rate", f"{result.metrics['win_rate']:.1%}")
                col_m7.metric("Max Drawdown", f"{result.metrics['max_drawdown']:.2%}")
                col_m8.metric("Total Return", f"{result.metrics['total_return_strategy']:.2%}")

                st.caption("""
                **Metrics Interpretation:**
                - **Sharpe Ratio:** Risk-adjusted returns (>1.0 = good, >2.0 = excellent)
                - **Information Ratio:** Excess return vs benchmark per unit of tracking error
                - **Sortino Ratio:** Like Sharpe, but only penalizes downside volatility
                - **Calmar Ratio:** Total return / Max Drawdown (risk-adjusted)
                - **Win Rate:** % of profitable periods
                """)

                # Cumulative returns chart
                st.markdown("---")
                st.markdown("### Cumulative Returns (Strategy vs Benchmark)")

                cum_strategy = (1 + pd.Series(result.strategy_returns)).cumprod()
                cum_benchmark = (1 + pd.Series(result.benchmark_returns)).cumprod()
                cum_dates = pd.date_range(end=end_date, periods=len(cum_strategy), freq='D')

                cum_df = pd.DataFrame({
                    'Date': cum_dates,
                    'Strategy': cum_strategy.values,
                    'Benchmark': cum_benchmark.values
                })

                if HAVE_PLOTLY:
                    fig_cum = px.line(
                        cum_df,
                        x='Date',
                        y=['Strategy', 'Benchmark'],
                        title=f"Cumulative Returns ({backtest_method})",
                        labels={'value': 'Cumulative Return', 'variable': 'Portfolio'}
                    )
                    fig_cum.update_traces(line=dict(width=2))
                    st.plotly_chart(fig_cum, use_container_width=True)
                else:
                    st.line_chart(cum_df.set_index('Date'))

                # Drawdown chart
                st.markdown("---")
                st.markdown("### Drawdown Analysis")

                running_max_strategy = cum_strategy.cummax()
                drawdown_strategy = (cum_strategy - running_max_strategy) / running_max_strategy

                running_max_benchmark = cum_benchmark.cummax()
                drawdown_benchmark = (cum_benchmark - running_max_benchmark) / running_max_benchmark

                dd_df = pd.DataFrame({
                    'Date': cum_dates,
                    'Strategy DD': drawdown_strategy.values,
                    'Benchmark DD': drawdown_benchmark.values
                })

                if HAVE_PLOTLY:
                    fig_dd = px.line(
                        dd_df,
                        x='Date',
                        y=['Strategy DD', 'Benchmark DD'],
                        title="Drawdown Over Time",
                        labels={'value': 'Drawdown', 'variable': 'Portfolio'}
                    )
                    fig_dd.update_traces(line=dict(width=2))
                    st.plotly_chart(fig_dd, use_container_width=True)
                else:
                    st.line_chart(dd_df.set_index('Date'))

                # Download results
                st.markdown("---")
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    metrics_df = pd.DataFrame([result.metrics])
                    st.download_button(
                        "📥 Download Metrics",
                        metrics_df.to_csv(index=False).encode(),
                        file_name=f"backtest_metrics_{datetime.now().strftime('%Y-%m-%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_dl2:
                    returns_export = pd.DataFrame({
                        'date': cum_dates,
                        'strategy_return': result.strategy_returns,
                        'benchmark_return': result.benchmark_returns
                    })
                    st.download_button(
                        "📥 Download Returns",
                        returns_export.to_csv(index=False).encode(),
                        file_name=f"backtest_returns_{datetime.now().strftime('%Y-%m-%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"❌ Backtest error: {e}")
            st.exception(e)

    # Available snapshots
    st.markdown("---")
    st.markdown("### Historical Portfolio Snapshots")
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
