# Portfolio Optimization End-to-End

Sistema completo de gestión de cartera con:
- **Kelly Fraccional Robusto** (binomial + continuo blend)
- **Macro Régimen Overlay** (z-score compuesto + Markov)
- **Quality Score 3D** (Liquidity + Fundamental + Technical)
- **Exit Monitoring** (MA200 + Momentum 12-1 + VFQ degradation)
- **Transaction Cost Model** (Spread + Market Impact)
- **Persistencia CSV** (6 archivos: portfolio, macro, quality, exits, risk, orders)

---

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys in Streamlit secrets
# .streamlit/secrets.toml:
FRED_API_KEY = "your_fred_key"
FMP_API_KEY = "your_fmp_key"
```

### 2. Run Macro Monitor (generate bundle)

```bash
streamlit run ma_streamlit.py
```

- Click "Ejecutar pipeline"
- Download `macro_monitor_bundle.csv`

### 3. Run Portfolio E2E

```bash
streamlit run portfolio_e2e_streamlit.py
```

- Upload `macro_monitor_bundle.csv` (or auto-detects if in `./`, `./data/`, or `./snapshots/`)
- Configure Kelly parameters in sidebar
- Click **"Run Portfolio Optimization"**
- Explore 5 tabs:
  1. **Portfolio Overview**: weights, sizing, downloads
  2. **Macro Monitor**: z-score, regime, overlay signal
  3. **Risk Analytics**: correlation heatmap, CVaR (coming soon)
  4. **Asset Quality & Exits**: 3D scores, exit signals (EXIT/TRIM/HOLD)
  5. **Backtest & Reports**: historical snapshots, walk-forward (coming soon)

---

## 📁 Architecture

```
MVQ/
├── portfolio_manager/              # NEW modules
│   ├── quality/
│   │   └── composite.py           # Quality Score 3D (Liq + Fund + Tech)
│   ├── execution/
│   │   └── cost_model.py          # Transaction costs (spread + impact)
│   ├── monitor/
│   │   └── persistence.py         # CSV save/load (6 files)
│   ├── data/
│   │   └── orchestrator_enhanced.py  # Kelly + Quality caps integration
│   └── backtest/                  # (future) walk-forward engine
├── qvm_trend/                     # Existing modules (REUSED)
│   ├── macro/
│   │   └── macro_score.py         # Regime mapping (z → M_macro, caps)
│   ├── pm/
│   │   ├── orchestrator.py        # Kelly robusto base
│   │   └── exits.py               # Exit signals (MA200, Mom, VFQ)
│   └── fquality/
│       └── fmp_quality.py         # FMP fundamentals fetcher
├── snapshots/                     # CSV persistence (auto-created)
│   ├── portfolio_state_YYYY-MM-DD.csv
│   ├── macro_scores_YYYY-MM-DD.csv
│   ├── quality_scores_YYYY-MM-DD.csv
│   ├── exit_signals_YYYY-MM-DD.csv
│   ├── risk_metrics_YYYY-MM-DD.csv
│   └── allocation_history.csv
├── ma_streamlit.py                # Macro Monitor (generates bundle)
├── pm_streamlit.py                # Old PM app (legacy, still works)
└── portfolio_e2e_streamlit.py     # NEW integrated app (5 tabs)
```

---

## 🎯 User Flow (Weekly Routine)

### **Monday AM** (5 min): Generate Macro Bundle

1. Run `ma_streamlit.py`
2. Click "Ejecutar pipeline"
3. Review regime (ON/NEUTRAL/OFF)
4. Download `macro_monitor_bundle.csv`

### **Tuesday AM** (10 min): Build Portfolio

1. Run `portfolio_e2e_streamlit.py`
2. Upload macro bundle (or auto-detects)
3. Configure symbols + Kelly parameters
4. Click "Run Portfolio Optimization"
5. Review Tab 1 (weights, quality scores)
6. Click **"Save Complete State"** → generates 6 CSVs

### **Wednesday AM** (5 min): Review Exit Signals

1. Go to Tab 4 (Asset Quality & Exits)
2. Filter by EXIT/TRIM actions
3. Note activos con alertas
4. Download `exit_signals_YYYY-MM-DD.csv`

### **Thursday AM** (10 min): Execute Orders

1. Open `sizing_YYYY-MM-DD.csv` (from Tab 1 download)
2. Compare with current broker positions
3. Generate orders (SELL exits first, then BUY)
4. Log executed orders in `rebalance_orders_YYYY-MM-DD.csv`

### **Friday AM** (5 min): Update State

1. Upload previous CSVs to Tab 5
2. Review changes (detect_changes feature)
3. Generate weekly report

**Total time:** ~35 min/week

---

## 📊 CSV Files Generated

### 1. `portfolio_state_YYYY-MM-DD.csv`
```csv
date,symbol,weight,shares,price,value,quality_score,kelly_fraction,sector,beta
2025-11-05,AAPL,0.08,100,180.50,18050,85,0.65,Tech,1.15
...
```

### 2. `macro_scores_YYYY-MM-DD.csv`
```csv
date,macro_z,regime,M_macro,beta_cap_sug,pos_cap_sug,overlay_signal
2025-11-05,0.75,ON,1.25,1.25,0.07,1
```

### 3. `quality_scores_YYYY-MM-DD.csv`
```csv
date,symbol,quality_score,liq_score,fund_score,tech_score,position_cap,ADV,spread_bps
2025-11-05,AAPL,85,90,82,83,0.10,50000000,3
```

### 4. `exit_signals_YYYY-MM-DD.csv`
```csv
date,symbol,price,MA200,ma_flag,mom_12_1,mom_flag,vfq_delta,quality_flag,action,reason
2025-11-05,PYPL,65.30,72.50,True,-0.08,True,-12,Degrading,EXIT,"Close<200MA; Mom<0; Cal↓"
```

### 5. `risk_metrics_YYYY-MM-DD.csv`
```csv
date,sharpe,sortino,max_dd,cvar_95,volatility,turnover
2025-11-05,1.85,2.10,-0.082,0.012,0.105,0.12
```

### 6. `allocation_history.csv` (append-only)
```csv
date,symbol,weight
2025-11-05,AAPL,0.08
2025-11-05,GOOGL,0.06
2025-11-12,AAPL,0.09
...
```

---

## 🔧 Key Components Explained

### **Quality Score 3D** (0-100)

**Liquidity Score (40%):**
- ADV (average daily volume USD)
- Bid-ask spread estimate (via volatility)
- Volume stability (CV)
- Days-to-liquidate (for $100k position)

**Fundamental Score (30%):**
- Market cap tier (large > mid > small)
- Debt/Equity ratio
- ROE consistency
- Sector defensiveness

**Technical Score (30%):**
- Volatility regime (current vs historical)
- Correlation stability (vs benchmark)
- Drawdown depth
- Momentum consistency

**Position Cap Mapping:**
- Quality 80-100 → cap 10%
- Quality 60-80 → cap 6%
- Quality 40-60 → cap 4%
- Quality <40 → cap 2% (or exclude)

### **Transaction Cost Model**

**Spread Cost:**
- Tiered by ADV (mega liquid 3-5 bps, illiquid 20-50 bps)
- Adjusted by volatility

**Market Impact:**
```
impact_cost = 0.1 × (shares/ADV)^1.5 × volatility × notional
```
- Non-linear: 5% ADV → ~5-10 bps, 20% ADV → ~30-50 bps

### **Exit Signals Rules**

- **EXIT** if: (MA200 breach AND Mom<0) OR (MA200 breach AND Quality degrading)
- **TRIM** if: MA200 breach OR Mom<0 OR Quality degrading
- **HOLD** otherwise

Reviewed quarterly (next_review date auto-calculated).

---

## 🎨 Streamlit Tabs Breakdown

### Tab 1: Portfolio Overview
- Weights table (with quality caps, lambda_quality)
- KPIs: N_eff, #assets, Σ(β·w), β-cap utilization
- Charts: weights bar, beta contrib bar
- Position sizing (with M_macro multiplier)
- Downloads: portfolio CSV, sizing CSV
- **Save Complete State** button → 6 CSVs

### Tab 2: Macro Monitor
- Z-score gauge (M_macro visualizer)
- Regime status (ON/NEUTRAL/OFF)
- Composite Z timeline
- Overlay signal (0/1 step chart)
- Markov probabilities (if in bundle)

### Tab 3: Risk Analytics
- Correlation heatmap (60d rolling)
- (Future) Marginal CVaR contributions
- (Future) VaR backtest
- (Future) Stress scenarios (2008, 2020, 2022)

### Tab 4: Asset Quality & Exits
- Quality scores 3D table
- Quality scatter plot (quality vs liquidity)
- Exit signals table (EXIT/TRIM/HOLD filter)
- Download exit_signals CSV
- Save to snapshots

### Tab 5: Backtest & Reports
- List available snapshots (by date)
- Load historical state (portfolio, macro, quality, exits, risk)
- (Future) Walk-forward backtest engine
- (Future) Performance reports (PDF)

---

## 🔄 Integration with Existing Apps

### **Macro Monitor (ma_streamlit.py)** → **REUTILIZADO 100%**
- Generates `macro_monitor_bundle.csv`
- Consumed by portfolio_e2e_streamlit.py (Tab 2)
- No code changes needed

### **PM Streamlit (pm_streamlit.py)** → **LEGACY (still works)**
- Old portfolio manager (Kelly + Exits)
- Can co-exist with new app
- Migrate workflows to portfolio_e2e_streamlit.py for integrated experience

### **QVM Trend (qvm_trend/)** → **CORE REUSED**
- `orchestrator.py`: Kelly base (extended by orchestrator_enhanced.py)
- `exits.py`: Exit signals (MA200, Mom, VFQ) - no changes
- `macro_score.py`: Regime mapping - no changes
- `fquality/fmp_quality.py`: FMP fetcher - no changes

**Backward compatible:** All existing code works unchanged.

---

## 🚧 Coming Soon (Future Enhancements)

### Backtest Module
- Walk-forward expanding window (12m train)
- Stress scenarios (2008, 2020, 2022)
- Cost simulation (realistic slippage)
- Performance attribution

### Risk Analytics
- Marginal CVaR contributions (treemap)
- VaR backtest (coverage analysis)
- Factor exposures (PCA)
- Tail risk metrics

### Execution Module
- Rebalance scheduler (monthly + bands)
- TWAP/VWAP order generation
- %ADV constraints (5% default)
- Broker integration (IBKR, Alpaca)

### Alerts System
- Regime shift alerts (z-score Δ > 1.0 in 5d)
- CVaR breach
- Quality degradation (>20 pts in 30d)
- Position cap violations

---

## 📖 References & Theory

### Kelly Criterion
- **Binomial:** `f* = p - (1-p)/b` (shrinkage to 0.5/1.0)
- **Continuo:** `f* = μ/σ²` (EWMA, winsorized, costs deducted)
- **Blend:** `k_raw = 0.5·k_bin + 0.5·k_cont`
- **Penalty:** `k' = k_raw / (1 + λ·max(0, ρ_i,proto))`

### Macro Overlay
- Z-score compuesto (Term, Credit, Liquidity, USD)
- Markov 2-state (calma vs estrés)
- Overlay grid-search OOS (Sharpe optimization)
- Histéresis (enter/exit thresholds)

### Quality Caps
- Multi-dimensional scoring (liquidity, fundamentals, technicals)
- Dynamic position caps (quality → max weight)
- λ_quality penalty (low quality → reduced weight)

### Exit Rules
- MA200 breach (trend deterioration)
- Momentum 12-1 < 0 (performance decline)
- VFQ degradation 1Q (fundamental weakness)
- Trimestral review frequency

---

## 🙏 Credits

Built on top of:
- **MVQ existing codebase** (qvm_trend/)
- **ma_streamlit.py** (Macro Monitor)
- **pm_streamlit.py** (legacy PM)

New modules:
- `portfolio_manager/quality/` (Quality Score 3D)
- `portfolio_manager/execution/` (Transaction costs)
- `portfolio_manager/monitor/` (CSV persistence)
- `portfolio_e2e_streamlit.py` (integrated UI)

---

## 📧 Support

For issues or questions:
1. Check this README
2. Review code comments in modules
3. Test with sample data (AAPL, GOOGL, MSFT, NVDA, SPY)

**Happy portfolio optimization!** 🎯📈
