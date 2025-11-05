# Piotroski F-Score - Fundamental Degradation Detection

## 📊 Overview

Implementación completa del **Piotroski F-Score** para detectar degradación fundamental en activos del portafolio.

Basado en el paper académico seminal:
**"Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers"**
*Joseph D. Piotroski, Journal of Accounting Research, 2000*

---

## 🎯 ¿Qué es el Piotroski F-Score?

El F-Score es un número entre **0-9** que mide la fortaleza fundamental de una empresa usando 9 señales binarias (0/1):

### **Profitability (4 señales):**
1. **F_ROA:** ROA > 0 (rentabilidad positiva)
2. **F_CFO:** CFO > 0 (cash flow operativo positivo)
3. **F_ΔROA:** ΔROA > 0 (rentabilidad mejorando YoY)
4. **F_ACCRUAL:** Accrual < 0 (CFO > Net Income, señal de earnings quality)

### **Leverage/Liquidity (3 señales):**
5. **F_ΔLEVER:** ΔLEVER < 0 (apalancamiento reduciéndose)
6. **F_ΔLIQUID:** ΔLIQUID > 0 (liquidez mejorando, current ratio up)
7. **F_EQ_OFFER:** EQ_OFFER = 0 (no emisión de acciones nueva)

### **Operating Efficiency (2 señales):**
8. **F_ΔMARGIN:** ΔMARGIN > 0 (margen bruto mejorando)
9. **F_ΔTURN:** ΔTURNOVER > 0 (asset turnover mejorando)

---

## 📚 Interpretación (Según Paper Original)

| F-Score | Quality | Interpretation |
|---------|---------|----------------|
| **8-9** | Excellent | Alta calidad fundamental, baja probabilidad de quiebra |
| **7** | Strong | Fundamentales sólidos |
| **5-6** | Above Average | Por encima del promedio |
| **3-4** | Below Average | Por debajo del promedio |
| **0-2** | Weak | Fundamentales débiles, alta probabilidad de distress |

### **Degradation Detection:**

- **ΔF-Score ≥ +2:** Fundamentales mejorando ("Improving")
- **ΔF-Score ≤ -2:** Fundamentales degradando ("Degrading")
- **|ΔF-Score| < 2:** Fundamentales estables ("Flat")

---

## 🔧 Usage

### **1. Calculate Piotroski F-Score History**

```python
from portfolio_manager.fundamentals.piotroski import calculate_piotroski_history

# Calculate quarterly F-Scores for your portfolio
symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA']
fmp_api_key = "your_fmp_api_key"

piotroski_hist = calculate_piotroski_history(
    symbols=symbols,
    api_key=fmp_api_key
)

# Result: DataFrame with columns ['symbol', 'date', 'F_SCORE', ...9 signals...]
print(piotroski_hist.tail())
```

### **2. Detect Fundamental Degradation**

```python
from portfolio_manager.fundamentals.piotroski import detect_fundamental_degradation

# Detect degradation for a specific symbol
degradation_info = detect_fundamental_degradation(
    piotroski_hist=piotroski_hist,
    symbol='AAPL',
    degradation_threshold=2  # Drop of 2+ points = degrading
)

print(degradation_info)
# {
#     'f_score_last': 7.0,
#     'f_score_prev': 8.0,
#     'f_score_delta': -1.0,
#     'degradation_flag': 'Flat'  # or 'Degrading' / 'Improving'
# }
```

### **3. Enhanced Exit Monitoring**

```python
from portfolio_manager.monitor.exits_enhanced import build_exit_table_enhanced

# Build exit table with Piotroski-based degradation detection
exit_table = build_exit_table_enhanced(
    panel=price_panel,
    ma_window=200,
    mom_lookback=252,
    piotroski_hist=piotroski_hist,
    use_piotroski=True,
    degradation_threshold=2
)

# Columns:
# - symbol, price_last, MA200, ma_flag, Mom12-1, mom_flag
# - f_score_last, f_score_prev, f_score_delta, fundamental_flag
# - reason, action (EXIT/TRIM/HOLD), next_review
```

---

## 📈 Integration with Portfolio E2E

En `portfolio_e2e_streamlit.py`, Tab 4 (Asset Quality & Exits):

1. **Checkbox:** "Use Piotroski F-Score" (default: True)
2. **Auto-calculation:** Sistema obtiene datos de FMP y calcula F-Scores trimestrales
3. **Exit Signals:** Combina MA200 + Momentum + Piotroski degradation

**Exit Rules:**
- **EXIT:** (MA flag AND Momentum flag) OR (MA flag AND Fundamental degrading)
- **TRIM:** MA flag OR Momentum flag OR Fundamental degrading
- **HOLD:** Sin señales

---

## 🧪 Example: Full Workflow

```python
# 1. Calculate Piotroski history
from portfolio_manager.fundamentals.piotroski import (
    calculate_piotroski_history,
    interpret_fscore
)

symbols = ['AAPL', 'MSFT', 'JPM', 'XOM']
piotroski_hist = calculate_piotroski_history(symbols, fmp_api_key)

# 2. Check latest F-Scores
latest = piotroski_hist.sort_values('date').groupby('symbol').tail(1)
for _, row in latest.iterrows():
    print(f"{row['symbol']}: F-Score = {row['F_SCORE']} ({interpret_fscore(row['F_SCORE'])})")

# Output:
# AAPL: F-Score = 8 (Excellent)
# MSFT: F-Score = 7 (Strong)
# JPM: F-Score = 6 (Above Average)
# XOM: F-Score = 4 (Below Average)

# 3. Detect degradation
from portfolio_manager.fundamentals.piotroski import detect_fundamental_degradation

for sym in symbols:
    info = detect_fundamental_degradation(piotroski_hist, sym)
    if info['degradation_flag'] == 'Degrading':
        print(f"⚠️ {sym}: F-Score dropped from {info['f_score_prev']:.0f} to {info['f_score_last']:.0f}")

# 4. Build exit table
from portfolio_manager.monitor.exits_enhanced import build_exit_table_enhanced

exit_table = build_exit_table_enhanced(
    panel=price_panel,
    piotroski_hist=piotroski_hist,
    use_piotroski=True
)

# Filter EXIT/TRIM signals
alerts = exit_table[exit_table['action'].isin(['EXIT', 'TRIM'])]
print(alerts[['symbol', 'action', 'reason', 'f_score_delta']])
```

---

## 🔬 Academic Validation

**Piotroski (2000) Findings:**
- High F-Score (8-9) stocks outperform market by ~13% annually
- Low F-Score (0-2) stocks underperform by ~10% annually
- F-Score especially effective for **value stocks** (high B/M)

**Implementation Details:**
- Uses **TTM (trailing 12 months)** for income/cash flow ratios
- Uses **YoY changes** (4 quarters ago) for delta signals
- Conservative: Missing data → signal = 0

---

## 📊 Data Requirements (FMP API)

**Quarterly data needed:**
- Income Statement: `netIncome`, `revenue`, `grossProfit`
- Balance Sheet: `totalAssets`, `totalDebt`, `totalCurrentAssets`, `totalCurrentLiabilities`, shares outstanding
- Cash Flow Statement: `netCashProvidedByOperatingActivities`
- Ratios: (auto-derived if not available)

**FMP Plan:** Starter plan sufficient (20 quarters × 4 statements = 80 API calls per symbol)

---

## 🆚 Piotroski vs VFQ (Legacy)

| Feature | Piotroski F-Score | VFQ (Legacy) |
|---------|-------------------|--------------|
| Academic basis | ✅ Seminal 2000 paper | Custom composite |
| Signals | 9 binary (clear) | Continuous z-scores |
| Interpretability | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| Degradation threshold | 2 points drop | 10% change |
| Industry standard | ✅ Widely used | Custom |
| Calculation speed | Fast (TTM rolling) | Medium |

**Recommendation:** Use Piotroski for **transparency** and **academic validation**. Use VFQ if you need more granular scoring.

---

## 💡 Tips & Best Practices

1. **Combine with Technical Signals:** Piotroski works best when combined with MA200 and momentum (as implemented)

2. **Quarterly Review:** F-Score updates quarterly, so review frequency should be "Q"

3. **Value Stock Focus:** Original paper found strongest results in high B/M stocks (value)

4. **Degradation Threshold:** Default of 2 points is conservative. You can adjust based on portfolio risk tolerance.

5. **Data Quality:** Ensure FMP has complete quarterly data (at least 8 quarters for meaningful signals)

6. **False Positives:** Single-quarter degradation can be noise. Consider trend over 2-3 quarters.

---

## 📖 References

- Piotroski, J. D. (2000). "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers." *Journal of Accounting Research*, 38, 1-41.

- Mohanram, P. S. (2005). "Separating Winners from Losers among Low Book-to-Market Stocks using Financial Statement Analysis." *Review of Accounting Studies*, 10, 133-170. (Extension to growth stocks)

---

## 🚀 Future Enhancements

- [ ] Add Mohanram G-Score (for growth stocks)
- [ ] Multi-period degradation trend (2-3 quarters)
- [ ] Industry-adjusted F-Score
- [ ] Sector-specific signal weights
- [ ] Integration with Altman Z-Score (bankruptcy risk)

---

**For support:** Check `portfolio_e2e_streamlit.py` Tab 4 for UI integration example
