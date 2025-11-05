# Macro Z-Score Calculator - MacroArimax Compatible

## 📊 Overview

Este módulo (`macro_fred_compatible.py`) implementa el cálculo de z-score macro **100% compatible** con la especificación MacroArimax/FRED.

---

## 🎯 Specification Compliance

### **1. Normalización por Familias**

#### **Family $ (Billions)**
- `RRPONTSYD` - ON RRP (Reverse Repo)
- `WTREGEN` - TGA (Treasury General Account)
- `WRESBAL` - Bank Reserves
- `WALCL` - Fed Balance Sheet
- `NL` - Net Liquidity (derived)

#### **Family bp (Basis Points)**
- `TGCRRATE` - Repo GC tri-party
- `SOFR_EFFR` - SOFR - EFFR spread (derived)
- `OBFR_SOFR` - OBFR - SOFR spread (derived)
- `BAMLH0A0HYM2` - High Yield OAS
- `T10Y2Y` - 10y-2y yield curve

**Rationale:** Evita que un indicador domine por unidades ($ billions >> bp).

---

### **2. Signos Económicos**

#### **Drenan Liquidez (Invertir signo: mayor = peor)**
- ✅ `WTREGEN` (TGA): +ΔTGA drena (negativo)
- ✅ `RRPONTSYD` (RRP): +ΔRRP drena (negativo)
- ✅ `SOFR_EFFR`: mayor spread = tensión (negativo)
- ✅ `OBFR_SOFR`: mayor spread = tensión (negativo)
- ✅ `TGCRRATE`: mayor rate = tensión (negativo)
- ✅ `BAMLH0A0HYM2`: mayor OAS = estrés (negativo)

#### **Inyectan Liquidez (Signo positivo: mayor = mejor)**
- ✅ `WRESBAL` (Reserves): +ΔRes inyecta (positivo)
- ✅ `WALCL` (Fed Balance): +Δ inyecta (positivo)
- ✅ `NL` (Net Liquidity): +ΔNL mejor (positivo)
- ✅ `T10Y2Y` (Curve): informativo (no invertir por defecto)

**Implementación:**
```python
ECONOMIC_SIGNS_INVERT = {
    'WTREGEN': True,        # Drena
    'RRPONTSYD': True,      # Drena
    'SOFR_EFFR': True,      # Tensión
    'OBFR_SOFR': True,      # Tensión
    'TGCRRATE': True,       # Tensión
    'BAMLH0A0HYM2': True,   # Estrés

    'WRESBAL': False,       # Inyecta
    'WALCL': False,         # Inyecta
    'NL': False,            # Mejor
    'T10Y2Y': False         # Informativo
}
```

---

### **3. Net Liquidity**

**Formula:**
```
NL_t = WRESBAL_t - WTREGEN_t - RRPONTSYD_t
```

**Interpretation:**
- ↑ NL = más liquidez disponible en sistema (positivo)
- ↓ NL = menos liquidez (negativo)

**Implementation:**
```python
nl = calculate_net_liquidity(
    reserves=df['WRESBAL'],
    tga=df['WTREGEN'],
    rrp=df['RRPONTSYD']
)
# → Deriva NL y lo trata como indicador $ (familia dollars)
```

---

### **4. Z-Score Calculation Steps**

#### **For each indicator:**

1. **Δ Calculation (if flow indicator):**
   ```python
   # Family $ indicators: use Δ (daily or weekly)
   delta = series.diff(periods=1)  # 1 = daily, 7 = weekly

   # Family bp: typically levels (rates use Δ)
   ```

2. **Winsorization (p1-p99):**
   ```python
   low = series.quantile(0.01)
   high = series.quantile(0.99)
   series_winsor = series.clip(lower=low, upper=high)
   ```

3. **Rolling Z-Score:**
   ```python
   window = 252  # annual (or 126 semi-annual)
   roll = series.rolling(window=window, min_periods=60)
   mean = roll.mean()
   std = roll.std()
   zscore = (series - mean) / std
   ```

4. **Apply Economic Sign:**
   ```python
   if ECONOMIC_SIGNS_INVERT[indicator]:
       zscore = -zscore
   ```

5. **Clip:**
   ```python
   zscore = zscore.clip(lower=-3.5, upper=3.5)
   ```

---

### **5. Composite Z-Score**

**Method:**
```python
# Weighted average across families (balanced 50% $ + 50% bp)

composite_z = (
    0.50 * weighted_avg(family_dollars_zscores) +
    0.50 * weighted_avg(family_bp_zscores)
)

# Clip composite también
composite_z = composite_z.clip(lower=-3.5, upper=3.5)
```

**Default Weights (MacroArimax):**
```python
{
    # Family $ (50% total)
    'RRPONTSYD': 0.15,
    'WTREGEN': 0.10,
    'WRESBAL': 0.15,
    'NL': 0.10,

    # Family bp (50% total)
    'SOFR_EFFR': 0.10,
    'OBFR_SOFR': 0.05,
    'TGCRRATE': 0.05,
    'BAMLH0A0HYM2': 0.15,
    'T10Y2Y': 0.15
}
```

---

## 🔧 Usage

### **1. Basic Usage**

```python
from portfolio_manager.macro_fred_compatible import (
    calculate_macro_zscore_from_fred_csv,
    get_macroarimax_default_weights
)

# Calculate z-score from FRED CSV
result_df, macro_z_last = calculate_macro_zscore_from_fred_csv(
    filepath_or_buffer='fred_indicators.csv',
    window=252,  # annual rolling window
    weights=get_macroarimax_default_weights(),
    clip_z=3.5
)

print(f"Composite Z-Score: {macro_z_last:.2f}")

# result_df contiene:
# - RRPONTSYD_z, WTREGEN_z, ... (individual z-scores)
# - NL_z (Net Liquidity z-score)
# - composite_z (weighted composite)
```

### **2. CSV Format Expected**

```csv
Date,RRPONTSYD,WTREGEN,WRESBAL,SOFR,EFFR,OBFR,BAMLH0A0HYM2,T10Y2Y
2023-01-01,2500.5,450.2,3200.1,4.55,4.58,4.60,450,0.50
2023-01-02,2505.0,448.0,3205.5,4.56,4.59,4.61,455,0.48
...
```

**Required columns:**
- `Date` (parsed as datetime)
- Core: `RRPONTSYD`, `WTREGEN`, `WRESBAL` (for Net Liquidity)
- Rates: `SOFR`, `EFFR`, `OBFR` (for spreads)
- Credit: `BAMLH0A0HYM2` (High Yield OAS)
- Curve: `T10Y2Y` (10y-2y)

**Optional columns:**
- `WALCL` (Fed balance)
- `TGCRRATE` (Repo GC)
- `DGS10`, `TB3MS` (for manual spread calculation)

---

## 📈 Integration with Portfolio E2E

### **In `portfolio_e2e_streamlit.py`:**

```python
from portfolio_manager.macro_fred_compatible import (
    calculate_macro_zscore_from_fred_csv,
    get_macroarimax_default_weights
)

uploaded_file = st.file_uploader("Upload FRED CSV")

if uploaded_file:
    result_df, macro_z = calculate_macro_zscore_from_fred_csv(
        uploaded_file,
        window=252,
        weights=get_macroarimax_default_weights()
    )

    # Use macro_z for portfolio optimization
    regime = z_to_regime(macro_z)
    st.metric("Macro Z-Score", f"{macro_z:.2f}")
    st.metric("Regime", regime.label)
```

---

## 🔍 Advanced Features

### **1. Quarter-End Mode**

```python
from portfolio_manager.macro_fred_compatible import (
    is_quarter_end,
    adjust_thresholds_quarter_end
)

# Detect quarter-end
date = pd.Timestamp('2023-03-31')
is_qe = is_quarter_end(date, buffer_days=3)

# Adjust thresholds (spreads/repo más altos son normales en QE)
zscore_adjusted = adjust_thresholds_quarter_end(
    zscore=2.5,
    indicator_name='TGCRRATE',
    is_qe=True,
    qe_multiplier=1.5  # reduce magnitud 1.5×
)
```

### **2. Custom Weights**

```python
custom_weights = {
    'RRPONTSYD': 0.20,  # Más peso a RRP
    'WTREGEN': 0.15,
    'WRESBAL': 0.10,
    'NL': 0.05,
    'SOFR_EFFR': 0.15,
    'BAMLH0A0HYM2': 0.20,  # Más peso a credit stress
    'T10Y2Y': 0.15
}

result_df, macro_z = calculate_macro_zscore_from_fred_csv(
    'fred.csv',
    weights=custom_weights
)
```

### **3. Different Window Sizes**

```python
# Annual (default)
result_annual, z_annual = calculate_macro_zscore_from_fred_csv(
    'fred.csv',
    window=252
)

# Semi-annual (more responsive)
result_semi, z_semi = calculate_macro_zscore_from_fred_csv(
    'fred.csv',
    window=126
)
```

---

## 📊 Output Interpretation

### **Composite Z-Score Ranges:**

| Z-Score | Regime | M_macro | Interpretation |
|---------|--------|---------|----------------|
| ≥ +0.5 | **ON** (Risk-On) | 1.25 | Expansive liquidity |
| [-0.5, +0.5] | **NEUTRAL** | 1.0 | Normal conditions |
| ≤ -0.5 | **OFF** (Risk-Off) | 0.7 | Tight liquidity |

### **Individual Indicators:**

**Positive z-scores (better liquidity):**
- ✅ High WRESBAL_z (reserves increasing)
- ✅ High NL_z (net liquidity improving)
- ✅ Negative WTREGEN_z (TGA draining = inyectando)
- ✅ Negative RRPONTSYD_z (RRP falling = menos drena)

**Negative z-scores (worse liquidity):**
- ❌ High SOFR_EFFR_z (spreads widening)
- ❌ High BAMLH0A0HYM2_z (credit stress)
- ❌ Inverted T10Y2Y (curve stress)

---

## 🧪 Testing

```python
# Test with sample data
import pandas as pd

test_data = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=300),
    'RRPONTSYD': np.random.randn(300) * 50 + 2500,
    'WTREGEN': np.random.randn(300) * 30 + 450,
    'WRESBAL': np.random.randn(300) * 80 + 3200,
    'SOFR': np.random.randn(300) * 0.1 + 4.5,
    'EFFR': np.random.randn(300) * 0.1 + 4.55,
    'OBFR': np.random.randn(300) * 0.1 + 4.6,
    'BAMLH0A0HYM2': np.random.randn(300) * 20 + 450,
    'T10Y2Y': np.random.randn(300) * 0.2 + 0.5
})

test_data.to_csv('test_fred.csv', index=False)

# Calculate
result, z = calculate_macro_zscore_from_fred_csv('test_fred.csv')
print(f"Test Z-Score: {z:.2f}")
```

---

## 📚 References

- **MacroArimax Spec:** User specification document (FRED/NY Fed)
- **FRED Data:** https://fred.stlouisfed.org/
- **Net Liquidity:** Popular metric among macro traders
- **Quarter-End Effects:** Known anomalies in repo markets

---

## 🔄 Differences from `macro_simple.py`

| Feature | macro_simple.py | macro_fred_compatible.py |
|---------|-----------------|--------------------------|
| Normalization | Single pool | **By families ($ vs bp)** |
| Economic signs | Manual preset | **Automatic by indicator** |
| Net Liquidity | No | **Yes (derived)** |
| Spreads | Manual | **Auto-calculated** |
| Winsorization | No | **Yes (p1-p99)** |
| Clip | No | **Yes (|z| ≤ 3.5)** |
| Quarter-end | No | **Yes (optional)** |
| MacroArimax compatible | Partial | **100% ✅** |

---

## 💡 Tips

1. **Always include core indicators** (RRPONTSYD, WTREGEN, WRESBAL) for Net Liquidity
2. **252d window** is recommended for stability (126d for more reactive)
3. **Check z-scores individually** in expander to diagnose regime drivers
4. **Quarter-end mode** can prevent false alarms around Mar 31, Jun 30, Sep 30, Dec 31
5. **Weights are customizable** but defaults are well-balanced (50% $ + 50% bp)

---

**For support:** Check `portfolio_e2e_streamlit.py` for integration example
