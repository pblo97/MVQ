# CVaR & VaR Risk Analytics

## 📊 Overview

Comprehensive risk analytics module implementing CVaR (Conditional Value at Risk), VaR, marginal risk contributions, and stress testing based on academic best practices.

---

## 📚 Academic Foundation

### **Key Papers:**

1. **Rockafellar & Uryasev (2000, 2002)**
   - "Optimization of conditional value-at-risk"
   - CVaR as coherent risk measure
   - Euler decomposition for risk attribution

2. **Acerbi & Tasche (2002)**
   - "Expected Shortfall: A natural coherent alternative to Value at Risk"
   - Properties and mathematical foundations

3. **Jorion (2007)**
   - "Value at Risk: The New Benchmark for Managing Financial Risk"
   - Practical implementation and backtesting

4. **Kupiec (1995) & Christoffersen (1998)**
   - VaR backtesting methodologies
   - POF (Proportion of Failures) and independence tests

---

## 🎯 Implemented Features

### **1. VaR (Value at Risk)**

**Definition:** Maximum expected loss at a given confidence level over a specific time horizon.

**Methods:**
- **Historical:** Empirical quantile from historical returns
- **Parametric:** Assumes normal distribution (μ ± z×σ)
- **Cornish-Fisher:** Adjusts for skew and kurtosis (fat tails)

**Interpretation:**
- VaR(95%) = 5%: In 95% of cases, loss won't exceed VaR
- Example: VaR = 2.5% → expect to lose max 2.5% in 95% of days

### **2. CVaR (Conditional Value at Risk)**

**Definition:** Expected loss given that loss exceeds VaR (tail risk measure).

**Also known as:** Expected Shortfall (ES), Average VaR, Tail VaR

**Properties:**
- ✅ **Coherent risk measure** (satisfies subadditivity, monotonicity, translation invariance, homogeneity)
- ✅ **Convex** (optimizable)
- ✅ **Accounts for tail shape** (not just quantile)

**Interpretation:**
- CVaR(95%) = 4%: When losses exceed VaR, average loss is 4%
- Always CVaR ≥ VaR (worse than worst 5%)

### **3. Marginal CVaR Contributions**

**Definition:** ∂CVaR/∂w_i = sensitivity of portfolio CVaR to small change in asset i weight.

**Formula (Tasche 2002):**
```
Marginal CVaR_i = E[R_i | R_portfolio < -VaR]
```

**Interpretation:**
- High marginal CVaR = asset drives tail losses
- Can be positive or negative (hedges have negative)

### **4. Component CVaR**

**Definition:** Additive risk decomposition (Euler allocation).

**Formula:**
```
Component CVaR_i = w_i × Marginal CVaR_i

Property: Σ Component CVaR_i = Portfolio CVaR
```

**Interpretation:**
- Shows how much of total CVaR comes from each asset
- Percentage contribution = Component CVaR_i / Portfolio CVaR

### **5. VaR Backtesting**

**Tests implemented:**

#### **Kupiec POF Test (Proportion of Failures)**
- **H0:** Exception rate = expected rate
- **Statistic:** LR (Likelihood Ratio) ~ χ²(1)
- **Reject H0 if:** p-value < 0.05 (model underestimates risk)

#### **Christoffersen Independence Test**
- **H0:** Exceptions are independent (not clustered)
- **Statistic:** LR ~ χ²(1)
- **Reject H0 if:** p-value < 0.05 (exceptions cluster → model misspecified)

### **6. Stress Testing**

**Default scenarios (based on historical events):**

| Scenario | Event | Typical Loss |
|----------|-------|--------------|
| **2008 Financial Crisis** | Oct 2008 crash | -30% uniform |
| **2020 COVID Crash** | Mar 2020 pandemic | -35% uniform |
| **2022 Rate Hike** | Fed tightening | -25% average |
| **3-Sigma Down** | Statistical (μ - 3σ) | Varies by asset |
| **5-Sigma Down** | Tail event | Severe losses |
| **Correlation One** | All assets down | -20% uniform |

---

## 🔧 Usage

### **1. Calculate VaR & CVaR**

```python
from portfolio_manager.risk.cvar_analysis import calculate_var, calculate_cvar

# Portfolio returns
portfolio_returns = (returns_df * weights).sum(axis=1)

# VaR at 95% confidence
var_95 = calculate_var(
    returns=portfolio_returns,
    confidence_level=0.95,
    method='historical'  # or 'parametric', 'cornish_fisher'
)

# CVaR at 95% confidence
cvar_95 = calculate_cvar(
    returns=portfolio_returns,
    confidence_level=0.95,
    method='historical'
)

print(f"VaR(95%): {var_95*100:.2f}%")
print(f"CVaR(95%): {cvar_95*100:.2f}%")
```

### **2. Marginal CVaR Contributions**

```python
from portfolio_manager.risk.cvar_analysis import calculate_percentage_cvar_contribution

# Calculate % contribution by asset
pct_contrib = calculate_percentage_cvar_contribution(
    returns_df=returns_df,  # DataFrame of asset returns
    weights=weights,         # Portfolio weights
    confidence_level=0.95,
    method='historical'
)

# Results
print(pct_contrib.sort_values(ascending=False))

# Example output:
# NVDA    35.2%  <- Drives 35% of tail risk
# GOOGL   28.1%
# AAPL    22.5%
# JPM     14.2%
```

### **3. Stress Testing**

```python
from portfolio_manager.risk.cvar_analysis import stress_test_scenarios

# Run stress tests
stress_results = stress_test_scenarios(
    returns_df=returns_df,
    weights=weights,
    scenarios=None  # Use default scenarios
)

print(stress_results[['scenario', 'portfolio_loss_pct']])

# Example output:
# scenario                portfolio_loss_pct
# 5_sigma_down            -45.2%
# 2008_financial_crisis   -30.0%
# 2020_covid_crash        -35.0%
```

### **4. Risk Metrics Summary**

```python
from portfolio_manager.risk.cvar_analysis import calculate_risk_metrics_summary

# Calculate comprehensive risk metrics
risk_summary = calculate_risk_metrics_summary(
    returns_df=returns_df,
    weights=weights,
    confidence_levels=[0.95, 0.99]
)

print(risk_summary)

# Example output:
# confidence_level  VaR_historical  CVaR_historical  VaR_parametric  CVaR_parametric
# 95%               2.34%           3.89%            2.12%           2.98%
# 99%               4.56%           6.21%            3.89%           4.52%
```

---

## 📈 Integration with Portfolio E2E

In **Tab 3 (Risk Analytics)** of `portfolio_e2e_streamlit.py`:

### **Features:**

1. **VaR & CVaR Summary Table**
   - Historical vs Parametric methods
   - 95% and 99% confidence levels
   - Side-by-side comparison

2. **Marginal CVaR Contributions**
   - Table showing % contribution by asset
   - Bar chart visualization
   - Identifies which assets drive tail risk

3. **Stress Testing**
   - 6 historical/statistical scenarios
   - Portfolio loss projections
   - Worst-case scenario highlighted

4. **Correlation Heatmap**
   - 60-day rolling correlations
   - Identifies concentration risk

---

## 🎓 When to Use Each Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Historical VaR/CVaR** | Recent market conditions | No distribution assumptions | Past ≠ future |
| **Parametric VaR/CVaR** | Normal markets | Fast, simple | Underestimates tail risk |
| **Cornish-Fisher VaR** | Fat-tailed returns | Accounts for skew/kurtosis | Still assumes distribution |

**Recommendation:** Use **historical CVaR** for conservative risk management (captures actual tail behavior).

---

## 📊 Interpretation Guidelines

### **VaR vs CVaR Comparison:**

```
Portfolio A:
VaR(95%) = 2.0%
CVaR(95%) = 2.5%
→ Tail risk is mild (CVaR only 25% worse than VaR)

Portfolio B:
VaR(95%) = 2.0%
CVaR(95%) = 5.0%
→ Tail risk is SEVERE (CVaR 150% worse than VaR)
→ Fat-tailed distribution (extreme events likely)
```

### **Marginal CVaR Interpretation:**

```
Asset      Weight  Marginal CVaR  Component CVaR  % Contribution
NVDA       15%     -8.2%          -1.23%          40%
AAPL       25%     -3.1%          -0.78%          25%
JPM        20%     -2.8%          -0.56%          18%
TLT        40%     +1.1%          +0.44%          -14%  <- Hedge!

Insights:
- NVDA drives 40% of tail risk despite only 15% weight
- TLT provides tail hedge (negative contribution)
- Consider reducing NVDA or increasing TLT
```

### **Stress Test Interpretation:**

```
Scenario              Portfolio Loss
3_sigma_down          -12.5%  ← Normal stress
2020_covid_crash      -28.0%  ← Severe but historical
5_sigma_down          -45.0%  ← Tail event (0.3% probability)

Actions:
- If cannot tolerate -45%, reduce leverage/beta
- Consider tail hedge (put options, TLT, gold)
- Review concentration in risky assets
```

---

## 🔬 Mathematical Details

### **CVaR Formula (Continuous)**

```
CVaR_α(X) = E[X | X ≤ VaR_α(X)]

Where:
- X = portfolio return (negative = loss)
- α = confidence level (e.g., 0.95)
- VaR_α = α-quantile of X
```

### **Marginal CVaR (Tasche 2002)**

```
∂CVaR/∂w_i = E[R_i | R_p ≤ -VaR]

Where:
- R_i = asset i return
- R_p = portfolio return
- Computed as conditional expectation in tail
```

### **Euler Decomposition**

```
CVaR = Σ w_i × (∂CVaR/∂w_i)

Property: Sum of component CVaR equals total CVaR
```

---

## 💡 Best Practices

1. **Use Multiple Methods:** Compare historical, parametric, and Cornish-Fisher
2. **Monitor Tail Ratio:** CVaR/VaR > 1.5 indicates fat tails
3. **Backtest Regularly:** Run Kupiec/Christoffersen tests quarterly
4. **Stress Test:** Run scenarios monthly, update for current events
5. **Marginal CVaR:** Rebalance if any asset > 30% tail risk contribution
6. **Confidence Levels:** Use 95% for daily, 99% for regulatory capital

---

## 🆚 CVaR vs Alternatives

| Risk Measure | Coherent? | Subadditive? | Tail-aware? | Optimizable? |
|--------------|-----------|--------------|-------------|--------------|
| **CVaR** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **VaR** | ❌ No | ❌ No | ⚠️ Quantile only | ❌ No |
| **Volatility (σ)** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **Semi-Variance** | ✅ Yes | ✅ Yes | ⚠️ Downside only | ✅ Yes |
| **Max Drawdown** | ❌ No | ❌ No | ⚠️ Historical | ❌ No |

**Winner:** CVaR for tail risk management (coherent + tail-aware + optimizable)

---

## 📖 References

- Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk." *Journal of risk*, 2, 21-42.

- Acerbi, C., & Tasche, D. (2002). "On the coherence of expected shortfall." *Journal of Banking & Finance*, 26(7), 1487-1503.

- Tasche, D. (2002). "Expected shortfall and beyond." *Journal of Banking & Finance*, 26(7), 1519-1533.

- Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd ed.). McGraw-Hill.

- Kupiec, P. H. (1995). "Techniques for verifying the accuracy of risk measurement models." *Journal of Derivatives*, 3(2), 73-84.

- Christoffersen, P. F. (1998). "Evaluating interval forecasts." *International Economic Review*, 841-862.

---

**For support:** Check `portfolio_e2e_streamlit.py` Tab 3 for UI integration example
