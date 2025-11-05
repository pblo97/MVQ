# Troubleshooting Guide - Portfolio E2E

## 🐛 Problem: Streamlit Cloud no carga la app

### Síntomas:
```
🐍 Python dependencies were installed
📦 Processed dependencies!
🔌 Disconnecting...
🖥 Provisioning machine...
⛓ Spinning up manager process...
[App no carga]
```

---

## ✅ SOLUCIÓN: Requirements.txt actualizado (DONE)

**Cambios aplicados en commit `ae5c4cd`:**

1. ✅ **Agregado `fredapi`** (necesario para Macro Monitor)
2. ✅ **Comentado `arch>=6.4`** (requiere compilación, es opcional)
3. ✅ **Agregados upper bounds** (`<2.0`, `<3.0`) para evitar breaking changes
4. ✅ **Reordenadas dependencias** (numpy/pandas primero)
5. ✅ **Versiones más permisivas** (scikit-learn 1.3 en vez de 1.4)

---

## 🧪 Testing Steps

### 1. Test Dependencies (local)
```bash
python test_dependencies.py
```
**Expected output:**
```
✓ streamlit         OK
✓ pandas            OK
✓ numpy             OK
✓ fredapi           OK
...
🎉 All dependencies installed successfully!
```

### 2. Test Minimal Streamlit App
```bash
streamlit run test_streamlit_minimal.py
```
**Should show:**
- Basic components (metrics, dataframe, charts)
- All imports working (Plotly, Statsmodels, Scikit-learn, Fredapi)
- "🎉 All tests passed!"

### 3. Test Portfolio E2E App
```bash
streamlit run portfolio_e2e_streamlit.py
```
**Should load 5 tabs:**
- Tab 1: Portfolio Overview
- Tab 2: Macro Monitor
- Tab 3: Risk Analytics
- Tab 4: Asset Quality & Exits
- Tab 5: Backtest & Reports

---

## 🔍 Debugging Checklist

### Si aún falla en Streamlit Cloud:

#### ✅ Check 1: Verify requirements.txt is updated
```bash
cat requirements.txt | head -10
```
**Should show:**
```
# Core (install first - order matters)
numpy>=1.24,<2.0
pandas>=2.0,<3.0
streamlit>=1.37,<2.0
...
```

#### ✅ Check 2: Verify arch is commented
```bash
grep "arch" requirements.txt
```
**Should show:**
```
# arch>=6.4
```
(commented out)

#### ✅ Check 3: Check Streamlit Cloud logs
En Streamlit Cloud → Manage app → Logs → busca:
```
ERROR: Could not install packages due to...
ModuleNotFoundError: No module named...
```

#### ✅ Check 4: Python version
En Streamlit Cloud settings:
- **Python version:** 3.11 (recomendado) o 3.10

#### ✅ Check 5: Memory issues
Si logs muestran "Killed" o "Out of memory":
- Reduce universo de símbolos (usa 5-6 en vez de 10)
- Comenta imports pesados temporalmente

---

## 🚨 Common Errors & Solutions

### Error 1: `ModuleNotFoundError: No module named 'fredapi'`
**Solution:** Ya fixed en requirements.txt (commit ae5c4cd)
```bash
git pull  # Asegúrate de tener la versión actualizada
```

### Error 2: `arch` compilation fails
**Solution:** Ya comentado en requirements.txt
- GARCH es opcional
- `ma_streamlit.py` maneja ausencia gracefully (línea 16-20)

### Error 3: `ImportError: sklearn`
**Solution:** Ya fixed (scikit-learn>=1.3 en vez de 1.4)

### Error 4: App carga pero falla al click "Run Portfolio Optimization"
**Possible causes:**
1. **FMP_API_KEY missing:** Add to Streamlit secrets
   ```toml
   # .streamlit/secrets.toml
   FMP_API_KEY = "your_key"
   ```

2. **Símbolos inválidos:** Verifica que existen en yfinance
   ```python
   # Test símbolos válidos:
   AAPL,GOOGL,MSFT,NVDA,JPM,BAC
   ```

3. **Fechas inválidas:** Start date debe ser >= 3 años antes de end date
   ```python
   # Ejemplo válido:
   start_date = 2023-01-01
   end_date = 2025-11-05
   ```

### Error 5: "No module named 'portfolio_manager'"
**Solution:** Verifica estructura de archivos
```bash
ls -la portfolio_manager/
# Debe mostrar:
# quality/
# execution/
# monitor/
# data/
```

Si falta, verifica que commit f2d36f0 se aplicó:
```bash
git log --oneline -5
# Debe mostrar:
# ae5c4cd fix: Update requirements.txt
# f2d36f0 feat: Portfolio Optimization E2E
```

---

## 📝 Streamlit Cloud Deployment

### Settings recomendados:

1. **Python version:** 3.11 o 3.10
2. **Main file path:** `portfolio_e2e_streamlit.py` (o `test_streamlit_minimal.py` para test)
3. **Secrets required:**
   ```toml
   FRED_API_KEY = "your_fred_api_key"
   FMP_API_KEY = "your_fmp_api_key"
   ```

4. **Advanced settings:**
   - ✅ Usar latest Streamlit version
   - ✅ Enable auto-updates (off para producción)

---

## 🆘 Si nada funciona

### Plan B: Run locally

```bash
# 1. Clone repo
git clone https://github.com/pblo97/MVQ.git
cd MVQ

# 2. Create virtual env
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add secrets
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
FRED_API_KEY = "your_key"
FMP_API_KEY = "your_key"
EOF

# 5. Run test app
streamlit run test_streamlit_minimal.py

# 6. Run full app
streamlit run portfolio_e2e_streamlit.py
```

---

## 📞 Support

### Verificar versiones instaladas:
```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly

print(f"Streamlit: {st.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Plotly: {plotly.__version__}")
```

### Expected versions:
```
Streamlit: 1.37.x - 1.40.x
Pandas: 2.0.x - 2.2.x
NumPy: 1.24.x - 1.26.x
Plotly: 5.18.x - 5.24.x
```

---

## ✅ Final Checklist

Antes de deployar en Streamlit Cloud:

- [ ] `requirements.txt` updated (commit ae5c4cd)
- [ ] `arch` commented out
- [ ] `fredapi` included
- [ ] Python 3.10 or 3.11 selected
- [ ] Secrets added (FRED_API_KEY, FMP_API_KEY)
- [ ] Test with `test_streamlit_minimal.py` first
- [ ] Verificar logs de Streamlit Cloud para errores específicos

---

**Si el problema persiste, comparte los logs de Streamlit Cloud para diagnóstico detallado.** 🔍
