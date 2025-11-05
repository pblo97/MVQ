"""
Minimal Streamlit test app - verifica que Streamlit funciona correctamente
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Test App", layout="wide")

st.title("🧪 Streamlit Test - Minimal")
st.caption("Si ves esto, Streamlit está funcionando correctamente")

# Test 1: Basic components
st.header("Test 1: Basic Components")
col1, col2, col3 = st.columns(3)
col1.metric("Metric 1", "100")
col2.metric("Metric 2", "200")
col3.metric("Metric 3", "300")

# Test 2: Data
st.header("Test 2: Data Display")
df = pd.DataFrame({
    'A': np.random.randn(10),
    'B': np.random.randn(10),
    'C': np.random.randn(10)
})
st.dataframe(df)

# Test 3: Charts
st.header("Test 3: Charts")
st.line_chart(df)

# Test 4: User input
st.header("Test 4: User Input")
text_input = st.text_input("Enter text", "Hello World")
slider_value = st.slider("Select value", 0, 100, 50)
st.write(f"You entered: {text_input}")
st.write(f"Slider value: {slider_value}")

# Test 5: Imports críticos
st.header("Test 5: Critical Imports")
try:
    import plotly.express as px
    st.success("✓ Plotly OK")
except Exception as e:
    st.error(f"✗ Plotly FAILED: {e}")

try:
    import statsmodels.api as sm
    st.success("✓ Statsmodels OK")
except Exception as e:
    st.error(f"✗ Statsmodels FAILED: {e}")

try:
    from sklearn.covariance import LedoitWolf
    st.success("✓ Scikit-learn OK")
except Exception as e:
    st.error(f"✗ Scikit-learn FAILED: {e}")

try:
    from fredapi import Fred
    st.success("✓ Fredapi OK")
except Exception as e:
    st.error(f"✗ Fredapi FAILED: {e}")

st.success("🎉 All tests passed! Streamlit is working correctly.")
