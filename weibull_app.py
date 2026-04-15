import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(layout="wide", page_title="Weibull Distribution Explorer")

st.title("Continuous Weibull Distribution Explorer")

st.markdown("""
Adjust the **Shape (k)** and **Scale (λ)** parameters in the sidebar to see how they 
transform the PDF, Survival, and Hazard functions.
""")

# Sidebar Sliders
st.sidebar.header("Distribution Parameters")
k = st.sidebar.slider("Shape Parameter (k)", min_value=0.1, max_value=5.0, value=1.5, step=0.1, 
                      help="k < 1: Decreasing hazard; k = 1: Constant hazard (Exponential); k > 1: Increasing hazard")
lam = st.sidebar.slider("Scale Parameter (λ)", min_value=0.5, max_value=10.0, value=5.0, step=0.1,
                        help="Stretches or compresses the distribution along the time axis")

# Generate time range t
t = np.linspace(0.01, 15, 500) 

# --- Weibull Calculations ---
# PDF: f(t) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
pdf = (k / lam) * (t / lam)**(k - 1) * np.exp(-(t / lam)**k)

# Survival: S(t) = exp(-(t/λ)^k)
survival = np.exp(-(t / lam)**k)

# Hazard: h(t) = (k/λ) * (t/λ)^(k-1)
hazard = (k / lam) * (t / lam)**(k - 1)

# --- Plotting ---
fig = make_subplots(rows=1, cols=3, 
                    subplot_titles=("Probability Density (PDF)", 
                                    "Survival Function", 
                                    "Hazard Function"))

# 1. PDF
fig.add_trace(go.Scatter(x=t, y=pdf, mode='lines', name='PDF', line=dict(color='blue', width=3)), row=1, col=1)

# 2. Survival
fig.add_trace(go.Scatter(x=t, y=survival, mode='lines', name='Survival', line=dict(color='green', width=3)), row=1, col=2)

# 3. Hazard
fig.add_trace(go.Scatter(x=t, y=hazard, mode='lines', name='Hazard', line=dict(color='red', width=3)), row=1, col=3)

# Layout adjustments
fig.update_layout(height=500, showlegend=False, hovermode="x unified")

# Formatting axes
fig.update_xaxes(title_text="Time (t)")
fig.update_yaxes(range=[0, 1.1], row=1, col=2) # Survival always between 0 and 1

# Limit Y-axis for Hazard to prevent "explosion" when k < 1 and t is near 0
max_haz_display = max(2.0, np.percentile(hazard, 95))
fig.update_yaxes(range=[0, max_haz_display], row=1, col=3)

st.plotly_chart(fig, use_container_width=True)

# Documentation and Insights
st.info(f"""
### Quick Insights:
* **When k = 1:** The distribution is **Exponential**. The Hazard function is a flat line (constant risk).
* **When k < 1:** The Hazard is decreasing (Infant Mortality period).
* **When k > 1:** The Hazard is increasing (Wear-out/Aging period).
* **Scale (λ)**: Changing this value scales the time axis. The 63.2 percentile of the distribution is always equal to λ, regardless of k.
""")

# streamlit run weibull_app.py

