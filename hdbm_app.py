import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ==========================================
# 1. HDBM Model Definition
# ==========================================
class HDBM:
    def __init__(self, alpha_go=0.85, alpha_stop=0.85, eta=5.0, rho=0.6, k=2.5, gamma=0.8, 
                 fusion_type='additive_2', a0=5.0, b0=1.0, scale_lambda=4.5):
        self.alpha_go = alpha_go
        self.alpha_stop = alpha_stop
        self.eta = eta
        self.rho = rho
        self.k = k            
        self.gamma = gamma
        self.fusion_type = fusion_type
        self.a0 = a0
        self.b0 = b0
        self.scale_lambda = scale_lambda

        max_possible_run = 50
        self.hazard_values = np.zeros(max_possible_run + 1)
        for i in range(1, max_possible_run + 1):
            # Strict Weibull hazard formula
            val = (self.k / self.scale_lambda) * ((i / self.scale_lambda) ** (self.k - 1))
            self.hazard_values[i] = val

    def _get_hazard(self, run_length):
        idx = min(run_length, len(self.hazard_values) - 1)
        return self.hazard_values[idx]

    def simu_task(self, sequence, return_details=False):
        a, b = self.a0, self.b0
        run_length = 0
        
        r_traj, Er_traj, h_traj, r_raw_traj = [], [], [], []

        for n, trial in enumerate(sequence):
            # 1. Calculate Bayesian expectation (GLOBAL: a and b carry over)
            Er = b / (a + b)
            
            # 2. Calculate Hazard (LOCAL: based on current run_length)
            h = self._get_hazard(run_length)

            # 3. Fusion Logic
            if self.fusion_type == 'additive_1':
                r_raw = (1 - self.rho) * Er + self.rho * h
            elif self.fusion_type == 'additive_2':
                r_raw = Er + self.rho * h
            elif self.fusion_type == 'multiplicative':
                r_raw = Er * (1 + self.gamma * h)
            else:
                raise ValueError("Unknown fusion_type")

            # Clip for the final decision probability
            r_final = np.clip(r_raw, 0, 1-1e-4)

            r_raw_traj.append(r_raw)
            r_traj.append(r_final)
            Er_traj.append(Er)
            h_traj.append(h)

            # 4. State Updates
            if trial == 0:  # Go trial
                # Global update for Bayesian
                a = (1 - self.alpha_go) * self.a0 + self.alpha_go * (a + self.eta)
                b = (1 - self.alpha_go) * self.b0 + self.alpha_go * (b + 0)
                # Local accumulation for Hazard
                run_length += 1
            else:           # Stop trial
                # Global update for Bayesian
                a = (1 - self.alpha_stop) * self.a0 + self.alpha_stop * (a + 0)
                b = (1 - self.alpha_stop) * self.b0 + self.alpha_stop * (b + 1)
                # LOCAL RESET for Hazard
                run_length = 0  

        if return_details:
            return np.array(r_traj), np.array(Er_traj), np.array(h_traj), np.array(r_raw_traj)
        return np.array(r_traj)

# ==========================================
# 2. Streamlit UI
# ==========================================
st.set_page_config(layout="wide", page_title="HDBM State Viewer")

st.title("HDBM Internal State Visualization")
# Updated subheader
st.subheader("Sequence: 20 Go -> 1 Stop -> 1 Go")

st.markdown("""
**Notice the difference in dynamics:**
* The **Blue Line (Bayesian E[r])** is *Global*. It updates continuously and carries its memory across the whole session.
* The **Red Line (Hazard h)** is *Local*. It perfectly resets to 0 immediately after the Stop signal.
""")

# Sidebar Parameters
st.sidebar.header("Model Parameters")
fusion_type = st.sidebar.selectbox("Fusion Mode", ['additive_2', 'additive_1', 'multiplicative'])

st.sidebar.subheader("Bayesian Learning (Global)")
alpha_go = st.sidebar.slider("alpha_go", 0.0, 1.0, 0.85, 0.05)
alpha_stop = st.sidebar.slider("alpha_stop", 0.0, 1.0, 0.85, 0.05)
eta = st.sidebar.slider("eta (evidence for Go)", 0.0, 10.0, 5.0, 0.5)

st.sidebar.subheader("Weibull Hazard (Local)")
k = st.sidebar.slider("k (shape)", 0.1, 5.0, 2.5, 0.1)
scale_lambda = st.sidebar.slider("lambda (scale)", 1.0, 15.0, 4.5, 0.5)

st.sidebar.subheader("Weights")
rho = st.sidebar.slider("rho (hazard weight)", 0.0, 2.0, 0.6, 0.1)
gamma = st.sidebar.slider("gamma (multiplicative weight)", 0.0, 2.0, 0.8, 0.1)

# Display Mathematical Formula
st.write("---")
col1, col2 = st.columns(2)
with col1:
    st.write("**Current Fusion Formula:**")
    if fusion_type == 'additive_1':
        st.latex(r"r_{raw} = (1 - \rho) \cdot E[r] + \rho \cdot h(t)")
    elif fusion_type == 'additive_2':
        st.latex(r"r_{raw} = E[r] + \rho \cdot h(t)")
    elif fusion_type == 'multiplicative':
        st.latex(r"r_{raw} = E[r] \cdot (1 + \gamma \cdot h(t))")
with col2:
    st.write("**Strict Weibull Hazard formula:**")
    st.latex(r"h(t) = \frac{k}{\lambda} \left( \frac{t}{\lambda} \right)^{k-1}")
st.write("---")

# Run Simulation
# Simple Sequence: 20 Go (0), 1 Stop (1), 1 Go (0)
sequence = [0] * 20 + [1] + [0] * 1
trials_x = np.arange(1, len(sequence) + 1)

model = HDBM(alpha_go=alpha_go, alpha_stop=alpha_stop, eta=eta, rho=rho, 
             k=k, gamma=gamma, fusion_type=fusion_type, scale_lambda=scale_lambda)

r_final, Er, h, r_raw = model.simu_task(sequence, return_details=True)

# Plotting
fig = go.Figure()

# E[r]
fig.add_trace(go.Scatter(x=trials_x, y=Er, mode='lines+markers', 
                         name='E[r] (Bayesian - Global)', line=dict(color='blue', width=2)))

# Hazard
fig.add_trace(go.Scatter(x=trials_x, y=h, mode='lines+markers', 
                         name='Hazard (Local Reset)', line=dict(color='red', width=2)))

# Combined r (Unclipped)
fig.add_trace(go.Scatter(x=trials_x, y=r_raw, mode='lines+markers', 
                         name='Combined r (Unclipped)', line=dict(color='purple', width=3)))

# Final r (Clipped)
fig.add_trace(go.Scatter(x=trials_x, y=r_final, mode='lines', 
                         name='Final r (Clipped at 1.0)', line=dict(color='purple', width=1, dash='dash')))

# Markers
fig.add_hline(y=1.0, line_dash="dot", line_color="black")

# Add Vertical line for the Single Stop trial (Trial 21)
fig.add_vline(x=21, line_dash="dash", line_color="orange", annotation_text="Stop Signal", annotation_position="top right")

fig.update_layout(
    xaxis_title="Trial Number",
    yaxis_title="Value",
    height=600,
    hovermode="x unified",
    title="HDBM Component Trajectories: 20 Go -> 1 Stop -> 1 Go"
)
fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)

st.plotly_chart(fig, use_container_width=True)

# Summary Table
st.write("**Summary Data at Key Trials:**")
# Indices for: Trial 1, Trial 20 (Last Go), Trial 21 (Stop), Trial 22 (Final Go)
summary_indices = [0, 19, 20, 21]
summary_data = {
    "Trial": [i + 1 for i in summary_indices],
    "Type": ["First Go", "Last Go before Stop", "Stop Trial", "First Go after Stop"],
    "E[r] (Global)": [Er[i] for i in summary_indices],
    "Hazard (Local)": [h[i] for i in summary_indices],
    "Combined r": [r_raw[i] for i in summary_indices]
}
st.table(pd.DataFrame(summary_data))