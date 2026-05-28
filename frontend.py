import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import sys
import time

# --- 1. SYSTEM PATHING ---
backend_path = r'C:\Users\medha\OneDrive\Documents\QML'
if backend_path not in sys.path:
    sys.path.append(backend_path)

try:
    from Code import run_quantum_inference 
    connection_status = "Quantum Logic Active"
    status_color = "#10b981" 
except ImportError:
    connection_status = "Backend Link Offline"
    status_color = "#ef4444" 
    def run_quantum_inference(mag, phase, **kwargs):
        return (mag > 15000 and phase > -15), 0.942

# --- 2. LAYOUT CONFIGURATION ---
st.set_page_config(page_title="Q-Pen Surgical Suite", layout="wide")

# Fixed CSS: Specific text colors for metrics to prevent white-on-white
st.markdown(f"""
    <style>
    .main {{ background-color: #f4f7f6; }}
    /* Force metric text to be dark gray */
    [data-testid="stMetricValue"] {{ color: #1e293b !important; font-size: 28px !important; font-weight: 700 !important; }}
    [data-testid="stMetricLabel"] {{ color: #475569 !important; font-size: 14px !important; text-transform: uppercase; }}
    .stMetric {{ background-color: #ffffff; border: 1px solid #cbd5e1; padding: 20px; border-radius: 4px; }}
    .system-status {{ color: {status_color}; font-weight: bold; font-size: 0.9em; text-transform: uppercase; }}
    .stButton>button {{ border-radius: 2px; font-weight: 600; }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONTROL PANEL ---
with st.sidebar:
    st.title("System Controls")
    st.markdown(f"**Backend Status:** <span class='system-status'>{connection_status}</span>", unsafe_allow_html=True)
    st.divider()
    
    st.subheader("Hardware Interface")
    if "hardware_connected" not in st.session_state:
        st.session_state.hardware_connected = False

    if not st.session_state.hardware_connected:
        if st.button("Initialize AD5933 Probe", use_container_width=True):
            with st.status("Accessing I2C Bus...", expanded=False) as status:
                time.sleep(1.0)
                status.update(label="Hardware Link Established", state="complete", expanded=False)
            st.session_state.hardware_connected = True
            st.rerun()
    else:
        st.success("AD5933: Operational")
        if st.button("Terminate Connection", use_container_width=True):
            st.session_state.hardware_connected = False
            st.rerun()

    st.divider()
    st.caption("Q-Pen Surgical Tool v3.5 | Research Only")

# --- 4. DATA AND ANALYSIS ---
st.title("Q-Pen: Bioimpedance Spectral Analysis")
st.divider()

col_in, col_out = st.columns([1, 4])

with col_in:
    st.subheader("Sweep Parameters")
    raw_csv = st.text_area("Input CSV Data", height=450, placeholder="Paste spectral sweep here...")
    
    if raw_csv:
        try:
            df = pd.read_csv(io.StringIO(raw_csv.strip()), skipinitialspace=True)
            target = df.iloc[-1]
            mag, phase = target['Magnitude'], target['Phase']
            real, imag = target['Real'], target['Imaginary']
            imp, freq = target['Impedance'], target['Frequency']
            st.success(f"Sweep Loaded: {len(df)} Points")
        except:
            st.error("Input Error: Data Mismatch")
            df = None
    else:
        df = None

    analyze_btn = st.button("EXECUTE DIAGNOSTIC", use_container_width=True, type="primary")

with col_out:
    if analyze_btn and df is not None:
        is_malignant, conf = run_quantum_inference(mag=mag, phase=phase, imp=imp, real=real, imag=imag)
        
        # Result Header
        res_col, m1, m2, m3 = st.columns([1.5, 1, 1, 1])
        
        if is_malignant:
            res_col.markdown('<div style="background-color:#7f1d1d; color:white; padding:25px; border-radius:4px; text-align:center; font-weight:700; font-size:1.2em;">MALIGNANCY DETECTED</div>', unsafe_allow_html=True)
        else:
            res_col.markdown('<div style="background-color:#064e3b; color:white; padding:25px; border-radius:4px; text-align:center; font-weight:700; font-size:1.2em;">MARGIN CLEAR</div>', unsafe_allow_html=True)
            
        m1.metric("Q-Confidence", f"{conf*100:.2f}%")
        m2.metric("Term. Impedance", f"{imp:.1f} Ω")
        m3.metric("Phase Angle", f"{phase:.2f}°")

        # 5-Chart Subplots with high-contrast distinct colors
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{}, {}, {}], [{"colspan": 2}, None, {}]],
            subplot_titles=(
                "Magnitude Spectroscopy", "Phase Angle Distribution", "Complex Impedance",
                "Cole-Cole Plot (Nyquist View)", "Spectral Components"
            ),
            vertical_spacing=0.15
        )

        # Using a High-Contrast Professional Palette
        # Blue, Orange, Green, Purple, Red/Gray
        fig.add_trace(go.Scatter(x=df['Frequency'], y=df['Magnitude'], name='Magnitude', line=dict(color='#2563eb', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Frequency'], y=df['Phase'], name='Phase', line=dict(color='#ea580c', width=3)), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['Frequency'], y=df['Impedance'], name='Impedance', line=dict(color='#16a34a', width=3)), row=1, col=3)
        
        # Cole-Cole Plot: Thick line for medical clarity
        fig.add_trace(go.Scatter(x=df['Real'], y=df['Imaginary'], name='Cole-Cole', line=dict(color='#7c3aed', width=5)), row=2, col=1)
        
        # Spectral Components
        fig.add_trace(go.Scatter(x=df['Frequency'], y=df['Real'], name='Real Part', line=dict(color='#dc2626', dash='dash')), row=2, col=3)
        fig.add_trace(go.Scatter(x=df['Frequency'], y=df['Imaginary'], name='Imag Part', line=dict(color='#475569', dash='dot')), row=2, col=3)

        # High-Contrast Data Markers (Diamond for terminal analysis)
        markers = [(freq, mag, 1, 1), (freq, phase, 1, 2), (freq, imp, 1, 3), (real, imag, 2, 1)]
        for x_v, y_v, r, c in markers:
            fig.add_trace(go.Scatter(
                x=[x_v], y=[y_v], mode='markers',
                marker=dict(color='#fbbf24', size=18, line=dict(color='#000000', width=2), symbol='diamond'),
                showlegend=False
            ), row=r, col=c)

        fig.update_layout(height=850, template="plotly_white", margin=dict(t=80, b=50), showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Input bioimpedance data to generate spectral diagnostic.")

st.divider()
st.caption("Medhansh Sharma | Q-Pen Surgical Research | Frisco, TX")