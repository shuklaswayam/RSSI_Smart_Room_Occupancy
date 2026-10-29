import math
from pathlib import Path
import random
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
import streamlit as st
import streamlit.components.v1 as components

from occ.config import get_root_dir, load_config
from occ.db import fetch_predictions, init_db
from occ.inference.engine import RealtimeInferenceEngine
from occ.models.classifier import load_model

st.set_page_config(
    page_title="AURA-RF • Liquid Glass Telemetry",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

cfg = load_config()
root_dir = get_root_dir()
init_db(cfg)

# =============================================================================
# iOS 26 LIQUID GLASS THEME INJECTION (High-Contrast, WCAG AAA)
# =============================================================================
st.markdown("""
<style>
    /* Global Deep Midnight Space Canvas */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(2, 132, 199, 0.18), transparent 45%),
                    radial-gradient(circle at 85% 75%, rgba(67, 56, 202, 0.18), transparent 50%),
                    radial-gradient(circle at 50% 50%, rgba(13, 148, 136, 0.10), transparent 55%),
                    #070b18 !important;
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", sans-serif;
    }

    /* Universal Text Readability Enforcers - Fixes black-on-black / white-on-white */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, span, label, div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    div[data-testid="stCaptionContainer"] p, .stCaption {
        color: #cbd5e1 !important;
    }

    /* Liquid Glass Top Header */
    .header-bar {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(28px) saturate(200%);
        -webkit-backdrop-filter: blur(28px);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        padding: 14px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: inset 0 1px 1px 0 rgba(255, 255, 255, 0.22), 0 12px 32px rgba(0, 0, 0, 0.5);
    }
    .header-badge {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8 !important;
        font-size: 11.5px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.3);
    }

    /* Frosted Liquid Glass Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(24px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        box-shadow: inset 0 1px 1px 0 rgba(255, 255, 255, 0.2), 0 8px 24px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px !important;
        color: #94a3b8 !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
    }

    /* Segmented Glass Tabs */
    button[data-baseweb="tab"] {
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        padding: 8px 18px !important;
        border-radius: 9999px !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.06) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.14) !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: inset 0 1px 1px 0 rgba(255, 255, 255, 0.3), 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    }

    /* Glass Action Buttons */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 9999px !important;
        color: #ffffff !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        box-shadow: inset 0 1px 1px 0 rgba(255, 255, 255, 0.18) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        padding: 8px 18px !important;
    }
    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.35), inset 0 1px 1px 0 rgba(255, 255, 255, 0.3) !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Dataframe and Tables */
    div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }

    /* Dropdown Select Boxes */
    div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] * {
        color: #ffffff !important;
    }

    /* Info & Success Banners */
    div.stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Luminous Top Header
st.markdown("""
<div class="header-bar">
    <div>
        <span style="font-size:18px; font-weight:700; color:#ffffff; letter-spacing:-0.02em;">📡 AURA-RF</span>
        <span style="font-size:13px; color:#cbd5e1; margin-left:10px;">Smart Room Occupancy • iOS 26 Liquid Glass</span>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span class="header-badge">10 Hz Live Telemetry</span>
        <a href="http://localhost:8080" target="_blank" style="background:rgba(56,189,248,0.18); border:1px solid rgba(56,189,248,0.4); color:#38bdf8; text-decoration:none; padding:5px 14px; border-radius:9999px; font-size:12px; font-weight:600; box-shadow:0 0 12px rgba(56,189,248,0.25);">
            Fullscreen HUD (Port 8080) ↗
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Inference Engine
if "engine" not in st.session_state:
    try:
        model = load_model()
    except Exception:
        from occ.models.baselines import ThresholdClassifier
        model = ThresholdClassifier()
    st.session_state.engine = RealtimeInferenceEngine(cfg=cfg, model=model)

# 4 Comprehensive Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌊 Liquid Glass HUD",
    "⚡ Stream Controls & Injection",
    "🔬 Sim-to-Real Calibration",
    "⚖️ Adaptive Calibration"
])

# -----------------------------------------------------------------------------
# TAB 1: LIQUID GLASS MISSION CONTROL (EMBEDDED HUD)
# -----------------------------------------------------------------------------
with tab1:
    web_dir = Path(__file__).parent / "web"
    index_file = web_dir / "index.html"
    css_file = web_dir / "style.css"
    js_file = web_dir / "app.js"
    
    if index_file.is_file() and css_file.is_file() and js_file.is_file():
        html_content = index_file.read_text(encoding="utf-8")
        css_content = css_file.read_text(encoding="utf-8")
        js_content = js_file.read_text(encoding="utf-8")
        
        # Inline the CSS and JS for hermetic rendering inside Streamlit's iframe
        bundled_html = (
            html_content.replace(
                '<link rel="stylesheet" href="style.css?v=liquid_glass_v4">',
                f'<style>\n{css_content}\n</style>',
            )
            .replace(
                '<link rel="stylesheet" href="style.css?v=liquid_glass_v3">',
                f'<style>\n{css_content}\n</style>',
            )
            .replace(
                '<link rel="stylesheet" href="style.css?v=clean_norm">',
                f'<style>\n{css_content}\n</style>',
            )
            .replace(
                '<link rel="stylesheet" href="style.css">',
                f'<style>\n{css_content}\n</style>',
            )
            .replace(
                '<script src="app.js?v=liquid_glass_v4"></script>',
                f'<script>\n{js_content}\n</script>',
            )
            .replace(
                '<script src="app.js?v=liquid_glass_v3"></script>',
                f'<script>\n{js_content}\n</script>',
            )
            .replace(
                '<script src="app.js?v=clean_norm"></script>',
                f'<script>\n{js_content}\n</script>',
            )
            .replace(
                '<script src="app.js"></script>',
                f'<script>\n{js_content}\n</script>',
            )
        )
        
        components.html(bundled_html, height=890, scrolling=True)
    else:
        st.info("Liquid Glass Web HUD active at http://localhost:8080")

# -----------------------------------------------------------------------------
# TAB 2: STREAM CONTROLS & INJECTION
# -----------------------------------------------------------------------------
with tab2:
    preds_df = fetch_predictions(limit=5)
    latest_pred = preds_df.iloc[0] if not preds_df.empty else None
    
    curr_label = latest_pred["voted_label"] if latest_pred is not None else "Empty"
    curr_conf = latest_pred["confidence"] if latest_pred is not None else 0.96
    
    is_occ = (curr_label in ["Occupied", "Transition"])
    actuator_text = "💡 Lights ON | 🌀 HVAC ON" if is_occ else "🌙 Lights OFF | 💤 HVAC Standby"
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Occupancy State", curr_label)
    m2.metric("Confidence", f"{curr_conf * 100:.1f}%")
    m3.metric("Building Actuation", actuator_text)

    # Clean Waveform Buffer in Session State
    if "wave_history" not in st.session_state:
        st.session_state.wave_history = []

    # One-Click Injection Controls
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    base = st.session_state.engine.calibrator.get_baseline()
    
    def inject_pattern(state_name: str, loss: float, noise: float):
        t_now = time.time()
        for i in range(50):
            t_pkt = t_now + (i * 0.1)
            d1 = base["R1"] - loss + random.gauss(0, noise)
            d2 = base["R2"] - loss + random.gauss(0, noise)
            st.session_state.engine.push_sample(t_pkt, "R1", d1)
            st.session_state.engine.push_sample(t_pkt + 0.005, "R2", d2)
            st.session_state.wave_history.append({"t": round(i * 0.1, 2), "R1": d1, "R2": d2, "State": state_name})
        st.session_state.wave_history = st.session_state.wave_history[-100:]
        st.session_state.engine.process_window(run_id="live_demo", origin="real")
        st.rerun()

    with c_btn1:
        if st.button("🟢 Inject Vacant Room (Empty)", use_container_width=True):
            inject_pattern("Empty", 0.0, 1.1)
    with c_btn2:
        if st.button("🟡 Inject Doorway Transition", use_container_width=True):
            inject_pattern("Transition", 4.2, 3.5)
    with c_btn3:
        if st.button("🔴 Inject Seated Occupant", use_container_width=True):
            inject_pattern("Occupied", 6.8, 1.8)

    # Real-Time Dual Oscilloscope with High-Contrast Dark Liquid Glass Theme
    if st.session_state.wave_history and HAS_PLOTLY:
        df_wave = pd.DataFrame(st.session_state.wave_history)
        fig_osc = go.Figure()
        fig_osc.add_trace(go.Scatter(
            y=df_wave["R1"], mode="lines", name="Sniffer R1 (dBm)",
            line=dict(color="#38bdf8", width=2.4)
        ))
        fig_osc.add_trace(go.Scatter(
            y=df_wave["R2"], mode="lines", name="Sniffer R2 (dBm)",
            line=dict(color="#fbbf24", width=2.4)
        ))
        fig_osc.update_layout(
            title=dict(text="Real-Time Dual-Channel RF Oscilloscope (10 Hz Grid)", font=dict(color="#ffffff", size=14)),
            xaxis=dict(title="Recent Samples (5.0s Window)", gridcolor="rgba(255, 255, 255, 0.08)", color="#cbd5e1"),
            yaxis=dict(title="Signal Strength (dBm)", gridcolor="rgba(255, 255, 255, 0.08)", color="#cbd5e1"),
            template="plotly_dark",
            paper_bgcolor="rgba(8, 13, 34, 0.6)",
            plot_bgcolor="rgba(8, 13, 34, 0.85)",
            font=dict(color="#cbd5e1"),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_osc, use_container_width=True)

    # Compact History Audit Trail
    if not preds_df.empty:
        st.caption("Recent Real-Time Decisions (Audit Trail)")
        st.dataframe(
            preds_df[["timestamp", "voted_label", "confidence", "latency_ms"]].head(4),
            use_container_width=True,
            hide_index=True
        )

# -----------------------------------------------------------------------------
# TAB 3: NS-2 SIM-TO-REAL
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("#### Sim-to-Real Calibration")

    col_sim_left, col_sim_right = st.columns([1, 1])

    with col_sim_left:
        st.markdown("**2D Fresnel Zone Obstruction Raytracer**")
        pos_sel = st.selectbox(
            "Move Human in Room (6.0m × 4.5m)",
            ["Seated at Desk (4.0, 3.2)", "Blocking Link 1 (2.5, 0.35)", "Blocking Link 2 (1.6, 2.2)", "Vacant / Outside (-1.0, 2.2)"]
        )
        if "Desk" in pos_sel:
            hx, hy = 4.0, 3.2
        elif "Link 1" in pos_sel:
            hx, hy = 2.5, 0.35
        elif "Link 2" in pos_sel:
            hx, hy = 1.6, 2.2
        else:
            hx, hy = -1.0, 2.2

        # Physics calculation
        tx = np.array([0.3, 0.3])
        r1 = np.array([5.7, 0.4])
        r2 = np.array([3.0, 4.2])
        p = np.array([hx, hy])
        lambda_rf = 3e8 / 2.437e9

        def calc_fresnel(tx_pt, rx_pt, pt):
            vec = rx_pt - tx_pt
            L = float(np.linalg.norm(vec))
            s = float(np.dot(pt - tx_pt, vec) / (L * L))
            if s <= 0.0 or s >= 1.0:
                return 0.0, "Clear"
            h = float(np.linalg.norm(pt - (tx_pt + s * vec)))
            r1 = math.sqrt((lambda_rf * s * (1 - s) * L))
            boundary = 1.31 * r1 + 0.20
            if h < boundary:
                frac = 1.0 - (h / boundary)
                return 6.18 * (frac ** 2), ("Blocked" if h < 0.2 else "Diffracted")
            return 0.0, "Clear"

        loss_l1, stat_l1 = calc_fresnel(tx, r1, p)
        loss_l2, stat_l2 = calc_fresnel(tx, r2, p)

        st.write(f"• **Link 1 (Tx → R1)**: `+{loss_l1:.2f} dB` excess loss ({stat_l1})")
        st.write(f"• **Link 2 (Tx → R2)**: `+{loss_l2:.2f} dB` excess loss ({stat_l2})")

        if HAS_PLOTLY:
            fig_r = go.Figure()
            # Walls
            fig_r.add_trace(go.Scatter(x=[0, 6, 6, 0, 0], y=[0, 0, 4.5, 4.5, 0], mode="lines", line=dict(color="#cbd5e1", width=2), name="Walls"))
            # LOS Rays
            fig_r.add_trace(go.Scatter(x=[tx[0], r1[0]], y=[tx[1], r1[1]], mode="lines", line=dict(color="#38bdf8", dash="dash", width=1.5), name="Link 1"))
            fig_r.add_trace(go.Scatter(x=[tx[0], r2[0]], y=[tx[1], r2[1]], mode="lines", line=dict(color="#fbbf24", dash="dash", width=1.5), name="Link 2"))
            # Nodes
            fig_r.add_trace(go.Scatter(x=[tx[0]], y=[tx[1]], mode="markers+text", text=["Tx"], textposition="bottom center", marker=dict(size=14, color="#fbbf24", symbol="diamond"), name="Tx AP"))
            fig_r.add_trace(go.Scatter(x=[r1[0]], y=[r1[1]], mode="markers+text", text=["R1"], textposition="bottom center", marker=dict(size=14, color="#38bdf8"), name="Rx1"))
            fig_r.add_trace(go.Scatter(x=[r2[0]], y=[r2[1]], mode="markers+text", text=["R2"], textposition="top center", marker=dict(size=14, color="#c084fc"), name="Rx2"))
            # Human
            h_col = "#f87171" if (loss_l1 > 2.0 or loss_l2 > 2.0) else "#34d399"
            fig_r.add_trace(go.Scatter(x=[hx], y=[hy], mode="markers+text", text=["Person"], textposition="top center", marker=dict(size=16, color=h_col), name="Human"))
            fig_r.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_dark",
                paper_bgcolor="rgba(8, 13, 34, 0.6)",
                plot_bgcolor="rgba(8, 13, 34, 0.85)",
                font=dict(color="#cbd5e1"),
                xaxis=dict(range=[-1.5, 6.5], gridcolor="rgba(255, 255, 255, 0.08)"),
                yaxis=dict(range=[-0.5, 5.0], gridcolor="rgba(255, 255, 255, 0.08)")
            )
            st.plotly_chart(fig_r, use_container_width=True)

    with col_sim_right:
        st.markdown("**Transfer Learning Performance**")
        sr_file = root_dir / "results" / "tables" / "e16_simreal_learning_curves.csv"
        if sr_file.is_file() and HAS_PLOTLY:
            df_sr = pd.read_csv(sr_file)
            fig_sr = px.line(
                df_sr, x="n_real_sessions", y="mean_f1", color="regime",
                template="plotly_dark", markers=True,
                color_discrete_map={"real_only": "#94a3b8", "pretrain_finetune": "#34d399", "mixed": "#38bdf8", "sim_only": "#c084fc"},
                labels={"n_real_sessions": "Real Sessions (N)", "mean_f1": "Macro-F1"}
            )
            fig_sr.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(8, 13, 34, 0.6)",
                plot_bgcolor="rgba(8, 13, 34, 0.85)",
                font=dict(color="#cbd5e1"),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)"),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)")
            )
            st.plotly_chart(fig_sr, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: ADAPTIVE CALIBRATION
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("#### Adaptive Drift Compensation")

    calib = st.session_state.engine.calibrator
    curr_b = calib.get_baseline()
    static_b = calib.static_baseline
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Adaptive Baseline R1", f"{curr_b['R1']:.2f} dBm", f"{curr_b['R1'] - static_b['R1']:+.2f} dB drift")
    k2.metric("Adaptive Baseline R2", f"{curr_b['R2']:.2f} dBm", f"{curr_b['R2'] - static_b['R2']:+.2f} dB drift")
    
    is_locked = (calib.consecutive_empty_count < calib.min_empty_windows)
    lock_status = "🔒 LOCKED (Guard Active)" if is_locked else "🟢 TRACKING (Empty Room)"
    k3.metric("Calibration Engine State", lock_status)
