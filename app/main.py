import os
import sys

# Add project root and app directory to sys.path so imports resolve correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
app_dir = os.path.abspath(os.path.dirname(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

import uuid
import streamlit as st
from dotenv import load_dotenv
from src.utils.logger import log_environment_snapshot, log_config_snapshot, cleanup_temp_files
from src.services.knowledge_base.kb_manager import KnowledgeBaseManager

load_dotenv()
from ui_components import inject_css, render_section_label

st.set_page_config(
    page_title="CropGuardian AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# Session Management
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 CropGuardian AI")
    st.caption("AI-powered crop disease diagnosis platform")
    st.divider()
    st.markdown("**Navigate**")
    st.page_link("pages/1_Disease_Detection.py", label="Disease Detection",  icon="🔍")
    st.page_link("pages/2_Feedback.py",          label="Submit Feedback",    icon="📝")
    st.page_link("pages/3_History.py",           label="Prediction History", icon="📜")
    st.page_link("pages/4_Model_Insights.py",    label="Model Insights",     icon="📊")

    st.divider()
    st.markdown("**🛠️ Developer Mode**")
    st.session_state["dev_mode"] = st.toggle("Enable Developer Mode", value=st.session_state.get("dev_mode", False))

# Log Environment on startup and cleanup temp files
log_environment_snapshot()
cleanup_temp_files(max_age_hours=24)

# Initialize and validate Knowledge Base on application startup
if "kb_manager" not in st.session_state:
    try:
        st.session_state["kb_manager"] = KnowledgeBaseManager(validate_on_startup=True)
    except Exception as e:
        st.error(f"Critical Error: Failed to initialize Knowledge Base. {e}")
        st.stop()

# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <div style="font-size:0.78rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:#4ade80;margin-bottom:12px;">
        AI-Powered Agricultural Intelligence
    </div>
    <div style="font-size:2.6rem;font-weight:800;letter-spacing:-0.03em;color:#f1f5f9;line-height:1.15;margin-bottom:16px;">
        Instant Crop Disease<br>Diagnosis & Advisory
    </div>
    <div style="font-size:1.05rem;color:#9ba3b2;max-width:580px;line-height:1.7;margin-bottom:28px;">
        Upload a photo of any crop leaf. CropGuardian AI detects the disease,
        evaluates environmental risk using real-time weather, and generates
        expert-level treatment plans — powered by MobileNetV2 and Gemini AI.
    </div>
</div>
""", unsafe_allow_html=True)

col_cta1, col_cta2, _ = st.columns([1, 1, 3])
with col_cta1:
    st.page_link("pages/1_Disease_Detection.py", label="→ Start Diagnosis", icon="🔍")
with col_cta2:
    st.page_link("pages/4_Model_Insights.py", label="View Model Stats", icon="📊")

st.divider()

# ─── How It Works ─────────────────────────────────────────────────────────────
render_section_label("How It Works")
st.markdown("")

steps = [
    ("📷", "Upload Image",        "Take or upload a photo of an affected crop leaf — JPG or PNG."),
    ("🤖", "AI Detection",        "MobileNetV2 classifies the disease across 38 crop-disease classes with confidence scoring."),
    ("🌤", "Weather Intelligence", "Real-time weather (temp, humidity, rain, wind) is fetched for your farm location via Open-Meteo."),
    ("⚠️",  "Risk Assessment",    "Environmental risk is calculated: fungal, bacterial, and heat-stress likelihood scores."),
    ("💊", "Gemini Advisory",     "Gemini 2.5 Flash generates a tailored treatment plan, prevention steps, and fertilizer guidance."),
]

cols = st.columns(5, gap="medium")
for col, (icon, title, desc) in zip(cols, steps):
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="cg-step-icon">{icon}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cg-step-title">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cg-step-desc">{desc}</div>', unsafe_allow_html=True)

st.divider()

# ─── Key Stats ────────────────────────────────────────────────────────────────
render_section_label("Platform Capabilities")
st.markdown("")

sc1, sc2, sc3, sc4 = st.columns(4)

with sc1:
    with st.container(border=True):
        st.markdown('<div class="cg-stat-num">38</div><div class="cg-stat-lbl">Disease Classes</div>', unsafe_allow_html=True)
with sc2:
    with st.container(border=True):
        st.markdown('<div class="cg-stat-num">14</div><div class="cg-stat-lbl">Crop Species</div>', unsafe_allow_html=True)
with sc3:
    with st.container(border=True):
        st.markdown('<div class="cg-stat-num">5</div><div class="cg-stat-lbl">AI Agents</div>', unsafe_allow_html=True)
with sc4:
    with st.container(border=True):
        st.markdown('<div class="cg-stat-num">2</div><div class="cg-stat-lbl">Orchestration Engines</div>', unsafe_allow_html=True)

st.divider()

# ─── Architecture Note ────────────────────────────────────────────────────────
render_section_label("Multi-Agent Architecture")

arch_col1, arch_col2 = st.columns([2, 1], gap="large")
with arch_col1:
    st.markdown("""
    The system runs a **sequential multi-agent pipeline** orchestrated by either the
    **Google ADK Workflow** (Gemini-powered agents) or the **Legacy Coordinator** (deterministic Python).

    Both engines produce identical output schemas. The ADK engine automatically falls back to
    the Legacy Coordinator if the Gemini API is unavailable — ensuring zero downtime.
    """)
with arch_col2:
    with st.container(border=True):
        st.caption("**Pipeline Trace**")
        for step in ["DiseaseDetectionAgent", "WeatherAgent", "SeverityAgent", "EnvironmentalRiskAgent", "AdvisoryAgent"]:
            st.markdown(f"`{step}`")
