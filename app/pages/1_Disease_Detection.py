import os
import sys
import csv
import datetime
import streamlit as st

# ─── Path Setup ───────────────────────────────────────────────────────────────
# Project root (d:/agricare) for backend agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# app/ directory for ui_components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ─── Backend Imports (unchanged) ──────────────────────────────────────────────
from agents.coordinator_agent.coordinator_agent import CoordinatorAgent
from agents.adk_workflow.adk_coordinator import ADKCoordinatorAgent
from agents.feedback_agent.feedback_agent import FeedbackAgent

# ─── UI Helpers ──────────────────────────────────────────────────────────────
from ui_components import (
    inject_css,
    render_badge,
    render_section_label,
    render_advisory_list,
    render_callout,
    format_disease_name,
    is_healthy,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Detection | CropGuardian AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ─── Cached Agent Initialisation — LAZY (load only when first needed) ────────
@st.cache_resource
def get_coordinator_agent():
    """Loads CoordinatorAgent (includes TF MobileNetV2). Cached for session lifetime."""
    return CoordinatorAgent()

@st.cache_resource
def get_adk_coordinator_agent():
    """Loads ADKCoordinatorAgent. Cached for session lifetime."""
    return ADKCoordinatorAgent()

@st.cache_resource
def get_feedback_agent():
    """Loads FeedbackAgent. Lightweight — no TF dependency."""
    return FeedbackAgent()

@st.cache_data
def get_class_names() -> list:
    """
    Loads disease class names directly from the numpy file.
    Does NOT require loading the TF model or CoordinatorAgent.
    Used only for the feedback dropdown.
    """
    import numpy as np
    return list(np.load("models/class_names.npy", allow_pickle=True))

# NOTE: Agents are NOT initialized at module level.
# They are loaded on first use inside the button handler and results section.
# @st.cache_resource guarantees they are loaded only once per session.


# ─── History Logger (backend-unchanged logic) ─────────────────────────────────
def log_prediction_history(result: dict, img_name: str):
    history_dir = "data/history"
    os.makedirs(history_dir, exist_ok=True)
    csv_path    = os.path.join(history_dir, "prediction_log.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "image_name", "predicted_class",
                      "confidence", "confidence_level", "model_version"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":       result["timestamp"],
            "image_name":      img_name,
            "predicted_class": result["disease"],
            "confidence":      result["confidence"],
            "confidence_level":result["confidence_level"],
            "model_version":   result["model_version"],
        })


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Settings
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌿 CropGuardian AI")
    st.divider()

    st.markdown("**⚙️ Settings**")
    workflow_mode = st.radio(
        "Orchestration Engine",
        ("Google ADK Workflow", "Legacy Coordinator"),
        help="ADK uses Gemini-powered agents. Falls back to Legacy Coordinator automatically if the Gemini API is unavailable.",
    )
    st.caption(
        "**ADK:** Gemini 2.5 Flash agents with tool calling.\n\n"
        "**Legacy:** Deterministic Python orchestration."
    )

    st.divider()
    st.markdown("**📍 Location Tips**")
    st.caption(
        "Providing your farm location unlocks:\n"
        "- Real-time weather data\n"
        "- Environmental disease risk\n"
        "- Weather-aware Gemini advice"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔍 Disease Detection")
st.caption("Upload a crop image · Add your farm location · Get a complete AI diagnosis in seconds.")


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT PANEL
# ═══════════════════════════════════════════════════════════════════════════════
input_col, loc_col = st.columns([5, 4], gap="large")

with input_col:
    render_section_label("Crop Image")
    uploaded_file = st.file_uploader(
        "Upload crop leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.image(uploaded_file, width=240, caption=uploaded_file.name)

with loc_col:
    render_section_label("Farm Location (Optional)")
    location_mode = st.radio(
        "Input method",
        ["City / State / Country", "Latitude + Longitude"],
        horizontal=True,
        label_visibility="collapsed",
    )

    location_input = None
    lat = lon = None

    if location_mode == "City / State / Country":
        location_input = st.text_input(
            "Location",
            placeholder="e.g. Cherthala, Kerala, India",
            label_visibility="collapsed",
        )
    else:
        la_col, lo_col = st.columns(2)
        with la_col:
            lat_raw = st.text_input("Latitude",  placeholder="e.g. 9.68")
        with lo_col:
            lon_raw = st.text_input("Longitude", placeholder="e.g. 76.33")
        if lat_raw and lon_raw:
            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
            except ValueError:
                st.error("Latitude and Longitude must be valid numbers.")

st.markdown("")

# ─── Run Button ───────────────────────────────────────────────────────────────
if uploaded_file:
    analyze_btn = st.button(
        "🔬  Analyze Crop Disease",
        type="primary",
        use_container_width=True,
    )
    if analyze_btn:
        # Save image temporarily
        temp_img_path = f"temp_{uploaded_file.name}"
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Lazy-load agents — triggers TF model load only here, on first use
        _coordinator     = get_coordinator_agent()
        _adk_coordinator = get_adk_coordinator_agent()

        with st.spinner("Running AI analysis pipeline — Disease · Weather · Severity · Risk · Advisory …"):
            if workflow_mode == "Google ADK Workflow":
                response = _adk_coordinator.process_image(
                    temp_img_path, location_input, lat, lon
                )
            else:
                response = _coordinator.process_image(
                    temp_img_path, location_input, lat, lon
                )

        if response["status"] == "error":
            st.error("The analysis pipeline encountered a critical error.")
            for w in response.get("warnings", []):
                st.error(f"  ↳ {w}")
        else:
            st.session_state["pipeline_response"]  = response
            st.session_state["uploaded_img_path"]  = temp_img_path
            log_prediction_history(response["prediction"], uploaded_file.name)
            # NOTE: No st.rerun() — results render in this same execution cycle.
            # st.rerun() would cause a full second page execution (doubled render time).
else:
    st.info("📂 Upload a crop leaf image above to begin.")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS — only shown after a successful run
# ═══════════════════════════════════════════════════════════════════════════════
if "pipeline_response" not in st.session_state:
    st.stop()

response      = st.session_state["pipeline_response"]
result        = response.get("prediction", {})
severity_res  = response.get("severity", {}) or {}
advice_res    = response.get("advice", {})   or {}
weather_data  = response.get("weather")
risk_data     = response.get("environmental_risk")
temp_img_path = st.session_state.get("uploaded_img_path", "")

# Derived values
disease_class     = result.get("disease", "Unknown")
confidence        = result.get("confidence", 0)
confidence_level  = result.get("confidence_level", "Unknown")
severity          = severity_res.get("severity", "Unknown")
spread_prob       = risk_data.get("spread_probability", 0) if (risk_data and "error" not in risk_data) else 0
overall_risk      = risk_data.get("overall_risk", "N/A")   if (risk_data and "error" not in risk_data) else "N/A"
healthy           = is_healthy(disease_class)

# Non-critical pipeline warnings (partial success)
non_fatal = [w for w in response.get("warnings", []) if "Fallback" not in w]
if non_fatal:
    for w in non_fatal:
        st.warning(w)

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# ① DIAGNOSIS HERO  ← above the fold priority
# ═══════════════════════════════════════════════════════════════════════════════
render_section_label("Diagnosis")

with st.container(border=True):
    hero_name_col, hero_conf_col, hero_sev_col, hero_risk_col = st.columns(
        [3, 1.2, 1.2, 1.2], gap="large"
    )

    with hero_name_col:
        # Disease name — largest element on the page
        name_class = "cg-disease-healthy" if healthy else ""
        st.markdown(
            f'<div class="cg-disease-hero {name_class}">'
            f'{format_disease_name(disease_class)}</div>',
            unsafe_allow_html=True,
        )
        # Badge row: Confidence Level · Severity · Overall Risk
        badges_html = f"""
        <div class="cg-badge-row">
            <span class="cg-badge-label">Confidence:</span>
            {render_badge(confidence_level)}
            <span class="cg-badge-label" style="margin-left:6px;">Severity:</span>
            {render_badge(severity)}
            <span class="cg-badge-label" style="margin-left:6px;">Risk:</span>
            {render_badge(overall_risk)}
        </div>
        """
        st.markdown(badges_html, unsafe_allow_html=True)

        # Low-confidence warning
        if result.get("warning"):
            st.markdown("")
            render_callout(result["warning"], kind="warning")

    with hero_conf_col:
        st.metric("Confidence", f"{confidence}%")
        st.progress(confidence / 100)

    with hero_sev_col:
        sev_score = severity_res.get("severity_score", 0)
        st.metric("Severity Score", f"{sev_score} / 3")
        st.progress(sev_score / 3)

    with hero_risk_col:
        st.metric("Spread Risk", f"{spread_prob}%")
        st.progress(spread_prob / 100)


# ═══════════════════════════════════════════════════════════════════════════════
# ② ENVIRONMENTAL INTELLIGENCE + TOP PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
env_col, pred_col = st.columns(2, gap="large")

with env_col:
    render_section_label("Environmental Intelligence")
    if weather_data and "error" not in weather_data:
        w = weather_data.get("weather", {})
        st.caption(f"📍 {weather_data.get('location', 'Unknown')}")
        wc1, wc2 = st.columns(2)
        wc1.metric("🌡 Temperature",   f"{w.get('temperature','—')}°C",
                   delta=f"Feels like {w.get('feels_like','—')}°C", delta_color="off")
        wc2.metric("💧 Humidity",      f"{w.get('humidity','—')}%")
        wc3, wc4 = st.columns(2)
        wc3.metric("🌧 Precipitation", f"{w.get('precipitation','—')} mm")
        wc4.metric("💨 Wind Speed",    f"{w.get('wind_speed','—')} km/h")
        if w.get("cloud_cover") is not None:
            st.caption(f"☁ Cloud cover: {w.get('cloud_cover')}%  ·  "
                       f"💨 Gusts: {w.get('wind_gusts','—')} km/h  ·  "
                       f"🔵 Pressure: {w.get('pressure','—')} hPa")
    else:
        with st.container(border=True):
            st.markdown("**🌤 Weather data unavailable**")
            st.caption(
                "Enter your farm location in the input panel above "
                "to unlock real-time weather and environmental risk data."
            )

with pred_col:
    render_section_label("Top Predictions")
    top3 = result.get("top_predictions", [])
    rank_icons = ["🥇", "🥈", "🥉"]
    for i, p in enumerate(top3):
        icon  = rank_icons[i] if i < 3 else f"#{i+1}"
        label = format_disease_name(p["class"])
        conf  = p["confidence"]
        st.markdown(f"**{icon} {label}**")
        st.progress(conf / 100, text=f"{conf}%")
    st.caption(
        f"Model: `{result.get('model_version','—')}`  ·  "
        f"Run: `{result.get('timestamp','—')}`"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ③ ENVIRONMENTAL RISK BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
if risk_data and "error" not in risk_data:
    render_section_label("Risk Breakdown")
    rc1, rc2, rc3 = st.columns(3, gap="medium")

    risk_items = [
        (rc1, "🍄 Fungal Risk",      risk_data.get("fungal_risk",      "Unknown")),
        (rc2, "🦠 Bacterial Risk",   risk_data.get("bacterial_risk",   "Unknown")),
        (rc3, "🌡 Heat Stress Risk", risk_data.get("heat_stress_risk", "Unknown")),
    ]
    for col, label, level in risk_items:
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.markdown(render_badge(level), unsafe_allow_html=True)

    for reason in risk_data.get("reasons", []):
        st.markdown(f'<div class="cg-reason">• {reason}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ④ GEMINI ADVISORY — Tabs
# ═══════════════════════════════════════════════════════════════════════════════
if advice_res:
    render_section_label("AI Advisory")

    # Immediate action + weather warning — always above the tabs
    immediate = advice_res.get("immediate_actions", "")
    if immediate and immediate.lower() not in ("none required.", "none required", "none", ""):
        render_callout(f"<strong>Immediate Action Required:</strong> {immediate}", kind="action")

    weather_warn = advice_res.get("weather_based_warnings", "")
    if weather_warn and weather_warn.lower() not in ("none.", "none", ""):
        render_callout(f"<strong>Weather Advisory:</strong> {weather_warn}", kind="warning")

    # Tabbed advisory cards
    tab_overview, tab_treatment, tab_prevention, tab_fertilizer = st.tabs(
        ["📋 Overview", "💊 Treatment", "🛡 Prevention", "🌱 Fertilizer"]
    )

    with tab_overview:
        st.markdown(advice_res.get("disease_description", ""))
        if severity_res.get("details"):
            st.markdown("")
            with st.container(border=True):
                st.markdown(f"**⚠️ Severity Assessment** — `{severity}`")
                st.markdown(severity_res.get("details", ""))
                st.caption(f"Assessment method: {severity_res.get('assessment_method','—')}")
        st.caption(f"Advisory source: {advice_res.get('source','Unknown')}")

    with tab_treatment:
        render_advisory_list(advice_res.get("treatment", []))

    with tab_prevention:
        render_advisory_list(advice_res.get("prevention", []))

    with tab_fertilizer:
        render_advisory_list(advice_res.get("fertilizer_recommendations", []))

    expert = advice_res.get("expert_consultation", "")
    if expert and expert.lower() not in ("not necessary at this time.", "not necessary at this time", ""):
        st.caption(f"🧑‍🔬 Expert consultation: {expert}")


# ═══════════════════════════════════════════════════════════════════════════════
# FEEDBACK — Expander
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Was this prediction correct?", expanded=False):
    st.markdown("Your feedback helps improve the model for future predictions.")
    fb_yes_col, fb_no_col, _ = st.columns([1, 1.5, 4])
    with fb_yes_col:
        if st.button("✅ Yes, correct", key="fb_yes"):
            st.success("Thank you for confirming!")
            st.session_state.pop("show_feedback_form", None)
    with fb_no_col:
        if st.button("❌ Needs correction", key="fb_no"):
            st.session_state["show_feedback_form"] = True

    if st.session_state.get("show_feedback_form"):
        st.divider()
        st.markdown("**Select the correct disease class:**")
        # get_class_names() loads from numpy — does NOT require the TF model
        class_names  = get_class_names()
        actual_class = st.selectbox("Actual class", class_names, label_visibility="collapsed")
        if st.button("Submit Correction", type="primary", key="fb_submit"):
            _feedback_agent = get_feedback_agent()
            success = _feedback_agent.log_feedback(
                original_image_path=temp_img_path,
                predicted_class=result["disease"],
                actual_class=actual_class,
                confidence=result["confidence"],
                timestamp=datetime.datetime.now().replace(microsecond=0).isoformat(),
                model_version=result["model_version"],
            )
            if success:
                render_callout("Feedback submitted. Image preserved for model retraining.", kind="success")
                st.session_state["show_feedback_form"] = False
            else:
                st.error("This image was already submitted for feedback.")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE DIAGNOSTICS — Expander (hidden by default, for technical users)
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Pipeline Diagnostics", expanded=False):
    dc1, dc2, dc3 = st.columns(3)
    engine = response.get("workflow_engine", "legacy_coordinator").replace("_", " ").title()
    dc1.metric("Engine",         engine)
    dc2.metric("Execution Time", f"{response.get('execution_time_ms', 0)} ms")
    dc3.metric("Status",         response.get("status", "—").replace("_", " ").title())

    trace = response.get("agent_trace", [])
    if trace:
        st.markdown("**Agent Trace:**")
        st.markdown("  →  ".join([f"`{a}`" for a in trace]))

    fallback_warnings = [w for w in response.get("warnings", []) if "Fallback" in w or "fallback" in w]
    if fallback_warnings:
        for w in fallback_warnings:
            st.info(f"ℹ️ {w}")
