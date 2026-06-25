import os
import sys
import datetime
import streamlit as st

# ─── Path Setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.feedback_agent.feedback_agent import FeedbackAgent
from agents.disease_detection_agent.disease_detection_agent import DiseaseDetectionAgent
from ui_components import inject_css, render_section_label, render_callout

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Submit Feedback | CropGuardian AI",
    page_icon="📝",
    layout="wide",
)
inject_css()


# ─── Cached Agents ────────────────────────────────────────────────────────────
@st.cache_resource
def get_agents():
    return FeedbackAgent(), DiseaseDetectionAgent()

feedback_agent, disease_agent = get_agents()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 CropGuardian AI")
    st.divider()
    st.markdown("**📝 Feedback**")
    st.caption(
        "Use this page to manually submit a misclassification report "
        "when you already know the correct label for an image."
    )
    st.divider()
    st.markdown("**ℹ️ How It Helps**")
    st.caption(
        "Each submission:\n"
        "- Stores the image with its correct label\n"
        "- Logs it for the retraining pipeline\n"
        "- Prevents duplicate submissions via hash checking"
    )


# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("## 📝 Submit Feedback")
st.caption("Help improve the model by correcting a misclassified prediction.")
st.divider()


# ─── Main Layout — Two Columns ────────────────────────────────────────────────
upload_col, form_col = st.columns([2, 3], gap="large")

temp_img_path = None

with upload_col:
    render_section_label("Image")
    uploaded_file = st.file_uploader(
        "Upload the misclassified image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        # Save temporarily so the feedback agent can hash it
        temp_img_path = f"temp_feedback_{uploaded_file.name}"
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

with form_col:
    render_section_label("Correction Details")

    if uploaded_file:
        predicted_class = st.selectbox(
            "What did the model predict? (leave as 'Unknown' if unsure)",
            ["Unknown"] + list(disease_agent.class_names),
        )
        actual_class = st.selectbox(
            "What is the correct disease class?",
            disease_agent.class_names,
        )
        confidence = st.number_input(
            "Model confidence at the time (% — leave at 0 if unknown)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
        )
        st.markdown("")

        if st.button("Submit Correction", type="primary", use_container_width=True):
            if temp_img_path and os.path.exists(temp_img_path):
                success = feedback_agent.log_feedback(
                    original_image_path=temp_img_path,
                    predicted_class=predicted_class,
                    actual_class=actual_class,
                    confidence=confidence,
                    timestamp=datetime.datetime.now().replace(microsecond=0).isoformat(),
                    model_version=disease_agent.model_version,
                )
                if success:
                    render_callout(
                        "Feedback submitted successfully. "
                        "The image has been preserved in the retraining dataset.",
                        kind="success",
                    )
                else:
                    st.error(
                        "This image has already been submitted. "
                        "Duplicate submissions are prevented by image hash."
                    )
            else:
                st.error("Could not locate the uploaded file. Please try again.")
    else:
        with st.container(border=True):
            st.markdown("**👈 Upload an image first**")
            st.caption(
                "Select the misclassified crop image on the left, "
                "then complete the form here to submit your correction."
            )
