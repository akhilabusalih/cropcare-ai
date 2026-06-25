import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui_components import inject_css, render_section_label, render_badge

st.set_page_config(
    page_title="Prediction History | CropGuardian AI",
    page_icon="📜",
    layout="wide",
)
inject_css()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 CropGuardian AI")
    st.divider()
    st.markdown("**📜 Prediction History**")
    st.caption(
        "All predictions made through the Disease Detection page "
        "are logged here automatically."
    )

# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("## 📜 Prediction History")
st.caption("A complete log of every AI analysis run through this platform.")
st.divider()

HISTORY_CSV = "data/history/prediction_log.csv"

if not os.path.exists(HISTORY_CSV):
    st.info(
        "No prediction history found yet. "
        "Run a diagnosis on the Disease Detection page to see results here."
    )
    st.stop()

# ─── Load Data ───────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(HISTORY_CSV)
except Exception as e:
    st.error(f"Could not load history log: {e}")
    st.stop()

if df.empty:
    st.info("The history log is empty. No predictions have been recorded yet.")
    st.stop()

if "timestamp" in df.columns:
    df = df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)


# ─── Summary Stats ────────────────────────────────────────────────────────────
render_section_label("Summary")
s1, s2, s3, s4 = st.columns(4)

total       = len(df)
most_recent = df["timestamp"].iloc[0]        if "timestamp"      in df.columns else "—"
top_disease = (
    df["predicted_class"].value_counts().idxmax()
    if "predicted_class" in df.columns and not df["predicted_class"].empty else "—"
)
avg_conf    = (
    round(df["confidence"].astype(float).mean(), 1)
    if "confidence" in df.columns else 0
)

s1.metric("Total Predictions",  total)
s2.metric("Avg Confidence",     f"{avg_conf}%")
s3.metric("Most Common Disease", top_disease if len(top_disease) < 28 else top_disease[:25] + "…")
s4.metric("Most Recent Run",     str(most_recent)[:16])

st.divider()

# ─── Filtered Table ───────────────────────────────────────────────────────────
render_section_label("All Records")

# Confidence-level filter
conf_levels = ["All"] + sorted(df["confidence_level"].dropna().unique().tolist()) \
    if "confidence_level" in df.columns else ["All"]
filter_col, dl_col = st.columns([2, 1])
with filter_col:
    selected_level = st.selectbox(
        "Filter by Confidence Level",
        conf_levels,
        label_visibility="collapsed",
    )
with dl_col:
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False),
        file_name="cropguardian_prediction_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

filtered_df = df.copy()
if selected_level != "All" and "confidence_level" in df.columns:
    filtered_df = filtered_df[filtered_df["confidence_level"] == selected_level]

# Configure column display
column_config = {}
if "confidence" in filtered_df.columns:
    column_config["confidence"] = st.column_config.ProgressColumn(
        "Confidence",
        min_value=0,
        max_value=100,
        format="%.1f%%",
    )
if "timestamp" in filtered_df.columns:
    column_config["timestamp"] = st.column_config.TextColumn("Timestamp", width="medium")
if "predicted_class" in filtered_df.columns:
    column_config["predicted_class"] = st.column_config.TextColumn("Detected Disease", width="large")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
)
st.caption(f"Showing {len(filtered_df)} of {total} records.")
