"""
ui_components.py — CropGuardian AI Shared Design System

Provides:
  - inject_css(): Global CSS injection (fonts, cards, badges, layout tweaks)
  - render_badge(level): Color-coded HTML badge for risk/severity/confidence levels
  - render_section_label(text): Uppercase section label with accent color
  - format_disease_name(disease_class): "Tomato___Late_blight" → "Tomato — Late Blight"
  - render_advisory_list(items): Styled bullet list for advisory items

Only used by app/ files. No backend imports.
"""

import streamlit as st

# ─── Color map for risk/severity/confidence levels ──────────────────────────

LEVEL_COLORS = {
    "High":     {"bg": "rgba(239,68,68,0.14)",   "text": "#f87171", "border": "rgba(239,68,68,0.35)"},
    "Very High":{"bg": "rgba(239,68,68,0.20)",   "text": "#ef4444", "border": "rgba(239,68,68,0.50)"},
    "Medium":   {"bg": "rgba(245,158,11,0.14)",  "text": "#fbbf24", "border": "rgba(245,158,11,0.35)"},
    "Low":      {"bg": "rgba(34,197,94,0.14)",   "text": "#4ade80", "border": "rgba(34,197,94,0.35)"},
    "None":     {"bg": "rgba(107,114,128,0.14)", "text": "#9ca3af", "border": "rgba(107,114,128,0.30)"},
    "N/A":      {"bg": "rgba(107,114,128,0.14)", "text": "#9ca3af", "border": "rgba(107,114,128,0.30)"},
    "Unknown":  {"bg": "rgba(107,114,128,0.14)", "text": "#9ca3af", "border": "rgba(107,114,128,0.30)"},
    "Healthy":  {"bg": "rgba(34,197,94,0.14)",   "text": "#4ade80", "border": "rgba(34,197,94,0.35)"},
}


# ─── Global CSS ──────────────────────────────────────────────────────────────

def inject_css():
    """Injects the CropGuardian AI global design system CSS into the page."""
    st.markdown("""
    <style>
    /* ── Font override ───────────────────────────────────────────────────────
       IMPORTANT: Do NOT use [class*="st-"] or !important here.
       Streamlit 1.41+ uses Material Icons as a glyph font on elements
       with class names like "st-*". Overriding font-family on those classes
       (especially with !important) breaks icon rendering — glyphs fall back
       to their raw text names: "arrow_right", "keyboard_double_arrow_right",
       "upload", etc.
       
       We only override font on known text-content elements. ──────────────── */
    .stApp,
    .stApp p,
    .stApp li,
    .stApp label,
    .stApp input,
    .stApp textarea,
    .stApp select,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"],
    [data-testid="stNotification"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter",
                     "Helvetica Neue", Arial, sans-serif;
    }

    /* ── Layout & Spacing ─────────────────────────────────────────────────── */
    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1280px;
    }

    /* ── Metric Component Overrides ──────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.70rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        color: #9ba3b2 !important;
    }
    [data-testid="stMetricDeltaIcon"],
    [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
    }

    /* ── Tabs ─────────────────────────────────────────────────────────────── */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 4px;
    }
    [data-testid="stTabs"] [role="tab"] {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 6px 14px !important;
        border-radius: 6px 6px 0 0 !important;
    }

    /* ── Progress Bar ─────────────────────────────────────────────────────── */
    [data-testid="stProgress"] > div > div {
        border-radius: 4px !important;
    }

    /* ── Button ──────────────────────────────────────────────────────────── */
    [data-testid="baseButton-primary"] {
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }

    /* ── File Uploader ───────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] section {
        border-radius: 10px !important;
    }

    /* ── Section Label ───────────────────────────────────────────────────── */
    .cg-section-label {
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #4ade80;
        margin-bottom: 8px;
        margin-top: 4px;
    }

    /* ── Disease Hero ────────────────────────────────────────────────────── */
    .cg-disease-hero {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #f1f5f9;
        line-height: 1.2;
        margin-bottom: 4px;
    }
    .cg-disease-healthy {
        color: #4ade80;
    }

    /* ── Badge Row ───────────────────────────────────────────────────────── */
    .cg-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        align-items: center;
        margin-top: 10px;
    }
    .cg-badge-label {
        font-size: 0.72rem;
        color: #9ba3b2;
        font-weight: 500;
    }

    /* ── Advisory List Items ─────────────────────────────────────────────── */
    .cg-adv-list { margin: 0; padding: 0; }
    .cg-adv-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        font-size: 0.88rem;
        color: #d1d5db;
        line-height: 1.5;
    }
    .cg-adv-item:last-child { border-bottom: none; }
    .cg-adv-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #4ade80;
        margin-top: 7px;
        flex-shrink: 0;
    }

    /* ── Risk Reason Bar ─────────────────────────────────────────────────── */
    .cg-reason {
        font-size: 0.82rem;
        color: #9ca3af;
        padding: 5px 0 5px 12px;
        border-left: 2px solid #374151;
        margin: 3px 0;
        line-height: 1.5;
    }

    /* ── Callout Boxes ───────────────────────────────────────────────────── */
    .cg-action-box {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.28);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.87rem;
        color: #fca5a5;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    .cg-warning-box {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.28);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.87rem;
        color: #fcd34d;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    .cg-success-box {
        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.28);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.87rem;
        color: #86efac;
        line-height: 1.5;
    }

    /* ── How-It-Works Step Cards (home page) ─────────────────────────────── */
    .cg-step-icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
    }
    .cg-step-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .cg-step-desc {
        font-size: 0.80rem;
        color: #9ba3b2;
        line-height: 1.5;
    }

    /* ── Sidebar Branding ────────────────────────────────────────────────── */
    [data-testid="stSidebarContent"] {
        padding-top: 1rem;
    }

    /* ── Stat number on home ─────────────────────────────────────────────── */
    .cg-stat-num {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4ade80;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .cg-stat-lbl {
        font-size: 0.80rem;
        color: #9ba3b2;
        margin-top: 4px;
    }

    </style>
    """, unsafe_allow_html=True)


# ─── Component Helpers ────────────────────────────────────────────────────────

def render_badge(level: str) -> str:
    """Returns an inline HTML badge string for a risk/severity/confidence level."""
    c = LEVEL_COLORS.get(level, LEVEL_COLORS["Unknown"])
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:20px;'
        f'background:{c["bg"]};color:{c["text"]};border:1px solid {c["border"]};'
        f'font-size:0.75rem;font-weight:600;letter-spacing:0.03em;">{level}</span>'
    )


def render_section_label(text: str):
    """Renders a small uppercase section label with the brand green accent."""
    st.markdown(f'<div class="cg-section-label">{text}</div>', unsafe_allow_html=True)


def format_disease_name(disease_class: str) -> str:
    """
    Formats a raw PlantVillage class name into a readable label.
    Examples:
      'Tomato___Late_blight'       → 'Tomato — Late Blight'
      'Apple___Apple_scab'         → 'Apple — Apple Scab'
      'Tomato___healthy'           → 'Tomato — Healthy'
      'Pepper,_bell___Bacterial_spot' → 'Pepper Bell — Bacterial Spot'
    """
    if not disease_class:
        return "Unknown"
    if "___" in disease_class:
        plant_raw, disease_raw = disease_class.split("___", 1)
        plant = plant_raw.replace(",", "").replace("_", " ").strip().title()
        disease = disease_raw.replace("_", " ").strip().title()
        return f"{plant} — {disease}"
    return disease_class.replace("_", " ").title()


def is_healthy(disease_class: str) -> bool:
    """Returns True if the prediction is a healthy plant."""
    return "healthy" in (disease_class or "").lower()


def render_advisory_list(items: list):
    """
    Renders a list of advisory string items as styled HTML bullet points.
    Falls back to st.caption if the list is empty.
    """
    if not items:
        st.caption("No specific recommendations available.")
        return
    html = '<div class="cg-adv-list">'
    for item in items:
        html += f'<div class="cg-adv-item"><div class="cg-adv-dot"></div><div>{item}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_callout(text: str, kind: str = "warning"):
    """
    Renders a styled callout box.
    kind: 'action' (red), 'warning' (amber), 'success' (green)
    """
    icons = {"action": "⚡", "warning": "🌦", "success": "✅"}
    css_class = {"action": "cg-action-box", "warning": "cg-warning-box", "success": "cg-success-box"}
    icon = icons.get(kind, "ℹ️")
    cls = css_class.get(kind, "cg-warning-box")
    st.markdown(f'<div class="{cls}">{icon} {text}</div>', unsafe_allow_html=True)
