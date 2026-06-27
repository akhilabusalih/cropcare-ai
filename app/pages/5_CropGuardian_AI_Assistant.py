import os
import sys

# Ensure root and app directories are in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

import json
import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Imports from project
from agents.advisory_agent.advisory_agent import AdvisoryAgent, GENAI_AVAILABLE
from ui_components import inject_css, render_section_label

# Set page config
st.set_page_config(
    page_title="CropGuardian AI Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium styling
inject_css()

# Cache the AdvisoryAgent initialization to reuse client connection
@st.cache_resource
def get_advisory_agent():
    return AdvisoryAgent()

# Guard: Ensure active diagnosis context exists
if "chat_context" not in st.session_state or not st.session_state["chat_context"]:
    st.markdown("")
    st.warning("🌿 No active diagnosis found. Please analyze a crop image first.")
    if st.button("Go to Disease Detection", type="primary"):
        st.switch_page("pages/1_Disease_Detection.py")
    st.stop()

context = st.session_state["chat_context"]
crop = context.get("crop", "Unknown")
disease = context.get("disease", "Unknown")
disease_id = context.get("disease_id", "Unknown")
severity = context.get("severity", "Unknown")
weather = context.get("weather", {})
kb_summary = context.get("knowledge_summary", {})

# Initialize message history list and diagnostics timestamp
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "chat_created_at" not in st.session_state:
    st.session_state["chat_created_at"] = datetime.datetime.now().isoformat()

# Sidebar options
st.sidebar.markdown("### ⚙️ Chat Settings")

# Model selector
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "gemini-2.5-flash"

model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]
selected_model = st.sidebar.selectbox(
    "Select AI Model",
    model_options,
    index=model_options.index(st.session_state["selected_model"])
)
st.session_state["selected_model"] = selected_model

# Quick action button: Reset chat
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["chat_messages"] = []
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
    "💬 **CropGuardian AI** is context-aware. It automatically knows your crop "
    "type, weather conditions, severity score, and treatment options from the "
    "active diagnosis."
)

# Main page layout
st.title("🌿 CropGuardian AI Assistant")
st.markdown("Ask follow-up questions about disease recovery, chemical dosages, organic alternatives, or prevention steps.")

# Display Active Diagnosis Summary Card
st.markdown("### 🌿 Current Diagnosis")
model_display_names = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-1.5-flash": "Gemini 1.5 Flash"
}
with st.container(border=True):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"**Crop**  \n`{crop}`")
    with col2:
        # Format disease name to look nicer
        formatted_disease = disease.replace("_", " ").title()
        st.markdown(f"**Disease**  \n`{formatted_disease}`")
    with col3:
        st.markdown(f"**Severity**  \n`{severity}`")
    with col4:
        model_display = model_display_names.get(selected_model, selected_model)
        st.markdown(f"**Model**  \n`{model_display}`")
    with col5:
        if weather:
            w_text = f"{weather.get('temperature', '—')}°C, {weather.get('humidity', '—')}% RH"
        else:
            w_text = "Unavailable"
        st.markdown(f"**Weather**  \n`{w_text}`")

st.divider()

# Render chat history
for message in st.session_state["chat_messages"]:
    role = message["role"]
    avatar = "🤖" if role == "model" else "🧑‍🌾"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

# Formulate system instruction based on compact context
weather_summary = f"{weather.get('temperature')}°C, {weather.get('humidity')}% Humidity, {weather.get('precipitation')}mm precipitation" if weather else "Unavailable"
compact_kb_str = json.dumps(kb_summary, indent=2) if kb_summary else "No additional scientific info loaded."

system_instruction_text = f"""
You are CropGuardian AI, a context-aware agricultural AI advisory assistant.
You are helping a farmer with the following active diagnosis:
- Crop: {crop}
- Disease: {disease} (ID: {disease_id})
- Severity: {severity}
- Weather: {weather_summary}

Knowledge Base Reference Details:
{compact_kb_str}

Farmer Advisory Guidelines:
1. Provide concise, practical, and highly actionable advice tailored specifically to the diagnosed crop, disease, severity, and current weather.
2. If weather conditions are bad (e.g. rain forecasted, high humidity, extreme temperatures), warn the farmer if they should delay chemical spraying or take protective actions.
3. Offer treatment options including chemical, biological, and organic methods based on the knowledge base.
4. Keep a helpful, empathetic, and professional tone suitable for supporting farmers.
5. Answer follow-up questions directly using this context. Do not offer advice unrelated to agriculture or this crop/disease context unless asked to compare with similar crop issues.
"""

# User Input
if prompt := st.chat_input("Ask about organic remedies, spray timing, fertilizer doses..."):
    # Display user input bubble
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(prompt)
    
    # Save user message to history
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})

    # Prepare client and contents for Gemini
    advisory_agent = get_advisory_agent()
    
    if not advisory_agent or not advisory_agent.client:
        st.error("Gemini client is not initialized. Please verify your GEMINI_API_KEY environment variable.")
        st.stop()
        
    client = advisory_agent.client

    # Formulate contents array (sliding window: last 10 messages)
    contents = []
    # Grab the last 10 messages from history (excluding the one we just added, which will be appended as the current query)
    recent_messages = st.session_state["chat_messages"][-10:]
    
    for msg in recent_messages:
        role = "user" if msg["role"] == "user" else "model"
        # google-genai SDK types.Content structure
        try:
            from google.genai import types
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )
        except ImportError:
            # Fallback format if types is not directly importable
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

    # Query Gemini model with spinner
    with st.spinner("CropGuardian AI is thinking..."):
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system_instruction_text,
                temperature=0.4
            )
            
            response = client.models.generate_content(
                model=selected_model,
                contents=contents,
                config=config
            )
            
            assistant_response = response.text
            
            # Save assistant message to history
            st.session_state["chat_messages"].append({"role": "model", "content": assistant_response})
            
            # Rerun to update chat screen
            st.rerun()

        except Exception as e:
            st.error(f"⚠️ CropGuardian AI encountered an error: {e}")
            st.info("Your message and chat history have been preserved. You can try sending your question again.")
