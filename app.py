import streamlit as st
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

st.set_page_config(
    page_title="VisionMate",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #aaa;
        border-radius: 6px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    .stButton > button { border-radius: 6px; font-weight: 500; }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 12px;
    }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

from pages.auth_page import render_auth_page
from pages.monitoring_tab import render_monitoring_tab
from pages.comparison_tab import render_comparison_tab
from pages.analytics_tab import render_analytics_tab

@st.cache_resource(show_spinner="Loading AI models...")
def _load_models_cached():
    """
    Load all six AI models once per server session.
    Cached via st.cache_resource — survives reruns and multiple users.
    """
    from utils.model_loader import load_all_eye_models, load_all_posture_models
    eye_models     = load_all_eye_models()
    posture_models = load_all_posture_models()
    return eye_models, posture_models

def init_session_state():
    defaults = {
        "logged_in": False,
        "user": None,
        "user_id": None,
        "username": None,
        "monitoring_active": False,
        "session_start": None,
        "last_metric_save": 0,
        "active_eye_model_name": "Custom CNN",
        "active_posture_model_name": "Custom LSTM/DNN",
        "captured_embedding": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_sidebar():
    with st.sidebar:
        st.markdown("### VisionMate")
        st.divider()
        if st.session_state.get("logged_in"):
            user = st.session_state.get("user", {})
            st.markdown("Logged in as:")
            st.markdown(f"**{user.get('full_name', 'User')}**")
            st.caption(f"@{user.get('username', '')}")
            st.divider()
            if st.button("Logout", use_container_width=True, key="logout_btn"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                init_session_state()
                st.rerun()

def main():
    init_session_state()

    if not st.session_state.get("logged_in"):
        render_auth_page()
        return

    eye_models, posture_models = _load_models_cached()

    render_sidebar()

    user_id      = st.session_state["user_id"]
    eye_name     = st.session_state["active_eye_model_name"]
    posture_name = st.session_state["active_posture_model_name"]

    tab1, tab2, tab3 = st.tabs([
        "Live Monitoring",
        "Model Comparison",
        "Analytics Dashboard",
    ])

    with tab1:
        render_monitoring_tab(
            processor          = None,   # WebRTC transformer handles this internally
            eye_model_name     = eye_name,
            posture_model_name = posture_name,
            user_id            = user_id,
        )

    with tab2:
        selected_eye, selected_posture = render_comparison_tab(
            eye_models_loaded     = eye_models,
            posture_models_loaded = posture_models,
            current_eye_model     = eye_name,
            current_posture_model = posture_name,
        )
        if selected_eye != eye_name or selected_posture != posture_name:
            st.session_state["active_eye_model_name"]     = selected_eye
            st.session_state["active_posture_model_name"] = selected_posture
            st.rerun()

    with tab3:
        render_analytics_tab(user_id=user_id)

if __name__ == "__main__":
    main()
