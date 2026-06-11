"""
Live Monitoring Tab
Uses streamlit-webrtc with Twilio TURN for reliable cloud deployment.
Live feed on the left, real-time analysis panel on the right.
"""

import streamlit as st
import time
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.frame_processor import FrameResult, load_mediapipe_landmarkers
from utils.voice_guidance import voice_guidance
from database.db_manager import save_health_metric


# ── Twilio TURN ice config ─────────────────────────────────────────────────────
# Set these two environment variables on Streamlit Cloud / Render:
#   TWILIO_ACCOUNT_SID  — from console.twilio.com
#   TWILIO_AUTH_TOKEN   — from console.twilio.com
# Twilio free tier is enough for development and testing.

def _get_rtc_configuration():
    try:
        from twilio.rest import Client
        sid   = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
        token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        if sid and token:
            client    = Client(sid, token)
            token_obj = client.tokens.create()
            return {"iceServers": token_obj.ice_servers}
    except Exception as e:
        print(f"Twilio TURN fetch failed, falling back to STUN: {e}")

    # Fallback: public STUN only (works on most networks)
    return {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
        ]
    }


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _get_status_color(status: str, good_value: str = "Normal") -> str:
    return "#2ecc71" if status == good_value else "#e74c3c"


def _get_health_color(score: float) -> str:
    if score >= 75:   return "#2ecc71"
    elif score >= 50: return "#f39c12"
    return "#e74c3c"


def _metric_card(label: str, value: str, color: str, sub_text: str = ""):
    sub_html = (
        f"<p style='font-size:12px;color:#aaa;margin:2px 0 0 0;'>{sub_text}</p>"
        if sub_text else ""
    )
    st.markdown(
        f"""
        <div style="
            background:#1e2130;
            border-left:4px solid {color};
            border-radius:8px;
            padding:14px 16px;
            margin-bottom:10px;
        ">
            <p style="font-size:11px;color:#aaa;margin:0 0 4px 0;
                      text-transform:uppercase;letter-spacing:1px;">{label}</p>
            <p style="font-size:24px;font-weight:bold;color:{color};margin:0;">{value}</p>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics_panel(result: FrameResult, eye_model_name: str, posture_model_name: str):
    st.markdown(
        "<p style='font-size:13px;font-weight:600;color:#ccc;"
        "text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>"
        "Live Analysis</p>",
        unsafe_allow_html=True,
    )

    _metric_card(
        "Eye Status",
        result.eye_status,
        _get_status_color(result.eye_status, "Normal"),
        f"EAR: {result.ear_value:.3f}",
    )
    _metric_card(
        "Posture Status",
        result.posture_status,
        _get_status_color(result.posture_status, "Good"),
        f"Neck angle: {result.posture_angle:.1f} deg",
    )
    _metric_card(
        "Health Score",
        f"{result.health_score:.0f} / 100",
        _get_health_color(result.health_score),
    )

    st.markdown(
        f"""
        <div style="font-size:11px;color:#aaa;margin-top:10px;
                    background:#1e2130;border-radius:6px;padding:10px;line-height:1.8;">
            <b>Eye model</b>: {eye_model_name}<br>
            <b>Posture model</b>: {posture_model_name}<br>
            Eye latency: {result.eye_latency_ms:.1f} ms<br>
            Posture latency: {result.posture_latency_ms:.1f} ms
        </div>
        """,
        unsafe_allow_html=True,
    )

    face_color = "#2ecc71" if result.face_detected else "#e74c3c"
    face_text  = "✓ Face Detected" if result.face_detected else "✗ No Face"
    st.markdown(
        f"""
        <div style="margin-top:8px;padding:8px 12px;
                    background:{face_color}22;border-radius:6px;
                    border:1px solid {face_color};
                    color:{face_color};font-size:12px;font-weight:600;">
            {face_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main tab renderer ──────────────────────────────────────────────────────────

def render_monitoring_tab(
    processor,
    eye_model_name: str,
    posture_model_name: str,
    user_id: int,
):
    st.header("Live Monitoring")

    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        from utils.frame_processor import VisionMateTransformer, WEBRTC_AVAILABLE
        if not WEBRTC_AVAILABLE:
            raise ImportError("VisionMateTransformer unavailable")
    except ImportError as err:
        st.error(
            f"streamlit-webrtc is not installed or failed to import: {err}\n"
            "Add `streamlit-webrtc==0.47.0` and `av` to requirements.txt."
        )
        return

    # ── Session controls ───────────────────────────────────────────────────────
    col_start, col_stop, col_voice = st.columns([1, 1, 2])

    with col_start:
        if st.button("Start Session", use_container_width=True, key="mon_start"):
            st.session_state["monitoring_active"] = True
            st.session_state["session_start"]     = time.time()
            voice_guidance.reset_all()

    with col_stop:
        if st.button("Stop Session", use_container_width=True, key="mon_stop"):
            st.session_state["monitoring_active"] = False
            voice_guidance.reset_all()

    with col_voice:
        if st.button("Test Voice Alert", use_container_width=True, key="mon_voice"):
            voice_guidance.speak_now("break_reminder")

    st.divider()

    if not st.session_state.get("monitoring_active", False):
        st.info("Click **Start Session** to begin live monitoring.")
        return

    # ── Load MediaPipe landmarkers once per server session ─────────────────────
    if "mp_landmarkers" not in st.session_state:
        with st.spinner("Loading MediaPipe landmarkers…"):
            st.session_state["mp_landmarkers"] = load_mediapipe_landmarkers()
    face_lm, pose_lm = st.session_state["mp_landmarkers"]

    # ── Resolve AI models ──────────────────────────────────────────────────────
    from utils.model_loader import load_all_eye_models, load_all_posture_models

    if "eye_models_rt" not in st.session_state:
        st.session_state["eye_models_rt"]     = load_all_eye_models()
        st.session_state["posture_models_rt"] = load_all_posture_models()

    eye_model     = st.session_state["eye_models_rt"].get(eye_model_name)
    posture_model = st.session_state["posture_models_rt"].get(posture_model_name)

    # ── CSS: constrain the webrtc widget to 55% width so it stays left ───────
    st.markdown(
        """
        <style>
        /* Shrink the webrtc video container to ~55% of page width */
        div[data-testid="stVerticalBlock"] > div:has(> iframe) {
            max-width: 55% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── webrtc_streamer MUST be called at the top level, not inside a column ──
    # Calling it inside st.columns causes the double-box bug seen in the screenshot.
    rtc_config = RTCConfiguration(_get_rtc_configuration())

    ctx = webrtc_streamer(
        key="visionmate-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_transformer_factory=VisionMateTransformer,
        media_stream_constraints={
            "video": {"width": {"ideal": 480}, "height": {"ideal": 360}},
            "audio": False,
        },
        async_processing=True,
    )

    # ── Inject models into the running transformer ─────────────────────────────
    if ctx.video_transformer:
        t = ctx.video_transformer
        t.face_landmarker    = face_lm
        t.pose_landmarker    = pose_lm
        t.eye_model          = eye_model
        t.eye_model_name     = eye_model_name
        t.posture_model      = posture_model
        t.posture_model_name = posture_model_name

    # ── Two-column layout: left = session info, right = live analysis ──────────
    feed_col, analysis_col = st.columns([3, 2])

    with feed_col:
        if "session_start" in st.session_state:
            elapsed    = int(time.time() - st.session_state["session_start"])
            mins, secs = divmod(elapsed, 60)
            hrs,  mins = divmod(mins, 60)
            timer_str  = (
                f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs > 0
                else f"{mins:02d}:{secs:02d}"
            )
            st.caption(f"Session duration: {timer_str}")

    with analysis_col:
        result = ctx.video_transformer.get_result() if ctx.video_transformer else FrameResult()
        _render_metrics_panel(result, eye_model_name, posture_model_name)

        # ── Voice guidance ─────────────────────────────────────────────────────
        voice_guidance.update_condition("eye_strain", result.eye_status     == "Strained")
        voice_guidance.update_condition("slouching",  result.posture_status == "Slouching")
        if "session_start" in st.session_state:
            session_mins = (time.time() - st.session_state["session_start"]) / 60.0
            voice_guidance.update_condition("break_reminder", session_mins > 20)

        # ── Persist metrics to DB every 5 s ───────────────────────────────────
        last_save = st.session_state.get("last_metric_save", 0)
        if time.time() - last_save >= 5 and result.face_detected:
            save_health_metric(
                user_id              = user_id,
                eye_status           = result.eye_status,
                ear_value            = result.ear_value,
                posture_status       = result.posture_status,
                posture_angle        = result.posture_angle,
                health_score         = result.health_score,
                active_eye_model     = eye_model_name,
                active_posture_model = posture_model_name,
            )
            st.session_state["last_metric_save"] = time.time()

    # ── Auto-rerun every 2 s to refresh analysis panel ────────────────────────
    if st.session_state.get("monitoring_active", False):
        time.sleep(2)
        st.rerun()
