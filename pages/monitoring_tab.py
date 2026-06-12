"""
Live Monitoring Tab
Uses streamlit-webrtc (pure WebRTC, no Twilio) to stream webcam frames
through the server, run AI model inference on every frame, and display
results in real-time. DB metrics are saved every 5 seconds.
"""

import streamlit as st
import time
import cv2
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.frame_processor import FrameResult, load_mediapipe_landmarkers
from utils.voice_guidance import voice_guidance
from database.db_manager import save_health_metric


# ── ICE server config — Fixed for Metered Port 443 ───────────────────────────
# Pulls clean credentials from st.secrets. 
try:
    METERED_USER = st.secrets["metered"]["username"].strip()
    METERED_CRED = st.secrets["metered"]["credential"].strip()
except Exception as e:
    st.error(f"Missing or misconfigured Streamlit Secrets: {e}")
    METERED_USER, METERED_CRED = "", ""

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"},
        {"urls": "stun:stun3.l.google.com:19302"},
        {"urls": "stun:stun4.l.google.com:19302"},
        # FIXED: Changed prefix from 'turn:' to 'turns:' and appended transport parameter
        # This allows WebRTC to successfully parse your Metered port 443 server.
        {
            "urls": "turns:global.relay.metered.ca:443?transport=tcp",
            "username": METERED_USER,
            "credential": METERED_CRED,
        }
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
    health_color = _get_health_color(result.health_score)
    _metric_card(
        "Health Score",
        f"{result.health_score:.0f} / 100",
        health_color,
    )
    st.markdown(
        f"""
        <div style="font-size:11px;color:#aaa;margin-top:10px;
                    background:#1e2130;border-radius:6px;padding:10px;">
            <b>Eye model</b>: {eye_model_name}<br>
            <b>Posture model</b>: {posture_model_name}<br>
            Eye latency: {result.eye_latency_ms:.1f} ms<br>
            Posture latency: {result.posture_latency_ms:.1f} ms
        </div>
        """,
        unsafe_allow_html=True,
    )
    face_color = "#2ecc71" if result.face_detected else "#e74c3c"
    face_text  = "Face Detected" if result.face_detected else "No Face"
    st.markdown(
        f"""
        <div style="margin-top:8px;padding:6px 10px;
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

    # ── Resolve AI models from the app-level cache ─────────────────────────────
    # _load_models_cached() is called once in app.py via st.cache_resource.
    # We reuse whatever was already loaded — no second disk read.
    from utils.model_loader import load_all_eye_models, load_all_posture_models

    if "eye_models_rt" not in st.session_state:
        st.session_state["eye_models_rt"]     = load_all_eye_models()
        st.session_state["posture_models_rt"] = load_all_posture_models()

    eye_model     = st.session_state["eye_models_rt"].get(eye_model_name)
    posture_model = st.session_state["posture_models_rt"].get(posture_model_name)

    # ── Start WebRTC streamer ──────────────────────────────────────────────────
    rtc_config = RTCConfiguration(RTC_CONFIGURATION)

    ctx = webrtc_streamer(
        key="visionmate-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_transformer_factory=VisionMateTransformer,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=True,
    )

    # ── Inject models into the running transformer every rerun ─────────────────
    if ctx.video_transformer:
        t = ctx.video_transformer
        t.face_landmarker    = face_lm
        t.pose_landmarker    = pose_lm
        t.eye_model          = eye_model
        t.eye_model_name     = eye_model_name
        t.posture_model      = posture_model
        t.posture_model_name = posture_model_name

    st.divider()

    # ── Metrics + session timer ────────────────────────────────────────────────
    _, metrics_col = st.columns([2, 1])

    with metrics_col:
        result = ctx.video_transformer.get_result() if ctx.video_transformer else FrameResult()
        _render_metrics_panel(result, eye_model_name, posture_model_name)

        # Session timer
        if "session_start" in st.session_state:
            elapsed    = int(time.time() - st.session_state["session_start"])
            mins, secs = divmod(elapsed, 60)
            hrs,  mins = divmod(mins, 60)
            timer_str  = (
                f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs > 0
                else f"{mins:02d}:{secs:02d}"
            )
            st.caption(f"Session duration: {timer_str}")

        # ── Voice guidance ─────────────────────────────────────────────────────
        voice_guidance.update_condition("eye_strain", result.eye_status     == "Strained")
        voice_guidance.update_condition("slouching",  result.posture_status == "Slouching")
        if "session_start" in st.session_state:
            session_mins = (time.time() - st.session_state["session_start"]) / 60.0
            voice_guidance.update_condition("break_reminder", session_mins > 20)

        # ── Persist metrics to DB every 5 s (only when face is detected) ──────
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

    # ── Auto-rerun every 2 s to refresh metrics panel ─────────────────────────
    if st.session_state.get("monitoring_active", False):
        time.sleep(2)
        st.rerun()
