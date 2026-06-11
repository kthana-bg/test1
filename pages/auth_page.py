import streamlit as st
import cv2
import numpy as np
import time
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from utils.auth_utils import register_user_face, login_by_face

DISPLAY_W = 320
DISPLAY_H = 240


def _decode_camera_image(img_file) -> np.ndarray:
    """Convert st.camera_input file buffer → BGR numpy array."""
    bytes_data = img_file.getvalue()
    arr        = np.frombuffer(bytes_data, np.uint8)
    bgr        = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr


def _get_face_embedding(bgr_frame: np.ndarray):
    """
    Extract face embedding from a BGR frame.
    Returns embedding array or None if no face found / library missing.
    """
    try:
        import face_recognition
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        small  = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        locs   = face_recognition.face_locations(small, model="hog")
        if not locs:
            return None
        locs_full = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in locs]
        encs      = face_recognition.face_encodings(rgb, locs_full)
        return np.array(encs[0]) if encs else None
    except ImportError:
        # face_recognition not installed — return dummy embedding
        # so registration/login still works without face verification
        return np.zeros(128, dtype=np.float64)


def _show_static_placeholder(placeholder, text: str):
    placeholder.markdown(
        f"""
        <div style="
            background:#1a1d2e;
            border:2px dashed #3498db;
            border-radius:10px;
            width:100%;
            height:{DISPLAY_H}px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            color:#3498db;
            font-size:13px;
            text-align:center;
            padding:16px;
        ">
            <div style="font-size:36px;margin-bottom:10px;">&#128247;</div>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Login Tab 

def render_login_tab():
    st.markdown(
        "<p style='color:#aaa;text-align:center;margin-bottom:16px;'>"
        "Take a photo and click Sign In to identify yourself.</p>",
        unsafe_allow_html=True,
    )

    status_box = st.empty()

    # st.camera_input — browser captures photo, no server camera needed
    photo = st.camera_input(
        label="Take a photo to sign in",
        key="login_camera",
        label_visibility="collapsed",
    )

    if st.button(
        "Sign In with Face",
        use_container_width=True,
        key="login_face_btn",
        type="primary",
    ):
        if photo is None:
            status_box.warning("Please take a photo first using the camera above.")
            return

        status_box.info("Scanning face...")
        bgr       = _decode_camera_image(photo)
        embedding = _get_face_embedding(bgr)

        if embedding is None:
            status_box.error(
                "No face detected in the photo. "
                "Make sure your face is well-lit and centred, then retake."
            )
            return

        with st.spinner("Identifying..."):
            result = login_by_face(embedding)

        if result["success"]:
            u = result["user"]
            st.session_state["logged_in"] = True
            st.session_state["user"]      = u
            st.session_state["user_id"]   = u["id"]
            st.session_state["username"]  = u["username"]
            status_box.success(result["message"])
            time.sleep(0.6)
            st.rerun()
        else:
            status_box.error(result["message"])


# Register Tab

def render_register_tab():
    st.markdown(
        "<p style='color:#aaa;text-align:center;margin-bottom:16px;'>"
        "Enter your name, take a photo, then click Create Account.</p>",
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        full_name = st.text_input(
            "Full Name",
            key="reg_fullname",
            placeholder="e.g. Keerthana Bale Murali",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    status_box = st.empty()

    # st.camera_input for registration
    photo = st.camera_input(
        label="Take a photo to register",
        key="reg_camera",
        label_visibility="collapsed",
    )

    # Preview captured photo with face detection feedback
    if photo is not None:
        bgr      = _decode_camera_image(photo)
        rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            import face_recognition
            small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            locs  = face_recognition.face_locations(small, model="hog")
            if locs:
                # Draw green box around detected face
                t, r, b, l = [v*2 for v in locs[0]]
                cv2.rectangle(rgb, (l, t), (r, b), (0, 255, 80), 2)
                status_box.success("Face detected — ready to register.")
            else:
                status_box.warning("No face detected. Please retake the photo.")
        except ImportError:
            status_box.info("Photo captured.")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "Create Account",
        use_container_width=True,
        key="reg_submit_btn",
        type="primary",
    ):
        if not full_name or not full_name.strip():
            status_box.error("Please enter your full name.")
            return

        if photo is None:
            status_box.error("Please take a photo before registering.")
            return

        bgr       = _decode_camera_image(photo)
        embedding = _get_face_embedding(bgr)

        with st.spinner("Creating account..."):
            result = register_user_face(full_name.strip(), embedding)

        if result["success"]:
            status_box.success(
                f"Account created! Your username is: {result['username']}\n"
                "Please go to the Login tab and sign in with your face."
            )
            time.sleep(2.0)
            st.rerun()
        else:
            status_box.error(result["message"])


# Main entry point 

def render_auth_page():
    st.markdown(
        """
        <div style="text-align:center;padding:24px 0 12px 0;">
            <h1 style="color:#3498db;margin-bottom:4px;font-size:2.2rem;">
                VisionMate
            </h1>
            <p style="color:#888;font-size:14px;margin:0;">
                AI Eye-Strain Monitor and Ergonomic Coach
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            render_login_tab()

    with register_tab:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            render_register_tab()
