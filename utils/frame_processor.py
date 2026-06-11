"""
Frame processor for VisionMate.
Provides:
  - process_frame()          : run face/pose landmarkers + AI model inference on one BGR frame
  - VisionMateTransformer    : streamlit-webrtc VideoTransformerBase subclass
  - load_mediapipe_landmarkers(): download + initialise MediaPipe task models
"""

import threading
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.eye_detection import (
    calculate_ear,
    extract_eye_landmarks,
    classify_eye_status_by_ear,
    draw_eye_landmarks,
    run_eye_model_inference,
    get_eye_roi,
)
from utils.posture_detection import (
    calculate_neck_tilt_angle,
    classify_posture_by_angle,
    draw_posture_overlay,
    extract_landmark_feature_vector,
    run_posture_model_inference,
)

ASSETS_DIR      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
FACE_MODEL_PATH = os.path.join(ASSETS_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(ASSETS_DIR, "pose_landmarker_lite.task")

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """Holds the output of one processed frame."""
    frame_bgr:          Optional[np.ndarray] = None
    eye_status:         str   = "Unknown"
    ear_value:          float = 0.0
    eye_confidence:     float = 0.0
    eye_latency_ms:     float = 0.0
    posture_status:     str   = "Unknown"
    posture_angle:      float = 0.0
    posture_confidence: float = 0.0
    posture_latency_ms: float = 0.0
    health_score:       float = 100.0
    face_detected:      bool  = False
    timestamp:          float = field(default_factory=time.time)


def compute_health_score(eye_status: str, posture_status: str) -> float:
    eye_score     = 50.0
    posture_score = 50.0
    if eye_status     == "Strained":  eye_score     = 20.0
    elif eye_status   == "Blinking":  eye_score     = 40.0
    if posture_status == "Slouching": posture_score = 20.0
    return eye_score + posture_score


# ── MediaPipe loader ───────────────────────────────────────────────────────────

def _download_model(url: str, dest_path: str) -> bool:
    if os.path.exists(dest_path):
        return True
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        import urllib.request
        print(f"Downloading {os.path.basename(dest_path)} ...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded: {dest_path}")
        return True
    except Exception as e:
        print(f"Model download failed: {e}")
        return False


def load_mediapipe_landmarkers():
    """Download (if needed) and initialise MediaPipe FaceLandmarker and PoseLandmarker."""
    face_lm = None
    pose_lm = None
    try:
        import mediapipe as mp
        BaseOptions = mp.tasks.BaseOptions
        vision      = mp.tasks.vision
        RunningMode = vision.RunningMode

        if _download_model(FACE_MODEL_URL, FACE_MODEL_PATH):
            face_opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            face_lm = vision.FaceLandmarker.create_from_options(face_opts)
            print("FaceLandmarker loaded")

        if _download_model(POSE_MODEL_URL, POSE_MODEL_PATH):
            pose_opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)
            print("PoseLandmarker loaded")

    except Exception as e:
        print(f"MediaPipe load error: {e}")

    return face_lm, pose_lm


# ── Core frame processing ──────────────────────────────────────────────────────

def process_frame(
    frame_bgr: np.ndarray,
    face_landmarker,
    pose_landmarker,
    eye_model,
    eye_model_name: str,
    posture_model,
    posture_model_name: str,
    ear_consec_counter: int,
) -> tuple:
    """
    Run detection + inference on a single BGR frame.
    Returns (FrameResult, updated_ear_consec_counter).
    """
    import mediapipe as mp

    result           = FrameResult()
    result.timestamp = time.time()
    h, w             = frame_bgr.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
    )

    # ── Eye Detection ──────────────────────────────────────────────────────────
    eye_status    = "Unknown"
    ear_val       = 0.0
    eye_conf      = 0.0
    eye_lat       = 0.0
    face_detected = False

    if face_landmarker is not None:
        try:
            face_result = face_landmarker.detect(mp_image)
            if face_result.face_landmarks:
                face_detected = True
                landmarks_478 = face_result.face_landmarks[0]

                class _LMProxy:
                    def __init__(self, lms): self.landmark = lms
                proxy = _LMProxy(landmarks_478)

                left_eye, right_eye = extract_eye_landmarks(proxy, w, h)
                ear_val = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

                # EAR rule gives a fallback status; model overrides when loaded
                eye_status, ear_consec_counter = classify_eye_status_by_ear(
                    ear_val, ear_consec_counter
                )

                # Use trained model when available (takes priority over EAR rules)
                use_model = (
                    eye_model is not None
                    and "Rule"      not in eye_model_name
                    and "MediaPipe" not in eye_model_name
                )
                if use_model:
                    roi = get_eye_roi(frame_bgr, left_eye)
                    if roi is not None:
                        try:
                            inf        = run_eye_model_inference(eye_model, roi, eye_model_name)
                            eye_status = inf["label"]
                            eye_conf   = inf["confidence"]
                            eye_lat    = inf["latency_ms"]
                        except Exception as me:
                            print(f"Eye model inference error: {me}")

                draw_eye_landmarks(frame_bgr, left_eye, right_eye)

        except Exception as e:
            print(f"Face detection error: {e}")

    # ── Posture Detection ──────────────────────────────────────────────────────
    posture_status = "Unknown"
    posture_angle  = 0.0
    posture_conf   = 0.0
    posture_lat    = 0.0

    if pose_landmarker is not None:
        try:
            pose_result = pose_landmarker.detect(mp_image)
            if pose_result.pose_landmarks:
                lms = pose_result.pose_landmarks[0]

                def to_px(idx):
                    return (int(lms[idx].x * w), int(lms[idx].y * h))

                lm_dict = {
                    "nose":           to_px(0),
                    "left_ear":       to_px(7),
                    "right_ear":      to_px(8),
                    "left_shoulder":  to_px(11),
                    "right_shoulder": to_px(12),
                }
                ear_mid = (
                    (lm_dict["left_ear"][0]  + lm_dict["right_ear"][0])  // 2,
                    (lm_dict["left_ear"][1]  + lm_dict["right_ear"][1])  // 2,
                )
                shoulder_mid = (
                    (lm_dict["left_shoulder"][0] + lm_dict["right_shoulder"][0]) // 2,
                    (lm_dict["left_shoulder"][1] + lm_dict["right_shoulder"][1]) // 2,
                )
                posture_angle  = calculate_neck_tilt_angle(ear_mid, shoulder_mid)
                posture_status = classify_posture_by_angle(posture_angle)

                # Use trained model when available (takes priority over angle rules)
                use_model = (
                    posture_model is not None
                    and "Rule"      not in posture_model_name
                    and "MediaPipe" not in posture_model_name
                )
                if use_model:
                    try:
                        feat_vec       = extract_landmark_feature_vector(lm_dict)
                        inf            = run_posture_model_inference(posture_model, feat_vec, posture_model_name)
                        posture_status = inf["label"]
                        posture_conf   = inf["confidence"]
                        posture_lat    = inf["latency_ms"]
                    except Exception as me:
                        print(f"Posture model inference error: {me}")

                draw_posture_overlay(frame_bgr, lm_dict, posture_angle, posture_status)

        except Exception as e:
            print(f"Pose detection error: {e}")

    # ── Overlay text ───────────────────────────────────────────────────────────
    health_score  = compute_health_score(eye_status, posture_status)
    eye_color     = (0, 255, 0) if eye_status     == "Normal" else (0, 0, 255)
    posture_color = (0, 255, 0) if posture_status == "Good"   else (0, 0, 255)

    cv2.putText(frame_bgr, f"Eye: {eye_status}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
    cv2.putText(frame_bgr, f"EAR: {ear_val:.3f}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame_bgr, f"Posture: {posture_status}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, posture_color, 2)
    cv2.putText(frame_bgr, f"Health: {health_score:.0f}/100",
                (10, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    result.frame_bgr          = frame_bgr
    result.eye_status         = eye_status
    result.ear_value          = ear_val
    result.eye_confidence     = eye_conf
    result.eye_latency_ms     = eye_lat
    result.posture_status     = posture_status
    result.posture_angle      = posture_angle
    result.posture_confidence = posture_conf
    result.posture_latency_ms = posture_lat
    result.health_score       = health_score
    result.face_detected      = face_detected

    return result, ear_consec_counter


# ── WebRTC Video Transformer ───────────────────────────────────────────────────

try:
    from streamlit_webrtc import VideoTransformerBase
    import av

    class VisionMateTransformer(VideoTransformerBase):
        """
        Called by streamlit-webrtc for every frame received from the browser.
        Runs inference on every FRAME_SKIP-th frame and returns the annotated frame.
        Stores the latest FrameResult in self._result for the metrics panel.
        """

        FRAME_SKIP = 3  # process every 3rd frame to keep latency low

        def __init__(self):
            self._result      = FrameResult()
            self._lock        = threading.Lock()
            self._ear_consec  = 0
            self._frame_count = 0

            # Injected from monitoring_tab after construction
            self.face_landmarker    = None
            self.pose_landmarker    = None
            self.eye_model          = None
            self.eye_model_name     = "Custom CNN"
            self.posture_model      = None
            self.posture_model_name = "Custom LSTM/DNN"

        def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
            img_bgr = frame.to_ndarray(format="bgr24")
            img_bgr = cv2.flip(img_bgr, 1)  # mirror so it feels natural

            self._frame_count += 1

            if self._frame_count % self.FRAME_SKIP == 0:
                with self._lock:
                    face_lm      = self.face_landmarker
                    pose_lm      = self.pose_landmarker
                    eye_m        = self.eye_model
                    eye_mn       = self.eye_model_name
                    posture_m    = self.posture_model
                    posture_mn   = self.posture_model_name
                    ear_consec   = self._ear_consec

                fr, new_ear_consec = process_frame(
                    img_bgr,
                    face_lm, pose_lm,
                    eye_m, eye_mn,
                    posture_m, posture_mn,
                    ear_consec,
                )

                with self._lock:
                    self._result    = fr
                    self._ear_consec = new_ear_consec

                img_bgr = fr.frame_bgr if fr.frame_bgr is not None else img_bgr

            else:
                # On skipped frames, re-draw last known status text quickly
                with self._lock:
                    last = self._result
                eye_color     = (0, 255, 0) if last.eye_status     == "Normal" else (0, 0, 255)
                posture_color = (0, 255, 0) if last.posture_status == "Good"   else (0, 0, 255)
                cv2.putText(img_bgr, f"Eye: {last.eye_status}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
                cv2.putText(img_bgr, f"EAR: {last.ear_value:.3f}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img_bgr, f"Posture: {last.posture_status}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, posture_color, 2)
                cv2.putText(img_bgr, f"Health: {last.health_score:.0f}/100",
                            (10, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

        def get_result(self) -> FrameResult:
            with self._lock:
                return self._result

    WEBRTC_AVAILABLE = True

except ImportError:
    WEBRTC_AVAILABLE = False
    VisionMateTransformer = None
