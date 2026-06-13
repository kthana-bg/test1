"""
Posture detection module.
Uses body landmark coordinates to compute head-neck-shoulder angles
and runs inference using whichever of the three posture models is selected.
"""

import time
import numpy as np
import cv2
import math

SLOUCH_ANGLE_THRESHOLD = 20.0

NOSE_IDX           = 0
LEFT_EAR_IDX       = 7
RIGHT_EAR_IDX      = 8
LEFT_SHOULDER_IDX  = 11
RIGHT_SHOULDER_IDX = 12


def calculate_angle(point_a: tuple, point_b: tuple, point_c: tuple) -> float:
    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return math.degrees(math.acos(cosine_angle))


def calculate_neck_tilt_angle(ear_midpoint: tuple, shoulder_midpoint: tuple) -> float:
    dx = ear_midpoint[0] - shoulder_midpoint[0]
    dy = shoulder_midpoint[1] - ear_midpoint[1]
    return math.degrees(math.atan2(abs(dx), max(dy, 1)))


def extract_pose_landmarks(pose_results, image_width: int, image_height: int) -> dict | None:
    if not pose_results.pose_landmarks:
        return None
    lm = pose_results.pose_landmarks.landmark

    def to_pixel(idx):
        return (int(lm[idx].x * image_width), int(lm[idx].y * image_height))

    return {
        "nose":           to_pixel(NOSE_IDX),
        "left_ear":       to_pixel(LEFT_EAR_IDX),
        "right_ear":      to_pixel(RIGHT_EAR_IDX),
        "left_shoulder":  to_pixel(LEFT_SHOULDER_IDX),
        "right_shoulder": to_pixel(RIGHT_SHOULDER_IDX),
    }


def classify_posture_by_angle(angle: float) -> str:
    return "Slouching" if angle > SLOUCH_ANGLE_THRESHOLD else "Good"


def extract_landmark_feature_vector(landmarks: dict) -> np.ndarray:
    """
    Returns exactly 3 features matching the training dataset (alnatron/posture-detection):
      [angle_y, angle_z, emg]

    angle_y — forward/backward head tilt (neck tilt from vertical, degrees, normalised)
    angle_z — lateral (side) head tilt (ear asymmetry relative to shoulders, degrees, normalised)
    emg     — muscle activity proxy (we have no sensor so we approximate from angle magnitude)

    All values are scaled to roughly the same range as the original sensor dataset
    (angle values ~0–90 deg, emg ~0–1) so the model outputs are meaningful.
    """
    # ── Pixel coordinates ──────────────────────────────────────────────────────
    nose           = landmarks["nose"]
    left_ear       = landmarks["left_ear"]
    right_ear      = landmarks["right_ear"]
    left_shoulder  = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]

    ear_mid = (
        (left_ear[0]  + right_ear[0])  / 2.0,
        (left_ear[1]  + right_ear[1])  / 2.0,
    )
    shoulder_mid = (
        (left_shoulder[0]  + right_shoulder[0]) / 2.0,
        (left_shoulder[1]  + right_shoulder[1]) / 2.0,
    )
    shoulder_width = max(
        abs(left_shoulder[0] - right_shoulder[0]), 1.0
    )

    # ── angle_y: forward/backward neck tilt from vertical (degrees) ───────────
    # Matches the dataset's angle_y (pitch / sagittal plane tilt).
    dx_y = ear_mid[0] - shoulder_mid[0]
    dy_y = shoulder_mid[1] - ear_mid[1]    # positive = head above shoulders
    angle_y = math.degrees(math.atan2(abs(dx_y), max(dy_y, 1.0)))

    # ── angle_z: lateral tilt — asymmetry between left and right ears ─────────
    # Matches the dataset's angle_z (roll / frontal plane tilt).
    # Positive when left ear is higher than right (head tilted right).
    ear_height_diff = right_ear[1] - left_ear[1]   # pixels, +ve = right ear lower
    angle_z = math.degrees(math.atan2(abs(ear_height_diff), shoulder_width))

    # ── emg: muscle activity proxy ────────────────────────────────────────────
    # We have no EMG sensor. We approximate as a normalised composite of
    # angle_y and angle_z, clamped to [0, 1].
    # Higher angles → more muscle effort → higher EMG proxy.
    emg = float(np.clip((angle_y + angle_z) / 90.0, 0.0, 1.0))

    return np.array([angle_y, angle_z, emg], dtype=np.float32)


def run_posture_model_inference(model, feature_vector: np.ndarray, model_name: str) -> dict:
    """
    Run posture inference using the selected model.
    feature_vector must be shape (3,): [angle_y, angle_z, emg]
    """
    start = time.perf_counter()

    # All trained posture models expect (batch, 3) — flat input, not LSTM sequence.
    # The LSTM branch inside custom_lstm handles the Reshape internally.
    input_tensor = feature_vector.reshape(1, -1)   # shape (1, 3)
    prediction   = model.predict(input_tensor, verbose=0)[0]

    # Output is [P(Good), P(Slouching/Bad)]
    slouch_conf = float(prediction[1]) if len(prediction) > 1 else float(prediction[0])
    label       = "Slouching" if slouch_conf > 0.5 else "Good"
    confidence  = slouch_conf if label == "Slouching" else 1.0 - slouch_conf

    latency_ms = (time.perf_counter() - start) * 1000.0
    return {"label": label, "confidence": confidence, "latency_ms": latency_ms}


def draw_posture_overlay(frame: np.ndarray, landmarks: dict, angle: float, status: str):
    color = (0, 255, 0) if status == "Good" else (0, 0, 255)

    left_ear       = landmarks["left_ear"]
    right_ear      = landmarks["right_ear"]
    left_shoulder  = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    nose           = landmarks["nose"]

    ear_mid = (
        (left_ear[0]  + right_ear[0])  // 2,
        (left_ear[1]  + right_ear[1])  // 2,
    )
    shoulder_mid = (
        (left_shoulder[0]  + right_shoulder[0]) // 2,
        (left_shoulder[1]  + right_shoulder[1]) // 2,
    )

    cv2.line(frame, ear_mid, shoulder_mid, color, 2)
    cv2.line(frame, left_shoulder, right_shoulder, color, 2)
    cv2.circle(frame, nose, 5, color, -1)
    cv2.circle(frame, ear_mid, 5, color, -1)
    cv2.circle(frame, shoulder_mid, 5, color, -1)
    cv2.putText(
        frame,
        f"Angle: {angle:.1f}deg",
        (shoulder_mid[0] - 60, shoulder_mid[1] + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
    )
