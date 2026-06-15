import time
import numpy as np
import cv2
import math
import os
import joblib

# Slouching angle threshold
SLOUCH_ANGLE_THRESHOLD = 20.0

# Landmark indices
NOSE_IDX = 0
LEFT_EAR_IDX = 7
RIGHT_EAR_IDX = 8
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

# Load scaler for posture features
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "posture", "scaler.joblib")
try:
    posture_scaler = joblib.load(SCALER_PATH)
except Exception:
    posture_scaler = None

def calculate_angle(point_a: tuple, point_b: tuple, point_c: tuple) -> float:
    # Calculate angle between three points
    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return math.degrees(math.acos(cosine_angle))

def calculate_neck_tilt_angle(ear_midpoint: tuple, shoulder_midpoint: tuple) -> float:
    # Calculate neck tilt from vertical
    dx = ear_midpoint[0] - shoulder_midpoint[0]
    dy = shoulder_midpoint[1] - ear_midpoint[1]
    return math.degrees(math.atan2(abs(dx), max(dy, 1)))

def extract_pose_landmarks(pose_results, image_width: int, image_height: int) -> dict | None:
    # Extract posture landmarks
    if not pose_results.pose_landmarks:
        return None
    lm = pose_results.pose_landmarks.landmark
    
    def to_pixel(idx):
        return (int(lm[idx].x * image_width), int(lm[idx].y * image_height))
        
    return {
        "nose": to_pixel(NOSE_IDX),
        "left_ear": to_pixel(LEFT_EAR_IDX),
        "right_ear": to_pixel(RIGHT_EAR_IDX),
        "left_shoulder": to_pixel(LEFT_SHOULDER_IDX),
        "right_shoulder": to_pixel(RIGHT_SHOULDER_IDX),
    }

def classify_posture_by_angle(angle: float) -> str:
    # Rule-based posture classification
    return "Slouching" if angle > SLOUCH_ANGLE_THRESHOLD else "Good"

def extract_landmark_feature_vector(landmarks: dict) -> np.ndarray:
    # Calculate angle_y, angle_z, and emg features
    nose = landmarks["nose"]
    left_ear = landmarks["left_ear"]
    right_ear = landmarks["right_ear"]
    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]

    ear_mid = ((left_ear[0] + right_ear[0]) / 2.0, (left_ear[1] + right_ear[1]) / 2.0)
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2.0, (left_shoulder[1] + right_shoulder[1]) / 2.0)
    shoulder_width = max(abs(left_shoulder[0] - right_shoulder[0]), 1.0)

    dx_y = ear_mid[0] - shoulder_mid[0]
    dy_y = shoulder_mid[1] - ear_mid[1]
    angle_y = math.degrees(math.atan2(abs(dx_y), max(dy_y, 1.0)))

    ear_height_diff = right_ear[1] - left_ear[1]
    angle_z = math.degrees(math.atan2(abs(ear_height_diff), shoulder_width))

    emg = float(np.clip((angle_y + angle_z) / 90.0, 0.0, 1.0))
    return np.array([angle_y, angle_z, emg], dtype=np.float32)

def run_posture_model_inference(model, feature_vector: np.ndarray, model_name: str) -> dict:
    # Run posture model inference with scaling
    start = time.perf_counter()
    input_tensor = feature_vector.reshape(1, -1)
    
    if posture_scaler is not None:
        input_tensor = posture_scaler.transform(input_tensor)
        
    prediction = model.predict(input_tensor, verbose=0)[0]
    slouch_conf = float(prediction[1]) if len(prediction) > 1 else float(prediction[0])
    label = "Slouching" if slouch_conf > 0.5 else "Good"
    confidence = slouch_conf if label == "Slouching" else 1.0 - slouch_conf

    latency_ms = (time.perf_counter() - start) * 1000.0
    return {"label": label, "confidence": confidence, "latency_ms": latency_ms}

def draw_posture_overlay(frame: np.ndarray, landmarks: dict, angle: float, status: str):
    # Draw posture lines and angles
    color = (0, 255, 0) if status == "Good" else (0, 0, 255)
    left_ear = landmarks["left_ear"]
    right_ear = landmarks["right_ear"]
    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    nose = landmarks["nose"]

    ear_mid = ((left_ear[0] + right_ear[0]) // 2, (left_ear[1] + right_ear[1]) // 2)
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)

    cv2.line(frame, ear_mid, shoulder_mid, color, 2)
    cv2.line(frame, left_shoulder, right_shoulder, color, 2)
    cv2.circle(frame, nose, 5, color, -1)
    cv2.circle(frame, ear_mid, 5, color, -1)
    cv2.circle(frame, shoulder_mid, 5, color, -1)
    cv2.putText(frame, f"Angle: {angle:.1f}deg", (shoulder_mid[0] - 60, shoulder_mid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
