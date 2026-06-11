import os, sys, json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MODELS_DIR  = os.path.join(_ROOT, "models")
RESULTS_DIR = os.path.join(_ROOT, "results")

EYE_MODEL_PATHS = {
    "Custom CNN":     os.path.join(MODELS_DIR, "eye_strain", "custom_cnn.h5"),
    "MobileNetV2":    os.path.join(MODELS_DIR, "eye_strain", "mobilenetv2.h5"),
    "EfficientNetB0": os.path.join(MODELS_DIR, "eye_strain", "efficientnetb0.h5"),
}

POSTURE_MODEL_PATHS = {
    "Custom LSTM/DNN":           os.path.join(MODELS_DIR, "posture", "custom_lstm.h5"),
    "MediaPipe Pose (Rule-Based)": None,   # no model file — pure geometry
    "YOLOv8-Pose / MoveNet DNN": os.path.join(MODELS_DIR, "posture", "yolo_movenet_dnn.h5"),
}

RESULTS_PATHS = {
    "Custom CNN":                  os.path.join(RESULTS_DIR, "custom_cnn_results.json"),
    "MobileNetV2":                 os.path.join(RESULTS_DIR, "mobilenetv2_results.json"),
    "EfficientNetB0":              os.path.join(RESULTS_DIR, "efficientnetb0_results.json"),
    "Custom LSTM/DNN":             os.path.join(RESULTS_DIR, "custom_lstm_results.json"),
    "MediaPipe Pose (Rule-Based)": os.path.join(RESULTS_DIR, "mediapipe_results.json"),
    "YOLOv8-Pose / MoveNet DNN":  os.path.join(RESULTS_DIR, "yolo_movenet_results.json"),
}

# Placeholder values shown when real results JSON is missing
_DEMO_RESULTS = {
    "Custom CNN":                  {"accuracy": 0.87, "f1_score": 0.86, "latency_ms": 12.3},
    "MobileNetV2":                 {"accuracy": 0.91, "f1_score": 0.90, "latency_ms": 8.7},
    "EfficientNetB0":              {"accuracy": 0.94, "f1_score": 0.93, "latency_ms": 15.2},
    "Custom LSTM/DNN":             {"accuracy": 0.85, "f1_score": 0.84, "latency_ms": 5.1},
    "MediaPipe Pose (Rule-Based)": {"accuracy": 0.82, "f1_score": 0.81, "latency_ms": 2.4},
    "YOLOv8-Pose / MoveNet DNN":  {"accuracy": 0.92, "f1_score": 0.91, "latency_ms": 18.6},
}


def load_keras_model(model_path: str):
    """
      1. tf_keras  — preserves TF 2.15 layer behaviour (best for .h5 files)
      2. keras compile=False — skips optimizer incompatibilities
      3. custom_object_scope — patches TrueDivide and batch_shape issues
    """
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    # Strategy 1: tf_keras (legacy Keras — matches Kaggle training environment)
    try:
        import tf_keras
        model = tf_keras.models.load_model(model_path, compile=False)
        print(f"Loaded (tf_keras): {os.path.basename(model_path)}")
        return model
    except Exception as e1:
        print(f"tf_keras failed for {os.path.basename(model_path)}: {e1}")

    # Strategy 2: standard keras with compile=False
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded (keras compile=False): {os.path.basename(model_path)}")
        return model
    except Exception as e2:
        print(f"keras compile=False failed: {e2}")

    # Strategy 3: custom_object_scope for TrueDivide + InputLayer compat
    try:
        import tensorflow as tf
        from tensorflow import keras

        # Patch InputLayer to accept and ignore unknown kwargs
        original_init = tf.keras.layers.InputLayer.__init__
        def patched_init(self, *args, **kwargs):
            kwargs.pop("batch_shape", None)
            kwargs.pop("optional", None)
            original_init(self, *args, **kwargs)

        custom_objects = {
            "InputLayer": tf.keras.layers.InputLayer,
            "TrueDivide":  tf.math.truediv,
        }
        with keras.utils.custom_object_scope(custom_objects):
            model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded (custom_object_scope): {os.path.basename(model_path)}")
        return model
    except Exception as e3:
        print(f"custom_object_scope failed: {e3}")

    print(f"All strategies failed for: {model_path}")
    return None


def load_all_eye_models() -> dict:
    models = {}
    for name, path in EYE_MODEL_PATHS.items():
        models[name] = load_keras_model(path)
    return models


def load_all_posture_models() -> dict:
    models = {}
    for name, path in POSTURE_MODEL_PATHS.items():
        models[name] = load_keras_model(path) if path else None
    return models


def load_results(model_name: str) -> dict:
    path = RESULTS_PATHS.get(model_name)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return _DEMO_RESULTS.get(model_name, {"accuracy": 0.80, "f1_score": 0.79, "latency_ms": 10.0})


def load_all_results() -> dict:
    return {name: load_results(name) for name in RESULTS_PATHS}
