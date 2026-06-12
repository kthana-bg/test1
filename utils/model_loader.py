import os
import sys
import json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MODELS_DIR  = os.path.join(_ROOT, "models")
RESULTS_DIR = os.path.join(_ROOT, "results")

# Paths unchanged
EYE_MODEL_PATHS = {
    "Custom CNN":     os.path.join(MODELS_DIR, "eye_strain", "custom_cnn.h5"),
    "MobileNetV2":    os.path.join(MODELS_DIR, "eye_strain", "mobilenetv2.h5"),
    "EfficientNetB0": os.path.join(MODELS_DIR, "eye_strain", "efficientnetb0.h5"),
}

POSTURE_MODEL_PATHS = {
    "Custom LSTM/DNN":           os.path.join(MODELS_DIR, "posture", "custom_lstm.h5"),
    "MediaPipe Pose (Rule-Based)": None,
    "YOLOv8-Pose / MoveNet DNN": os.path.join(MODELS_DIR, "posture", "yolo_movenet_dnn.h5"),
}

RESULTS_PATHS = {
    "Custom CNN":                  os.path.join(RESULTS_DIR, "custom_cnn_results.json"),
    "MobileNetV2":                 os.path.join(RESULTS_DIR, "mobilenetv2_results.json"),
    "EfficientNetB0":              os.path.join(RESULTS_DIR, "efficientnetb0_results.json"),
    "Custom LSTM/DNN":             os.path.join(RESULTS_DIR, "custom_lstm_results.json"),
    "MediaPipe Pose (Rule-Based)": os.path.join(RESULTS_DIR, "mediapipe_results.json"),
    "YOLOv8-Pose / MoveNet DNN":   os.path.join(RESULTS_DIR, "yolo_movenet_results.json"),
}

_DEMO_RESULTS = {
    "Custom CNN":                  {"accuracy": 0.87, "f1_score": 0.86, "latency_ms": 12.3},
    "MobileNetV2":                 {"accuracy": 0.91, "f1_score": 0.90, "latency_ms": 8.7},
    "EfficientNetB0":              {"accuracy": 0.94, "f1_score": 0.93, "latency_ms": 15.2},
    "Custom LSTM/DNN":             {"accuracy": 0.85, "f1_score": 0.84, "latency_ms": 5.1},
    "MediaPipe Pose (Rule-Based)": {"accuracy": 0.82, "f1_score": 0.81, "latency_ms": 2.4},
    "YOLOv8-Pose / MoveNet DNN":   {"accuracy": 0.92, "f1_score": 0.91, "latency_ms": 18.6},
}

def load_keras_model(model_path: str):
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    # Strategy 1: Attempt standard clean load first
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded (Standard): {os.path.basename(model_path)}")
        return model
    except Exception:
        pass

    # Strategy 2: Ultra Fallback for legacy environment parsing Keras 3 files
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        # Intercept and scrub dictionary configs globally on deserialization
        from tensorflow.python.keras.layers import deserialize as deserialize_layer
        
        original_deserialize = deserialize_layer
        
        def scrubbed_deserialize(config, custom_objects=None):
            if isinstance(config, dict) and "config" in config:
                inner_cfg = config["config"]
                if isinstance(inner_cfg, dict):
                    # Strip Keras 3 specific properties
                    inner_cfg.pop("batch_shape", None)
                    inner_cfg.pop("optional", None)
                    inner_cfg.pop("quantization_config", None)
                    if "dtype" in inner_cfg and isinstance(inner_cfg["dtype"], dict):
                        inner_cfg["dtype"] = inner_cfg["dtype"].get("config", {}).get("name", "float32")
            return original_deserialize(config, custom_objects)
            
        # Temporarily inject the structural scrubbing monkey-patch
        import tensorflow.python.keras.layers as legacy_layers
        legacy_layers.deserialize = scrubbed_deserialize
        
        model = keras.models.load_model(model_path, compile=False)
        
        # Restore normal state
        legacy_layers.deserialize = original_deserialize
        print(f"Loaded via config scrubbing: {os.path.basename(model_path)}")
        return model
    except Exception as final_err:
        print(f"All loading approaches failed for {os.path.basename(model_path)}: {final_err}")
        
    return None

def load_all_eye_models() -> dict:
    return {name: load_keras_model(path) for name, path in EYE_MODEL_PATHS.items()}

def load_all_posture_models() -> dict:
    return {name: (load_keras_model(path) if path else None) for name, path in POSTURE_MODEL_PATHS.items()}

def load_results(model_name: str) -> dict:
    path = RESULTS_PATHS.get(model_name)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return _DEMO_RESULTS.get(model_name, {"accuracy": 0.80, "f1_score": 0.79, "latency_ms": 10.0})

def load_all_results() -> dict:
    return {name: load_results(name) for name in RESULTS_PATHS}
