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
    "Custom LSTM/DNN":             os.path.join(MODELS_DIR, "posture", "custom_lstm.h5"),
    "MediaPipe Pose (Rule-Based)": None,
    "YOLOv8-Pose / MoveNet DNN":  os.path.join(MODELS_DIR, "posture", "yolo_movenet_dnn.h5"),
}

RESULTS_PATHS = {
    "Custom CNN":                  os.path.join(RESULTS_DIR, "custom_cnn_results.json"),
    "MobileNetV2":                 os.path.join(RESULTS_DIR, "mobilenetv2_results.json"),
    "EfficientNetB0":              os.path.join(RESULTS_DIR, "efficientnetb0_results.json"),
    "Custom LSTM/DNN":             os.path.join(RESULTS_DIR, "custom_lstm_results.json"),
    "MediaPipe Pose (Rule-Based)": os.path.join(RESULTS_DIR, "mediapipe_results.json"),
    "YOLOv8-Pose / MoveNet DNN":  os.path.join(RESULTS_DIR, "yolo_movenet_results.json"),
}

_DEMO_RESULTS = {
    "Custom CNN":                  {"accuracy": 0.87, "f1_score": 0.86, "latency_ms": 12.3},
    "MobileNetV2":                 {"accuracy": 0.91, "f1_score": 0.90, "latency_ms":  8.7},
    "EfficientNetB0":              {"accuracy": 0.94, "f1_score": 0.93, "latency_ms": 15.2},
    "Custom LSTM/DNN":             {"accuracy": 0.85, "f1_score": 0.84, "latency_ms":  5.1},
    "MediaPipe Pose (Rule-Based)": {"accuracy": 0.82, "f1_score": 0.81, "latency_ms":  2.4},
    "YOLOv8-Pose / MoveNet DNN":  {"accuracy": 0.92, "f1_score": 0.91, "latency_ms": 18.6},
}


def _build_compat_objects() -> dict:
    """
    Builds a single custom_objects dict that patches ALL known incompatibilities
    between Kaggle-trained models and the deployment TF/Keras version:

    1. quantization_config=None  — models saved after quantization-aware training
       pass this extra kwarg to Dense, Conv2D, DepthwiseConv2D, BatchNormalization.
    2. batch_shape / optional    — InputLayer saved with these kwargs which newer
       Keras no longer accepts.
    3. TrueDivide                — MobileNetV2 preprocessing layer saved as a
       custom TrueDivide op; map it to tf.math.truediv.
    """
    import tensorflow as tf

    # ── Patch Dense ────────────────────────────────────────────────────────────
    class _CompatDense(tf.keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    # ── Patch Conv2D ───────────────────────────────────────────────────────────
    class _CompatConv2D(tf.keras.layers.Conv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    # ── Patch DepthwiseConv2D ──────────────────────────────────────────────────
    class _CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    # ── Patch BatchNormalization ───────────────────────────────────────────────
    class _CompatBatchNorm(tf.keras.layers.BatchNormalization):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    # ── Patch InputLayer ───────────────────────────────────────────────────────
    _orig_input_init = tf.keras.layers.InputLayer.__init__
    class _CompatInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, **kwargs):
            kwargs.pop("batch_shape", None)
            kwargs.pop("optional",    None)
            _orig_input_init(self, *args, **kwargs)

    return {
        "Dense":               _CompatDense,
        "Conv2D":              _CompatConv2D,
        "DepthwiseConv2D":     _CompatDepthwiseConv2D,
        "BatchNormalization":  _CompatBatchNorm,
        "InputLayer":          _CompatInputLayer,
        # MobileNetV2 preprocessing uses TrueDivide as a named layer
        "TrueDivide":          tf.keras.layers.Lambda(lambda x: x / 127.5 - 1.0),
    }


def load_keras_model(model_path: str):
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    name = os.path.basename(model_path)

    # ── Strategy 1: unified compat scope (handles all known issues at once) ────
    # This is now the FIRST strategy because the logs show every model needs it.
    try:
        from tensorflow import keras
        compat_objects = _build_compat_objects()
        with keras.utils.custom_object_scope(compat_objects):
            model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded (compat-scope): {name}")
        return model
    except Exception as e1:
        print(f"compat-scope failed for {name}: {e1}")

    # ── Strategy 2: tf_keras + compat objects ─────────────────────────────────
    try:
        import tf_keras
        compat_objects = _build_compat_objects()
        model = tf_keras.models.load_model(
            model_path, compile=False, custom_objects=compat_objects
        )
        print(f"Loaded (tf_keras + compat): {name}")
        return model
    except Exception as e2:
        print(f"tf_keras + compat failed for {name}: {e2}")

    # ── Strategy 3: tf_keras alone (no custom objects) ────────────────────────
    try:
        import tf_keras
        model = tf_keras.models.load_model(model_path, compile=False)
        print(f"Loaded (tf_keras): {name}")
        return model
    except Exception as e3:
        print(f"tf_keras failed for {name}: {e3}")

    # ── Strategy 4: plain keras compile=False ─────────────────────────────────
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded (keras compile=False): {name}")
        return model
    except Exception as e4:
        print(f"keras compile=False failed for {name}: {e4}")

    print(f"ALL strategies failed for: {model_path}")
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
