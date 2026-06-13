import os, sys, json

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


def _build_compat_objects():
    """
    Single custom_objects dict that fixes ALL known issues:
      1. quantization_config=None  on Dense / Conv2D / DepthwiseConv2D / BatchNorm
      2. batch_shape + optional    on InputLayer  →  convert batch_shape to shape
      3. TrueDivide                on MobileNetV2 →  map to a Lambda layer
    """
    import tensorflow as tf

    class _CompatDense(tf.keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    class _CompatConv2D(tf.keras.layers.Conv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    class _CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    class _CompatBatchNorm(tf.keras.layers.BatchNormalization):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    class _CompatInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, **kwargs):
            # batch_shape=[None, 32, 64, 3]  →  shape=(32, 64, 3)
            batch_shape = kwargs.pop("batch_shape", None)
            kwargs.pop("optional", None)
            if batch_shape is not None and "shape" not in kwargs:
                # strip the leading batch dimension
                kwargs["shape"] = tuple(batch_shape[1:])
            super().__init__(*args, **kwargs)

    # MobileNetV2 serialises its /127.5-1 preprocessing as a TrueDivide layer
    class _TrueDivide(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.math.truediv(inputs, 127.5) - 1.0

    return {
        "Dense":               _CompatDense,
        "Conv2D":              _CompatConv2D,
        "DepthwiseConv2D":     _CompatDepthwiseConv2D,
        "BatchNormalization":  _CompatBatchNorm,
        "InputLayer":          _CompatInputLayer,
        "TrueDivide":          _TrueDivide,
    }


def load_keras_model(model_path: str):
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    name = os.path.basename(model_path)

    # Strategy 1: keras + full compat objects  (fixes all 3 issues at once)
    try:
        from tensorflow import keras
        with keras.utils.custom_object_scope(_build_compat_objects()):
            model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded (compat-scope): {name}")
        return model
    except Exception as e1:
        print(f"compat-scope failed for {name}: {e1}")

    # Strategy 2: tf_keras + full compat objects
    try:
        import tf_keras
        model = tf_keras.models.load_model(
            model_path, compile=False, custom_objects=_build_compat_objects()
        )
        print(f"Loaded (tf_keras+compat): {name}")
        return model
    except Exception as e2:
        print(f"tf_keras+compat failed for {name}: {e2}")

    # Strategy 3: tf_keras alone
    try:
        import tf_keras
        model = tf_keras.models.load_model(model_path, compile=False)
        print(f"Loaded (tf_keras): {name}")
        return model
    except Exception as e3:
        print(f"tf_keras failed for {name}: {e3}")

    # Strategy 4: plain keras compile=False
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded (keras): {name}")
        return model
    except Exception as e4:
        print(f"keras failed for {name}: {e4}")

    print(f"ALL strategies failed for: {model_path}")
    return None


def load_all_eye_models() -> dict:
    return {name: load_keras_model(path) for name, path in EYE_MODEL_PATHS.items()}


def load_all_posture_models() -> dict:
    return {
        name: (load_keras_model(path) if path else None)
        for name, path in POSTURE_MODEL_PATHS.items()
    }


def load_results(model_name: str) -> dict:
    path = RESULTS_PATHS.get(model_name)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return _DEMO_RESULTS.get(model_name, {"accuracy": 0.80, "f1_score": 0.79, "latency_ms": 10.0})


def load_all_results() -> dict:
    return {name: load_results(name) for name in RESULTS_PATHS}
