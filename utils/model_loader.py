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


def _compat_objects_for_tf_keras():
    """
    custom_objects for tf.keras (TF 2.15+).
    InputLayer: accepts  shape=  NOT input_shape=
    """
    import tensorflow as tf

    class _Dense(tf.keras.layers.Dense):
        def __init__(self, *a, **kw):
            kw.pop("quantization_config", None)
            super().__init__(*a, **kw)

    class _Conv2D(tf.keras.layers.Conv2D):
        def __init__(self, *a, **kw):
            kw.pop("quantization_config", None)
            super().__init__(*a, **kw)

    class _DWConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *a, **kw):
            kw.pop("quantization_config", None)
            super().__init__(*a, **kw)

    class _BN(tf.keras.layers.BatchNormalization):
        def __init__(self, *a, **kw):
            kw.pop("quantization_config", None)
            super().__init__(*a, **kw)

    class _InputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *a, **kw):
            # batch_shape=[None,32,64,3] → shape=(32,64,3)
            bs = kw.pop("batch_shape", None)
            kw.pop("optional", None)
            if bs is not None and "shape" not in kw and "input_shape" not in kw:
                kw["shape"] = tuple(bs[1:])   # strip batch dim
            super().__init__(*a, **kw)

    class _TrueDivide(tf.keras.layers.Layer):
        """MobileNetV2 preprocessing saved as TrueDivide op."""
        def call(self, x):
            return tf.math.truediv(tf.cast(x, tf.float32), 127.5) - 1.0
        def get_config(self):
            return super().get_config()

    return {
        "Dense":              _Dense,
        "Conv2D":             _Conv2D,
        "DepthwiseConv2D":    _DWConv2D,
        "BatchNormalization": _BN,
        "InputLayer":         _InputLayer,
        "TrueDivide":         _TrueDivide,
    }


def _compat_objects_for_tf_keras_legacy():
    """
    custom_objects for tf_keras (legacy Keras 2).
    InputLayer: accepts  input_shape=  NOT shape=
    """
    import tf_keras as tfk

    class _Dense(tfk.layers.Dense):
        def __init__(self, *a, **kw):
            kw.pop("quantization_config", None)
            super().__init__(*a, **kw)

    class _Conv2D(tfk.layers.Conv2D):
        def __init__(self, *a, **kw):
            kw.pop("quantization_config", None)
            super().__init__(*a, **kw)

    class _DWConv2D(tfk.layers.DepthwiseConv2D):
        def __init__(self, *a, **kw):
            kw.pop("quantization_config", None)
            super().__init__(*a, **kw)

    class _BN(tfk.layers.BatchNormalization):
        def __init__(self, *a, **kw):
            kw.pop("quantization_config", None)
            super().__init__(*a, **kw)

    class _InputLayer(tfk.layers.InputLayer):
        def __init__(self, *a, **kw):
            # batch_shape=[None,32,64,3] → input_shape=(32,64,3)
            bs = kw.pop("batch_shape", None)
            kw.pop("optional", None)
            if bs is not None and "input_shape" not in kw and "shape" not in kw:
                kw["input_shape"] = tuple(bs[1:])
            super().__init__(*a, **kw)

    import tensorflow as tf
    class _TrueDivide(tfk.layers.Layer):
        def call(self, x):
            return tf.math.truediv(tf.cast(x, tf.float32), 127.5) - 1.0
        def get_config(self):
            return super().get_config()

    return {
        "Dense":              _Dense,
        "Conv2D":             _Conv2D,
        "DepthwiseConv2D":    _DWConv2D,
        "BatchNormalization": _BN,
        "InputLayer":         _InputLayer,
        "TrueDivide":         _TrueDivide,
    }


def load_keras_model(model_path: str):
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    name = os.path.basename(model_path)

    # Strategy 1: tf.keras + compat objects (shape= for InputLayer)
    try:
        from tensorflow import keras
        with keras.utils.custom_object_scope(_compat_objects_for_tf_keras()):
            model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded (tf.keras compat): {name}")
        return model
    except Exception as e1:
        print(f"tf.keras compat failed [{name}]: {e1}")

    # Strategy 2: tf_keras + compat objects (input_shape= for InputLayer)
    try:
        import tf_keras
        model = tf_keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=_compat_objects_for_tf_keras_legacy(),
        )
        print(f"Loaded (tf_keras compat): {name}")
        return model
    except Exception as e2:
        print(f"tf_keras compat failed [{name}]: {e2}")

    # Strategy 3: tf_keras plain
    try:
        import tf_keras
        model = tf_keras.models.load_model(model_path, compile=False)
        print(f"Loaded (tf_keras plain): {name}")
        return model
    except Exception as e3:
        print(f"tf_keras plain failed [{name}]: {e3}")

    # Strategy 4: tf.keras plain
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded (tf.keras plain): {name}")
        return model
    except Exception as e4:
        print(f"tf.keras plain failed [{name}]: {e4}")

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
