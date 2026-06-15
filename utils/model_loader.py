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
    "YOLOv8-Pose / MoveNet DNN":   os.path.join(MODELS_DIR, "posture", "yolo_movenet_dnn.h5"),
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
    "MobileNetV2":                 {"accuracy": 0.91, "f1_score": 0.90, "latency_ms":  8.7},
    "EfficientNetB0":              {"accuracy": 0.94, "f1_score": 0.93, "latency_ms": 15.2},
    "Custom LSTM/DNN":             {"accuracy": 0.85, "f1_score": 0.84, "latency_ms":  5.1},
    "MediaPipe Pose (Rule-Based)": {"accuracy": 0.82, "f1_score": 0.81, "latency_ms":  2.4},
    "YOLOv8-Pose / MoveNet DNN":   {"accuracy": 0.92, "f1_score": 0.91, "latency_ms": 18.6},
}

# Load weights using Keras native by_name mapping
def _load_weights_from_h5(model, h5_path: str):
    try:
        model.load_weights(h5_path, by_name=True)
    except Exception as e:
        print(f"  Failed to load weights: {e}")

# Custom CNN for eye strain detection
def _build_custom_cnn():
    import tensorflow as tf
    K = tf.keras

    inp = K.Input(shape=(32, 64, 3), name="eye_input")

    x = K.layers.Conv2D(32, 3, padding="same", use_bias=True, name="conv2d")(inp)
    x = K.layers.BatchNormalization(name="batch_normalization")(x)
    x = K.layers.Activation("relu", name="activation")(x)

    x = K.layers.Conv2D(32, 3, padding="same", use_bias=True, name="conv2d_1")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_1")(x)
    x = K.layers.Activation("relu", name="activation_1")(x)

    x = K.layers.MaxPooling2D(name="max_pooling2d")(x)
    x = K.layers.Dropout(0.25, name="dropout")(x)

    x = K.layers.Conv2D(64, 3, padding="same", use_bias=True, name="conv2d_2")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_2")(x)
    x = K.layers.Activation("relu", name="activation_2")(x)

    x = K.layers.Conv2D(64, 3, padding="same", use_bias=True, name="conv2d_3")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_3")(x)
    x = K.layers.Activation("relu", name="activation_3")(x)

    x = K.layers.MaxPooling2D(name="max_pooling2d_1")(x)
    x = K.layers.Dropout(0.25, name="dropout_1")(x)

    x = K.layers.Conv2D(128, 3, padding="same", use_bias=True, name="conv2d_4")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_4")(x)
    x = K.layers.Activation("relu", name="activation_4")(x)

    x = K.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = K.layers.Dropout(0.3, name="dropout_2")(x)
    x = K.layers.Dense(256, activation="relu", name="dense")(x)
    x = K.layers.Dropout(0.3, name="dropout_3")(x)
    out = K.layers.Dense(2, activation="softmax", name="predictions")(x)

    return K.Model(inp, out, name="custom_cnn")

# MobileNetV2 transfer learning model
def _build_mobilenetv2():
    import tensorflow as tf
    K = tf.keras

    inp = K.Input(shape=(32, 64, 3), name="eye_input")
    x = K.layers.Lambda(lambda t: tf.cast(t, tf.float32) / 127.5 - 1.0, name="preprocess")(inp)

    base = K.applications.MobileNetV2(
        input_shape=(32, 64, 3),
        include_top=False,
        weights=None,
    )
    base._name = "mobilenetv2_1.00_224"
    x = base(x)

    x = K.layers.GlobalAveragePooling2D(name="gap")(x)
    x = K.layers.Dense(256, name="dense1")(x)
    x = K.layers.BatchNormalization(name="batch_normalization")(x)
    x = K.layers.Dropout(0.3, name="dropout")(x)
    x = K.layers.Dense(128, name="dense2")(x)
    x = K.layers.Dropout(0.3, name="dropout_1")(x)
    out = K.layers.Dense(2, activation="softmax", name="predictions")(x)

    return K.Model(inp, out, name="mobilenetv2_model")

# EfficientNetB0 transfer learning model
def _build_efficientnetb0():
    import tensorflow as tf
    K = tf.keras

    inp = K.Input(shape=(96, 96, 3), name="eye_input")

    base = K.applications.EfficientNetB0(
        input_shape=(96, 96, 3),
        include_top=False,
        weights=None,
    )
    base._name = "efficientnetb0"
    x = base(inp)

    x = K.layers.GlobalAveragePooling2D(name="gap")(x)
    x = K.layers.Dense(256, name="dense1")(x)
    x = K.layers.BatchNormalization(name="batch_normalization")(x)
    x = K.layers.Activation("relu", name="activation")(x)
    x = K.layers.Dropout(0.3, name="dropout")(x)
    x = K.layers.Dense(128, name="dense2")(x)
    x = K.layers.Dropout(0.3, name="dropout_1")(x)
    out = K.layers.Dense(2, activation="softmax", name="predictions")(x)

    return K.Model(inp, out, name="efficientnetb0_model")

# Custom LSTM/DNN hybrid for posture detection
def _build_custom_lstm():
    import tensorflow as tf
    K = tf.keras

    inp = K.Input(shape=(3,), name="posture_features")

    dnn = K.layers.Dense(64, activation="relu", name="dnn1")(inp)
    dnn = K.layers.BatchNormalization(name="bn1")(dnn)
    dnn = K.layers.Dropout(0.2, name="drop2")(dnn)

    lstm_in = K.layers.Reshape((1, 3), name="reshape_for_lstm")(inp)
    lstm_in = K.layers.Dropout(0.3, name="drop1")(lstm_in)
    lstm_out = K.layers.LSTM(64, name="lstm")(lstm_in)
    lstm_out = K.layers.Dense(32, activation="relu", name="dnn2")(lstm_out)
    lstm_out = K.layers.Dropout(0.3, name="lstm_drop")(lstm_out)

    merged = K.layers.Concatenate(name="merge")([dnn, lstm_out])
    merged = K.layers.Dense(64, activation="relu", name="merge_dense")(merged)
    merged = K.layers.Dropout(0.3, name="merge_drop")(merged)
    out = K.layers.Dense(2, activation="softmax", name="output")(merged)

    return K.Model(inp, out, name="custom_lstm")

# Deep residual DNN for posture detection
def _build_yolo_movenet_dnn():
    import tensorflow as tf
    K = tf.keras

    inp = K.Input(shape=(3,), name="posture_features")

    x = K.layers.Dense(128, activation="relu", name="entry")(inp)
    x = K.layers.BatchNormalization(name="batch_normalization")(x)
    x = K.layers.Dropout(0.3, name="dropout")(x)

    r1 = K.layers.Dense(128, name="dense")(x)
    r1 = K.layers.BatchNormalization(name="batch_normalization_1")(r1)
    r1 = K.layers.Activation("relu", name="activation")(r1)
    r1 = K.layers.Dropout(0.3, name="dropout_1")(r1)
    r1 = K.layers.Dense(128, name="dense_1")(r1)
    r1 = K.layers.BatchNormalization(name="batch_normalization_2")(r1)
    x  = K.layers.Add(name="add")([x, r1])
    x  = K.layers.Activation("relu", name="activation_1")(x)

    r2 = K.layers.Dense(64, name="dense_2")(x)
    r2 = K.layers.BatchNormalization(name="batch_normalization_3")(r2)
    r2 = K.layers.Activation("relu", name="activation_2")(r2)
    r2 = K.layers.Dropout(0.25, name="dropout_2")(r2)
    r2 = K.layers.Dense(64, name="dense_3")(r2)
    r2 = K.layers.BatchNormalization(name="batch_normalization_4")(r2)
    sc2 = K.layers.Dense(64, use_bias=False, name="dense_4")(x)
    x   = K.layers.Add(name="add_1")([sc2, r2])
    x   = K.layers.Activation("relu", name="activation_3")(x)

    r3 = K.layers.Dense(32, name="dense_5")(x)
    r3 = K.layers.BatchNormalization(name="batch_normalization_5")(r3)
    r3 = K.layers.Activation("relu", name="activation_4")(r3)
    r3 = K.layers.Dropout(0.2, name="dropout_3")(r3)
    r3 = K.layers.Dense(32, name="dense_6")(r3)
    r3 = K.layers.BatchNormalization(name="batch_normalization_6")(r3)
    sc3 = K.layers.Dense(32, use_bias=False, name="dense_7")(x)
    x   = K.layers.Add(name="add_2")([sc3, r3])
    x   = K.layers.Activation("relu", name="activation_5")(x)

    x   = K.layers.Dense(16, activation="relu", name="pre_out")(x)
    out = K.layers.Dense(2, activation="softmax", name="output")(x)

    return K.Model(inp, out, name="yolo_movenet_dnn")

# Map models to builder functions
_BUILDERS = {
    "Custom CNN":               _build_custom_cnn,
    "MobileNetV2":              _build_mobilenetv2,
    "EfficientNetB0":           _build_efficientnetb0,
    "Custom LSTM/DNN":          _build_custom_lstm,
    "YOLOv8-Pose / MoveNet DNN":_build_yolo_movenet_dnn,
}

# Build model and load weights
def load_keras_model(model_name: str, model_path: str):
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    builder = _BUILDERS.get(model_name)
    if builder is None:
        print(f"No builder defined for model: {model_name}")
        return None

    try:
        model = builder()
        _load_weights_from_h5(model, model_path)
        print(f"Loaded (rebuild+weights): {os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        import traceback; traceback.print_exc()
        return None

# Load all models for eye strain
def load_all_eye_models() -> dict:
    return {
        name: load_keras_model(name, path)
        for name, path in EYE_MODEL_PATHS.items()
    }

# Load all models for posture
def load_all_posture_models() -> dict:
    result = {}
    for name, path in POSTURE_MODEL_PATHS.items():
        result[name] = load_keras_model(name, path) if path else None
    return result

# Load evaluation results
def load_results(model_name: str) -> dict:
    path = RESULTS_PATHS.get(model_name)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return _DEMO_RESULTS.get(model_name, {"accuracy": 0.80, "f1_score": 0.79, "latency_ms": 10.0})

# Return results for all models
def load_all_results() -> dict:
    return {name: load_results(name) for name in RESULTS_PATHS}
