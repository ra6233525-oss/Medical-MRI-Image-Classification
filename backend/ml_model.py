"""
NeuroScan AI — VGG16 inference and Grad-CAM explainability engine.

The architecture and weights are loaded from the existing artifacts in
``backend/model``.  No retraining or architecture substitution is performed.
"""

import base64
import io
import json
import os
import threading
import time

import numpy as np
from PIL import Image, UnidentifiedImageError

# TensorFlow is imported lazily at module import time, matching the original
# project.  The environment variable is set by the run scripts when needed.
import tensorflow as tf
import keras
from keras.models import model_from_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_CONFIG = os.path.join(MODEL_DIR, "config.json")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "model.weights.h5")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

# Folder ordering used during training.  This mapping must not be changed
# without retraining the model.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor Detected by AI Model.",
    "pituitary": "Pituitary Tumour",
}

MODEL_NAME = "VGG16 Transfer Learning"
GRADCAM_LAYER = "block5_conv3"
INPUT_WIDTH = 128
INPUT_HEIGHT = 128
CHANNELS = 3

MODEL_ARCHITECTURE = (
    "VGG16 (ImageNet transfer learning, 13 convolutional layers) → "
    "Flatten → Dropout(0.30) → Dense(128, ReLU) → Dropout(0.20) → "
    "Dense(4, Softmax)"
)

model = None
gradcam_model = None
_model_lock = threading.Lock()


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as a supported image."""


def get_model_metadata():
    """Returns factual metadata about the loaded model for API/report display."""
    return {
        "model_name": MODEL_NAME,
        "architecture": MODEL_ARCHITECTURE,
        "backbone": "VGG16",
        "input_shape": [INPUT_WIDTH, INPUT_HEIGHT, CHANNELS],
        "input_resize": "Bilinear",
        "normalization": "[0.0, 1.0] (pixel / 255.0)",
        "classes": list(CLASS_NAMES),
        "display_classes": dict(CLASS_DISPLAY_NAMES),
        "gradcam_layer": GRADCAM_LAYER,
        "pretrained_on": "ImageNet (VGG16 backbone)",
        "fine_tuned_blocks": "block5 (convolutional fine-tuning)",
    }


def load_model():
    """Loads the existing architecture, weights and Grad-CAM graph exactly once."""
    global model, gradcam_model

    if model is not None:
        return True

    with _model_lock:
        if model is not None:
            return True

        try:
            if not os.path.exists(MODEL_CONFIG):
                raise FileNotFoundError(f"Model config not found at: {MODEL_CONFIG}")
            if not os.path.exists(MODEL_WEIGHTS):
                raise FileNotFoundError(f"Model weights not found at: {MODEL_WEIGHTS}")

            with open(MODEL_CONFIG, "r", encoding="utf-8") as file_handle:
                config = json.load(file_handle)

            candidate = model_from_json(json.dumps(config))
            candidate.load_weights(MODEL_WEIGHTS)

            # The trained Sequential model contains the complete VGG16
            # Functional block as its first layer.
            vgg = candidate.layers[0]
            last_conv = vgg.get_layer(GRADCAM_LAYER)

            # Rebuild the prediction graph symbolically.  A deserialized
            # Sequential model does not always expose ``.output`` before its
            # first call, so the classifier head is replayed explicitly.
            grad_inputs = keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, CHANNELS))
            feature_model = keras.Model(
                inputs=vgg.input,
                outputs=[last_conv.output, vgg.output],
            )
            conv_output, features = feature_model(grad_inputs)
            logits = features
            for layer in candidate.layers[1:]:
                logits = layer(logits, training=False)
            grad_model = keras.Model(
                inputs=grad_inputs,
                outputs=[conv_output, logits],
            )

            model = candidate
            gradcam_model = grad_model
            print("Model weights loaded successfully.")
            print(f"Grad-CAM sub-model initialized at layer '{GRADCAM_LAYER}'.")
            return True
        except Exception as error:  # pragma: no cover - depends on local artifacts
            print(f"Failed to load model: {error}")
            model = None
            gradcam_model = None
            return False


def preprocess_pil(image_pil):
    """
    Applies the training preprocessing pipeline:
    RGB conversion, 128x128 bilinear resize, [0, 1] normalization, batch axis.
    """
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    image_resized = image_pil.resize(
        (INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.BILINEAR
    )
    image_array = np.asarray(image_resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def preprocess_image(image_file):
    """Loads and preprocesses an image from a path or file-like object."""
    with Image.open(image_file) as image_pil:
        image_pil = image_pil.convert("RGB")
        return preprocess_pil(image_pil)


def _jet_colormap(normalized_cam):
    """Converts a [0, 1] activation map to an RGB uint8 Jet colormap."""
    cam = np.clip(normalized_cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return (np.dstack([red, green, blue]) * 255.0).astype(np.uint8)


def _to_jpeg_data_uri(rgb_array, quality=90):
    buffer = io.BytesIO()
    Image.fromarray(rgb_array).save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def generate_gradcam_visuals(image_pil, image_array, pred_index):
    """
    Computes a genuine Grad-CAM for the predicted class.

    Returns ``(heatmap_data_uri, overlay_data_uri)``.  The heatmap is the raw
    Jet-colormapped activation map; the overlay blends it with the uploaded
    MRI.  Neither image is fabricated or altered beyond the documented
    visualization blending.
    """
    if gradcam_model is None or model is None:
        return None, None

    try:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = gradcam_model(image_array, training=False)
            tape.watch(conv_outputs)
            class_score = predictions[:, pred_index]

        gradients = tape.gradient(class_score, conv_outputs)
        if gradients is None:
            return None, None

        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_outputs[0] * pooled_gradients, axis=-1)
        cam = tf.maximum(cam, 0.0)

        maximum = tf.reduce_max(cam)
        if float(maximum) > 0.0:
            cam = cam / maximum

        cam_array = cam.numpy().astype(np.float32)

        original_width, original_height = image_pil.size
        cam_image = Image.fromarray((cam_array * 255.0).astype(np.uint8)).resize(
            (original_width, original_height), resample=Image.Resampling.BILINEAR
        )
        cam_normalized = np.asarray(cam_image, dtype=np.float32) / 255.0

        heatmap_rgb = _jet_colormap(cam_normalized)
        original_rgb = np.asarray(image_pil.convert("RGB"), dtype=np.float32)
        blended = np.clip(
            0.55 * original_rgb + 0.45 * heatmap_rgb.astype(np.float32), 0, 255
        ).astype(np.uint8)

        return _to_jpeg_data_uri(heatmap_rgb), _to_jpeg_data_uri(blended)
    except Exception as error:  # pragma: no cover - runtime safety
        print(f"Warning: Grad-CAM generation encountered an error: {error}")
        return None, None


def predict_image(image_file):
    """
    Runs genuine VGG16 inference on an uploaded MRI image.

    Returns the predicted class, confidence percentage, all four class
    probabilities, a genuine Grad-CAM heatmap and overlay, and latency.
    """
    if not load_model():
        raise RuntimeError("Machine learning model could not be loaded.")

    start_time = time.time()

    try:
        with Image.open(image_file) as opened_image:
            image_pil = opened_image.convert("RGB")
        image_array = preprocess_pil(image_pil)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError,
            SyntaxError, ValueError) as error:
        # A malformed or unsupported upload is a client error, not an
        # inference failure.  Keep the model API honest about the distinction.
        raise InvalidImageError(
            "The uploaded file is not a readable JPG or PNG image."
        ) from error

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100.0

    probabilities = {
        CLASS_NAMES[index]: round(float(predictions[index]) * 100.0, 2)
        for index in range(len(CLASS_NAMES))
    }

    heatmap_data_url, overlay_data_url = generate_gradcam_visuals(
        image_pil, image_array, predicted_index
    )

    elapsed_ms = round((time.time() - start_time) * 1000.0, 1)

    return {
        "prediction": predicted_class,
        "predicted_index": predicted_index,
        "display_classification": CLASS_DISPLAY_NAMES.get(
            predicted_class, predicted_class.title()
        ),
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "gradcam_image": overlay_data_url,
        "gradcam_heatmap": heatmap_data_url,
        "model_name": MODEL_NAME,
        "gradcam_layer": GRADCAM_LAYER,
        "processing_time_ms": elapsed_ms,
    }
