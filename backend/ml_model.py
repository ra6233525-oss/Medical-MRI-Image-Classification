import json
import os
import time
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras.models import model_from_json

# Configure file paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_CONFIG = os.path.join(MODEL_DIR, "config.json")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "model.weights.h5")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

model = None
gradcam_model = None

def load_model():
    """Loads the model architecture, weights, and prepares the Grad-CAM submodel."""
    global model, gradcam_model
    if model is not None:
        return True

    try:
        if not os.path.exists(MODEL_CONFIG):
            raise FileNotFoundError(f"Model config not found at: {MODEL_CONFIG}")
        if not os.path.exists(MODEL_WEIGHTS):
            raise FileNotFoundError(f"Model weights not found at: {MODEL_WEIGHTS}")

        # Load architecture
        with open(MODEL_CONFIG, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Load model from json config
        model = model_from_json(json.dumps(config))
        
        # Load weights
        model.load_weights(MODEL_WEIGHTS)
        print("Model weights loaded successfully.")

        # Build Grad-CAM feature extractor using the last convolutional layer (block5_conv3 of VGG16)
        try:
            vgg = model.layers[0]
            last_conv = vgg.get_layer("block5_conv3")
            gradcam_model = keras.Model(inputs=vgg.input, outputs=[last_conv.output, vgg.output])
            print("Grad-CAM sub-model initialized.")
        except Exception as e:
            print(f"Warning: Could not initialize Grad-CAM extractor: {e}")
            gradcam_model = None

        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

def preprocess_pil(image_pil):
    """
    Preprocesses a PIL Image according to the training specifications:
    - RGB 3-channel
    - 128x128 resolution
    - Normalized to [0.0, 1.0]
    - Batch dimension added: shape (1, 128, 128, 3)
    """
    image_resized = image_pil.resize((128, 128), Image.Resampling.BILINEAR)
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

def preprocess_image(image_file):
    """Loads and preprocesses image from a file path or file-like object."""
    image_pil = Image.open(image_file).convert("RGB")
    return preprocess_pil(image_pil)

def generate_gradcam_overlay(image_pil, image_array, pred_index):
    """
    Generates genuine Grad-CAM heatmap overlay for the specified class index.
    Computes gradients of the target class score w.r.t. the last conv layer (block5_conv3).
    Returns a data URL base64 JPEG string, or None if computation fails.
    """
    global model, gradcam_model
    if gradcam_model is None or model is None:
        return None

    try:
        with tf.GradientTape() as tape:
            conv_outputs, vgg_out = gradcam_model(image_array)
            tape.watch(conv_outputs)
            
            # Pass intermediate features through subsequent classifier layers
            x = vgg_out
            for layer in model.layers[1:]:
                x = layer(x, training=False)
            score = x[:, pred_index]

        grads = tape.gradient(score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
        cam = tf.maximum(cam, 0.0)
        
        max_val = tf.reduce_max(cam)
        if max_val > 0:
            cam = cam / max_val
        cam_np = cam.numpy()

        # Resize heatmap back to match original image dimensions
        orig_w, orig_h = image_pil.size
        cam_img = Image.fromarray((cam_np * 255.0).astype(np.uint8)).resize(
            (orig_w, orig_h), resample=Image.Resampling.BILINEAR
        )
        cam_normalized = np.array(cam_img, dtype=np.float32) / 255.0

        # Vectorized Jet colormap calculation
        r = np.clip(1.5 - np.abs(4.0 * cam_normalized - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * cam_normalized - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * cam_normalized - 1.0), 0.0, 1.0)
        color_heatmap = np.dstack([r, g, b]) * 255.0

        # Blend original image and heatmap overlay (60% MRI, 40% heat map)
        orig_rgb = np.array(image_pil.convert("RGB"), dtype=np.float32)
        blended = np.clip(0.6 * orig_rgb + 0.4 * color_heatmap, 0, 255).astype(np.uint8)

        # Encode as JPEG in memory
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="JPEG", quality=88)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    except Exception as e:
        print(f"Warning: Grad-CAM overlay generation encountered an error: {e}")
        return None

def predict_image(image_file):
    """
    Takes an image file or stream, processes it, and returns:
    - prediction: predicted tumor class name
    - confidence: confidence percentage rounded to 2 decimals
    - probabilities: breakdown of all 4 class probabilities
    - gradcam_image: base64 data URL for Grad-CAM overlay visualization
    - processing_time_ms: inference latency in milliseconds
    """
    if not load_model():
        raise RuntimeError("Machine learning model could not be loaded.")
    
    start_time = time.time()
    
    # Open PIL image
    image_pil = Image.open(image_file).convert("RGB")
    
    # Preprocess
    image_array = preprocess_pil(image_pil)
    
    # Predict
    predictions = model.predict(image_array, verbose=0)[0]
    
    # Results
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100.0
    
    # Class probabilities dictionary
    probabilities = {
        CLASS_NAMES[i]: round(float(predictions[i]) * 100.0, 2)
        for i in range(len(CLASS_NAMES))
    }
    
    # Grad-CAM heatmap visualization
    gradcam_data_url = generate_gradcam_overlay(image_pil, image_array, predicted_index)
    
    elapsed_ms = round((time.time() - start_time) * 1000.0, 1)
    
    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "gradcam_image": gradcam_data_url,
        "processing_time_ms": elapsed_ms
    }
