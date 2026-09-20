import json
import os
import numpy as np
from PIL import Image
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

def load_model():
    """Loads the model architecture and weights."""
    global model
    if model is not None:
        return True

    try:
        # Load architecture
        with open(MODEL_CONFIG, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Load model from json config
        model = model_from_json(json.dumps(config))
        
        # Load weights
        model.load_weights(MODEL_WEIGHTS)
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

def preprocess_image(image_file):
    """
    Preprocesses the image for the current model.
    Note: This is isolated so it can easily be replaced when the final 
    preprocessing steps are confirmed by the teammate.
    """
    # Open image
    image = Image.open(image_file).convert("RGB")
    
    # Resize to 128x128 as expected by the current temporary model
    image = image.resize((128, 128))
    
    # Convert to array
    image_array = np.array(image, dtype=np.float32)
    
    # Normalize pixel values
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(image_file):
    """
    Takes an image file (e.g. from Flask request.files), processes it,
    and returns a dictionary with the prediction and confidence.
    """
    if not load_model():
        raise RuntimeError("Machine learning model could not be loaded.")
    
    # Preprocess
    image_array = preprocess_image(image_file)
    
    # Predict
    predictions = model.predict(image_array, verbose=0)[0]
    
    # Extract results
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100
    
    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    }
