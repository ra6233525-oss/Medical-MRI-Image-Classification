import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import ml_model

app = Flask(__name__)

# Max payload size: 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Enable CORS for local dev and production frontend
CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jpe'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "Uploaded image file exceeds the 16MB size limit."}), 413

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify backend status and model readiness."""
    model_loaded = ml_model.load_model()
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "classes": ml_model.CLASS_NAMES,
        "framework": "TensorFlow/Keras",
        "version": "1.0.0"
    }), 200 if model_loaded else 503

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts an uploaded MRI image (multipart/form-data field 'image').
    Returns prediction, confidence percentage, class probabilities, and Grad-CAM overlay.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' field provided in upload request."}), 400
    
    file = request.files['image']
    
    if not file or file.filename == '':
        return jsonify({"error": "No file selected for analysis."}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG images are supported."}), 400
    
    try:
        # In-memory buffer: do not save patient images to disk for privacy
        image_bytes = io.BytesIO(file.read())
        
        # Perform inference and generate Grad-CAM
        result = ml_model.predict_image(image_bytes)
        
        # Add metadata
        result["filename"] = secure_filename(file.filename)
        return jsonify(result), 200
        
    except RuntimeError as e:
        return jsonify({"error": f"Model inference error: {str(e)}"}), 503
    except Exception as e:
        return jsonify({"error": "Failed to parse or analyze image. Ensure the file is an undamaged image."}), 500

if __name__ == '__main__':
    # Pre-warm model on startup
    print("Pre-warming model weights...")
    ml_model.load_model()
    
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting NeuroScan AI backend server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
