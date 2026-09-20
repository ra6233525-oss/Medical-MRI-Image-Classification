from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import ml_model
import io

app = Flask(__name__)
# Enable CORS so the frontend can communicate with the backend during local development
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify backend status."""
    model_loaded = ml_model.load_model()
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Accepts an image and returns a medical prediction."""
    # Validate that an image was uploaded in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image field in the request"}), 400
    
    file = request.files['image']
    
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400
    
    # Validate that it's a supported image type
    if not allowed_file(file.filename):
        return jsonify({"error": "Allowed image types are png, jpg, jpeg"}), 400
    
    try:
        # Use io.BytesIO to keep the file in memory without saving to disk
        image_bytes = io.BytesIO(file.read())
        
        # Call the isolated ML logic
        result = ml_model.predict_image(image_bytes)
        
        return jsonify(result), 200
        
    except RuntimeError as e:
        # Model loading failed
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        # Other unexpected errors (e.g., malformed image)
        return jsonify({"error": "Failed to process the image. Ensure it is a valid image file."}), 500

if __name__ == '__main__':
    # Attempt to load the model on startup
    ml_model.load_model()
    app.run(debug=True, port=5000)
