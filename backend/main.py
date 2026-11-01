import os
import io
import base64
import datetime
from flask import Flask, request, jsonify, render_template_string, Response, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import ml_model
import database
import report_generator

app = Flask(__name__)

# Initialize the existing SQLite schema on import so WSGI deployments and the
# development server behave identically.  The call is idempotent.
database.init_db()

# Max payload size: 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Enable CORS for local dev and production frontend
CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jpe'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_text(value):
    """Returns a stripped string for text fields, or '' for non-string JSON values."""
    return value.strip() if isinstance(value, str) else ""

def parse_analysis_date(value):
    """
    Validates an optional YYYY-MM-DD exam date.

    Returns the normalised ISO date string, today's date when omitted, or
    ``None`` when the supplied value is not a realistic calendar date.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return datetime.date.today().isoformat()
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.date.fromisoformat(value.strip())
    except ValueError:
        return None
    # Reject clearly impossible exam dates while allowing a one-day buffer for
    # clients in timezones ahead of the server.
    if parsed.year < 1900 or parsed > datetime.date.today() + datetime.timedelta(days=1):
        return None
    return parsed.isoformat()

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "Uploaded image file exceeds the 16MB size limit."}), 413

@app.route('/', methods=['GET'])
def index():
    """Service root endpoint."""
    return jsonify({
        "service": "NeuroScan AI - Brain MRI Classification Inference API",
        "status": "online",
        "version": "1.2.0",
        "framework": "Flask / TensorFlow 2.21",
        "database": "SQLite (neuroscan.db)",
        "documentation": "/docs",
        "health_check": "/health",
        "prediction_endpoint": "/predict",
        "patient_endpoints": {
            "verify_patient": "/patients/<patient_id>",
            "register_patient": "/patients",
            "patient_scans": "/patients/<patient_id>/scans",
            "get_scan": "/scans/<scan_id>"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify backend status, model readiness, and database connectivity."""
    model_loaded = ml_model.load_model()
    db_connected = False
    try:
        conn = database.get_db_connection()
        conn.execute("SELECT 1;")
        conn.close()
        db_connected = True
    except Exception as e:
        db_connected = False

    is_healthy = model_loaded and db_connected
    metadata = ml_model.get_model_metadata()
    return jsonify({
        "status": "healthy" if is_healthy else "degraded",
        "model_loaded": model_loaded,
        "database_connected": db_connected,
        "database_type": "SQLite",
        "classes": ml_model.CLASS_NAMES,
        "framework": "TensorFlow/Keras",
        "backbone": "VGG16",
        "model": metadata,
        "version": "1.2.0"
    }), 200 if is_healthy else 503

# ===================================================================
# PATIENT MANAGEMENT ENDPOINTS
# ===================================================================

@app.route('/patients/<patient_id>', methods=['GET'])
def get_patient_endpoint(patient_id):
    """Verifies and retrieves patient record by Patient ID."""
    if not patient_id or not patient_id.strip():
        return jsonify({"error": "Patient ID parameter cannot be empty."}), 400

    patient = database.get_patient(patient_id.strip())
    if not patient:
        return jsonify({
            "found": False,
            "patient_id": patient_id.strip().upper(),
            "message": f"Patient ID '{patient_id.strip().upper()}' is not registered in the database."
        }), 404

    return jsonify({
        "found": True,
        "patient": patient
    }), 200

@app.route('/patients', methods=['POST'])
def create_patient_endpoint():
    """Registers or updates a patient in the database."""
    data = request.get_json(silent=True)
    if data is None:
        data = request.form.to_dict()
    if not isinstance(data, dict) or not data:
        return jsonify({"error": "Request body must contain patient information."}), 400

    patient_id = clean_text(data.get("patient_id"))
    name = clean_text(data.get("name"))
    age = data.get("age")
    gender = clean_text(data.get("gender"))

    if not patient_id:
        return jsonify({"error": "Field 'patient_id' is required."}), 400
    if len(patient_id) < 3:
        return jsonify({"error": "Patient ID must be at least 3 characters in length."}), 400
    if not name:
        return jsonify({"error": "Field 'name' is required."}), 400
    if age is None or str(age).strip() == "":
        return jsonify({"error": "Field 'age' is required."}), 400
    if gender not in ["Male", "Female", "Other"]:
        return jsonify({"error": "Field 'gender' must be 'Male', 'Female', or 'Other'."}), 400

    try:
        patient = database.create_or_update_patient(patient_id, name, age, gender)
        return jsonify({
            "message": "Patient registered successfully.",
            "patient": patient
        }), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/patients/<patient_id>/scans', methods=['GET'])
def get_patient_scans_endpoint(patient_id):
    """Retrieves all previous MRI scans and reports for a given Patient ID."""
    if not patient_id or not patient_id.strip():
        return jsonify({"error": "Patient ID is required."}), 400

    patient = database.get_patient(patient_id.strip())
    if not patient:
        return jsonify({"error": f"Patient with ID '{patient_id.strip().upper()}' does not exist."}), 404

    scans = database.get_patient_scans(patient_id.strip())
    return jsonify({
        "patient_id": patient["patient_id"],
        "patient_name": patient["name"],
        "total_scans": len(scans),
        "scans": scans
    }), 200

@app.route('/scans/<scan_id>', methods=['GET'])
def get_scan_endpoint(scan_id):
    """Retrieves full details and generated report reference of an MRI scan by Scan ID."""
    if not scan_id or not scan_id.strip():
        return jsonify({"error": "Scan ID is required."}), 400

    scan = database.get_mri_scan(scan_id.strip())
    if not scan:
        return jsonify({"error": f"Scan with ID '{scan_id.strip()}' not found."}), 404

    return jsonify(scan), 200

# ===================================================================
# PREDICTION & SCAN RECORDING ENDPOINT
# ===================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts an uploaded MRI image (multipart/form-data field 'image')
    associated with a required 'patient_id'.
    Performs VGG16 inference, generates Grad-CAM overlay, assigns unique Scan ID,
    and persistently associates the report with the patient in the database.
    """
    # 1. Validate Image Field
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' field provided in upload request."}), 400
    
    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected for analysis."}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG images are supported."}), 400

    # 2. Validate Patient ID Field
    patient_id = request.form.get('patient_id') or request.args.get('patient_id')
    if not patient_id or not patient_id.strip():
        return jsonify({"error": "Patient ID is required to perform and associate this MRI analysis."}), 400

    patient_id = patient_id.strip().upper()
    if len(patient_id) < 3:
        return jsonify({"error": "Patient ID must be at least 3 characters in length."}), 400

    # 3. Verify Patient ID Exists (or register if demographics provided)
    patient = database.get_patient(patient_id)
    if not patient:
        # Check if user provided patient demographic fields to auto-register
        p_name = request.form.get('patient_name') or request.form.get('name')
        p_age = request.form.get('patient_age') or request.form.get('age')
        p_gender = request.form.get('patient_gender') or request.form.get('gender')

        if p_name and p_age and p_gender:
            try:
                patient = database.create_or_update_patient(patient_id, p_name, p_age, p_gender)
            except Exception as e:
                return jsonify({"error": f"Failed to register patient '{patient_id}': {str(e)}"}), 400
        else:
            return jsonify({
                "error": f"Patient ID '{patient_id}' is not registered in the database. Please verify or register patient information first.",
                "patient_id": patient_id,
                "requires_registration": True
            }), 404

    # 4. Validate the optional exam date and generate identifiers
    analysis_date = parse_analysis_date(request.form.get('analysis_date'))
    if analysis_date is None:
        return jsonify({
            "error": "Field 'analysis_date' must be a valid ISO date (YYYY-MM-DD) "
                     "between 1900-01-01 and today."
        }), 400

    scan_id = database.generate_unique_scan_id()
    report_reference = database.generate_report_reference(scan_id)
    clean_filename = secure_filename(file.filename) or "mri_scan.jpg"

    # 5. Perform Deep Learning Inference & Prepare Image Data
    try:
        raw_bytes = file.read()
        image_bytes = io.BytesIO(raw_bytes)
        result = ml_model.predict_image(image_bytes)

        # Build base64 data URI of uploaded MRI slice for persistent self-contained reporting
        img_ext = clean_filename.rsplit('.', 1)[-1].lower() if '.' in clean_filename else 'jpeg'
        mime_type = "image/png" if img_ext == 'png' else "image/jpeg"
        image_data_uri = f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('utf-8')}"
    except ml_model.InvalidImageError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": f"Model inference error: {str(e)}"}), 503
    except Exception as e:
        return jsonify({"error": f"Failed to parse or analyze image: {str(e)}"}), 500

    # 6. Persistently Save MRI Scan Record to Database
    model_name = "VGG16 Transfer Learning"
    try:
        saved_record = database.save_mri_scan(
            scan_id=scan_id,
            patient_id=patient["patient_id"],
            filename=clean_filename,
            predicted_class=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_name=model_name,
            gradcam_image=result.get("gradcam_image"),
            gradcam_heatmap=result.get("gradcam_heatmap"),
            report_reference=report_reference,
            analysis_date=analysis_date
        )
    except Exception as e:
        return jsonify({"error": f"Failed to associate and save MRI scan in database: {str(e)}"}), 500

    # 7. Automatically Generate & Persist Patient-Specific Report
    analysis_datetime = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    display_classification = report_generator.format_display_classification(result["prediction"])
    classification_summary = report_generator.generate_classification_summary(
        predicted_class=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"]
    )

    try:
        report_payload = {
            "project_name": report_generator.PROJECT_NAME,
            "patient_id": patient["patient_id"],
            "patient_name": patient["name"],
            "patient_age": patient["age"],
            "patient_gender": patient["gender"],
            "scan_id": scan_id,
            "report_reference": report_reference,
            "analysis_datetime": analysis_datetime,
            "filename": clean_filename,
            "image_data": image_data_uri,
            "gradcam_data": result.get("gradcam_image"),
            "gradcam_heatmap": result.get("gradcam_heatmap"),
            "predicted_class": result["prediction"],
            "display_classification": display_classification,
            "confidence": result["confidence"],
            "model_architecture": report_generator.MODEL_ARCHITECTURE,
            "model_information": report_generator.MODEL_INFORMATION,
            "classification_summary": classification_summary,
            "medical_disclaimer": report_generator.MEDICAL_DISCLAIMER,
            "probabilities": result["probabilities"],
            "processing_time_ms": result.get("processing_time_ms"),
        }

        html_report = report_generator.build_html_report(report_payload)

        database.save_report(
            report_id=report_reference,
            report_reference=report_reference,
            scan_id=scan_id,
            patient_id=patient["patient_id"],
            project_name=report_generator.PROJECT_NAME,
            analysis_datetime=analysis_datetime,
            image_data=image_data_uri,
            gradcam_data=result.get("gradcam_image"),
            gradcam_heatmap=result.get("gradcam_heatmap"),
            predicted_class=result["prediction"],
            display_classification=display_classification,
            confidence=result["confidence"],
            model_architecture=report_generator.MODEL_ARCHITECTURE,
            classification_summary=classification_summary,
            medical_disclaimer=report_generator.MEDICAL_DISCLAIMER,
            html_content=html_report
        )
    except Exception as e:
        print(f"Warning: Failed to save automated report: {e}")

    # 8. Construct Response with Identifiers, Report Links, and Predictions
    response_payload = {
        "status": "success",
        "scan_id": scan_id,
        "patient_id": patient["patient_id"],
        "patient_name": patient["name"],
        "patient_age": patient["age"],
        "patient_gender": patient["gender"],
        "filename": clean_filename,
        "prediction": result["prediction"],
        "predicted_class": result["prediction"],
        "display_classification": display_classification,
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "model_name": model_name,
        "model": ml_model.get_model_metadata(),
        "gradcam_layer": ml_model.GRADCAM_LAYER,
        "image_data": image_data_uri,
        "gradcam_image": result.get("gradcam_image"),
        "gradcam_heatmap": result.get("gradcam_heatmap"),
        "report_reference": report_reference,
        "report_url": f"/reports/{report_reference}",
        "report_download_url": f"/reports/{report_reference}/download",
        "classification_summary": classification_summary,
        "analysis_date": analysis_date,
        "analysis_datetime": analysis_datetime,
        "processing_time_ms": result.get("processing_time_ms")
    }

    return jsonify(response_payload), 200

# ===================================================================
# AUTOMATED CLINICAL REPORT ENDPOINTS
# ===================================================================

@app.route('/reports/<report_ref>', methods=['GET'])
def get_report_endpoint(report_ref):
    """
    Renders or returns the patient-specific MRI analysis report.
    By default returns the publication-quality medical HTML report page.
    If ?format=json is passed, returns structured JSON.
    """
    if not report_ref or not report_ref.strip():
        return jsonify({"error": "Report reference is required."}), 400

    report = database.get_report(report_ref.strip())
    if not report:
        return jsonify({"error": f"Report '{report_ref.strip()}' not found."}), 404

    if request.args.get('format') == 'json':
        return jsonify(report), 200

    html_content = report["html_content"]
    if request.args.get('print') in ('1', 'true', 'yes'):
        # The frontend opens this URL in a new tab for the "Print report"
        # action.  The report itself owns the print stylesheet.
        print_script = (
            "<script>window.addEventListener('load', function () {"
            "  setTimeout(function () { window.print(); }, 350);"
            "});</script>"
        )
        html_content = html_content.replace("</body>", print_script + "</body>")

    return Response(
        html_content,
        mimetype='text/html',
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Content-Type-Options": "nosniff",
        },
    )

@app.route('/reports/<report_ref>/download', methods=['GET'])
def download_report_endpoint(report_ref):
    """Downloads the self-contained clinical HTML report as an offline attachment."""
    if not report_ref or not report_ref.strip():
        return jsonify({"error": "Report reference is required."}), 400

    report = database.get_report(report_ref.strip())
    if not report:
        return jsonify({"error": f"Report '{report_ref.strip()}' not found."}), 404

    filename = f"NeuroScan_Report_{report['scan_id']}.html"
    return Response(
        report["html_content"],
        mimetype='text/html',
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store, max-age=0",
            "X-Content-Type-Options": "nosniff",
        },
    )

@app.route('/reports/<report_ref>/json', methods=['GET'])
def get_report_json_endpoint(report_ref):
    """Returns JSON representation of the report."""
    if not report_ref or not report_ref.strip():
        return jsonify({"error": "Report reference is required."}), 400

    report = database.get_report(report_ref.strip())
    if not report:
        return jsonify({"error": f"Report '{report_ref.strip()}' not found."}), 404

    return jsonify(report), 200

@app.route('/patients/<patient_id>/reports', methods=['GET'])
def get_patient_reports_endpoint(patient_id):
    """Retrieves list of all reports for a specific Patient ID."""
    if not patient_id or not patient_id.strip():
        return jsonify({"error": "Patient ID is required."}), 400

    patient = database.get_patient(patient_id.strip())
    if not patient:
        return jsonify({"error": f"Patient with ID '{patient_id.strip().upper()}' not found."}), 404

    reports = database.get_patient_reports(patient_id.strip())
    return jsonify({
        "patient_id": patient["patient_id"],
        "patient_name": patient["name"],
        "total_reports": len(reports),
        "reports": reports
    }), 200

@app.route('/scans/<scan_id>/report', methods=['GET'])
def get_scan_report_redirect(scan_id):
    """Redirects or serves report associated with an MRI Scan ID."""
    if not scan_id or not scan_id.strip():
        return jsonify({"error": "Scan ID is required."}), 400

    report = database.get_report(scan_id.strip())
    if not report:
        return jsonify({"error": f"No report found for Scan ID '{scan_id.strip()}'."}), 404

    return redirect(f"/reports/{report['report_reference']}")

# ===================================================================
# OPENAPI & SWAGGER UI DOCUMENTATION
# ===================================================================

@app.route('/openapi.json', methods=['GET'])
def openapi_spec():
    """OpenAPI 3.0 specification for NeuroScan AI backend with Patient & Scan Integration."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "NeuroScan AI — Brain MRI Classification & Patient Records API",
            "version": "1.2.0",
            "description": "Flask-powered deep learning inference backend using VGG16 with full Patient ID and MRI Scan ID relational database tracking in SQLite."
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "System, Model & Database Health Check",
                    "responses": {
                        "200": {"description": "All systems healthy, model loaded, database connected."}
                    }
                }
            },
            "/patients": {
                "post": {
                    "summary": "Register or Update Patient Demographics",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "patient_id": {"type": "string", "example": "PID-1001"},
                                        "name": {"type": "string", "example": "Eleanor Vance"},
                                        "age": {"type": "integer", "example": 48},
                                        "gender": {"type": "string", "enum": ["Male", "Female", "Other"]}
                                    },
                                    "required": ["patient_id", "name", "age", "gender"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {"description": "Patient successfully registered or updated."}
                    }
                }
            },
            "/patients/{patient_id}": {
                "get": {
                    "summary": "Verify Patient ID and Fetch Demographics",
                    "parameters": [
                        {"name": "patient_id", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "Patient found."},
                        "404": {"description": "Patient not found."}
                    }
                }
            },
            "/patients/{patient_id}/scans": {
                "get": {
                    "summary": "Retrieve Previous MRI Scans for Patient",
                    "parameters": [
                        {"name": "patient_id", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "List of historical scans and predictions."}
                    }
                }
            },
            "/scans/{scan_id}": {
                "get": {
                    "summary": "Retrieve Specific MRI Scan by Scan ID",
                    "parameters": [
                        {"name": "scan_id", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "MRI scan details with report reference."}
                    }
                }
            },
            "/predict": {
                "post": {
                    "summary": "Run MRI Tumor Classification Associated with Patient ID",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "image": {"type": "string", "format": "binary"},
                                        "patient_id": {"type": "string", "example": "PID-1001"},
                                        "patient_name": {"type": "string", "example": "Eleanor Vance"},
                                        "patient_age": {"type": "integer", "example": 48},
                                        "patient_gender": {"type": "string", "example": "Female"},
                                        "analysis_date": {
                                            "type": "string",
                                            "format": "date",
                                            "example": "2026-10-30",
                                            "description": "Optional ISO exam date (YYYY-MM-DD). Defaults to today."
                                        }
                                    },
                                    "required": ["image", "patient_id"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Inference complete, Scan ID generated, saved to database."}
                    }
                }
            },
            "/reports/{report_ref}": {
                "get": {
                    "summary": "View Patient-Specific Medical Report (HTML / JSON)",
                    "parameters": [
                        {"name": "report_ref", "in": "path", "required": True, "schema": {"type": "string"}},
                        {"name": "format", "in": "query", "required": False, "schema": {"type": "string", "enum": ["html", "json"]}},
                        {"name": "print", "in": "query", "required": False, "schema": {"type": "string", "enum": ["1", "true", "yes"]},
                         "description": "Inject a load-triggered window.print() call."}
                    ],
                    "responses": {
                        "200": {"description": "Full clinical report returned."},
                        "404": {"description": "Report not found."}
                    }
                }
            },
            "/reports/{report_ref}/json": {
                "get": {
                    "summary": "Retrieve Structured Report JSON",
                    "parameters": [
                        {"name": "report_ref", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "Structured report including probabilities and patient context."},
                        "404": {"description": "Report not found."}
                    }
                }
            },
            "/reports/{report_ref}/download": {
                "get": {
                    "summary": "Download Clinical Report HTML File",
                    "parameters": [
                        {"name": "report_ref", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "Self-contained HTML file attachment."},
                        "404": {"description": "Report not found."}
                    }
                }
            },
            "/scans/{scan_id}/report": {
                "get": {
                    "summary": "Redirect from a Scan ID to its Report",
                    "parameters": [
                        {"name": "scan_id", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "302": {"description": "Redirect to the report HTML page."},
                        "404": {"description": "No report exists for this scan."}
                    }
                }
            },
            "/patients/{patient_id}/reports": {
                "get": {
                    "summary": "Retrieve All Reports for a Patient",
                    "parameters": [
                        {"name": "patient_id", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "List of all historical reports for patient."}
                    }
                }
            }
        }
    }
    return jsonify(spec)

@app.route('/docs', methods=['GET'])
def swagger_ui():
    """Interactive Swagger UI documentation page."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NeuroScan AI — API Documentation</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
  <style>
    body { margin: 0; background: #fafafa; font-family: sans-serif; }
    .topbar { display: none !important; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js" crossorigin></script>
  <script>
    window.onload = () => {
      window.ui = SwaggerUIBundle({
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIBundle.SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout"
      });
    };
  </script>
</body>
</html>"""
    return render_template_string(html)

if __name__ == '__main__':
    # Initialize database
    database.init_db()

    # Pre-warm model on startup
    print("Pre-warming model weights...")
    ml_model.load_model()
    
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting NeuroScan AI backend server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
