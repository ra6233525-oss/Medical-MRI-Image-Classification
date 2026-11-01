# NeuroScan AI

**A Flask-based four-class brain MRI classification system using VGG16 transfer learning and genuine Grad-CAM explainability.**

NeuroScan AI is an academic research prototype. It accepts a single MRI slice, classifies it into one of four trained categories, stores the result against a patient record in SQLite, and generates a self-contained report containing the uploaded image, a real Grad-CAM heatmap, a Grad-CAM overlay, and the complete softmax distribution.

> **Not a medical device. Not a diagnosis.** Model output must not be used for clinical decisions. See [Limitations](#limitations-and-safety).

---

## Verified results

The deployed weights were evaluated on all **1,600 images** in `dataset/Testing` (400 per class). Nothing was sampled or excluded.

| Metric | Value |
|---|---:|
| Overall accuracy | **90.94%** (1,455 / 1,600) |
| Macro precision | 92.00% |
| Macro recall | 90.94% |
| Macro F1 | 90.64% |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 97.54% | **69.25%** | 80.99% | 400 |
| Meningioma | 78.94% | 96.50% | 86.84% | 400 |
| No tumor | 95.20% | 99.25% | 97.18% | 400 |
| Pituitary | 96.34% | 98.75% | 97.53% | 400 |

**Confusion matrix** (rows = ground truth, columns = prediction):

| True \ Predicted | Glioma | Meningioma | No tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Glioma | **277** | 97 | 18 | 8 | 400 |
| Meningioma | 6 | **386** | 1 | 7 | 400 |
| No tumor | 1 | 2 | **397** | 0 | 400 |
| Pituitary | 0 | 4 | 1 | **395** | 400 |

The main weakness is **glioma sensitivity**: 69.25%. Most missed gliomas (97 of 123 errors) were predicted as meningioma. Overall accuracy alone hides this failure mode, which is why the interface and reports show per-class results.

These figures are from the project test split only. No external validation has been performed.

---

## What the application does

- **Real inference.** Predictions come from the existing Keras VGG16 artifact in `backend/model`. No mock predictions or simulated probabilities.
- **Four trained classes.** `glioma`, `meningioma`, `notumor`, `pituitary` in that fixed index order.
- **Genuine Grad-CAM.** Gradients are taken from `block5_conv3` for the predicted class. Both the raw Jet-colormapped heatmap and the 55/45 MRI-heatmap overlay are returned and persisted.
- **Patient records.** Patients, scans, probabilities, heatmaps, and reports are stored in SQLite. History searches query the database directly; there are no hardcoded example patients.
- **Self-contained reports.** Each report embeds patient/scan/report identifiers, the original MRI, both Grad-CAM views, all four probabilities, model information, an objective model-output summary, and a medical disclaimer.
- **Exact normal label.** The normal class is displayed as **“No Tumor Detected by AI Model.”**
- **Responsive interface.** Dark navy/cyan/turquoise/purple/emerald visual system with accessible reduced-motion handling. Animations are limited to the marketing page; MRI and Grad-CAM images are never animated.
- **Backward-compatible schema.** Existing databases are upgraded additively with `gradcam_heatmap` columns. Legacy records remain readable and are not given fabricated heatmaps.

---

## Architecture

```text
Browser
  └── frontend/index.html + style.css + app.js
        │  multipart image upload + patient fields
        ▼
Flask API  (backend/main.py)
  ├── validation and patient lookup/registration
  ├── VGG16 inference  (backend/ml_model.py)
  │     └── Grad-CAM from block5_conv3
  ├── SQLite persistence  (backend/database.py)
  │     ├── patients
  │     ├── mri_scans
  │     └── reports
  └── report generation  (backend/report_generator.py)
        └── self-contained HTML with embedded images
```

### Model

```text
Input 128 × 128 × 3 RGB
  → bilinear resize, pixel values / 255.0
  → VGG16 backbone (13 Conv2D layers, ImageNet weights, block5 fine-tuned)
  → Flatten
  → Dropout(0.30)
  → Dense(128, ReLU)
  → Dropout(0.20)
  → Dense(4, Softmax)
```

- Model config: `backend/model/config.json`
- Weights: `backend/model/model.weights.h5` (tracked with Git LFS)
- Classes: `glioma`, `meningioma`, `notumor`, `pituitary`
- Grad-CAM layer: `block5_conv3`
- The application does **not** retrain, replace, or modify the trained architecture.

---

## Repository layout

```text
backend/
  main.py              Flask routes, validation, inference/report orchestration
  ml_model.py          model loading, preprocessing, inference, Grad-CAM
  database.py          SQLite schema, additive migrations, persistence
  report_generator.py  HTML report, objective summaries, disclaimer
  model/               trained Keras config/weights/metadata
  data/                runtime SQLite database (git-ignored)
frontend/
  index.html           homepage, analysis workspace, records, limitations
  style.css            responsive visual system and reduced-motion rules
  app.js               API client, upload flow, results, history, API settings
docs/
  NeuroScan_AI_Academic_Report.pdf
  academic_report.html
README.md
PROJECT_DOCUMENTATION.md
PHASE_TEST_REPORT.md
requirements.txt
vercel.json
```

`dataset/`, `model_test/`, SQLite files, uploads, logs, and `.env` files are git-ignored. Never commit patient data or datasets.

---

## Requirements

Verified environment:

- Windows 11
- Python 3.13.7
- TensorFlow 2.21.0
- Keras 3.15.1
- Flask 3.1.3
- Pillow 12.3.0
- NumPy 2.5.3
- Node.js 24.21.0 only for the optional browser audit

The backend runs on CPU. TensorFlow GPU support is not available on native Windows for TensorFlow ≥ 2.11.

---

## Quick start (Windows PowerShell)

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend\main.py
```

If script execution is blocked for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

The backend starts on `http://127.0.0.1:5000` and pre-warms the model. The first model load can take several seconds.

In a second terminal, serve the frontend:

```powershell
python -m http.server 8000 --directory frontend
```

Open **http://localhost:8000**.

> Do not open `frontend/index.html` with `file://`. The API settings default to `http://127.0.0.1:5000`, and browser security rules may block local requests from a file origin.

### Optional environment overrides

```powershell
$env:PORT = "5000"
$env:HOST = "127.0.0.1"
$env:NEUROSCAN_DB_PATH = "C:\path\to\isolated_test.db"
python backend\main.py
```

`NEUROSCAN_DB_PATH` is intended for tests. It prevents integration tests from writing to the production database.

---

## API

Swagger UI: `http://127.0.0.1:5000/docs`  
OpenAPI JSON: `http://127.0.0.1:5000/openapi.json`

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service metadata and route index |
| `GET` | `/health` | Model, database, and version health |
| `POST` | `/patients` | Register or update a patient |
| `GET` | `/patients/<patient_id>` | Verify patient and count scans |
| `GET` | `/patients/<patient_id>/scans` | Patient scan history |
| `GET` | `/patients/<patient_id>/reports` | Patient report history |
| `GET` | `/scans/<scan_id>` | Full scan record |
| `POST` | `/predict` | Upload image, infer, persist scan and report |
| `GET` | `/reports/<report_ref>` | HTML report |
| `GET` | `/reports/<report_ref>/json` | Structured report JSON |
| `GET` | `/reports/<report_ref>/download` | Download self-contained HTML |
| `GET` | `/scans/<scan_id>/report` | Redirect scan to its report |

### Example prediction

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict `
  -F "patient_id=PID-EXAMPLE-001" `
  -F "patient_name=Example Patient" `
  -F "patient_age=52" `
  -F "patient_gender=Female" `
  -F "image=@dataset/Testing/meningioma/Te-aug-me_34.jpg"
```

Successful responses include:

```json
{
  "status": "success",
  "patient_id": "PID-EXAMPLE-001",
  "scan_id": "SCAN-YYYYMMDD-XXXXXXXX",
  "prediction": "meningioma",
  "confidence": 98.02,
  "probabilities": {
    "glioma": 0.01,
    "meningioma": 98.02,
    "notumor": 0.01,
    "pituitary": 1.96
  },
  "gradcam_layer": "block5_conv3",
  "gradcam_heatmap": "data:image/jpeg;base64,...",
  "gradcam_image": "data:image/jpeg;base64,...",
  "image_data": "data:image/jpeg;base64,...",
  "report_reference": "REP-YYYYMMDD-XXXXXXXX",
  "report_url": "/reports/REP-YYYYMMDD-XXXXXXXX"
}
```

Confidence and probabilities are percentages in the range 0–100, not fractions.

### Upload constraints

- One image per request.
- Allowed extensions: `.jpg`, `.jpeg`, `.jpe`, `.png`.
- Maximum payload: 16 MB.
- Patient ID is required and must be at least 3 characters.
- `analysis_date` is optional and must be a real ISO date (`YYYY-MM-DD`) from 1900-01-01 through today; it defaults to the server date.
- If the patient does not exist, demographics may be supplied to register them during the same request.
- Unreadable or non-image content is rejected with **400**, not treated as an inference failure.
- JSON patient fields must use the documented string/integer types; wrong types return **400**.

---

## Database

Default path: `backend/data/neuroscan.db` (git-ignored).

| Table | Purpose |
|---|---|
| `patients` | Patient ID, name, age, gender, timestamps |
| `mri_scans` | Scan IDs, probabilities, prediction, Grad-CAM images, report reference |
| `reports` | Full self-contained report HTML plus structured fields |

Relationships use foreign keys with cascade deletes. Scan IDs use the format `SCAN-YYYYMMDD-XXXXXXXX`; report references use `REP-YYYYMMDD-XXXXXXXX`.

Migrations are additive and idempotent. Existing databases gain `gradcam_heatmap` columns on startup. Legacy scans and reports keep `NULL` heatmaps because no original heatmap exists for them; the application does not invent one.

---

## Tests performed

| Test | Result |
|---|---|
| Backend integration, isolated SQLite DB | **58 / 58 passed** |
| Headless browser audit (homepage, upload, results, report, history, console) | **48 / 48 passed** |
| Responsive/motion audit at 1440, 1024, 768, 390 px | **24 / 24 passed** |
| Generated report layout audit (desktop, mobile, print) | **12 / 12 passed** |
| Production DB copy compatibility | **passed**; production hash unchanged |
| Full held-out evaluation | **1,600 / 1,600 images**; 90.94% accuracy |

Detailed methodology and limitations are in [PHASE_TEST_REPORT.md](PHASE_TEST_REPORT.md).

Local audit scripts live in `scratch/`, which is git-ignored because it contains generated screenshots and test databases.

### Production safety

- Browser and backend tests used isolated databases via `NEUROSCAN_DB_PATH`.
- The production database was copied, hashed before and after, and never modified during compatibility testing.
- No patient records, datasets, model logs, credentials, or SQLite files should be committed.

---

## Limitations and safety

- **Single-slice input.** The model sees one 128×128 image, not a volumetric MRI study or clinical context.
- **Four classes only.** Other tumours, infections, artefacts, and normal variants are outside the label space.
- **Weak glioma recall.** 30.75% of glioma test images were missed; most were predicted as meningioma.
- **Dataset-bound evaluation.** Results are from the project test split and have not been externally validated.
- **No calibration study.** Softmax probabilities are not clinically calibrated.
- **Grad-CAM is not segmentation.** It shows gradient-weighted model attention, not a lesion boundary or radiological finding.
- **No regulatory clearance.** This is not a certified medical device and must not guide patient care.
- **Prototype security.** CORS is open, there is no authentication/authorization layer, and the SQLite database contains identifiable data. Do not expose this deployment to untrusted networks without adding authentication, TLS, access controls, audit logging, encryption, and a HIPAA/GDPR review.

---

## Documentation

- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) — architecture, data flow, API contracts, database design, security, deployment, and maintenance.
- [PHASE_TEST_REPORT.md](PHASE_TEST_REPORT.md) — exact commands, results, metrics, and test limitations.
- [docs/NeuroScan_AI_Academic_Report.pdf](docs/NeuroScan_AI_Academic_Report.pdf) — 6-page academic report.
- [docs/academic_report.html](docs/academic_report.html) — editable source for the academic report.

---

## Status

Version **1.2.0**. The project includes the redesigned frontend, corrected documentation, Grad-CAM heatmap support, and verified integration tests. The trained model artifacts are unchanged; see [PHASE_TEST_REPORT.md](PHASE_TEST_REPORT.md) for the current verification evidence.
