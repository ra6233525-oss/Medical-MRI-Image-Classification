# NeuroScan AI — Technical Project Documentation

**Version:** 1.2.0  
**Document date:** 1 November 2026  
**System type:** Academic research prototype — not a medical device

---

## 1. Purpose and scope

NeuroScan AI classifies a single brain MRI slice into four trained categories and produces inspectable evidence for the prediction:

1. `glioma`
2. `meningioma`
3. `notumor`
4. `pituitary`

The system combines:

- an existing VGG16 transfer-learning model,
- genuine Grad-CAM explainability,
- a Flask inference API,
- a SQLite patient/scan/report store,
- a self-contained HTML report generator,
- a responsive static frontend.

The project deliberately preserves the trained model and class order. This work does **not** retrain the network, substitute a different architecture, or fabricate predictions, probabilities, patient records, segmentation masks, or radiological findings.

The output is a statistical model ranking. It is not a diagnosis, a radiological read, or evidence that a disease is absent.

---

## 2. System architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│ Browser                                                          │
│ frontend/index.html · style.css · app.js                         │
│   patient intake · image validation · results · history          │
└───────────────────────────────┬──────────────────────────────────┘
                                │ HTTP / multipart
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ Flask API — backend/main.py                                      │
│                                                                  │
│  /health         model + DB readiness                            │
│  /patients       registration and lookup                         │
│  /predict        validation → inference → persistence → report   │
│  /scans          scan records                                    │
│  /reports        HTML, JSON, download, print mode                │
│  /docs           Swagger UI                                      │
└───────────────┬─────────────────────┬────────────────────────────┘
                │                     │
                ▼                     ▼
┌───────────────────────────┐  ┌───────────────────────────────────┐
│ ml_model.py               │  │ database.py                       │
│ VGG16 + Grad-CAM          │  │ SQLite                            │
│                           │  │ patients · mri_scans · reports    │
└───────────────────────────┘  └───────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ report_generator.py                                              │
│ self-contained HTML report with embedded image data URIs         │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 Runtime components

| Component | File | Responsibility |
|---|---|---|
| Flask application | `backend/main.py` | Routes, validation, orchestration, response contracts |
| Inference engine | `backend/ml_model.py` | Model load, preprocessing, prediction, Grad-CAM |
| Persistence | `backend/database.py` | Schema, migrations, CRUD, identifiers |
| Report engine | `backend/report_generator.py` | Report HTML, objective summary, disclaimer |
| Frontend | `frontend/*` | Homepage, workspace, results, history, API settings |
| Model artifacts | `backend/model/*` | Keras config, weights, metadata |

---

## 3. Model

### 3.1 Architecture

The deployed model is the existing Keras artifact described by `backend/model/config.json`:

```text
Input: 128 × 128 × 3 RGB
  → VGG16 backbone
      13 convolutional layers
      5 max-pooling layers
      ImageNet pretrained
      block5 convolutional layers fine-tuned
  → Flatten                 (4 × 4 × 512 = 8,192 features)
  → Dropout(0.30)
  → Dense(128, ReLU)
  → Dropout(0.20)
  → Dense(4, Softmax)
```

Class index mapping is fixed:

| Index | Raw class | Display label |
|---:|---|---|
| 0 | `glioma` | Glioma |
| 1 | `meningioma` | Meningioma |
| 2 | `notumor` | **No Tumor Detected by AI Model.** |
| 3 | `pituitary` | Pituitary Tumour |

Changing the class order without retraining would invalidate the output.

### 3.2 Preprocessing

The preprocessing path is implemented in `ml_model.preprocess_pil`:

1. Convert the image to RGB.
2. Resize to **128 × 128** with **bilinear** interpolation.
3. Convert to `float32`.
4. Normalize pixel values to `[0.0, 1.0]` by dividing by 255.
5. Add a batch dimension.

No ImageNet mean/std normalization is applied by the deployed inference path. This matches the code, not an assumption about training.

### 3.3 Prediction output

`ml_model.predict_image` returns:

- raw predicted class and index,
- display label,
- highest-class confidence as a percentage (0–100),
- all four probabilities as percentages,
- raw Grad-CAM heatmap data URI,
- Grad-CAM overlay data URI,
- model name,
- Grad-CAM layer,
- inference latency in milliseconds.

The softmax outputs are rounded to two decimal places for transport. Their sum is approximately 100%, subject to rounding.

### 3.4 Model loading and thread safety

- The model is loaded once and cached.
- A process-level lock prevents concurrent first-load races.
- The Grad-CAM graph is rebuilt symbolically from the deserialized model so the classifier head can be replayed explicitly.
- The application pre-warms the model when `python backend/main.py` starts.

The intended production process model is one worker with multiple threads because TensorFlow models are not safely forked across multiple gunicorn workers.

---

## 4. Grad-CAM explainability

### 4.1 Computation

For predicted class \(c\), Grad-CAM uses the feature maps \(A^k\) from `block5_conv3`:

1. Run a forward pass and capture the convolutional feature maps and class scores.
2. Compute the gradient of the score for class \(c\) with respect to each feature map.
3. Global-average-pool the gradients over spatial dimensions to obtain weights \(\alpha_k^c\).
4. Form a weighted sum of the feature maps.
5. Apply ReLU and normalize by the maximum activation.
6. Resize the activation map to the original image dimensions.
7. Render it with a Jet colormap.
8. Blend the original MRI and heatmap at **55% original / 45% heatmap**.

The raw heatmap and the overlay are different images. Both are returned to the browser and stored in SQLite. Neither is generated from a segmentation model.

### 4.2 What Grad-CAM does and does not show

Grad-CAM highlights regions that contributed to the predicted class score. It does **not**:

- identify a lesion boundary,
- measure tumour size,
- prove that a highlighted region is pathological,
- provide a radiological finding,
- validate that the prediction is correct.

A plausible-looking heatmap can accompany an incorrect prediction. The interface and report state this explicitly.

---

## 5. Request flow

### 5.1 Prediction sequence

```text
User enters patient details and selects one image
        │
        ▼
Frontend validates extension and size (≤ 16 MB)
        │
        ▼
POST /predict (multipart)
        │
        ├── validate image field and extension
        ├── require patient_id (≥ 3 characters)
        ├── look up patient; optionally register if demographics supplied
        ├── generate scan ID and report reference
        ├── run VGG16 inference
        ├── compute Grad-CAM heatmap + overlay
        ├── build original-image data URI
        ├── persist mri_scans row
        ├── build objective report summary
        ├── persist reports row
        └── return identifiers, probabilities, images, report links
        │
        ▼
Frontend renders original MRI, raw heatmap, overlay, four probabilities,
summary, actions, and disclaimer
```

### 5.2 Report actions

| Action | Behavior |
|---|---|
| View report | Opens `/reports/<ref>` in a new tab |
| Print report | Opens `/reports/<ref>?print=1`; Flask injects a load-triggered `window.print()` |
| Download HTML | Downloads the self-contained report as an attachment |
| Patient history | Queries `/patients/<id>` and `/patients/<id>/scans` |

The backend report also contains a print button, so it works without the app shell.

---

## 6. API contract

Base URL: `http://127.0.0.1:5000`

### 6.1 Health

`GET /health`

```json
{
  "status": "healthy",
  "version": "1.2.0",
  "model_loaded": true,
  "database_connected": true,
  "backbone": "VGG16",
  "classes": ["glioma", "meningioma", "notumor", "pituitary"],
  "model": {
    "architecture": "VGG16 ... → Dense(4, Softmax)",
    "input_shape": [128, 128, 3],
    "input_resize": "Bilinear",
    "normalization": "[0.0, 1.0] (pixel / 255.0)",
    "gradcam_layer": "block5_conv3"
  }
}
```

Returns `503` if the model or database is unavailable.

### 6.2 Patients

`POST /patients`

```json
{
  "patient_id": "PID-001",
  "name": "Patient Name",
  "age": 52,
  "gender": "Female"
}
```

- `gender` must be `Male`, `Female`, or `Other`.
- `age` must be an integer from 0 to 150.
- `patient_id` and `name` must be JSON strings; wrong types return `400`.
- Patient IDs are normalized to uppercase and must be at least 3 characters.
- Existing patients are updated.

`GET /patients/<id>` returns `{found, patient}` or `404`.

`GET /patients/<id>/scans` returns patient metadata and all scans, newest first.

`GET /patients/<id>/reports` returns report metadata, newest first.

### 6.3 Prediction

`POST /predict` with `multipart/form-data`:

| Field | Required | Notes |
|---|---|---|
| `image` | Yes | JPG/JPEG/JPE/PNG, ≤ 16 MB |
| `patient_id` | Yes | At least 3 characters |
| `patient_name` | Conditional | Required only to auto-register a new patient |
| `patient_age` | Conditional | Same |
| `patient_gender` | Conditional | Same |
| `analysis_date` | No | ISO date `YYYY-MM-DD`, 1900-01-01 through today; defaults to today |

Response fields include `scan_id`, `report_reference`, `prediction`, `display_classification`, `confidence`, `probabilities`, `image_data`, `gradcam_heatmap`, `gradcam_image`, `model`, `gradcam_layer`, `classification_summary`, and report URLs.

Failure behavior:

| Condition | Status |
|---|---:|
| Missing image field | 400 |
| Empty filename | 400 |
| Unsupported extension | 400 |
| Missing/short patient ID | 400 |
| Unknown patient without demographics | 404 |
| Invalid patient demographics or non-string JSON fields | 400 |
| Invalid `analysis_date` | 400 |
| Payload over 16 MB | 413 |
| Model unavailable | 503 |
| Invalid image content / unreadable image | 400 |
| Persistence failure | 500 |

### 6.4 Reports

| Endpoint | Response |
|---|---|
| `GET /reports/<ref>` | Self-contained HTML |
| `GET /reports/<ref>?format=json` | Direct report JSON |
| `GET /reports/<ref>/json` | Direct report JSON |
| `GET /reports/<ref>/download` | HTML attachment |
| `GET /scans/<scan_id>/report` | Redirect to HTML report |

Report responses set `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.

### 6.5 Documentation

`GET /openapi.json` returns the OpenAPI 3.0 specification.  
`GET /docs` renders Swagger UI.

---

## 7. Database design

### 7.1 Tables

#### `patients`

| Column | Type | Notes |
|---|---|---|
| `patient_id` | TEXT PK | Uppercase |
| `name` | TEXT | Required |
| `age` | INTEGER | 0–150 |
| `gender` | TEXT | Male/Female/Other |
| `created_at` | TIMESTAMP | Default current time |
| `updated_at` | TIMESTAMP | Updated on write |

#### `mri_scans`

| Column | Type | Notes |
|---|---|---|
| `scan_id` | TEXT PK | `SCAN-YYYYMMDD-XXXXXXXX` |
| `patient_id` | TEXT FK | Cascade delete |
| `filename` | TEXT | Sanitized upload name |
| `predicted_class` | TEXT | Raw class |
| `confidence` | REAL | Percentage |
| `probabilities` | TEXT | JSON object |
| `model_name` | TEXT | `VGG16 Transfer Learning` |
| `gradcam_image` | TEXT | Overlay data URI |
| `gradcam_heatmap` | TEXT | Raw heatmap data URI; added additively |
| `report_reference` | TEXT UNIQUE | `REP-YYYYMMDD-XXXXXXXX` |
| `analysis_date` | TEXT | User/current date |
| `created_at` | TIMESTAMP | Insert time |

#### `reports`

Stores report identifiers, patient/scan references, embedded `image_data`, `gradcam_data`, `gradcam_heatmap`, predicted class, display label, confidence, model architecture, summary, disclaimer, and full `html_content`.

### 7.2 Relationships and integrity

- `mri_scans.patient_id` → `patients.patient_id` (cascade)
- `reports.scan_id` → `mri_scans.scan_id` (cascade)
- `reports.patient_id` → `patients.patient_id` (cascade)
- Foreign keys are enabled on every connection.
- Lookup indices cover patient scans, creation time, report references, and scan/report joins.

### 7.3 Migrations and legacy compatibility

`init_db()` is idempotent and runs on startup. It:

1. Creates missing tables and indices.
2. Adds `gradcam_heatmap` to `mri_scans` if absent.
3. Adds `gradcam_heatmap` to `reports` if absent.

Existing scans/reports keep `NULL` heatmaps because the historical artifact does not contain one. The application renders only what exists. A compatibility test copied the production database, migrated the copy, exercised real routes, and verified that the original file hash did not change.

### 7.4 Database isolation for tests

`NEUROSCAN_DB_PATH` overrides the default path before `database` is imported:

```powershell
$env:NEUROSCAN_DB_PATH = "$env:TEMP\neuroscan_test.db"
python backend\main.py
```

Tests should always use this override. Production is `backend/data/neuroscan.db` and must not be committed.

---

## 8. Report generation

### 8.1 Report contents

Each report includes:

- project and report title,
- patient name, ID, age, gender,
- scan ID and report reference,
- analysis timestamp,
- uploaded filename,
- original MRI,
- raw Grad-CAM heatmap,
- Grad-CAM overlay,
- highest-class probability,
- all four softmax probabilities,
- model architecture and input/Grad-CAM facts,
- objective AI summary,
- medical disclaimer,
- print and download controls.

### 8.2 Summary policy

The AI summary describes only:

- which class received the highest softmax probability,
- the probability value,
- the complete ranking,
- the fact that the output is statistical.

It does not invent anatomical findings, lesion descriptions, measurements, staging, or clinical recommendations. The normal class is reported exactly as **“No Tumor Detected by AI Model.”**

### 8.3 HTML escaping and portability

All text fields are escaped before insertion into HTML. The report embeds images as data URIs, so it remains readable offline and does not depend on the original upload file. That also means reports are large and contain identifiable image data; they must be protected accordingly.

---

## 9. Frontend

### 9.1 Structure

- `index.html` — semantic sections: hero, overview, classes, technology, workflow, analysis workspace, records, limitations, footer.
- `style.css` — dark navy/cyan/turquoise/purple/emerald design system, responsive breakpoints at 1060, 860, 680 px, print styles.
- `app.js` — API client, validation, upload flow, result rendering, history, API settings modal, health polling.

### 9.2 Required content

The homepage includes:

- the exact heading **“Intelligent Brain MRI Analysis”**,
- CTAs **“Start MRI Analysis”** and **“Explore Technology”**,
- project overview,
- four-class explanation,
- VGG16 architecture and Grad-CAM explanation,
- six-step workflow,
- verified metrics and limitations,
- footer disclaimer.

### 9.3 Results view

The results view displays:

- real uploaded original MRI,
- genuine raw Grad-CAM heatmap,
- genuine Grad-CAM overlay,
- four probability bars,
- patient, scan, report, and latency identifiers,
- AI summary,
- view report, print report, download HTML, and patient history actions,
- visible “not a diagnosis” disclaimer.

### 9.4 Motion policy

- Page-load, hero, and reveal animations are restricted to the marketing/homepage experience.
- `.scan-image` elements have no animation or transition.
- `prefers-reduced-motion: reduce` suppresses animations/transitions globally, forces reveal content visible, and is mirrored by a `no-motion` body class.
- No MRI or Grad-CAM image is animated.

### 9.5 API configuration

The frontend defaults to `http://127.0.0.1:5000`. A settings modal allows another endpoint, stored in `localStorage` under `neuroscan_api_base`. This is a convenience for local development, not an authentication mechanism.

---

## 10. Security and privacy

### 10.1 Current controls

- Filenames are sanitized with `secure_filename`.
- File extension allow-list and 16 MB request cap.
- PIL must successfully decode the image.
- SQL values are parameterized.
- Report text is HTML-escaped.
- Report responses set `no-store` and `nosniff`.
- Database foreign keys are enabled.
- `.env`, databases, datasets, uploads, logs, and test artifacts are git-ignored.
- Tests use an isolated database path and production compatibility testing was read-only.

### 10.2 Prototype gaps

This is not production-ready for real PHI:

- No authentication or authorization.
- CORS allows all origins.
- No TLS termination in the development server.
- No rate limiting or abuse controls.
- SQLite is not encrypted at rest.
- No audit log of who viewed which patient record.
- No retention/deletion policy.
- No CSRF strategy for cookie-based deployment (the current API is token-less).
- No malware scanning beyond image decoding.
- No formal threat model or penetration test.

Before any real deployment, add authentication, role-based access, TLS, CSRF protection where applicable, encrypted storage, audit logging, backups, retention controls, least-privilege network policy, and a HIPAA/GDPR/clinical governance review. Do not expose the development server publicly.

---

## 11. Configuration and deployment

### 11.1 Local configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | Flask listening port |
| `HOST` | `0.0.0.0` | Flask bind address |
| `NEUROSCAN_DB_PATH` | `backend/data/neuroscan.db` | SQLite override |
| `TF_ENABLE_ONEDNN_OPTS` | unset | Set `0` to disable oneDNN custom ops |

### 11.2 Local run

```powershell
.\.venv\Scripts\python.exe backend\main.py
python -m http.server 8000 --directory frontend
```

### 11.3 Container

`backend/Dockerfile` builds a Python 3.11 image, runs as a non-root user, exposes port 5000, includes a health check, and starts gunicorn with one worker and four threads.

```powershell
cd backend
docker build -t neuroscan-ai .
docker run --rm -p 5000:5000 -e PORT=5000 neuroscan-ai
```

Do not bake a production database or patient data into the image. Mount a protected volume or use a managed database for persistence.

### 11.4 Static frontend hosting

`vercel.json` rewrites `/` and static requests to `frontend/`. Update the API base URL in the frontend configuration for the deployed backend. The default localhost URL will not reach a remote service.

### 11.5 Model artifact handling

`backend/model/model.weights.h5` is about 128 MB and is tracked with Git LFS (`.gitattributes`). A clone without LFS will contain a pointer file and model loading will fail. Run `git lfs install` and `git lfs pull` before starting the backend.

---

## 12. Evaluation methodology

The deployed weights were evaluated on all 1,600 test images:

- 400 glioma
- 400 meningioma
- 400 no tumor
- 400 pituitary

Preprocessing matched inference: RGB, bilinear resize to 128 × 128, divide by 255. Batched inference used batch size 64. Predictions were taken as `argmax` over the four softmax outputs.

### 12.1 Results

| Metric | Value |
|---|---:|
| Accuracy | 90.94% (1,455/1,600) |
| Macro precision | 92.00% |
| Macro recall | 90.94% |
| Macro F1 | 90.64% |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Glioma | 97.54% | 69.25% | 80.99% |
| Meningioma | 78.94% | 96.50% | 86.84% |
| No tumor | 95.20% | 99.25% | 97.18% |
| Pituitary | 96.34% | 98.75% | 97.53% |

### 12.2 Confusion matrix

| True \ Predicted | Glioma | Meningioma | No tumor | Pituitary |
|---|---:|---:|---:|---:|
| Glioma | **277** | 97 | 18 | 8 |
| Meningioma | 6 | **386** | 1 | 7 |
| No tumor | 1 | 2 | **397** | 0 |
| Pituitary | 0 | 4 | 1 | **395** |

### 12.3 Interpretation

- Meningioma, no-tumor, and pituitary classes perform strongly on this split.
- Glioma recall is the dominant weakness. 97 glioma images were classified as meningioma.
- Meningioma precision is lower (78.94%) because it absorbs many glioma errors.
- High overall accuracy does not imply safe glioma screening. A missed glioma is the most consequential error pattern in this evaluation.
- The test split is internal to the project. No external, multi-site, or prospective validation exists.

---

## 13. Testing summary

| Suite | Result |
|---|---|
| Backend integration (isolated DB) | 58/58 |
| Browser UI audit | 48/48 |
| Responsive/motion audit | 24/24 |
| Generated report layout audit | 12/12 |
| Production DB copy compatibility | Passed |
| Full test-set evaluation | 1,600 images |

See [PHASE_TEST_REPORT.md](PHASE_TEST_REPORT.md) for exact commands, assertions, environment, and untested areas.

---

## 14. Known limitations and future work

### Model and data

- Glioma recall is 69.25%; this is not sufficient for clinical use.
- The model accepts one 128×128 slice, losing volumetric and acquisition context.
- Only four classes are represented.
- No external validation, calibration analysis, subgroup analysis, or prospective study.
- No uncertainty estimation beyond the softmax distribution.
- Grad-CAM is explanatory but not a segmentation or lesion-localization ground truth.

### Engineering

- SQLite is suitable for a single-node prototype, not concurrent clinical workloads.
- No authentication, audit trail, or encryption at rest.
- The frontend depends on a separately hosted static server.
- No automated CI pipeline is configured.
- Test scripts in `scratch/` are local and not part of the tracked repository.
- Report persistence failures are logged but not transactional with scan persistence.

### Responsible next steps

1. Add authentication, authorization, audit logging, and encrypted storage before handling real PHI.
2. Validate on an external, multi-site dataset and report subgroup performance.
3. Improve glioma sensitivity with class-balanced training or targeted augmentation, then re-evaluate the full matrix.
4. Calibrate probabilities and add uncertainty reporting.
5. Add automated CI for backend, frontend, migration, and regression tests.
6. Obtain clinical and regulatory review before any patient-facing use.

---

## 15. Maintenance notes

- Keep class order unchanged unless the model is retrained.
- Keep preprocessing aligned with training; do not silently switch interpolation or normalization.
- Keep `block5_conv3` as the Grad-CAM layer unless the model graph changes.
- Run `init_db()` after schema changes; migrations must remain additive and idempotent.
- Never commit `backend/data/neuroscan.db`, `dataset/`, `model_test/`, uploads, logs, or `.env`.
- Verify the model weights with Git LFS after cloning.
- Re-run the full evaluation whenever weights, preprocessing, or class mapping changes.

---

## 16. References

1. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*.
2. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV*.
3. TensorFlow and Keras documentation — model serialization and gradient tape.
4. Flask documentation — routing, request handling, and responses.
5. SQLite documentation — foreign keys, schema alteration, and transactions.

---

**Medical disclaimer:** NeuroScan AI is an academic research prototype. Its output is a statistical classification, not a diagnosis or radiological finding. “No Tumor Detected by AI Model” does not rule out disease. Clinical decisions require a qualified radiologist or physician. This software is not a certified medical device.
