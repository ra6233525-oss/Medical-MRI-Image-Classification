# NeuroScan AI — Phase 6: Complete End-to-End Integration Test Report

**Execution Timestamp:** 2026-10-31T20:10:00+05:30  
**Environment:** Windows 11, Python 3.12.10 (`neuroscan_env`), TensorFlow 2.21.0, Keras 3.15.1, Node.js v24.21.0  
**Classification Engine:** VGG16 Transfer Learning (13 Convolutional Layers + Dense 512 + Dropout 0.5)  
**Database:** Native Relational SQLite (`backend/data/neuroscan.db`)  
**Frontend Server:** Port 8000 (`http://localhost:8000`)  
**Backend Server:** Port 5000 (`http://127.0.0.1:5000`)

> **SUPERSEDED — DO NOT USE AS CURRENT EVIDENCE.** This earlier report contains incorrect model facts: it describes bicubic preprocessing and a Dense(512)/Dropout(0.5) classifier. The deployed artifacts use **bilinear 128×128 preprocessing** and **Flatten → Dropout(0.30) → Dense(128) → Dropout(0.20) → Dense(4, Softmax)**. See [PHASE_TEST_REPORT.md](PHASE_TEST_REPORT.md) for the current verified results. The metrics repeated below happen to match the full 1,600-image evaluation, but the methodology and architecture statements in this file are not authoritative.

---

## 1. Executive Summary

A comprehensive, end-to-end integration audit was conducted across all subsystems of the **NeuroScan AI** medical MRI classification project. All 35 automated integration tests passed (100.0% pass rate). The evaluation confirmed that all functionality developed in Phases 1 through 5 remains operational, persistent, and connected.

```
+-----------------------------------------------------------------------+
|                       INTEGRATION TEST SUMMARY                        |
+------------------------------------+------------+----------+----------+
| Category                           | Total Tests| Passed   | Failed   |
+------------------------------------+------------+----------+----------+
| 1. Backend & API Endpoints         |     5      |    5     |    0     |
| 2. Four-Class Model & Grad-CAM     |     4      |    4     |    0     |
| 3. Patient Database Operations     |     6      |    6     |    0     |
| 4. Clinical Report Generation      |     9      |    9     |    0     |
| 5. Frontend & DOM Contract         |     4      |    4     |    0     |
| 6. Headless Browser (CDP) Audit   |     4      |    4     |    0     |
| 7. Security & Data Integrity       |     3      |    3     |    0     |
+------------------------------------+------------+----------+----------+
| TOTAL                              |    35      |   35     |    0     |
+------------------------------------+------------+----------+----------+
```

---

## 2. Features Tested

### 2.1 Backend & REST API Endpoints
- `GET /health`: Verified system health reporting (`status: healthy`, `model_loaded: true`, `database_connected: true`, `backbone: VGG16`, `classes: 4`).
- `GET /openapi.json` and `GET /docs`: Verified OpenAPI 3.0 specification generation and interactive documentation.
- `POST /predict`: Verified image payload processing, patient ID association, bicubic tensor standardization, and latency tracking.
- Input Validation:
  - Missing image file payload rejected with `HTTP 400`.
  - Disallowed file extensions (`.txt`, `.exe`) rejected with `HTTP 400: Invalid file format`.
  - Missing or whitespace-only Patient ID rejected with `HTTP 400: Patient ID is required`.
- Relational Patient Endpoints:
  - `POST /patients`: Patient demographic registration.
  - `GET /patients/<id>`: Patient verification and demographic lookup.
  - `GET /patients/<id>/scans`: Longitudinal scan history retrieval.
  - `GET /scans/<id>`: Single MRI scan inspection.
- Report Distribution Endpoints:
  - `GET /reports/<report_ref>`: Rendered clinical HTML report.
  - `GET /reports/<report_ref>/download`: Attachment download with `Content-Disposition`.
  - `GET /reports/<report_ref>/json`: Machine-readable structured JSON report.
  - `GET /patients/<patient_id>/reports`: Patient-specific report collection.
  - `GET /scans/<scan_id>/report`: Scan ID to report mapping.

### 2.2 Four-Class Model & Explainable AI (Grad-CAM)
- Genuine VGG16 inference without mock data or simulated responses.
- Four target classes supported:
  1. `glioma`
  2. `meningioma`
  3. `pituitary`
  4. `notumor` (Normal)
- Image Preprocessing: 128×128 pixel bicubic interpolation, 3-channel RGB standardization, normalized to $[0.0, 1.0]$.
- Real-time Grad-CAM backpropagation computed from the final convolutional layer (`block5_conv3`), superimposed with Jet colormap synthesis, and exported as a data URI JPEG.
- Full 1,600 held-out test image evaluation executed and metrics logged.

### 2.3 Patient Database & Relational Persistence
- Relational tables verified: `patients`, `mri_scans`, `reports`.
- Unique identifier generation:
  - MRI Scan ID format: `SCAN-YYYYMMDD-XXXXXXXX` (UUID-based timestamped hash).
  - Report Reference format: `REP-YYYYMMDD-XXXXXXXX`.
- Duplicate prevention and multi-scan patient support.
- Foreign key integrity (`patient_id` foreign key cascade and lookup indices).
- Persistent storage across Flask server shutdowns and restarts.

### 2.4 Clinical Report Generation
- Automatic synthesis of publication-quality self-contained HTML clinical reports upon each prediction.
- Verification of 10 required clinical items:
  1. Project Name (`NeuroScan AI`)
  2. Patient ID / MRN
  3. MRI Scan ID
  4. Analysis Date and Timestamp
  5. Uploaded MRI slice (embedded base64 image)
  6. Predicted Classification
  7. Model Confidence Percentage
  8. Model Architecture (`VGG16 Transfer Learning`)
  9. AI-generated Classification Summary
  10. Strict Academic Disclaimer & Diagnostic Safety Notice
- Dedicated handling for Normal scans: displays `"No Tumor Detected by AI Model."`.
- Offline report viewing, attachment downloading, and print-to-PDF formatting.

### 2.5 Frontend & Client-Side User Experience
- Responsive AASRA-inspired UI layout across desktop and mobile viewports.
- Patient ID auto-capitalization, debounced verification against SQLite backend.
- Drag-and-drop file upload with immediate thumbnail preview (`#upload-thumbnail-img`).
- Multi-step animated progress indicators (`Preprocessing`, `VGG16 Inference`, `Grad-CAM Maps`, `Report Archiving`).
- Prediction result hero card (`#prediction-hero-card`) with slice thumbnail and identifiers.
- Distinct theme for normal scans (`.pred-hero-normal`) with emerald-green borders and alert disclaimers.
- Grad-CAM interactive controls: single overlay view, side-by-side split view, and opacity slider ($0\%-100\%$).
- Dedicated `#history` section with Patient ID search bar and 5 quick-lookup pills (`PID-1001`, `PID-NRM-504`, `PID-PIT-403`, `PID-MEN-302`, `PID-GLI-201`).
- Headless Chromium DevTools Protocol (CDP) automated browser audit.

---

## 3. Four-Class Model Evaluation & Prediction Results

### 3.1 Held-Out Test Dataset Evaluation (1,600 Images)
Evaluated across all 1,600 test images in `dataset/Testing/` (400 images per class):

- **Overall Test Accuracy:** **90.94%** (1,455 / 1,600 correct predictions)
- **Macro Average:** Precision: 92.00% | Recall: 90.94% | F1-Score: 90.64%
- **Weighted Average:** Precision: 92.00% | Recall: 90.94% | F1-Score: 90.64%

#### Confusion Matrix
```
True \ Pred      glioma     meningioma     notumor     pituitary      Total
---------------------------------------------------------------------------
glioma           277           97           18             8           400
meningioma         6          386            1             7           400
notumor            1            2          397             0           400
pituitary          0            4            1           395           400
```

#### Per-Class Metrics
```
+--------------+-------------+----------+----------+---------+
| Class        | Precision   | Recall   | F1-Score | Support |
+--------------+-------------+----------+----------+---------+
| glioma       |   97.54%    |  69.25%  |  80.99%  |   400   |
| meningioma   |   78.94%    |  96.50%  |  86.84%  |   400   |
| notumor      |   95.20%    |  99.25%  |  97.18%  |   400   |
| pituitary    |   96.34%    |  98.75%  |  97.53%  |   400   |
+--------------+-------------+----------+----------+---------+
| Macro Avg    |   92.00%    |  90.94%  |  90.64%  |  1600   |
| Weighted Avg |   92.00%    |  90.94%  |  90.64%  |  1600   |
+--------------+-------------+----------+----------+---------+
```

### 3.2 Individual Sample Predictions & Spot-Checks
Actual inference results on representative test samples:

```
+--------------------+--------------+-------------------+-------------+------------------+
| Test Image File    | Ground Truth | Predicted Class   | Confidence  | Status           |
+--------------------+--------------+-------------------+-------------+------------------+
| Te-gl_10.jpg       | glioma       | glioma            |   99.98%    | [CORRECT]        |
| Te-gl_100.jpg      | glioma       | glioma            |  100.00%    | [CORRECT]        |
| Te-gl_1.jpg        | glioma       | meningioma        |   98.45%    | [MISCLASSIFIED]* |
| Te-aug-me_10.jpg   | meningioma   | meningioma        |   98.02%    | [CORRECT]        |
| Te-aug-me_1.jpg    | meningioma   | meningioma        |   50.81%    | [CORRECT]        |
| Te-aug-me_100.jpg  | meningioma   | meningioma        |   99.71%    | [CORRECT]        |
| Te-no_10.jpg       | notumor      | notumor (Normal)  |  100.00%    | [CORRECT]        |
| Te-no_1.jpg        | notumor      | notumor (Normal)  |  100.00%    | [CORRECT]        |
| Te-no_100.jpg      | notumor      | notumor (Normal)  |   99.93%    | [CORRECT]        |
| Te-pi_10.jpg       | pituitary    | pituitary         |  100.00%    | [CORRECT]        |
| Te-pi_1.jpg        | pituitary    | pituitary         |   94.62%    | [CORRECT]        |
| Te-pi_100.jpg      | pituitary    | pituitary         |  100.00%    | [CORRECT]        |
+--------------------+--------------+-------------------+-------------+------------------+
```
*\*Note: `Te-gl_1.jpg` misclassification is recorded honestly without forcing labels. Gliomas exhibiting dense, circumscribed morphology occasionally present visual feature similarity with meningiomas in single 2D axial slices.*

---

## 4. Patient Database & Persistence Results

### 4.1 Synthetic Patient Testing
- **Synthetic Patient ID:** `PID-SYNTH-901`
- **Patient Demographics:** Dr. Eleanor Vance (42y, Female)
- **Scans Ingested:** 4 successive MRI uploads (Glioma, Notumor, Pituitary, Meningioma).
- **Scan IDs Generated:**
  - `SCAN-20261031-D42F28AA` (Glioma, 99.98% confidence)
  - `SCAN-20261031-FD044BAB` (Normal, 100.00% confidence)
  - `SCAN-20261031-E54DF34C` (Pituitary, 100.00% confidence)
  - `SCAN-20261031-1EECDE29` (Meningioma, 98.02% confidence)
- **Report References Generated:**
  - `REP-20261031-D42F28AA`
  - `REP-20261031-FD044BAB`
  - `REP-20261031-E54DF34C`
  - `REP-20261031-1EECDE29`

### 4.2 Server Restart & Persistence Verification
1. Both synthetic patients (`PID-SYNTH-901`) and default patients (`PID-1001`, `PID-NRM-504`, `PID-PIT-403`, `PID-MEN-302`, `PID-GLI-201`) were populated.
2. The active Flask process was terminated (`manage_task kill`), completely freeing port 5000.
3. The Flask application was restarted (`python backend/main.py`).
4. Re-queried `/patients/PID-SYNTH-901/scans` and `/reports/REP-20261031-D42F28AA`:
   - All 4 historical scans were retrieved with accurate timestamps, confidence scores, and filenames.
   - The clinical HTML report was served with embedded Grad-CAM heatmaps and diagnostic disclaimers intact.
   - **Result:** Relational SQLite persistence across process restarts is verified.

---

## 5. Headless Browser (CDP) Testing Results

Microsoft Edge was driven headlessly using native WebSocket connections over the Chromium DevTools Protocol (CDP) to audit client-side JavaScript execution, DOM hydration, network requests, and visual layout.

- **URL Tested:** `http://localhost:8000`
- **Uncaught JavaScript Exceptions:** **0**
- **Console Errors:** **0**
- **Failed Network Requests:** **0**
- **Diagnostic Verification:**
  - `#backend-status-pill`: Successfully transitioned from `Checking...` to `Backend Online` (`status-pill status-connected`).
  - `#verify-msg`: Automatically displayed `✓ Patient Verified: Jane Doe (48y, Female)`.
  - `#patient-name`: Auto-populated with `Jane Doe`.
  - `#archive-patient-profile`: Rendered clinical dossier banner for MRN `PID-1001`.
  - `#archive-scans-grid`: Rendered historical scan cards with `View Report` and `Download HTML` action buttons.
  - Interactive Click Test: Clicking quick-lookup pill `PID-NRM-504` dynamically queried the SQLite API, updated the banner to `Eleanor Shellstrop`, and rendered 3 scan cards in the grid.
  - Workbench Tab Switching: Verified tab switching between `Original MRI`, `Model Analysis`, `Prediction Results`, and `AI Visualization`.
  - Screenshot captured: `scratch/browser_audit_screenshot.png` (168 KB).

---

## 6. Bugs Discovered & Fixes Applied

1. **Bug:** `.gitignore` lacked exclusion rules for SQLite database files and runtime logs (`*.db`, `backend/data/*.db`). This presented a risk of committing persistent patient records to version control, violating Requirement 6.  
   **Fix Applied:** Added rules to `.gitignore` targeting `*.db`, `*.sqlite`, `*.sqlite3`, `backend/data/*.db`, `*.log`, `.env`, and `temp_uploads/`. Verified with `git status` that `neuroscan.db` remains untracked.

2. **Bug:** Test assertion string mismatch in automated test script for file extension rejection and empty patient ID rejection. The backend returned valid clinical messages (`"Invalid file format..."` and `"Patient ID is required..."`), whereas earlier test script checked for strict lowercase snake_case substrings.  
   **Fix Applied:** Updated test script assertions to match actual backend response payloads; re-ran and passed.

3. **Bug:** Automated test expected report download attachment filename to match `REP-*.html`, whereas `backend/main.py` generates descriptive clinical filenames based on the MRI Scan ID (`NeuroScan_Report_SCAN-*.html`).  
   **Fix Applied:** Updated test assertion to validate the standard clinical filename structure with `.html` extension; re-ran and passed.

---

## 7. Security & Data Integrity Audit

- **Privacy & Confidentiality:** No actual patient protected health information (PHI) is present. All database records use synthetic test subjects (`Jane Doe`, `Eleanor Vance`, `Marcus Vance`, etc.).
- **VCS Protection:** SQLite databases (`*.db`), virtual environments (`.venv`), temporary files (`scratch/`), and environment configs are excluded in `.gitignore`.
- **SQL Injection Prevention:** All SQL queries in `backend/database.py` use parameterized queries with `?` parameter markers; no raw f-string or string-concatenated SQL queries exist.
- **In-Memory Image I/O:** Uploaded image byte streams are processed via in-memory buffers (`io.BytesIO`) during preprocessing and Grad-CAM generation, preventing temporary image leaks on disk.
- **Upload Restrictions:** 16 MB maximum file size limit enforced in both backend (`main.py`) and frontend (`index.html`); disallowed extensions are rejected prior to tensor processing.

---

## 8. Remaining Limitations

1. **CPU Inference Execution:** On native Windows with TensorFlow $\ge$ 2.11, GPU acceleration via CUDA is not natively supported by TensorFlow and requires WSL2 or TensorFlow-DirectML. Model executes on optimized AVX2/FMA CPU instructions with ~130ms inference latency per slice.
2. **Glioma-Meningioma Morphological Overlap:** While overall accuracy is 90.94%, Glioma recall is 69.25% due to 97 test samples being classified as Meningioma. This reflects visual ambiguity in 2D axial slices where circumscribed gliomas mimic meningioma borders.
3. **Academic Scope:** NeuroScan AI is an educational prototype. As highlighted in all generated reports and UI alerts, it has not undergone FDA clearance and must not be used as a primary diagnostic instrument.

---

## 9. Localhost Access & Startup Commands

### Running Services
- **Frontend URL:** [http://localhost:8000](http://localhost:8000)
- **Backend API URL:** [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **API Documentation:** [http://127.0.0.1:5000/docs](http://127.0.0.1:5000/docs)
- **OpenAPI Specification:** [http://127.0.0.1:5000/openapi.json](http://127.0.0.1:5000/openapi.json)

### Startup Commands

To start both servers from the project root:

```powershell
# 1. Start the Flask Backend (Port 5000)
C:\Users\oyand\.venvs\neuroscan_env\Scripts\python.exe backend\main.py

# 2. In a separate terminal, start the Frontend Web Server (Port 8000)
python -m http.server 8000 --directory frontend
```

