# NeuroScan AI — Phase Test Report

**Report date:** 1 November 2026 (first verified 31 October 2026)  
**Application version:** 1.2.0  
**Scope:** backend, model inference, Grad-CAM, SQLite persistence, report generation, frontend behavior, responsive layout, generated-report layout, reduced motion, production database compatibility, and full held-out evaluation.

> This report supersedes the earlier `PHASE_6_INTEGRATION_TEST_REPORT.md`. The earlier document incorrectly described bicubic preprocessing and a Dense(512)/Dropout(0.5) classifier. The deployed artifacts use **bilinear 128×128 preprocessing** and **Flatten → Dropout(0.30) → Dense(128) → Dropout(0.20) → Dense(4)**.

---

## 1. Result summary

| Area | Result | Evidence |
|---|---:|---|
| Backend integration | **58 / 58 passed** | Isolated temporary SQLite DB |
| Browser/UI audit | **48 / 48 passed** | Headless Edge, real upload |
| Responsive/motion audit | **24 / 24 passed** | 1440, 1024, 768, 390 px |
| Generated report layout audit | **12 / 12 passed** | Real report at 1440 and 390 px, plus print media |
| Production DB compatibility | **Passed** | Byte-for-byte copy; original hash unchanged |
| Full held-out evaluation | **1,600 / 1,600 images** | 90.94% accuracy |
| Console/runtime errors in UI audit | **0** | No uncaught exceptions, console errors, or failed requests |

No test below is inferred from a previous run. Each result was produced during the verification pass on the deployed artifacts. The backend and report-layout suites were extended and re-run on 1 November 2026 after input-validation and report-CSS hardening.

---

## 2. Environment

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language |
| Python | 3.13.7 |
| TensorFlow | 2.21.0 (CPU) |
| Keras | 3.15.1 |
| Flask | 3.1.3 |
| Pillow | 12.3.0 |
| NumPy | 2.5.3 |
| Node.js | 24.21.0 |
| Browser | Microsoft Edge 156.0.4314.50 |
| Backend | `http://127.0.0.1:5000` |
| Frontend | `http://localhost:8000` |

TensorFlow GPU support is unavailable on native Windows for TensorFlow ≥ 2.11. All inference ran on CPU.

---

## 3. Test isolation and data safety

Separate database contexts were used:

| Context | Path | Purpose |
|---|---|---|
| Production | `backend/data/neuroscan.db` | Read-only source for compatibility copy |
| Backend integration | `%TEMP%\neuroscan_test_*` | Isolated route/model/DB tests |
| Browser audit | `%TEMP%\opencode\browser_audit.db` | UI-created patients/scans/reports |
| Report layout audit | `scratch\report_audit.db` | Real report for desktop/mobile/print checks |
| Production copy | `%TEMP%\neuroscan_prod_compat_*` | Additive migration and legacy route test |

The production database was never opened for writing during testing. Its SHA-256 hash was recorded before and after the compatibility test and compared byte-for-byte.

Audit-created patient records contain synthetic names/IDs. No real patient data is used in browser tests.

---

## 4. Backend integration — 58/58

Command:

```powershell
cmd /c ".venv\Scripts\python.exe scratch\test_backend_integration.py 2>&1"
```

The script sets `NEUROSCAN_DB_PATH` to a fresh temporary file before importing the backend. It then exercises the real Flask test client, model, Grad-CAM, database, and report renderer.

### 4.1 Checks covered

**Health and model metadata**

- `GET /health` returns 200.
- `model_loaded` is true.
- `database_connected` is true.
- Model class order is exactly `glioma, meningioma, notumor, pituitary`.

**Patient registration and lookup**

- `POST /patients` returns 201.
- Patient IDs are normalized to uppercase.
- `GET /patients/<id>` returns 200.

**Prediction and evidence**

- `POST /predict` returns 200 for a real test image.
- Scan ID starts with `SCAN-`.
- Report reference starts with `REP-`.
- Prediction is one of the four known classes.
- Four probabilities are returned.
- Probabilities sum to 100 within 0.5 percentage points.
- Original image is returned as a data URI.
- Raw Grad-CAM heatmap is returned as a data URI.
- Grad-CAM overlay is returned as a data URI.
- Heatmap and overlay are different images.
- Model metadata reports `block5_conv3`.
- Normal display label is exactly `No Tumor Detected by AI Model.` when applicable.

**Persistence**

- Scan row exists in SQLite.
- Raw heatmap persisted.
- Overlay persisted.
- Report row exists.
- Report heatmap persisted.
- Four probabilities persisted with the report.

**Report endpoints**

- HTML report returns 200.
- Report embeds the original MRI.
- Report embeds the raw heatmap.
- Report embeds the overlay.
- Report contains the scan ID and report reference.
- Report states VGG16.
- Report contains the disclaimer.
- HTML is escaped; no injected script executes.
- JSON report returns four probabilities.
- Download returns 200 and `Content-Disposition: attachment`.
- Patient scan history returns one scan.
- Patient report history returns one report.

**Validation and error handling**

- Missing image → 400.
- Missing patient ID → 400.
- Unknown patient → 404.
- Unknown report → 404.
- Invalid gender → 400.
- Invalid age → 400.
- Non-object JSON body → 400.
- Non-string `patient_id`, `name`, or `gender` → 400.
- Short patient ID → 400.
- Invalid `analysis_date` → 400.
- Unreadable/non-image upload → 400 (client error, not 500).
- Failed validation creates no scan or report row.

**Service metadata**

- `GET /` returns 200 and version 1.2.0.
- `/openapi.json` is valid JSON.
- `/docs` renders Swagger UI.

### 4.2 Result

```text
RESULT: 58 passed, 0 failed
ALL BACKEND INTEGRATION TESTS PASSED
EXIT_CODE=0
```

---

## 5. Browser/UI audit — 48/48

Command:

```powershell
node scratch\browser_audit_v2.js
```

The audit starts headless Edge, waits for the real backend to report healthy, seeds a real prediction, and drives the UI through Chrome DevTools Protocol.

### 5.1 Coverage

| Group | Checks |
|---|---:|
| Required homepage content | 9 |
| No hardcoded patient defaults | 4 |
| Backend health pill | 1 |
| Real SQLite patient history | 5 |
| Upload flow through file input | 4 |
| Results, images, probabilities, report actions | 18 |
| History action from results | 2 |
| Screenshots (desktop hero/results, mobile) | 2 |
| Console/runtime/network diagnostics | 3 |
| **Total** | **48** |

### 5.2 Key assertions

- Hero heading is exactly **“Intelligent Brain MRI Analysis.”**
- Both required CTAs are present: **“Start MRI Analysis”** and **“Explore Technology.”**
- Overview, classes, technology, workflow, analysis, records, and limitations sections exist.
- Verified 90.94% accuracy and 69.25% glioma recall are visible.
- VGG16 and Grad-CAM are explained.
- Patient ID, name, age, and gender fields start empty.
- Health pill reaches an online state.
- Records lookup returns the database patient name and stored scans.
- A real image is set on the file input, previewed, and submitted.
- Four probability values render and sum to approximately 100.
- Original MRI, raw heatmap, and overlay are all real `data:image/` URIs.
- Heatmap and overlay differ.
- Heatmap is not a placeholder.
- Summary and disclaimer are visible.
- Scan and report identifiers render.
- View, print, and download report links are correct.
- Print mode injects `window.print()`.
- Newly created scan appears in patient history.
- No uncaught JavaScript exceptions.
- No console errors.
- No failed network requests (excluding favicon).

### 5.3 Result

```text
RESULT: 48 passed, 0 failed
ALL BROWSER CHECKS PASSED
```

Screenshots were written to `scratch/ui_audit_hero.png`, `scratch/ui_audit_results.png`, and `scratch/ui_audit_mobile.png`.

---

## 6. Responsive and motion audit — 24/24

Command:

```powershell
node scratch\layout_audit.js
```

Viewports tested: **1440×1000**, **1024×900**, **768×900**, **390×844**.

Checks per viewport:

1. No horizontal overflow.
2. No off-screen elements.
3. Hero heading rendered.
4. Correct navigation mode:
   - desktop/laptop: full nav visible, hamburger hidden;
   - tablet/mobile: full nav hidden, hamburger visible.

Additional checks:

- Mobile menu opens at 768 and 390.
- `.scan-image` elements have no CSS animation.
- `.scan-image` elements have no transition.
- `prefers-reduced-motion: reduce` is detected.
- Reduced-motion body class is applied.
- Reveal elements are visible with reduced motion.
- Animation duration is effectively zero (`1e-06s`).

### 6.1 Result

```text
RESULT: 24 passed, 0 failed
```

### 6.2 Harness issues found

Two initial audit failures were traced to the test harness, not the product:

1. **Desktop nav check.** The first width was measured before the stylesheet finished applying. The harness now waits for `document.readyState === 'complete'` and a non-empty stylesheet list before asserting.
2. **Reduced-motion string comparison.** The browser reports `1e-06s` for the CSS value `0.001ms`. The assertion now parses seconds and accepts values ≤ 0.001s.

No application CSS change was required for either failure.

### 6.3 Generated report layout audit — 12/12

Command:

```powershell
node scratch\report_layout_audit.js
```

The audit seeds a real prediction, opens the backend-rendered report, and measures the actual generated HTML rather than a static mock.

Checks:

- Desktop 1440 px: no horizontal overflow, no off-screen elements, three visual-evidence images, three-column grid, disclaimer present.
- Mobile 390 px: no horizontal overflow, no off-screen elements, one-column grid, report page fits the viewport.
- Print media: action bar hidden, three-image grid preserved.
- Mobile screenshot captured at `scratch/report_audit_mobile.png`.

This audit caught a real defect: the report used an inline `grid-template-columns` value that overrode the mobile media query, so the three-image row stayed three columns at 390 px. The report now uses a CSS custom property, which the mobile and print rules can override correctly.

```text
RESULT: 12 passed, 0 failed
ALL REPORT LAYOUT CHECKS PASSED
```

---

## 7. Production database compatibility

Command:

```powershell
.\.venv\Scripts\python.exe scratch\test_production_db_compat.py
```

Method:

1. Hash `backend/data/neuroscan.db`.
2. Read table counts.
3. Copy the database to a temporary directory.
4. Point `NEUROSCAN_DB_PATH` at the copy.
5. Run `init_db()` twice to confirm idempotence.
6. Verify counts are unchanged.
7. Verify `gradcam_heatmap` columns were added additively.
8. Read a real legacy patient, scan, and report.
9. Exercise eight real Flask routes against the copy.
10. Re-hash the original production database.

Results:

```text
PASS production DB untouched (sha256 verified)
PASS record counts preserved: {'patients': 11, 'mri_scans': 29, 'reports': 26}
PASS additive migration added gradcam_heatmap to mri_scans and reports
PASS legacy patient/scan/report records readable with NULL heatmaps
PASS real Flask routes served 8 legacy-record requests
```

Routes exercised:

| Status | Route |
|---:|---|
| 200 | `/patients/PID-001` |
| 200 | `/patients/PID-001/scans` |
| 200 | `/scans/SCAN-20261030-938E875F` |
| 200 | `/reports/REP-20261030-95877A84` |
| 200 | `/reports/REP-20261030-95877A84/json` |
| 200 | `/reports/REP-20261030-95877A84/download` |
| 200 | `/patients/PID-001/reports` |
| 302 | `/scans/SCAN-20261030-95877A84/report` |

Legacy records have no original Grad-CAM artifact. Their `gradcam_heatmap` remains `NULL`; the system does not fabricate a replacement.

The production database itself was not migrated during testing. The real application applies the additive migration on its next startup.

---

## 8. Full held-out evaluation — 1,600 images

Command:

```powershell
.\.venv\Scripts\python.exe scratch\evaluate_test_set.py
```

Method:

- Dataset: `dataset/Testing`, exactly 400 images per class.
- Preprocessing: RGB, bilinear resize to 128×128, divide by 255.
- Batch size: 64.
- Prediction: `argmax` over four softmax outputs.
- No exclusions, sampling, or threshold tuning.
- CPU inference: 355.27 seconds total, approximately 222 ms/image.

### 8.1 Overall metrics

| Metric | Value |
|---|---:|
| Accuracy | **90.94%** (1,455 / 1,600) |
| Macro precision | 92.00% |
| Macro recall | 90.94% |
| Macro F1 | 90.64% |
| Weighted precision | 92.00% |
| Weighted recall | 90.94% |
| Weighted F1 | 90.64% |

### 8.2 Reserved per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 97.54% | **69.25%** | 80.99% | 400 |
| Meningioma | 78.94% | 96.50% | 86.84% | 400 |
| No tumor | 95.20% | 99.25% | 97.18% | 400 |
| Pituitary | 96.34% | 98.75% | 97.53% | 400 |

### 8.3 Confusion matrix

Rows are ground truth; columns are predictions.

| True \ Predicted | Glioma | Meningioma | No tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Glioma | **277** | 97 | 18 | 8 | 400 |
| Meningioma | 6 | **386** | 1 | 7 | 400 |
| No tumor | 1 | 2 | **397** | 0 | 400 |
| Pituitary | 0 | 4 | 1 | **395** | 400 |

### 8.4 Findings

- 277 correct, 123 incorrect glioma predictions.
- Glioma recall is **69.25%**, the dominant weakness.
- 97 of 123 glioma errors were predicted as meningioma.
- Meningioma precision is reduced to 78.94% because it absorbs those glioma false positives.
- No-tumor and pituitary classes are strong on this split (99.25% and 98.75% recall).
- The matrix is reported in full because aggregate accuracy alone obscures the glioma failure mode.

### 8.5 Interpretation limits

This is an internal test split. It is not external validation. No calibration, subgroup, site-shift, or prospective analysis was performed. These numbers must not be presented as evidence of clinical safety or effectiveness.

---

## 9. Issues found and resolved during this work

| Issue | Resolution |
|---|---|
| Documentation claimed bicubic preprocessing | Corrected to bilinear, matching `ml_model.py` and the evaluation script |
| Documentation claimed Dense(512)/Dropout(0.5) | Corrected to Dropout(0.30) → Dense(128) → Dropout(0.20) |
| Raw Grad-CAM heatmap was not separately available | Added genuine raw heatmap output and `gradcam_heatmap` persistence |
| Legacy production DB lacked heatmap columns | Added additive, idempotent migration with compatibility test |
| Patient UI contained hardcoded defaults | Removed; fields start empty and history queries the API |
| Reports contained inferred clinical language | Rewrote summary generation to describe model output only |
| Normal label varied | Standardized exact label “No Tumor Detected by AI Model.” |
| Print action downloaded/opened without printing | Added `?print=1` mode with a load-triggered print call |
| Responsive audit measured before CSS applied | Harness now waits for complete document/stylesheets |
| Reduced-motion assertion expected an exact string | Harness now parses effective animation duration |
| Model loss label in UI said categorical cross-entropy | Corrected to sparse categorical cross-entropy, matching the trained config |
| Dataset/uploads/PHI were not fully ignored | Expanded `.gitignore` for datasets, DICOM/NIfTI, uploads, DB journals |

No model retraining or architecture substitution was performed.

---

## 10. What was not tested

The following remain outside this verification:

- External or multi-site validation.
- Prospective clinical evaluation.
- Radiologist reader study.
- Probability calibration.
- Fairness/subgroup analysis by age, sex, scanner, or site.
- Security penetration testing, authentication/authorization, or PHI compliance.
- Load, concurrency, or stress testing.
- Multi-worker TensorFlow deployment.
- Cross-browser testing beyond Microsoft Edge.
- Physical mobile devices and screen readers.
- Database corruption or disk-failure recovery.
- Disaster recovery and backup restoration.
- Regulatory review or medical-device certification.

These are material gaps for any real-world use. The system remains a research prototype.

---

## 11. Reproduction checklist

1. Activate the virtual environment and install `requirements.txt`.
2. Ensure the model weights are present (Git LFS pull if needed).
3. Ensure `dataset/Testing` contains the 1,600 evaluation images.
4. Run the backend integration suite with an isolated DB.
5. Start the backend and static frontend for browser tests.
6. Run the browser, responsive, and generated-report audits.
7. Run the production-copy compatibility test.
8. Run the full evaluation.
9. Confirm the production database hash is unchanged.

Commands:

```powershell
.\.venv\Scripts\python.exe scratch\test_backend_integration.py
node scratch\browser_audit_v2.js
node scratch\layout_audit.js
node scratch\report_layout_audit.js
.\.venv\Scripts\python.exe scratch\test_production_db_compat.py
.\.venv\Scripts\python.exe scratch\evaluate_test_set.py
```

`scratch/` is git-ignored. These are local verification scripts, not a packaged CI suite.

---

## 12. Conclusion

On the verified environment, the deployed VGG16 artifact, Grad-CAM implementation, Flask API, SQLite persistence, report generation, and responsive frontend operate as documented:

- 58/58 backend integration checks,
- 48/48 browser checks,
- 24/24 responsive/motion checks,
- 12/12 generated-report layout checks,
- successful legacy database compatibility,
- 90.94% accuracy over all 1,600 held-out images,
- 69.25% glioma recall, reported without concealment.

The system is functionally complete as an academic prototype, but it is not clinically validated and must not be used for diagnosis or patient care.
