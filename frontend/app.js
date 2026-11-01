/* ==========================================================================
   NeuroScan AI — frontend controller
   All predictions, probabilities, heatmaps, and patient records come from the
   Flask API. Nothing in this file invents model output or patient data.
   ========================================================================== */

(function () {
    "use strict";

    var $ = function (id) { return document.getElementById(id); };
    var API_STORAGE_KEY = "neuroscan_api_base";
    var MAX_FILE_BYTES = 16 * 1024 * 1024;
    var VALID_EXTENSIONS = ["jpg", "jpeg", "jpe", "png"];

    var state = {
        selectedFile: null,
        selectedFileDataUrl: null,
        lastResult: null,
        loadingTimer: null,
        healthTimer: null,
        verifyTimer: null,
        verifiedPatient: null
    };

    /* ------------------------------ utilities ------------------------------ */
    function defaultApiBase() {
        if (window.location.protocol.indexOf("http") === 0 && window.location.port === "5000") {
            return window.location.origin;
        }
        return "http://127.0.0.1:5000";
    }

    function getApiBase() {
        try {
            var stored = window.localStorage.getItem(API_STORAGE_KEY);
            if (stored) { return stored.replace(/\/+$/, ""); }
        } catch (error) { /* storage may be unavailable */ }
        return defaultApiBase();
    }

    function setApiBase(value) {
        var clean = String(value || "").trim().replace(/\/+$/, "");
        try { window.localStorage.setItem(API_STORAGE_KEY, clean); } catch (error) { /* ignore */ }
    }

    function apiUrl(path) {
        var base = getApiBase();
        return base + (path.charAt(0) === "/" ? path : "/" + path);
    }

    function show(element) { if (element) { element.classList.remove("hidden"); } }
    function hide(element) { if (element) { element.classList.add("hidden"); } }

    function setText(id, value) {
        var element = $(id);
        if (element) { element.textContent = value == null || value === "" ? "—" : String(value); }
    }

    function toast(message) {
        var element = $("toast");
        if (!element) { return; }
        element.textContent = message;
        show(element);
        window.clearTimeout(toast._timer);
        toast._timer = window.setTimeout(function () { hide(element); }, 4200);
    }

    function formatBytes(bytes) {
        if (!bytes) { return "0 KB"; }
        if (bytes < 1024 * 1024) { return Math.max(1, Math.round(bytes / 1024)) + " KB"; }
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function displayClass(raw) {
        var key = String(raw || "").toLowerCase();
        if (key === "notumor" || key === "normal" || key === "no_tumor") {
            return "No Tumor Detected by AI Model.";
        }
        if (key === "pituitary") { return "Pituitary Tumour"; }
        if (key === "glioma") { return "Glioma"; }
        if (key === "meningioma") { return "Meningioma"; }
        return raw ? String(raw) : "Unknown";
    }

    function placeholderImage(label) {
        var safe = String(label).replace(/[<>&"]/g, "");
        var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="360" height="260" viewBox="0 0 360 260">' +
            '<rect width="360" height="260" fill="#02060d"/>' +
            '<rect x="1" y="1" width="358" height="258" fill="none" stroke="#1e293b"/>' +
            '<text x="180" y="124" text-anchor="middle" fill="#64748b" font-family="monospace" font-size="11">' + safe + '</text>' +
            '<text x="180" y="144" text-anchor="middle" fill="#334155" font-family="monospace" font-size="9">NOT RETURNED BY API</text>' +
            '</svg>';
        return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
    }

    /* ------------------------------ health ------------------------------ */
    function setBackendStatus(mode, text) {
        var pill = $("backend-status-pill");
        var label = $("backend-status-text");
        if (!pill || !label) { return; }
        pill.classList.remove("online", "offline", "status-checking");
        pill.classList.add(mode);
        label.textContent = text;
    }

    function checkBackendHealth() {
        var controller = new AbortController();
        var timer = window.setTimeout(function () { controller.abort(); }, 9000);
        return fetch(apiUrl("/health"), { signal: controller.signal, headers: { "Accept": "application/json" } })
            .then(function (response) {
                return response.json().catch(function () { return {}; }).then(function (body) {
                    return { ok: response.ok, body: body };
                });
            })
            .then(function (result) {
                if (result.ok && result.body && result.body.model_loaded) {
                    setBackendStatus("online", "Backend online");
                } else if (result.body && result.body.model_loaded === false) {
                    setBackendStatus("offline", "Model unavailable");
                } else {
                    setBackendStatus("offline", "Backend error");
                }
            })
            .catch(function () {
                setBackendStatus("offline", "Backend offline");
            })
            .finally(function () { window.clearTimeout(timer); });
    }

    /* ------------------------------ patient verification ------------------------------ */
    function resetVerifyBox(message) {
        var box = $("patient-verify-box");
        if (box) { box.classList.remove("ok", "warn", "err"); }
        setText("verify-msg", message || "Enter a patient ID to check the database.");
        hide($("btn-patient-history"));
        state.verifiedPatient = null;
    }

    function setVerifyBox(mode, message) {
        var box = $("patient-verify-box");
        if (!box) { return; }
        box.classList.remove("ok", "warn", "err");
        box.classList.add(mode);
        setText("verify-msg", message);
    }

    function verifyPatient(silent) {
        var input = $("patient-id");
        if (!input) { return Promise.resolve(null); }
        var patientId = input.value.trim().toUpperCase();
        if (patientId.length < 3) {
            resetVerifyBox("Patient ID must be at least 3 characters.");
            return Promise.resolve(null);
        }

        return fetch(apiUrl("/patients/" + encodeURIComponent(patientId)), { headers: { "Accept": "application/json" } })
            .then(function (response) {
                return response.json().catch(function () { return {}; }).then(function (body) {
                    return { status: response.status, body: body };
                });
            })
            .then(function (result) {
                if (result.status === 200 && result.body && result.body.patient) {
                    var patient = result.body.patient;
                    state.verifiedPatient = patient;
                    setVerifyBox("ok", "Verified: " + patient.name + " · " + patient.total_scans + " stored scan(s).");
                    var countButton = $("btn-patient-history");
                    if (countButton) {
                        setText("patient-scan-count", String(patient.total_scans || 0));
                        show(countButton);
                    }
                    var nameField = $("patient-name");
                    var ageField = $("patient-age");
                    var genderField = $("patient-gender");
                    if (nameField && !nameField.value.trim()) { nameField.value = patient.name || ""; }
                    if (ageField && !ageField.value) { ageField.value = patient.age != null ? patient.age : ""; }
                    if (genderField && !genderField.value) { genderField.value = patient.gender || ""; }
                    return patient;
                }
                if (result.status === 404) {
                    state.verifiedPatient = null;
                    setVerifyBox("warn", "Not registered yet. This ID will be created when the analysis runs.");
                    hide($("btn-patient-history"));
                    return null;
                }
                state.verifiedPatient = null;
                setVerifyBox("err", (result.body && result.body.error) || "Could not verify patient.");
                hide($("btn-patient-history"));
                return null;
            })
            .catch(function () {
                if (!silent) {
                    setVerifyBox("err", "Backend unreachable. Check the API endpoint.");
                }
                return null;
            });
    }

    /* ------------------------------ api settings ------------------------------ */
    function openApiModal() {
        var modal = $("api-modal");
        var input = $("api-url-input");
        var status = $("modal-ping-status");
        if (input) { input.value = getApiBase(); }
        if (status) { status.className = "inline-status hidden"; status.textContent = ""; }
        show(modal);
        if (input) { input.focus(); input.select(); }
    }

    function closeApiModal() { hide($("api-modal")); }

    function testApiConnection() {
        var input = $("api-url-input");
        var status = $("modal-ping-status");
        var url = String((input && input.value) || "").trim().replace(/\/+$/, "");
        if (!url) {
            if (status) { status.className = "inline-status err"; status.textContent = "Enter an API base URL first."; }
            return;
        }
        if (status) { status.className = "inline-status"; status.textContent = "Testing connection…"; }
        fetch(url + "/health", { headers: { "Accept": "application/json" } })
            .then(function (response) {
                return response.json().catch(function () { return {}; }).then(function (body) {
                    return { ok: response.ok, body: body };
                });
            })
            .then(function (result) {
                if (status) {
                    if (result.ok && result.body.model_loaded) {
                        status.className = "inline-status ok";
                        status.textContent = "Connected. Model loaded: " + (result.body.backbone || "VGG16") + " · v" + (result.body.version || "?");
                    } else {
                        status.className = "inline-status err";
                        status.textContent = "Reachable, but the model or database is not healthy.";
                    }
                }
            })
            .catch(function () {
                if (status) { status.className = "inline-status err"; status.textContent = "Could not reach " + url + "/health"; }
            });
    }

    /* ------------------------------ file handling ------------------------------ */
    function clearFile() {
        state.selectedFile = null;
        state.selectedFileDataUrl = null;
        var input = $("file-input");
        if (input) { input.value = ""; }
        hide($("selected-file-badge"));
        var thumb = $("upload-thumbnail-img");
        if (thumb) { thumb.removeAttribute("src"); }
        var button = $("analyze-btn");
        if (button) { button.disabled = true; }
        clearError();
    }

    function showError(message) {
        var banner = $("error-message");
        setText("error-text", message);
        show(banner);
    }

    function clearError() { hide($("error-message")); }

    function handleFile(file) {
        clearError();
        if (!file) { return; }
        var name = String(file.name || "");
        var extension = name.indexOf(".") >= 0 ? name.split(".").pop().toLowerCase() : "";
        var typeOk = VALID_EXTENSIONS.indexOf(extension) >= 0 || /^image\/(jpeg|png)$/.test(file.type || "");
        if (!typeOk) {
            showError("Unsupported file type. Upload a JPG, JPEG, or PNG image.");
            return;
        }
        if (file.size > MAX_FILE_BYTES) {
            showError("This file is " + formatBytes(file.size) + ". The maximum upload size is 16 MB.");
            return;
        }
        if (!file.size) {
            showError("The selected file is empty.");
            return;
        }

        state.selectedFile = file;
        setText("badge-file-name", name);
        setText("badge-file-size", formatBytes(file.size) + " · " + (extension || "image").toUpperCase());

        var reader = new FileReader();
        reader.onload = function () {
            state.selectedFileDataUrl = reader.result;
            var thumb = $("upload-thumbnail-img");
            if (thumb) { thumb.src = reader.result; }
            show($("selected-file-badge"));
            var button = $("analyze-btn");
            if (button) { button.disabled = false; }
        };
        reader.onerror = function () { showError("The browser could not read this file."); };
        reader.readAsDataURL(file);
    }

    /* ------------------------------ loading state ------------------------------ */
    function startLoading() {
        hide($("workspace-empty-state"));
        hide($("results-workbench"));
        show($("loading-state"));
        var button = $("analyze-btn");
        if (button) { button.disabled = true; }

        var progress = $("loading-progress");
        if (progress) { progress.style.width = "8%"; }
        var messages = [
            "Preprocessing the uploaded slice…",
            "Propagating through VGG16…",
            "Computing Grad-CAM gradients…",
            "Persisting scan and report…"
        ];
        var step = 0;
        var applyStep = function () {
            for (var index = 1; index <= 4; index += 1) {
                var pill = $("step-pill-" + index);
                if (pill) { pill.classList.toggle("active", index <= step + 1); }
            }
            setText("loading-step-text", messages[step]);
            if (progress) { progress.style.width = (12 + step * 24) + "%"; }
        };
        applyStep();
        window.clearInterval(state.loadingTimer);
        state.loadingTimer = window.setInterval(function () {
            if (step < messages.length - 1) {
                step += 1;
                applyStep();
            }
        }, 1400);
    }

    function stopLoading() {
        window.clearInterval(state.loadingTimer);
        state.loadingTimer = null;
        hide($("loading-state"));
        for (var index = 1; index <= 4; index += 1) {
            var pill = $("step-pill-" + index);
            if (pill) { pill.classList.remove("active"); }
        }
    }

    /* ------------------------------ prediction ------------------------------ */
    function populateProbabilities(probabilities) {
        var classes = ["glioma", "meningioma", "notumor", "pituitary"];
        var winner = state.lastResult && state.lastResult.prediction ? String(state.lastResult.prediction).toLowerCase() : "";
        classes.forEach(function (key) {
            var value = Number(probabilities && probabilities[key] != null ? probabilities[key] : 0);
            if (!isFinite(value)) { value = 0; }
            value = Math.max(0, Math.min(100, value));
            var bar = $("prob-bar-" + key);
            var label = $("prob-val-" + key);
            if (bar) {
                window.requestAnimationFrame(function () { bar.style.width = value.toFixed(2) + "%"; });
            }
            if (label) { label.textContent = value.toFixed(2) + "%"; }
            var row = document.querySelector('.prob-row[data-class="' + key + '"]');
            if (row) { row.classList.toggle("is-winner", key === winner); }
        });
    }

    function predictionNote(prediction) {
        var key = String(prediction || "").toLowerCase();
        if (key === "notumor") {
            return "The classifier ranked the No Tumor class highest. This is a model ranking only and does not rule out disease.";
        }
        return "The classifier ranked " + displayClass(key) + " highest. This is a statistical output, not a radiological finding.";
    }

    function displayResults(data) {
        state.lastResult = data;

        var original = data.image_data || state.selectedFileDataUrl;
        var heatmap = data.gradcam_heatmap;
        var overlay = data.gradcam_image;

        var originalImage = $("result-original-img");
        var heatmapImage = $("result-heatmap-img");
        var overlayImage = $("result-overlay-img");
        if (originalImage) { originalImage.src = original || placeholderImage("ORIGINAL MRI"); }
        if (heatmapImage) { heatmapImage.src = heatmap || placeholderImage("GRAD-CAM HEATMAP"); }
        if (overlayImage) { overlayImage.src = overlay || placeholderImage("GRAD-CAM OVERLAY"); }

        setText("result-filename", data.filename || (state.selectedFile && state.selectedFile.name) || "—");
        setText("hero-patient-id", data.patient_id || "—");
        setText("hero-scan-id", data.scan_id || "—");
        setText("hero-report-ref", data.report_reference || "—");
        setText("latency-badge", data.processing_time_ms != null ? data.processing_time_ms + " ms" : "—");

        var display = data.display_classification || displayClass(data.prediction);
        setText("hero-pred-class", display);
        setText("result-heading", display);
        var confidence = Number(data.confidence);
        if (!isFinite(confidence)) { confidence = 0; }
        setText("hero-pred-confidence", confidence.toFixed(2) + "%");
        var confidenceBar = $("confidence-bar");
        if (confidenceBar) {
            confidenceBar.style.width = "0%";
            window.requestAnimationFrame(function () { confidenceBar.style.width = confidence.toFixed(2) + "%"; });
        }
        setText("prediction-note", predictionNote(data.prediction));

        populateProbabilities(data.probabilities || {});
        setText("result-summary", data.classification_summary || "No summary was returned by the backend.");

        var viewLink = $("dossier-view-report-link");
        var printLink = $("dossier-print-report-link");
        var downloadLink = $("dossier-download-report-link");
        if (viewLink) { viewLink.href = data.report_url ? apiUrl(data.report_url) : "#"; }
        if (printLink) {
            printLink.href = data.report_url ? apiUrl(data.report_url) + "?print=1" : "#";
            if (!data.report_url) { printLink.classList.add("hidden"); } else { printLink.classList.remove("hidden"); }
        }
        if (downloadLink) {
            downloadLink.href = data.report_download_url ? apiUrl(data.report_download_url) : "#";
            if (!data.report_download_url) { downloadLink.classList.add("hidden"); } else { downloadLink.classList.remove("hidden"); }
        }

        hide($("workspace-empty-state"));
        hide($("loading-state"));
        show($("results-workbench"));

        // Keep the patient verification box in sync with the real persisted record.
        var previousScans = state.verifiedPatient && state.verifiedPatient.total_scans
            ? Number(state.verifiedPatient.total_scans)
            : 0;
        state.verifiedPatient = {
            patient_id: data.patient_id,
            name: data.patient_name,
            age: data.patient_age,
            gender: data.patient_gender,
            total_scans: previousScans + 1
        };
        setVerifyBox("ok", "Verified: " + (data.patient_name || data.patient_id) + " · scan " + (data.scan_id || "saved") + ".");
        setText("patient-scan-count", String(previousScans + 1));
        show($("btn-patient-history"));

        // Re-read the stored scan count from SQLite (source of truth).
        verifyPatient(true);

        var button = $("analyze-btn");
        if (button) { button.disabled = !state.selectedFile; }
    }

    function runAnalysis() {
        clearError();
        var form = $("patient-form");
        if (form && typeof form.reportValidity === "function" && !form.reportValidity()) {
            return;
        }
        var patientId = ($("patient-id") && $("patient-id").value || "").trim().toUpperCase();
        var name = ($("patient-name") && $("patient-name").value || "").trim();
        var age = ($("patient-age") && $("patient-age").value || "").trim();
        var gender = ($("patient-gender") && $("patient-gender").value || "").trim();
        var analysisDate = ($("analysis-date") && $("analysis-date").value || "").trim();

        if (patientId.length < 3) { showError("Patient ID must be at least 3 characters."); return; }
        if (!name) { showError("Patient full name is required."); return; }
        if (age === "" || Number(age) < 0 || Number(age) > 150) { showError("Enter a valid age between 0 and 150."); return; }
        if (!gender) { showError("Select a gender."); return; }
        if (!analysisDate) { showError("Select the exam date."); return; }
        if (!state.selectedFile) { showError("Select an MRI image before running inference."); return; }

        var formData = new FormData();
        formData.append("patient_id", patientId);
        formData.append("patient_name", name);
        formData.append("patient_age", age);
        formData.append("patient_gender", gender);
        formData.append("analysis_date", analysisDate);
        formData.append("image", state.selectedFile, state.selectedFile.name);

        startLoading();
        var controller = new AbortController();
        var timeout = window.setTimeout(function () { controller.abort(); }, 120000);

        fetch(apiUrl("/predict"), { method: "POST", body: formData, signal: controller.signal })
            .then(function (response) {
                return response.json().catch(function () { return {}; }).then(function (body) {
                    return { ok: response.ok, status: response.status, body: body };
                });
            })
            .then(function (result) {
                stopLoading();
                if (!result.ok) {
                    var message = (result.body && result.body.error) || ("Prediction failed with HTTP " + result.status + ".");
                    showError(message);
                    hide($("results-workbench"));
                    show($("workspace-empty-state"));
                    var button = $("analyze-btn");
                    if (button) { button.disabled = false; }
                    return;
                }
                displayResults(result.body || {});
                toast("Analysis complete. Report " + (result.body.report_reference || "") + " saved.");
            })
            .catch(function (error) {
                stopLoading();
                showError(error && error.name === "AbortError"
                    ? "The request timed out. The model may still be loading; try again."
                    : "Could not reach the backend. Check that the Flask server is running and the API URL is correct.");
                hide($("results-workbench"));
                show($("workspace-empty-state"));
                var button = $("analyze-btn");
                if (button) { button.disabled = false; }
            })
            .finally(function () { window.clearTimeout(timeout); });
    }

    function resetWorkbench() {
        stopLoading();
        clearFile();
        hide($("results-workbench"));
        show($("workspace-empty-state"));
        state.lastResult = null;
        window.scrollTo({ top: ($("analysis") || {}).offsetTop || 0, behavior: "smooth" });
    }

    /* ------------------------------ records ------------------------------ */
    function setRecordsStatus(mode, message) {
        var status = $("records-status");
        if (!status) { return; }
        if (!message) { hide(status); return; }
        status.className = "inline-status " + (mode || "");
        status.textContent = message;
        show(status);
    }

    function buildScanCard(scan) {
        var card = document.createElement("article");
        card.className = "scan-card";

        var top = document.createElement("div");
        top.className = "scan-card-top";
        var headingWrap = document.createElement("div");
        var heading = document.createElement("h4");
        heading.textContent = displayClass(scan.predicted_class);
        var scanId = document.createElement("p");
        scanId.className = "mono";
        scanId.textContent = scan.scan_id || "—";
        headingWrap.appendChild(heading);
        headingWrap.appendChild(scanId);
        top.appendChild(headingWrap);
        card.appendChild(top);

        var tags = document.createElement("div");
        tags.className = "scan-tags";
        var confidence = Number(scan.confidence);
        var confidenceTag = document.createElement("span");
        confidenceTag.className = "tag-confidence";
        confidenceTag.textContent = (isFinite(confidence) ? confidence.toFixed(2) : "—") + "% confidence";
        tags.appendChild(confidenceTag);
        var dateTag = document.createElement("span");
        dateTag.textContent = scan.analysis_date || "No date";
        tags.appendChild(dateTag);
        if (scan.model_name) {
            var modelTag = document.createElement("span");
            modelTag.textContent = scan.model_name;
            tags.appendChild(modelTag);
        }
        card.appendChild(tags);

        var foot = document.createElement("div");
        foot.className = "scan-card-foot";
        var filename = document.createElement("small");
        filename.textContent = scan.filename || "—";
        foot.appendChild(filename);
        if (scan.report_reference) {
            var link = document.createElement("a");
            link.href = apiUrl("/reports/" + encodeURIComponent(scan.report_reference));
            link.target = "_blank";
            link.rel = "noopener";
            link.textContent = "Open report →";
            foot.appendChild(link);
        }
        card.appendChild(foot);
        return card;
    }

    function searchRecords() {
        var input = $("archive-search-id");
        var patientId = input ? input.value.trim().toUpperCase() : "";
        if (patientId.length < 3) {
            setRecordsStatus("err", "Enter a patient ID of at least 3 characters.");
            return;
        }

        setRecordsStatus("", "Loading patient record…");
        hide($("records-profile"));
        hide($("records-empty"));
        var scansContainer = $("records-scans");
        if (scansContainer) { scansContainer.textContent = ""; }

        Promise.all([
            fetch(apiUrl("/patients/" + encodeURIComponent(patientId)), { headers: { "Accept": "application/json" } })
                .then(function (response) { return response.json().catch(function () { return {}; }).then(function (body) { return { status: response.status, body: body }; }); }),
            fetch(apiUrl("/patients/" + encodeURIComponent(patientId) + "/scans"), { headers: { "Accept": "application/json" } })
                .then(function (response) { return response.json().catch(function () { return {}; }).then(function (body) { return { status: response.status, body: body }; }); })
        ]).then(function (results) {
            var patientResult = results[0];
            var scansResult = results[1];
            if (patientResult.status === 404) {
                setRecordsStatus("err", "Patient ID “" + patientId + "” is not registered in the database.");
                var empty = $("records-empty");
                if (empty) {
                    empty.classList.remove("hidden");
                    empty.querySelector("p").textContent = "No patient record matched this ID.";
                }
                return;
            }
            if (patientResult.status !== 200 || !patientResult.body.patient) {
                setRecordsStatus("err", (patientResult.body && patientResult.body.error) || "Could not load this patient record.");
                return;
            }

            var patient = patientResult.body.patient;
            var scans = (scansResult.body && scansResult.body.scans) || [];
            setText("profile-name", patient.name);
            setText("profile-id", patient.patient_id);
            setText("profile-age", patient.age != null ? String(patient.age) : "—");
            setText("profile-gender", patient.gender || "—");
            setText("profile-scans", String(scans.length));
            show($("records-profile"));
            hide($("records-empty"));

            if (scansContainer) {
                scansContainer.textContent = "";
                scans.forEach(function (scan) { scansContainer.appendChild(buildScanCard(scan)); });
            }
            hide($("records-no-scans"));
            if (!scans.length) { show($("records-no-scans")); }
            setRecordsStatus("ok", "Loaded " + patient.name + " · " + scans.length + " stored scan(s).");
        }).catch(function () {
            setRecordsStatus("err", "Backend unreachable. Check that the Flask server is running.");
        });
    }

    function openPatientHistory() {
        var patientId = ($("patient-id") && $("patient-id").value || "").trim().toUpperCase();
        if (patientId.length < 3) { toast("Enter a patient ID first."); return; }
        var searchInput = $("archive-search-id");
        if (searchInput) { searchInput.value = patientId; }
        var records = $("records");
        if (records) { records.scrollIntoView({ behavior: "smooth", block: "start" }); }
        window.setTimeout(searchRecords, 420);
    }

    /* ------------------------------ init ------------------------------ */
    function init() {
        var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (reduced) { document.body.classList.add("no-motion"); }

        var dateField = $("analysis-date");
        if (dateField && !dateField.value) {
            var today = new Date();
            var local = new Date(today.getTime() - today.getTimezoneOffset() * 60000);
            dateField.value = local.toISOString().slice(0, 10);
        }

        // Header + mobile navigation
        var header = document.querySelector(".site-header");
        var menuButton = $("mobile-menu-btn");
        var navLinks = $("nav-links");
        var closeMenu = function () {
            if (navLinks) { navLinks.classList.remove("open"); }
            if (menuButton) { menuButton.setAttribute("aria-expanded", "false"); }
        };
        if (menuButton && navLinks) {
            menuButton.addEventListener("click", function () {
                var open = navLinks.classList.toggle("open");
                menuButton.setAttribute("aria-expanded", open ? "true" : "false");
            });
            navLinks.querySelectorAll("a").forEach(function (link) { link.addEventListener("click", closeMenu); });
        }
        window.addEventListener("scroll", function () {
            if (header) { header.classList.toggle("scrolled", window.scrollY > 12); }
        }, { passive: true });

        // Reveal on scroll (homepage shell only; result evidence is excluded)
        if (!reduced && "IntersectionObserver" in window) {
            var observer = new IntersectionObserver(function (entries) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("visible");
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
            document.querySelectorAll(".reveal").forEach(function (element) { observer.observe(element); });
        } else {
            document.querySelectorAll(".reveal").forEach(function (element) { element.classList.add("visible"); });
        }

        var year = $("footer-year");
        if (year) { year.textContent = String(new Date().getFullYear()); }

        // API settings
        ["api-settings-btn", "footer-config-btn"].forEach(function (id) {
            var button = $(id);
            if (button) { button.addEventListener("click", openApiModal); }
        });
        ["close-modal-btn", "save-api-btn"].forEach(function (id) {
            var button = $(id);
            if (button) {
                button.addEventListener("click", function () {
                    if (id === "save-api-btn") {
                        var input = $("api-url-input");
                        var value = input ? input.value.trim() : "";
                        if (!value) { toast("Enter an API base URL."); return; }
                        setApiBase(value);
                        closeApiModal();
                        checkBackendHealth();
                        toast("API endpoint saved: " + getApiBase());
                    } else {
                        closeApiModal();
                    }
                });
            }
        });
        var testButton = $("test-api-btn");
        if (testButton) { testButton.addEventListener("click", testApiConnection); }
        var modal = $("api-modal");
        if (modal) {
            modal.addEventListener("click", function (event) {
                if (event.target === modal) { closeApiModal(); }
            });
        }
        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") { closeApiModal(); }
        });

        // Patient verification
        var patientIdInput = $("patient-id");
        var verifyButton = $("btn-verify-patient");
        if (verifyButton) { verifyButton.addEventListener("click", function () { verifyPatient(false); }); }
        if (patientIdInput) {
            patientIdInput.addEventListener("input", function () {
                resetVerifyBox("Enter a patient ID to check the database.");
                patientIdInput.classList.remove("invalid");
                window.clearTimeout(state.verifyTimer);
            });
            patientIdInput.addEventListener("blur", function () {
                if (patientIdInput.value.trim().length >= 3) {
                    state.verifyTimer = window.setTimeout(function () { verifyPatient(true); }, 250);
                }
            });
        }
        var historyButton = $("btn-patient-history");
        if (historyButton) { historyButton.addEventListener("click", openPatientHistory); }

        // Upload interactions
        var dropZone = $("drop-zone");
        var fileInput = $("file-input");
        if (dropZone && fileInput) {
            dropZone.addEventListener("click", function () { fileInput.click(); });
            dropZone.addEventListener("keydown", function (event) {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    fileInput.click();
                }
            });
            ["dragenter", "dragover"].forEach(function (type) {
                dropZone.addEventListener(type, function (event) {
                    event.preventDefault();
                    dropZone.classList.add("dragover");
                });
            });
            ["dragleave", "drop"].forEach(function (type) {
                dropZone.addEventListener(type, function (event) {
                    event.preventDefault();
                    dropZone.classList.remove("dragover");
                });
            });
            dropZone.addEventListener("drop", function (event) {
                if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length) {
                    handleFile(event.dataTransfer.files[0]);
                }
            });
            fileInput.addEventListener("change", function (event) {
                handleFile(event.target.files && event.target.files[0]);
            });
        }
        var clearButton = $("clear-file-btn");
        if (clearButton) {
            clearButton.addEventListener("click", function (event) {
                event.stopPropagation();
                clearFile();
            });
        }

        var analyzeButton = $("analyze-btn");
        if (analyzeButton) { analyzeButton.addEventListener("click", runAnalysis); }
        var resetButton = $("reset-workbench-btn");
        if (resetButton) { resetButton.addEventListener("click", resetWorkbench); }
        var dossierHistoryButton = $("dossier-history-btn");
        if (dossierHistoryButton) { dossierHistoryButton.addEventListener("click", openPatientHistory); }

        // Records
        var archiveSearchButton = $("btn-archive-search");
        var archiveInput = $("archive-search-id");
        if (archiveSearchButton) { archiveSearchButton.addEventListener("click", searchRecords); }
        if (archiveInput) {
            archiveInput.addEventListener("keydown", function (event) {
                if (event.key === "Enter") { event.preventDefault(); searchRecords(); }
            });
        }
        var archiveClearButton = $("btn-archive-clear");
        if (archiveClearButton) {
            archiveClearButton.addEventListener("click", function () {
                if (archiveInput) { archiveInput.value = ""; }
                hide($("records-profile"));
                hide($("records-status"));
                var empty = $("records-empty");
                if (empty) {
                    empty.classList.remove("hidden");
                    empty.querySelector("p").textContent = "Enter a patient ID to load their demographics and every stored MRI scan.";
                }
            });
        }

        // Health polling
        checkBackendHealth();
        state.healthTimer = window.setInterval(checkBackendHealth, 30000);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
