import sqlite3
import os
import json
import uuid
import datetime

# Database directory and file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# The production database remains backend/data/neuroscan.db.  An environment
# override is provided so integration tests can run against a temporary copy
# without touching real patient records.
DB_PATH = os.environ.get("NEUROSCAN_DB_PATH", os.path.join(DATA_DIR, "neuroscan.db"))


def get_db_connection():
    """Returns a SQLite connection with foreign keys enabled and row access by column name."""
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initializes the database schema with patients and mri_scans tables and indices."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = get_db_connection()
    try:
        with conn:
            # 1. Patients Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. MRI Scans Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mri_scans (
                    scan_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    gradcam_image TEXT,
                    report_reference TEXT NOT NULL UNIQUE,
                    analysis_date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                );
            """)

            # 3. Reports Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    report_reference TEXT NOT NULL UNIQUE,
                    scan_id TEXT NOT NULL UNIQUE,
                    patient_id TEXT NOT NULL,
                    project_name TEXT NOT NULL DEFAULT 'NeuroScan AI',
                    analysis_datetime TEXT NOT NULL,
                    image_data TEXT NOT NULL,
                    gradcam_data TEXT,
                    predicted_class TEXT NOT NULL,
                    display_classification TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_architecture TEXT NOT NULL,
                    classification_summary TEXT NOT NULL,
                    medical_disclaimer TEXT NOT NULL,
                    html_content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id) ON DELETE CASCADE,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                );
            """)

            # 4. Indices for rapid lookups and relational integrity
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mri_patient_id ON mri_scans(patient_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mri_created_at ON mri_scans(created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mri_report_ref ON mri_scans(report_reference);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_patient_id ON reports(patient_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_scan_id ON reports(scan_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_ref ON reports(report_reference);")

            # 5. Lightweight, additive migrations for existing databases.
            #    ``gradcam_heatmap`` stores the raw Grad-CAM heatmap separately
            #    from the blended overlay (genuine model output, not a mask).
            def ensure_column(table_name, column_name, column_type):
                columns = {
                    row[1] for row in conn.execute(f"PRAGMA table_info({table_name});")
                }
                if column_name not in columns:
                    conn.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};"
                    )

            ensure_column("mri_scans", "gradcam_heatmap", "TEXT")
            ensure_column("reports", "gradcam_heatmap", "TEXT")


        print(f"Database initialized successfully at: {DB_PATH}")
        return True
    except sqlite3.Error as e:
        print(f"Failed to initialize database: {e}")
        return False
    finally:
        conn.close()

def generate_unique_scan_id():
    """Generates a unique, collision-proof MRI Scan ID (e.g. SCAN-20261030-9F2A1C)."""
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    unique_suffix = uuid.uuid4().hex[:8].upper()
    return f"SCAN-{date_str}-{unique_suffix}"

def generate_report_reference(scan_id):
    """Generates a unique clinical report reference identifier."""
    return f"REP-{scan_id.replace('SCAN-', '')}"

def get_patient(patient_id):
    """Retrieves patient record by patient_id. Returns dict or None."""
    if not patient_id:
        return None
    patient_id = patient_id.strip().upper()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT patient_id, name, age, gender, created_at, updated_at FROM patients WHERE patient_id = ?", (patient_id,))
        row = cur.fetchone()
        if row:
            # Count total scans for patient
            cur.execute("SELECT COUNT(*) AS total_scans FROM mri_scans WHERE patient_id = ?", (patient_id,))
            count_row = cur.fetchone()
            data = dict(row)
            data["total_scans"] = count_row["total_scans"] if count_row else 0
            return data
        return None
    finally:
        conn.close()

def create_or_update_patient(patient_id, name, age, gender):
    """
    Inserts a new patient or updates existing demographics.
    Validates patient_id, age, and name.
    """
    if not patient_id or not patient_id.strip():
        raise ValueError("Patient ID cannot be empty.")
    if len(patient_id.strip()) < 3:
        raise ValueError("Patient ID must be at least 3 characters in length.")
    if not name or not name.strip():
        raise ValueError("Patient name cannot be empty.")
    try:
        age = int(age)
        if age < 0 or age > 150:
            raise ValueError("Age must be between 0 and 150.")
    except (TypeError, ValueError):
        raise ValueError("Age must be a valid integer.")

    if not gender or gender.strip() not in ["Male", "Female", "Other"]:
        raise ValueError("Gender must be 'Male', 'Female', or 'Other'.")

    patient_id = patient_id.strip().upper()
    name = name.strip()
    gender = gender.strip()
    now_str = datetime.datetime.now().isoformat()

    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                INSERT INTO patients (patient_id, name, age, gender, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    name = excluded.name,
                    age = excluded.age,
                    gender = excluded.gender,
                    updated_at = excluded.updated_at;
            """, (patient_id, name, age, gender, now_str, now_str))
        return get_patient(patient_id)
    finally:
        conn.close()

def save_mri_scan(scan_id, patient_id, filename, predicted_class, confidence, probabilities, model_name, gradcam_image, report_reference, analysis_date, gradcam_heatmap=None):
    """
    Saves an MRI scan record associated with a verified patient_id.
    Ensures relational integrity and prevents duplicate scan IDs.
    """
    patient_id = patient_id.strip().upper()
    conn = get_db_connection()
    try:
        with conn:
            # Check patient existence
            cur = conn.cursor()
            cur.execute("SELECT patient_id FROM patients WHERE patient_id = ?", (patient_id,))
            if not cur.fetchone():
                raise ValueError(f"Patient with ID '{patient_id}' does not exist in the database.")

            # Check scan_id collision
            cur.execute("SELECT scan_id FROM mri_scans WHERE scan_id = ?", (scan_id,))
            if cur.fetchone():
                raise ValueError(f"Scan ID '{scan_id}' already exists.")

            prob_json = json.dumps(probabilities) if isinstance(probabilities, dict) else str(probabilities)

            conn.execute("""
                INSERT INTO mri_scans (
                    scan_id, patient_id, filename, predicted_class, confidence,
                    probabilities, model_name, gradcam_image, gradcam_heatmap,
                    report_reference, analysis_date, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scan_id,
                patient_id,
                filename,
                predicted_class,
                float(confidence),
                prob_json,
                model_name,
                gradcam_image,
                gradcam_heatmap,
                report_reference,
                analysis_date,
                datetime.datetime.now().isoformat()
            ))

        return get_mri_scan(scan_id)
    finally:
        conn.close()

def get_mri_scan(scan_id):
    """Retrieves full details of a single MRI scan by scan_id."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT s.scan_id, s.patient_id, p.name AS patient_name, p.age, p.gender,
                   s.filename, s.predicted_class, s.confidence, s.probabilities,
                   s.model_name, s.gradcam_image, s.gradcam_heatmap, s.report_reference,
                   s.analysis_date, s.created_at
            FROM mri_scans s
            JOIN patients p ON s.patient_id = p.patient_id
            WHERE s.scan_id = ?
        """, (scan_id,))
        row = cur.fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["probabilities"] = json.loads(data["probabilities"])
        except Exception:
            pass
        return data
    finally:
        conn.close()

def get_patient_scans(patient_id):
    """Retrieves all previous MRI scans associated with a Patient ID, ordered latest first."""
    patient_id = patient_id.strip().upper()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT s.scan_id, s.patient_id, s.filename, s.predicted_class, s.confidence,
                   s.probabilities, s.model_name, s.report_reference, s.analysis_date, s.created_at
            FROM mri_scans s
            WHERE s.patient_id = ?
            ORDER BY s.created_at DESC
        """, (patient_id,))
        rows = cur.fetchall()
        scans = []
        for r in rows:
            item = dict(r)
            try:
                item["probabilities"] = json.loads(item["probabilities"])
            except Exception:
                pass
            scans.append(item)
        return scans
    finally:
        conn.close()

def save_report(report_id, report_reference, scan_id, patient_id, project_name, analysis_datetime, image_data, gradcam_data, predicted_class, display_classification, confidence, model_architecture, classification_summary, medical_disclaimer, html_content, gradcam_heatmap=None):
    """
    Saves an automatic clinical report into the reports table.
    Associates the report persistently with Patient ID and MRI Scan ID.
    """
    patient_id = patient_id.strip().upper()
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                INSERT INTO reports (
                    report_id, report_reference, scan_id, patient_id, project_name,
                    analysis_datetime, image_data, gradcam_data, gradcam_heatmap,
                    predicted_class, display_classification, confidence,
                    model_architecture, classification_summary, medical_disclaimer,
                    html_content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_reference) DO UPDATE SET
                    display_classification = excluded.display_classification,
                    confidence = excluded.confidence,
                    classification_summary = excluded.classification_summary,
                    gradcam_heatmap = excluded.gradcam_heatmap,
                    html_content = excluded.html_content;
            """, (
                report_id,
                report_reference,
                scan_id,
                patient_id,
                project_name,
                analysis_datetime,
                image_data,
                gradcam_data,
                gradcam_heatmap,
                predicted_class,
                display_classification,
                float(confidence),
                model_architecture,
                classification_summary,
                medical_disclaimer,
                html_content,
                datetime.datetime.now().isoformat()
            ))
        return get_report(report_reference)
    finally:
        conn.close()

def get_report(identifier):
    """
    Retrieves a report by report_reference, scan_id, or report_id.
    Joins with patients table to provide full patient demographic information.
    """
    if not identifier:
        return None
    ident = identifier.strip()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.report_id, r.report_reference, r.scan_id, r.patient_id,
                   p.name AS patient_name, p.age AS patient_age, p.gender AS patient_gender,
                   r.project_name, r.analysis_datetime, r.image_data, r.gradcam_data,
                   r.gradcam_heatmap, r.predicted_class, r.display_classification, r.confidence,
                   r.model_architecture, r.classification_summary, r.medical_disclaimer,
                   r.html_content, r.created_at, s.filename, s.probabilities
            FROM reports r
            JOIN patients p ON r.patient_id = p.patient_id
            JOIN mri_scans s ON r.scan_id = s.scan_id
            WHERE r.report_reference = ? OR r.scan_id = ? OR r.report_id = ?
        """, (ident, ident, ident))
        row = cur.fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["probabilities"] = json.loads(data["probabilities"])
        except Exception:
            pass
        return data
    finally:
        conn.close()

def get_patient_reports(patient_id):
    """Retrieves all reports for a given Patient ID, latest first."""
    if not patient_id:
        return []
    patient_id = patient_id.strip().upper()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.report_id, r.report_reference, r.scan_id, r.patient_id,
                   p.name AS patient_name, p.age AS patient_age, p.gender AS patient_gender,
                   r.project_name, r.analysis_datetime, r.predicted_class,
                   r.display_classification, r.confidence, r.model_architecture,
                   r.created_at, s.filename
            FROM reports r
            JOIN patients p ON r.patient_id = p.patient_id
            JOIN mri_scans s ON r.scan_id = s.scan_id
            WHERE r.patient_id = ?
            ORDER BY r.created_at DESC
        """, (patient_id,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
