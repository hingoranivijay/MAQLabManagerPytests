import io
import os
from contextlib import contextmanager
from unittest.mock import MagicMock
import pytest
import pandas as pd
from docx import Document
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg2 import Error as Psycopg2Error

from modules.burn_in_module import burn_in_router, set_db_connection


# ============================================================================
# FIXTURES & ISOLATED ENVIRONMENT SETUP
# ============================================================================

@pytest.fixture
def api_client():
    """Mounts the router onto a FastAPI app instance for HTTP integration testing."""
    app = FastAPI()
    app.include_router(burn_in_router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_db():
    """Mocks the database connection boundary at the module level."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    @contextmanager
    def mock_db_factory():
        yield mock_conn

    set_db_connection(mock_db_factory)
    yield mock_cursor
    set_db_connection(None)


@pytest.fixture(autouse=True)
def redirect_storage_paths(tmp_path, monkeypatch):
    """
    Redirects M_DRIVE_BASE and output paths to a temporary directory 
    to isolate file system side-effects during testing.
    """
    csv_path = tmp_path / "CSV Data"
    graph_path = tmp_path / "Graphs"
    report_path = tmp_path / "Reports"

    csv_path.mkdir(parents=True, exist_ok=True)
    graph_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("modules.burn_in_module.CSV_SAVE_PATH", str(csv_path))
    monkeypatch.setattr("modules.burn_in_module.GRAPH_SAVE_PATH", str(graph_path))
    monkeypatch.setattr("modules.burn_in_module.REPORT_SAVE_PATH", str(report_path))


@pytest.fixture
def sample_csv_file():
    """Generates a valid burn-in test CSV file in memory."""
    csv_content = "Header Line\nTime_s,Temperature_C,Power_dBm,Voltage_V\n0,25.0,10.5,5.0\n3600,75.0,10.2,5.1\n"
    return ("test_data.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")


@pytest.fixture
def sample_docx_file(tmp_path):
    """Generates a Word document with the 'Test Result: _____' placeholder."""
    doc_path = tmp_path / "template.docx"
    doc = Document()
    doc.add_paragraph("Test Result: _____")
    doc.save(str(doc_path))
    
    with open(doc_path, "rb") as f:
        content = f.read()
    return ("template.docx", io.BytesIO(content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


# ============================================================================
# BEHAVIOR-DRIVEN TESTS: DEVICE LOOKUP API
# ============================================================================

@pytest.mark.parametrize(
    "serial_number, db_row, expected_status, expected_payload",
    [
        (
            "SN-12345",
            ("SensorTypeA", "MO-9988"),
            200,
            {"found": True, "device_type": "SensorTypeA", "manufacturing_order_number": "MO-9988"},
        ),
        (
            "SN-NULL-FIELDS",
            (None, None),
            200,
            {"found": True, "device_type": "", "manufacturing_order_number": ""},
        ),
        (
            "SN-UNKNOWN",
            None,
            200,
            {"found": False, "device_type": "", "manufacturing_order_number": ""},
        ),
    ],
    ids=["device_found", "device_null_fields", "device_not_found"],
)
def test_device_lookup_behavior(
    api_client, mock_db, serial_number, db_row, expected_status, expected_payload
):
    """Verifies looking up device details by serial number returns expected database mappings."""
    mock_db.fetchone.return_value = db_row

    response = api_client.get(f"/device-lookup/{serial_number}")

    assert response.status_code == expected_status
    assert response.json() == expected_payload


def test_device_lookup_unconfigured_db_raises_error(api_client):
    """Verifies that accessing lookup without an active database yields an internal server error."""
    set_db_connection(None)

    response = api_client.get("/device-lookup/SN-ERR-001")

    assert response.status_code == 500
    assert response.json()["detail"] == "Database connection not configured"


# ============================================================================
# BEHAVIOR-DRIVEN TESTS: REPORT SEARCH & LISTING
# ============================================================================

def test_list_and_search_reports_formatting(api_client, mock_db):
    """Ensures reports listing and filtering properly formats dates and returns structured records."""
    import datetime

    now = datetime.datetime(2026, 1, 1, 12, 0, 0)
    mock_db.description = [
        ("id",), ("device_serial_number",), ("manufacturing_order_number",),
        ("test_result",), ("report_path",), ("operator",),
        ("start_time",), ("end_time",), ("notes",), ("status",), ("created_at",)
    ]
    mock_db.fetchall.return_value = [
        (1, "SN-001", "MO-100", "pass", "/path/report.pdf", "Op1", now, now, "None", "Complete", now)
    ]

    # Test List Endpoint
    response = api_client.get("/reports")
    assert response.status_code == 200
    reports = response.json()["reports"]
    assert len(reports) == 1
    assert reports[0]["device_serial_number"] == "SN-001"
    assert reports[0]["created_at"] == "2026-01-01T12:00:00"

    # Test Search Endpoint
    search_resp = api_client.get("/reports/search?serial_number=SN-001")
    assert search_resp.status_code == 200
    assert len(search_resp.json()["reports"]) == 1


# ============================================================================
# BEHAVIOR-DRIVEN TESTS: REPORT GENERATION & FILE HANDLING
# ============================================================================

@pytest.mark.parametrize(
    "file_count, filenames, expected_status",
    [
        (4, ["f1.csv", "f2.csv", "f3.csv", "f4.csv"], 400),  # Exceeds max 3 files limit
        (1, ["unsupported.exe"], 400),                        # Unallowed file extension
    ],
    ids=["exceed_max_files", "invalid_extension"]
)
def test_generate_report_with_files_validation_failures(
    api_client, mock_db, file_count, filenames, expected_status
):
    """Validates boundary rules for file upload counts and extension restrictions."""
    files = [("files", (fname, io.BytesIO(b"data"), "text/plain")) for fname in filenames]
    data = {
        "device_serial_number": "SN-VAL-01",
        "test_result": "PASS"
    }

    response = api_client.post("/reports/generate-with-files", data=data, files=files)
    assert response.status_code == expected_status


def test_generate_report_with_files_end_to_end_success(
    api_client, mock_db, sample_csv_file, sample_docx_file
):
    """
    Tests end-to-end report generation with file processing, Word placeholder filling,
    and output PDF creation.
    """
    mock_db.fetchone.return_value = ("WidgetTypeX", "MO-888")

    data = {
        "device_serial_number": "SN-999",
        "manufacturing_order_number": "",
        "test_result": "PASS",
        "operator": "Alice",
        "notes": "Burn-in completed nominal"
    }
    files = [
        ("files", sample_csv_file),
        ("files", sample_docx_file)
    ]

    response = api_client.post("/reports/generate-with-files", data=data, files=files)

    assert response.status_code == 200
    payload = response.json()
    assert payload["device_type"] == "WidgetTypeX"
    assert payload["mo_number"] == "MO-888"
    assert os.path.exists(payload["report_path"])


def test_save_report_persists_to_db(api_client, mock_db):
    """Ensures report metadata is written to the database on save requests."""
    mock_db.fetchone.return_value = [42]  # Returned record ID

    payload = {
        "device_serial_number": "SN-777",
        "manufacturing_order_number": "MO-555",
        "test_result": "PASS",
        "operator": "Bob",
        "start_time": "2026-08-14T00:00:00",
        "end_time": "2026-08-14T08:00:00",
        "notes": "All clear",
        "report_path": "/tmp/test_report.pdf",
        "status": "Complete"
    }

    response = api_client.post("/reports/save", json=payload)

    assert response.status_code == 200
    assert response.json() == {"message": "Saved successfully", "id": 42}


# ============================================================================
# BEHAVIOR-DRIVEN TESTS: REPORT DOWNLOADS
# ============================================================================

def test_download_report_behavior(api_client, mock_db, tmp_path):
    """Tests downloading generated report files by ID."""
    # Scenario 1: Report record not in DB
    mock_db.fetchone.return_value = None
    response = api_client.get("/reports/999/download")
    assert response.status_code == 404
    assert response.json()["detail"] == "Report not found"

    # Scenario 2: DB record exists but file missing on disk
    mock_db.fetchone.return_value = ("/non/existent/path.pdf", "SN-100")
    response = api_client.get("/reports/100/download")
    assert response.status_code == 404
    assert response.json()["detail"] == "Report file not found on disk"

    # Scenario 3: Valid file on disk
    pdf_file = tmp_path / "valid_report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 sample content")
    mock_db.fetchone.return_value = (str(pdf_file), "SN-200")

    download_resp = api_client.get("/reports/200/download")
    assert download_resp.status_code == 200
    assert download_resp.headers["content-type"] == "application/pdf"
    assert download_resp.content == b"%PDF-1.4 sample content"