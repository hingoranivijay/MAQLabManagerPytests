import os
import io
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.burn_in_module import burn_in_router, set_db_connection

# ============================================================================
# FIXTURES & TEST SETUP
# ============================================================================

@pytest.fixture
def mock_db():
    """Mocks database connection boundary for database-dependent endpoints."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # Default mock lookup return value for device_information: (device_type, MO)
    mock_cursor.fetchone.return_value = ("TestDeviceType", "MO-2026-999")

    @contextmanager
    def _db_generator():
        yield mock_conn

    set_db_connection(_db_generator)
    return mock_cursor


@pytest.fixture
def client(tmp_path, monkeypatch, mock_db):
    """
    Sets up a FastAPI TestClient with temporary file paths redirected to isolated tmp_path.
    """
    app = FastAPI()
    app.include_router(burn_in_router)

    # Redirect M_DRIVE_BASE paths to temporary test directories
    test_csv_path = tmp_path / "CSV Data"
    test_report_path = tmp_path / "Reports"
    test_graph_path = tmp_path / "Graphs"

    monkeypatch.setattr("modules.burn_in_module.CSV_SAVE_PATH", str(test_csv_path))
    monkeypatch.setattr("modules.burn_in_module.REPORT_SAVE_PATH", str(test_report_path))
    monkeypatch.setattr("modules.burn_in_module.GRAPH_SAVE_PATH", str(test_graph_path))

    return TestClient(app)


# ============================================================================
# BEHAVIOR-DRIVEN TESTS: FILE VALIDATION & UPLOAD PROCESSING
# ============================================================================

@pytest.mark.parametrize(
    "file_list, expected_status, error_substring",
    [
        # Case 1: Exceeding maximum allowed files limit (MAX_FILES = 3)
        (
            [
                ("files", ("file1.csv", b"col1,col2\n1,2", "text/csv")),
                ("files", ("file2.csv", b"col1,col2\n1,2", "text/csv")),
                ("files", ("file3.csv", b"col1,col2\n1,2", "text/csv")),
                ("files", ("file4.csv", b"col1,col2\n1,2", "text/csv")),
            ],
            400,
            "Maximum 3 files allowed",
        ),
        # Case 2: Disallowed file extension (.exe)
        (
            [
                ("files", ("unauthorized.exe", b"binary_data", "application/octet-stream"))
            ],
            400,
            "is not allowed",
        ),
    ],
)
def test_generate_report_file_validation_rejects_invalid_inputs(
    client, file_list, expected_status, error_substring
):
    """Verifies that the endpoint rejects uploads violating maximum file limits or extension restrictions."""
    data = {
        "device_serial_number": "SN-INVALID-001",
        "test_result": "PASS",
    }

    response = client.post("/reports/generate-with-files", data=data, files=file_list)

    assert response.status_code == expected_status
    assert error_substring in response.json()["detail"]


@pytest.mark.parametrize(
    "uploaded_filename, file_content, content_type",
    [
        ("test_data.csv", b"Time_s,Temperature_C,Power_dBm,Voltage_V\n0,25,10,5\n", "text/csv"),
        ("test_report.docx", b"Dummy Docx Content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("test_graph.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "image/png"),
        ("test_summary.pdf", b"%PDF-1.4 header content", "application/pdf"),
    ],
)
def test_generate_report_with_files_valid_types_succeeds(
    client, uploaded_filename, file_content, content_type
):
    """
    Verifies public behavior: valid uploaded files are processed, PDF report is generated,
    and valid payload metadata is returned.
    """
    data = {
        "device_serial_number": "SN-2026-TEST",
        "manufacturing_order_number": "MO-5555",
        "test_result": "PASS",
        "operator": "John Doe",
        "notes": "Test completed successfully",
    }
    files = [("files", (uploaded_filename, file_content, content_type))]

    response = client.post("/reports/generate-with-files", data=data, files=files)

    assert response.status_code == 200
    res_json = response.json()

    assert res_json["device_serial_number"] == "SN-2026-TEST"
    assert res_json["test_result"] == "PASS"
    assert res_json["pdf_filename"].startswith("SN-2026-TEST_burn_in_report_")
    assert res_json["pdf_filename"].endswith(".pdf")
    assert os.path.exists(res_json["report_path"])


# ============================================================================
# BEHAVIOR-DRIVEN TESTS: DATABASE PERSISTENCE
# ============================================================================

def test_save_report_persists_to_database(client, mock_db):
    """Verifies that calling /reports/save correctly inserts record and returns new ID."""
    mock_db.fetchone.return_value = (101,)  # Simulated returned record ID from RETURNING id

    payload = {
        "device_serial_number": "SN-8888",
        "manufacturing_order_number": "MO-8888",
        "test_result": "PASS",
        "operator": "Alice",
        "report_path": "/fake/path/report.pdf",
        "status": "Complete",
    }

    response = client.post("/reports/save", json=payload)

    assert response.status_code == 200
    assert response.json() == {"message": "Saved successfully", "id": 101}
    mock_db.execute.assert_called_once()