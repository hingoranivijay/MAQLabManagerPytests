import os
import datetime
from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from burn_in_module import burn_in_router, get_db_connection

app = FastAPI()
app.include_router(burn_in_router)
client = TestClient(app, raise_server_exceptions=False)


# ============================================================================
# FIXTURES & ISOLATION HELPERS
# ============================================================================

@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """
    Redirects M_DRIVE_BASE to a temporary directory for safe file I/O tests.
    Ensures tests assert actual file creation without touching production storage.
    """
    m_drive = tmp_path / "M_Drive"
    csv_path = m_drive / "CSV Data"
    graph_path = m_drive / "Graphs"
    report_path = m_drive / "Reports"

    for path in [csv_path, graph_path, report_path]:
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("burn_in_module.M_DRIVE_BASE", str(m_drive))
    monkeypatch.setattr("burn_in_module.CSV_SAVE_PATH", str(csv_path))
    monkeypatch.setattr("burn_in_module.GRAPH_SAVE_PATH", str(graph_path))
    monkeypatch.setattr("burn_in_module.REPORT_SAVE_PATH", str(report_path))

    return {
        "base": m_drive,
        "csv": csv_path,
        "graph": graph_path,
        "report": report_path,
    }


@pytest.fixture
def mock_db_connection(monkeypatch):
    """Mocks the external DB context manager boundary."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    class DummyContextManager:
        def __enter__(self):
            return mock_conn

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr("burn_in_module.get_db_connection", lambda: DummyContextManager())
    return mock_cursor


@pytest.fixture
def valid_form_data():
    """Provides a baseline multipart form payload."""
    return {
        "device_serial_number": "SN-998811",
        "manufacturing_order_number": "MO-2026-001",
        "test_result": "PASS",
        "operator": "j_doe",
        "start_time": "2026-08-12T08:00:00",
        "end_time": "2026-08-12T16:00:00",
        "notes": "All specs within range.",
    }


# ============================================================================
# INPUT VALIDATION & GUARD TESTS
# ============================================================================

class TestFileValidationGuards:

    def test_exceeding_max_files_limit_returns_400(self, valid_form_data):
        """Rejects requests with more than 3 uploaded files."""
        files = [
            ("files", ("test1.csv", b"c1", "text/csv")),
            ("files", ("test2.png", b"c2", "image/png")),
            ("files", ("test3.docx", b"c3", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ("files", ("test4.pdf", b"c4", "application/pdf")),
        ]

        response = client.post("/reports/generate-with-files", data=valid_form_data, files=files)

        assert response.status_code == 400
        assert "Maximum 3 files allowed" in response.json()["detail"]

    @pytest.mark.parametrize("filename", [
        "malicious.exe",
        "script.py",
        "payload.bat",
        "archive.zip",
        "vector.svg",
    ])
    def test_disallowed_file_extensions_return_400(self, valid_form_data, filename):
        """Rejects files with extension types outside the explicit allowlist."""
        files = [("files", (filename, b"content", "application/octet-stream"))]

        response = client.post("/reports/generate-with-files", data=valid_form_data, files=files)

        assert response.status_code == 400
        assert "not allowed" in response.json()["detail"]

    @pytest.mark.parametrize("missing_field", ["device_serial_number", "test_result"])
    def test_missing_required_form_fields_returns_422(self, valid_form_data, missing_field):
        """Validates HTTP 422 standard response when mandatory fields are omitted."""
        payload = valid_form_data.copy()
        payload.pop(missing_field)

        response = client.post("/reports/generate-with-files", data=payload)

        assert response.status_code == 422


# ============================================================================
# ENDPOINT BEHAVIOR & INTEGRATION TESTS
# ============================================================================

class TestReportGenerationBehavior:

    def test_generate_report_without_files_creates_pdf_artifact(
        self, isolated_storage, mock_db_connection, valid_form_data
    ):
        """Validates PDF generation and payload output when no supplementary files are attached."""
        mock_db_connection.fetchone.return_value = ("Transceiver-A", "MO-2026-001")

        response = client.post("/reports/generate-with-files", data=valid_form_data)

        assert response.status_code == 200
        body = response.json()

        assert body["device_serial_number"] == "SN-998811"
        assert body["device_type"] == "Transceiver-A"
        assert body["mo_number"] == "MO-2026-001"
        assert body["test_result"] == "PASS"

        # Assert physical file creation on disk
        generated_pdf = body["report_path"]
        assert os.path.exists(generated_pdf)
        assert os.path.getsize(generated_pdf) > 0

    def test_generate_report_routes_files_and_writes_to_disk(
        self, isolated_storage, mock_db_connection, valid_form_data
    ):
        """
        Validates end-to-end file persistence: uploaded CSV, PNG, and PDF files
        are saved to their respective subdirectories based on device type and year.
        """
        mock_db_connection.fetchone.return_value = ("Optical-Transmitter", "MO-2026-001")
        current_year = str(datetime.datetime.now().year)

        files = [
            ("files", ("data.csv", b"time,temp\n1,25", "text/csv")),
            ("files", ("graph.png", b"\x89PNG\r\n\x1a\n...", "image/png")),
            ("files", ("existing_doc.pdf", b"%PDF-1.4...", "application/pdf")),
        ]

        response = client.post("/reports/generate-with-files", data=valid_form_data, files=files)

        assert response.status_code == 200

        # Assert CSV file routing
        expected_csv_path = isolated_storage["csv"] / current_year / "Optical-Transmitter" / "data.csv"
        assert expected_csv_path.exists()

        # Assert PNG file routing
        expected_img_path = isolated_storage["graph"] / current_year / "Optical-Transmitter" / "graph.png"
        assert expected_img_path.exists()

        # Assert attached PDF file routing
        expected_pdf_path = isolated_storage["report"] / current_year / "Optical-Transmitter" / "existing_doc.pdf"
        assert expected_pdf_path.exists()

    @pytest.mark.parametrize(
        "db_row, form_mo, expected_mo, expected_device_type",
        [
            (("Sensor-Type-B", "MO-FROM-DB"), "", "MO-FROM-DB", "Sensor-Type-B"),
            (("Sensor-Type-B", "MO-FROM-DB"), "MO-EXPLICIT", "MO-EXPLICIT", "Sensor-Type-B"),
            (None, "MO-EXPLICIT", "MO-EXPLICIT", ""),
            (None, "", "", ""),
        ],
    )
    def test_device_info_database_resolution_rules(
        self, isolated_storage, mock_db_connection, valid_form_data,
        db_row, form_mo, expected_mo, expected_device_type
    ):
        """
        Validates database resolution behaviors for Manufacturing Order and Device Type:
        1. Auto-populates MO from DB if omitted in form.
        2. Form MO overrides DB MO if both exist.
        3. Falls back gracefully when device is missing in DB.
        """
        mock_db_connection.fetchone.return_value = db_row
        valid_form_data["manufacturing_order_number"] = form_mo

        response = client.post("/reports/generate-with-files", data=valid_form_data)

        assert response.status_code == 200
        body = response.json()
        assert body["mo_number"] == expected_mo
        assert body["device_type"] == expected_device_type

    def test_db_connection_unconfigured_returns_500(self, monkeypatch, valid_form_data):
        """Ensures endpoint returns an HTTP 500 status when the DB layer fails."""
        def raise_db_error():
            raise Exception("Database Connection Failure")

        monkeypatch.setattr("burn_in_module.get_db_connection", raise_db_error)

        response = client.post("/reports/generate-with-files", data=valid_form_data)

        assert response.status_code == 500