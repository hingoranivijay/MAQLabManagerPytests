import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from burn_in_module import burn_in_router

# ============================================================================
# SETUP & FIXTURES
# ============================================================================

app = FastAPI()
app.include_router(burn_in_router)


@pytest.fixture
def client():
    """Provides a fresh FastAPI TestClient instance for route requests."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_db():
    """
    Mocks the database context manager boundary.
    Yields (mock_cursor, mock_connection) for query configuration.
    """
    with patch("burn_in_module.get_db_connection") as mock_conn_func:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_conn_func.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        yield mock_cursor, mock_conn


# ============================================================================
# BEHAVIOR TESTS: /reports/{report_id}/download
# ============================================================================

def test_download_report_success(client, mock_db, tmp_path):
    """
    Behavior: Successfully streams the report file as an attachment with correct headers
    when a valid record exists in the database and the file is present on disk.
    """
    cursor, _ = mock_db
    
    # Create an actual physical file in pytest's temporary directory
    pdf_filename = "SN998811_burn_in_report_20260812.pdf"
    pdf_file = tmp_path / pdf_filename
    pdf_content = b"%PDF-1.4 dummy pdf content"
    pdf_file.write_bytes(pdf_content)

    cursor.fetchone.return_value = (str(pdf_file), "SN998811")

    response = client.get("/reports/101/download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert f'attachment; filename="{pdf_filename}"' in response.headers["content-disposition"]
    assert response.content == pdf_content


@pytest.mark.parametrize(
    "db_result, expected_detail",
    [
        (None, "Report not found"),
        (("", "SN-EMPTY"), "Report file not found on disk"),
        (None, "Report not found"),
        (("/non/existent/path/report.pdf", "SN-MISSING"), "Report file not found on disk"),
    ],
    ids=["record_not_in_db", "null_path_in_db", "missing_record", "file_missing_from_disk"]
)
def test_download_report_not_found_scenarios(client, mock_db, db_result, expected_detail):
    """
    Behavior: Returns HTTP 404 with a clear error message whenever either
    the database entry or the physical file on disk cannot be found.
    """
    cursor, _ = mock_db
    
    # Handle the tuple setup if path is explicitly None
    if db_result is None:
        cursor.fetchone.return_value = None
    elif db_result[0] == "":
        cursor.fetchone.return_value = (None, db_result[1])
    else:
        cursor.fetchone.return_value = db_result

    response = client.get("/reports/102/download")

    assert response.status_code == 404
    assert response.json()["detail"] == expected_detail


@pytest.mark.parametrize(
    "endpoint_url, expected_status",
    [
        ("/reports/not-an-integer/download", 422),  # FastAPI schema validation failure
    ],
    ids=["invalid_path_parameter_type"]
)
def test_download_report_input_validation(client, endpoint_url, expected_status):
    """
    Behavior: Rejects invalid request parameter types before executing endpoint logic.
    """
    response = client.get(endpoint_url)
    assert response.status_code == expected_status


def test_download_report_handles_database_failure(client):
    """
    Behavior: Gracefully returns an HTTP 500 status when the database system fails.
    """
    with patch("burn_in_module.get_db_connection", side_effect=Exception("Database Connection Dropped")):
        response = client.get("/reports/101/download")

    assert response.status_code == 500