import datetime
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Ensure module import path
MODULES_DIR = Path(__file__).parent
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

from rework_module import (
    get_current_user,
    rework_router,
)

# App Setup for Testing
app = FastAPI()
app.include_router(rework_router)
client = TestClient(app, raise_server_exceptions=False)

# Test User Identities
ADMIN_USER = {"user_id": 1, "username": "admin_user", "role": "admin"}
OPERATOR_USER = {"user_id": 2, "username": "operator_user", "role": "operator"}
VIEWER_USER = {"user_id": 3, "username": "viewer_user", "role": "viewer"}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_dependency_overrides():
    """Guarantees FastAPI dependency overrides are wiped after every test execution."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_db():
    """Mocks PostgreSQL database connection context manager and cursor."""
    with patch("rework_module.get_db_connection") as mock_conn_func:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.rowcount = 1
        mock_conn_func.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        yield mock_cursor, mock_conn


@pytest.fixture
def valid_update_payload():
    """Baseline valid payload representing a standard rework update request."""
    return {
        "request_id": 500,
        "rework_type": "Wire Bonding Rework",
        "status": "completed",
        "notes": "Wire replaced and optical power verified.",
        "operator": "op_john",
        "disposition": "Continue Production",
        "device_serial_number": None,
        "scanned_mo_number": None,
    }


@pytest.fixture
def db_request_row():
    """Simulates an active rework request row fetched from the database."""
    return {
        "status": "in_progress",
        "mo_number": "MO-88100",
        "housing_serial": "HS-2026-99",
        "section": "Wire Bonding",
        "device_type": "TX_MODULE",
        "start_time": datetime.datetime(2026, 8, 13, 10, 0, 0),
    }


# ============================================================================
# 1. ACCESS CONTROL BEHAVIOR
# ============================================================================

class TestAccessControlBehavior:

    def test_viewer_role_is_forbidden_from_updating(self, valid_update_payload):
        """BEHAVIOR: Users with 'viewer' role must receive 403 Forbidden."""
        app.dependency_overrides[get_current_user] = lambda: VIEWER_USER

        response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 403
        assert response.json()["detail"] == "Viewers cannot update rework requests"

    @pytest.mark.parametrize("user_context", [ADMIN_USER, OPERATOR_USER], ids=["admin", "operator"])
    def test_authorized_roles_can_update_requests(
        self, user_context, mock_db, valid_update_payload, db_request_row
    ):
        """BEHAVIOR: Users with administrative or operator privileges are authorized."""
        app.dependency_overrides[get_current_user] = lambda: user_context
        cursor, _ = mock_db
        cursor.fetchone.return_value = db_request_row

        response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 200
        assert response.json()["success"] is True


# ============================================================================
# 2. BUSINESS RULE & VALIDATION BEHAVIOR
# ============================================================================

class TestBusinessRuleValidation:

    def test_returns_404_when_request_id_not_found(self, mock_db, valid_update_payload):
        """BEHAVIOR: Attempting to update a non-existent request yields 404 Not Found."""
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = None

        response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 404
        assert response.json()["detail"] == "Rework request 500 not found"

    @pytest.mark.parametrize("closed_status", ["completed", "cancelled", "rejected", "closed"])
    def test_rejects_updates_on_closed_or_terminal_requests(
        self, closed_status, mock_db, valid_update_payload, db_request_row
    ):
        """BEHAVIOR: Updates are rejected if the request is not in 'pending' or 'in_progress' state."""
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        db_request_row["status"] = closed_status
        cursor.fetchone.return_value = db_request_row

        response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 400
        assert f"already {closed_status}" in response.json()["detail"]

    @pytest.mark.parametrize("invalid_disposition", ["", "   ", "Scrap", "Scrap Device", "Return to Vendor"])
    def test_rejects_invalid_or_missing_disposition(
        self, invalid_disposition, mock_db, valid_update_payload, db_request_row
    ):
        """BEHAVIOR: Disposition must be explicitly 'NCM' or 'Continue Production'."""
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = db_request_row

        valid_update_payload["disposition"] = invalid_disposition

        response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 400
        assert "Next step is required" in response.json()["detail"]


# ============================================================================
# 3. MODULATOR CHIP REPLACEMENT SPECIFIC BEHAVIOR
# ============================================================================

class TestModulatorChipReplacementBehavior:

    @pytest.mark.parametrize(
        "device_serial, scanned_mo, expected_error_snippet",
        [
            (None, "MO-88100", "Device Serial Number is required"),
            ("", "MO-88100", "Device Serial Number is required"),
            ("   ", "MO-88100", "Device Serial Number is required"),
            ("DEV-9912", None, "MO Number scan is required"),
            ("DEV-9912", "", "MO Number scan is required"),
            ("DEV-9912", "   ", "MO Number scan is required"),
        ],
        ids=[
            "null-device-serial",
            "empty-device-serial",
            "whitespace-device-serial",
            "null-mo-scan",
            "empty-mo-scan",
            "whitespace-mo-scan",
        ]
    )
    def test_modulator_replacement_requires_valid_serials(
        self, device_serial, scanned_mo, expected_error_snippet, mock_db, valid_update_payload, db_request_row
    ):
        """BEHAVIOR: Modulator Chip Replacement strictly enforces scanned serial identifiers."""
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = db_request_row

        valid_update_payload["rework_type"] = "Modulator Chip Replacement"
        valid_update_payload["device_serial_number"] = device_serial
        valid_update_payload["scanned_mo_number"] = scanned_mo

        response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 400
        assert expected_error_snippet in response.json()["detail"]

    @patch("rework_module.notify_module_users", new_callable=AsyncMock)
    def test_modulator_replacement_forces_ncm_and_scraps_device(
        self, mock_notify, mock_db, valid_update_payload, db_request_row
    ):
        """
        BEHAVIOR:
        Modulator Chip Replacement automatically:
        1. Overrides disposition to 'NCM'.
        2. Updates device status to 'scrap'.
        3. Returns `device_scrapped: True` in the output contract.
        """
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = db_request_row
        cursor.rowcount = 1

        valid_update_payload["rework_type"] = "Modulator Chip Replacement"
        valid_update_payload["device_serial_number"] = "DEV-9912"
        valid_update_payload["scanned_mo_number"] = "MO-88100"
        valid_update_payload["disposition"] = "Continue Production"  # User attempts to bypass

        response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 200
        payload = response.json()

        assert payload["success"] is True
        assert payload["disposition"] == "NCM"
        assert payload["device_scrapped"] is True
        assert "marked as scrap" in payload["message"]


# ============================================================================
# 4. WORKFLOW & DOCUMENT GENERATION BEHAVIOR
# ============================================================================

class TestWorkflowOutcomes:

    @patch("rework_module.generate_rework_assembly_rider", new_callable=AsyncMock)
    @patch("rework_module.notify_module_users", new_callable=AsyncMock)
    def test_continue_production_triggers_assembly_rider_generation(
        self, mock_notify, mock_rider, mock_db, valid_update_payload, db_request_row
    ):
        """
        BEHAVIOR:
        When disposition is 'Continue Production', an Assembly Rider PDF is generated
        and its file path is returned in the API contract.
        """
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = db_request_row
        mock_rider.return_value = "/app/rework_results/Assembly_Rider_DEV123.pdf"

        valid_update_payload["disposition"] = "Continue Production"

        response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 200
        payload = response.json()

        assert payload["success"] is True
        assert payload["disposition"] == "Continue Production"
        assert payload["assembly_rider_generated"] is True
        assert payload["assembly_rider_path"] == "/app/rework_results/Assembly_Rider_DEV123.pdf"
        mock_rider.assert_awaited_once_with(
            housing_serial="HS-2026-99",
            section="Wire Bonding",
            device_type="TX_MODULE",
            mo_number="MO-88100",
            rework_operator="op_john"
        )

    @patch("rework_module.generate_rework_assembly_rider", new_callable=AsyncMock)
    @patch("rework_module.notify_module_users", new_callable=AsyncMock)
    def test_ncm_disposition_bypasses_assembly_rider_generation(
        self, mock_notify, mock_rider, mock_db, valid_update_payload, db_request_row
    ):
        """
        BEHAVIOR:
        When disposition is 'NCM', Assembly Rider PDF generation is bypassed.
        """
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = db_request_row

        valid_update_payload["disposition"] = "NCM"

        response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 200
        payload = response.json()

        assert payload["success"] is True
        assert payload["disposition"] == "NCM"
        assert payload["assembly_rider_generated"] is False
        assert payload["assembly_rider_path"] is None
        mock_rider.assert_not_called()


# ============================================================================
# 5. RESILIENCE & FAULT TOLERANCE BEHAVIOR
# ============================================================================

class TestSystemResilienceBehavior:

    @patch("rework_module.generate_rework_assembly_rider", side_effect=RuntimeError("PDF Rendering Fault"))
    @patch("rework_module.notify_module_users", new_callable=AsyncMock)
    def test_pdf_generation_failure_is_non_fatal(
        self, mock_notify, mock_rider, mock_db, valid_update_payload, db_request_row
    ):
        """
        BEHAVIOR:
        An unexpected exception during PDF generation does not fail the primary update transaction.
        The record update completes successfully while flagging `assembly_rider_generated: False`.
        """
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = db_request_row

        valid_update_payload["disposition"] = "Continue Production"

        response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 200
        payload = response.json()

        assert payload["success"] is True
        assert payload["assembly_rider_generated"] is False
        assert payload["assembly_rider_path"] is None

    def test_unhandled_database_exception_returns_500(self, valid_update_payload):
        """BEHAVIOR: Unhandled database failures map gracefully to HTTP 500."""
        app.dependency_overrides[get_current_user] = lambda: ADMIN_USER

        with patch("rework_module.get_db_connection", side_effect=Exception("Database Connection Timeout")):
            response = client.post("/update", json=valid_update_payload)

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to update rework request"