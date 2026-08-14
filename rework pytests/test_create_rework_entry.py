# test_create_rework_entry.py - Behavior-Driven Test Suite (Refactored)

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

MODULES_DIR = Path(__file__).parent
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

from rework_module import get_current_user, rework_router

# ============================================================================
# APP SETUP & CONSTANTS
# ============================================================================

app = FastAPI()
app.include_router(rework_router)
client = TestClient(app, raise_server_exceptions=False)

ADMIN_USER = {"user_id": 1, "username": "admin_user", "role": "admin"}
OPERATOR_USER = {"user_id": 2, "username": "operator_user", "role": "operator"}
VIEWER_USER = {"user_id": 3, "username": "viewer_user", "role": "viewer"}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_dependency_overrides():
    """Wipes FastAPI dependency overrides after every test execution."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_db():
    """
    Mocks database connection context manager and tracks executed queries.
    Provides utility functions to query recorded DB actions without asserting execution index.
    """
    with patch("rework_module.get_db_connection") as mock_conn_func:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.rowcount = 1
        mock_conn_func.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        def get_query_params(statement_keyword: str):
            """Returns parameter tuple for the query containing statement_keyword."""
            for call in mock_cursor.execute.call_args_list:
                args, _ = call
                if statement_keyword.lower() in args[0].lower():
                    return args[1]
            return None

        mock_cursor.get_query_params = get_query_params
        yield mock_cursor, mock_conn


@pytest.fixture
def valid_entry_payload():
    """Provides a baseline valid payload for creating a rework entry."""
    return {
        "housing_serial": "HS-2026-001",
        "mo_number": "MO-99123",
        "device_type": "TX_MODULATOR",
        "section": "Optical Alignment",
        "reason": "Insertion loss out of spec during burn-in test.",
        "module": "Optical Alignment",
        "chip_serial_number": "CHIP-881",
        "wafer_id": "WAFER-42",
        "device_known": True,
        "scrap_reason_code": "bonding_failure",
    }


# ============================================================================
# 1. ACCESS CONTROL BEHAVIOR
# ============================================================================

class TestAccessControlBehavior:

    def test_viewer_role_is_forbidden(self, valid_entry_payload):
        """BEHAVIOR: Users with 'viewer' role are rejected with HTTP 403 Forbidden."""
        app.dependency_overrides[get_current_user] = lambda: VIEWER_USER

        response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 403
        assert response.json()["detail"] == "Viewers cannot create rework entries"

    @pytest.mark.parametrize("user_context", [ADMIN_USER, OPERATOR_USER], ids=["admin", "operator"])
    def test_authorized_roles_can_create_entries(
        self, user_context, mock_db, valid_entry_payload
    ):
        """BEHAVIOR: Authorized roles (admin, operator) can create rework entries successfully."""
        app.dependency_overrides[get_current_user] = lambda: user_context
        cursor, _ = mock_db
        cursor.fetchone.return_value = {"request_id": 100, "created_at": "2026-08-13 10:00:00"}

        response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["request_id"] == 100


# ============================================================================
# 2. INPUT VALIDATION BEHAVIOR
# ============================================================================

class TestInputValidationBehavior:

    @pytest.mark.parametrize("missing_field", [
        "housing_serial",
        "mo_number",
        "device_type",
        "section",
        "reason",
    ])
    def test_missing_mandatory_schema_fields_returns_422(
        self, missing_field, valid_entry_payload
    ):
        """BEHAVIOR: Omitting required payload fields yields HTTP 422 Unprocessable Entity."""
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        del valid_entry_payload[missing_field]

        response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 422

    @pytest.mark.parametrize("whitespace_field", [
        "housing_serial",
        "mo_number",
        "device_type",
        "section",
        "reason",
    ])
    def test_blank_or_whitespace_mandatory_fields_returns_400(
        self, whitespace_field, valid_entry_payload
    ):
        """BEHAVIOR: Supplying blank or whitespace-only values yields HTTP 400 Bad Request."""
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        valid_entry_payload[whitespace_field] = "   "

        response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 400
        assert "housing_serial, mo_number, device_type, section, and reason are required" in response.json()["detail"]


# ============================================================================
# 3. FIELD TRANSFORMATION & SANITIZATION BEHAVIOR
# ============================================================================

class TestFieldHandlingBehavior:

    @pytest.mark.parametrize("code_input, expected_persisted_code", [
        ("photolithography_defect", "photolithography_defect"),
        ("bonding_failure", "bonding_failure"),
        ("invalid_custom_code", None),
        ("", None),
        (None, None),
    ])
    def test_scrap_reason_code_whitelisting(
        self, code_input, expected_persisted_code, mock_db, valid_entry_payload
    ):
        """
        BEHAVIOR: Valid scrap reason codes are preserved; unknown/empty codes fallback to None.
        Checks query contents semantically rather than assuming index position.
        """
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = {"request_id": 101, "created_at": "2026-08-13 10:00:00"}

        valid_entry_payload["scrap_reason_code"] = code_input

        response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 200
        insert_params = cursor.get_query_params("INSERT INTO rework_requests")
        assert insert_params is not None
        assert expected_persisted_code in insert_params

    @pytest.mark.parametrize("module_value", [None, "", "   "])
    def test_module_defaults_to_section_when_empty(
        self, module_value, mock_db, valid_entry_payload
    ):
        """BEHAVIOR: When 'module' is omitted or empty, it defaults to the 'section' value."""
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = {"request_id": 102, "created_at": "2026-08-13 10:00:00"}

        valid_entry_payload["module"] = module_value
        valid_entry_payload["section"] = "Laser Bonding"

        response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 200
        insert_params = cursor.get_query_params("INSERT INTO rework_requests")
        assert "Laser Bonding" in insert_params


# ============================================================================
# 4. DOMAIN STATE BRANCHING BEHAVIOR
# ============================================================================

class TestDeviceBranchingBehavior:

    def test_known_device_updates_device_status(
        self, mock_db, valid_entry_payload
    ):
        """BEHAVIOR: Known devices return device_status_updated=True and set status in device table."""
        app.dependency_overrides[get_current_user] = lambda: ADMIN_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = {"request_id": 201, "created_at": "2026-08-13 10:00:00"}

        valid_entry_payload["device_known"] = True

        response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 200
        res = response.json()
        assert res["success"] is True
        assert res["request_id"] == 201
        assert res["device_status_updated"] is True

        update_params = cursor.get_query_params("UPDATE device_information")
        assert update_params == ("HS-2026-001",)

    def test_unknown_device_updates_housing_prep(
        self, mock_db, valid_entry_payload
    ):
        """BEHAVIOR: Unknown devices update housing prep data and return device_status_updated=False."""
        app.dependency_overrides[get_current_user] = lambda: OPERATOR_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = {"request_id": 202, "created_at": "2026-08-13 10:00:00"}

        valid_entry_payload["device_known"] = False

        response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 200
        res = response.json()
        assert res["success"] is True
        assert res["request_id"] == 202
        assert res["device_status_updated"] is False

        update_params = cursor.get_query_params("UPDATE housing_preparation_data")
        assert update_params == ("HS-2026-001",)

    def test_unknown_device_returns_404_when_housing_record_missing(
        self, mock_db, valid_entry_payload
    ):
        """BEHAVIOR: If housing preparation record does not exist for unknown device, returns 404."""
        app.dependency_overrides[get_current_user] = lambda: ADMIN_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = {"request_id": 203, "created_at": "2026-08-13 10:00:00"}
        cursor.rowcount = 0

        valid_entry_payload["device_known"] = False

        response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 404
        assert "No housing preparation record found for serial 'HS-2026-001'" in response.json()["detail"]


# ============================================================================
# 5. RESILIENCE & FAULT TOLERANCE BEHAVIOR
# ============================================================================

class TestSystemResilienceBehavior:

    @patch("rework_module.log_production_event", side_effect=RuntimeError("Trail Log Timeout"))
    def test_production_trail_logging_failure_is_non_fatal(
        self, mock_log_prod, mock_db, valid_entry_payload
    ):
        """BEHAVIOR: Production trail logging exceptions do not prevent entry creation."""
        app.dependency_overrides[get_current_user] = lambda: ADMIN_USER
        cursor, _ = mock_db
        cursor.fetchone.return_value = {"request_id": 301, "created_at": "2026-08-13 10:00:00"}

        response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_database_connection_failure_returns_500(self, valid_entry_payload):
        """BEHAVIOR: Internal database failures convert to HTTP 500 error responses."""
        app.dependency_overrides[get_current_user] = lambda: ADMIN_USER

        with patch("rework_module.get_db_connection", side_effect=Exception("Database down")):
            response = client.post("/create-entry", json=valid_entry_payload)

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to create rework entry"