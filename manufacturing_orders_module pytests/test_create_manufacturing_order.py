# test_create_manufacturing_order.py - Refactored Behavior-Driven Pytest Suite

import io
import json
import warnings
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from manufacturing_orders_module import get_current_user, mo_router

# Setup test application
app = FastAPI()
app.include_router(mo_router)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    client = TestClient(app)


# ==============================================================================
# STATEFUL DATABASE MOCK
# ==============================================================================

class MockDBContext:
    """
    State-driven database mock that routes queries based on SQL content,
    eliminating rigid call-order assertion dependencies.
    """

    def __init__(self, is_team_lead=False, existing_mo=False, valid_device_types=True, fail_on_insert=False):
        self.is_team_lead = is_team_lead
        self.existing_mo = existing_mo
        self.valid_device_types = valid_device_types
        self.fail_on_insert = fail_on_insert
        self.conn = MagicMock()
        self.cursor = MagicMock()

        self.conn.cursor.return_value = self.cursor
        self.cursor.execute.side_effect = self._execute
        self.cursor.fetchone.side_effect = self._fetchone
        self.cursor.fetchall.side_effect = self._fetchall

        self._last_query = ""
        self._last_params = ()

    def _execute(self, query, params=()):
        self._last_query = query
        self._last_params = params

        if self.fail_on_insert and "INSERT INTO" in query:
            raise Exception("Database transaction constraint error")

    def _fetchone(self):
        if "FROM teams WHERE team_lead_id" in self._last_query:
            return (1,) if self.is_team_lead else None
        elif "FROM manufacturing_orders WHERE manufacturing_order_number" in self._last_query:
            return (self._last_params[0],) if self.existing_mo else None
        return None

    def _fetchall(self):
        if "FROM device_types" in self._last_query:
            if self.valid_device_types and len(self._last_params) >= 2:
                # Return match for all requested device types
                requested_types = self._last_params[1]
                return [(dt,) for dt in requested_types]
            return []
        return []


@pytest.fixture
def setup_db_mock():
    """Sets up the stateful database context manager mock."""
    def _setup(is_team_lead=False, existing_mo=False, valid_device_types=True, fail_on_insert=False):
        db_ctx = MockDBContext(
            is_team_lead=is_team_lead,
            existing_mo=existing_mo,
            valid_device_types=valid_device_types,
            fail_on_insert=fail_on_insert,
        )
        mock_get_db = patch("manufacturing_orders_module.get_db_connection").start()
        mock_get_db.return_value.__enter__.return_value = db_ctx.conn
        return db_ctx

    yield _setup
    patch.stopall()


@pytest.fixture
def valid_mo_payload():
    """Standard payload for manufacturing order creation."""
    return {
        "manufacturing_order_number": "MO-2026-001",
        "customer_name": "Acme Corp",
        "product_line": "Electronics",
        "device_details": json.dumps([
            {"device_type": "Sensor-A", "quantity": 10, "description": "Temp Sensor"}
        ]),
        "priority": "medium",
        "due_date": "2026-12-31",
        "notes": "Standard manufacturing delivery.",
    }


# ==============================================================================
# AUTHORIZATION TESTS
# ==============================================================================

@pytest.mark.parametrize(
    "user_role, is_lead, expected_status",
    [
        ("admin", False, 200),
        ("operator", True, 200),
        ("operator", False, 403),
        ("viewer", False, 403),
    ],
)
def test_create_mo_permissions_matrix(
    user_role, is_lead, expected_status, setup_db_mock, valid_mo_payload, tmp_path
):
    """
    Behavioral Test: Validates access matrix ensuring admins and designated team leads 
    can create orders, while unauthorized users receive HTTP 403.
    """
    setup_db_mock(is_team_lead=is_lead)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 42, "role": user_role}

    with patch("manufacturing_orders_module.UPLOAD_DIR", tmp_path):
        response = client.post("/manufacturing-orders", data=valid_mo_payload)

    assert response.status_code == expected_status
    if expected_status == 403:
        assert response.json()["detail"] == "Admin or team-lead access required"
    else:
        assert response.json()["success"] is True


# ==============================================================================
# INPUT VALIDATION & BUSINESS RULE TESTS
# ==============================================================================

@pytest.mark.parametrize(
    "device_details_input, expected_detail",
    [
        ("{invalid_json: true", "Invalid device details format"),
        (
            json.dumps([{"device_type": "INVALID_TYPE", "quantity": 1}]),
            "Some device types are not valid for the selected product line",
        ),
    ],
)
def test_create_mo_invalid_payload_returns_400(
    device_details_input, expected_detail, setup_db_mock, valid_mo_payload
):
    """
    Behavioral Test: Bad device JSON syntax or invalid device types for a product line
    must trigger HTTP 400 Bad Request before saving data.
    """
    setup_db_mock(valid_device_types=False)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 42, "role": "admin"}

    valid_mo_payload["device_details"] = device_details_input
    response = client.post("/manufacturing-orders", data=valid_mo_payload)

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail


def test_create_mo_duplicate_number_returns_400(setup_db_mock, valid_mo_payload):
    """Behavioral Test: Attempting to create an MO with an existing order number raises HTTP 400."""
    setup_db_mock(existing_mo=True)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 42, "role": "admin"}

    response = client.post("/manufacturing-orders", data=valid_mo_payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Manufacturing order number already exists"


@pytest.mark.parametrize(
    "due_date_input",
    ["None", "2026-12-31", None, ""],
)
def test_create_mo_due_date_variations(due_date_input, setup_db_mock, valid_mo_payload, tmp_path):
    """Behavioral Test: Verifies successful processing across multiple valid due date formats."""
    setup_db_mock()
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 42, "role": "admin"}

    payload = {**valid_mo_payload, "due_date": due_date_input}
    
    with patch("manufacturing_orders_module.UPLOAD_DIR", tmp_path):
        response = client.post("/manufacturing-orders", data=payload)

    assert response.status_code == 200
    assert response.json()["success"] is True


# ==============================================================================
# FILE ATTACHMENT & DISK STORAGE TESTS
# ==============================================================================

def test_create_mo_with_file_attachment_success(setup_db_mock, valid_mo_payload, tmp_path):
    """
    Behavioral Test: Ensures uploaded files are saved to disk and 
    the order creation succeeds.
    """
    setup_db_mock()
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 42, "role": "admin"}

    file_payload = {"file": ("spec.pdf", io.BytesIO(b"%PDF-1.4 sample content"), "application/pdf")}

    with patch("manufacturing_orders_module.UPLOAD_DIR", tmp_path):
        response = client.post("/manufacturing-orders", data=valid_mo_payload, files=file_payload)

    assert response.status_code == 200
    assert response.json()["success"] is True
    # Confirm file was written to disk inside target upload directory
    saved_files = list(tmp_path.glob("MO-2026-001_*.pdf"))
    assert len(saved_files) == 1


# ==============================================================================
# TRANSACTION ROLLBACK & ERROR RESILIENCE
# ==============================================================================

def test_create_mo_database_failure_returns_500(setup_db_mock, valid_mo_payload):
    """Behavioral Test: Unhandled database exceptions produce an HTTP 500 error response."""
    setup_db_mock(fail_on_insert=True)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 42, "role": "admin"}

    response = client.post("/manufacturing-orders", data=valid_mo_payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to create manufacturing order"


def test_create_mo_logging_failure_swallowed(setup_db_mock, valid_mo_payload, tmp_path):
    """Resilience Test: Non-critical log action failures do not block successful MO creation."""
    setup_db_mock()
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 42, "role": "admin"}

    with patch("manufacturing_orders_module.log_action", side_effect=Exception("System log failed")), \
         patch("manufacturing_orders_module.UPLOAD_DIR", tmp_path):
        response = client.post("/manufacturing-orders", data=valid_mo_payload)

    assert response.status_code == 200
    assert response.json()["success"] is True