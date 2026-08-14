# test_create_batch.py - Behavior-Driven Test Suite for Batch Creation Endpoint

import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from manufacturing_orders_module import mo_router, get_current_user

# =====================================================================
# FIXTURES & APPARATUS
# =====================================================================

app = FastAPI()
app.include_router(mo_router)
client = TestClient(app)


@pytest.fixture
def mock_db():
    """Mock the external database connection boundary."""
    with patch("manufacturing_orders_module.get_db_connection") as mock_conn_func:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_conn_func.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        yield mock_cursor, mock_conn


@pytest.fixture
def valid_batch_payload():
    """Standard valid payload for batch creation."""
    return {
        "batch_number": "BATCH-2026-001",
        "batch_type": "catalog",
        "manufacturing_orders": ["MO-1001", "MO-1002"],
        "device_types": {"TypeA": 5, "TypeB": 10},
        "created_by": "admin_user"
    }


def set_authenticated_user(role="admin", username="admin_user", user_id=1):
    """Helper to set authentication context override."""
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": user_id,
        "username": username,
        "role": role
    }


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Automatically reset dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


# =====================================================================
# 1. SUCCESS & BEHAVIORAL STATE TESTS
# =====================================================================

def test_create_batch_success(mock_db, valid_batch_payload):
    """
    Test successful batch creation:
    - User is authorized as admin
    - Target MOs exist and are in 'created' status
    - State mutations complete and return standard success payload
    """
    set_authenticated_user(role="admin", username="admin_user")
    cursor, conn = mock_db
    
    # Mock DB query: MOs exist and are in 'created' state
    cursor.fetchall.return_value = [
        ("MO-1001", "created"),
        ("MO-1002", "created")
    ]
    
    # Each UPDATE statement on manufacturing_order_devices updates 2 devices per MO
    # 2 MOs * 2 devices = 4 total updated
    cursor.rowcount = 2

    response = client.post("/batches", json=valid_batch_payload)

    assert response.status_code == 200
    res = response.json()
    assert res == {
        "success": True,
        "message": f"Batch {valid_batch_payload['batch_number']} created successfully",
        "batch_number": valid_batch_payload["batch_number"],
        "manufacturing_orders_count": 2,
        "devices_updated": 4
    }
    # Note: Called twice because log_action opens its own DB context and commits
    assert conn.commit.called


# =====================================================================
# 2. AUTHORIZATION & RBAC TESTS
# =====================================================================

@pytest.mark.parametrize(
    "user_context, expected_status, expected_detail",
    [
        (
            {"user_id": 2, "username": "operator_user", "role": "operator"},
            403,
            "Admin access required"
        ),
        (
            {"user_id": 3, "username": "guest_user", "role": "viewer"},
            403,
            "Admin access required"
        ),
    ],
    ids=["operator_forbidden", "viewer_forbidden"]
)
def test_create_batch_rbac_enforcement(user_context, expected_status, expected_detail, valid_batch_payload):
    """Verify non-admin roles are rejected with 403 Forbidden."""
    app.dependency_overrides[get_current_user] = lambda: user_context

    response = client.post("/batches", json=valid_batch_payload)

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail


# =====================================================================
# 3. INPUT VALIDATION & BUSINESS RULES
# =====================================================================

@pytest.mark.parametrize(
    "db_mo_records, payload_mos, expected_status, expected_detail_snippet",
    [
        (
            [("MO-1001", "created")],
            ["MO-1001", "MO-9999"],
            400,
            "One or more manufacturing orders not found"
        ),
        (
            [("MO-1001", "created"), ("MO-1002", "in_progress")],
            ["MO-1001", "MO-1002"],
            400,
            "MOs not in 'created' status: MO-1002"
        ),
        (
            [("MO-1001", "completed"), ("MO-1002", "cancelled")],
            ["MO-1001", "MO-1002"],
            400,
            "MOs not in 'created' status: MO-1001, MO-1002"
        ),
    ],
    ids=["missing_mo", "single_invalid_status", "multiple_invalid_statuses"]
)
def test_create_batch_business_rule_validations(
    mock_db, valid_batch_payload, db_mo_records, payload_mos, expected_status, expected_detail_snippet
):
    """Verify business requirements: all MOs must exist and be in 'created' status."""
    set_authenticated_user(role="admin")
    cursor, _ = mock_db
    cursor.fetchall.return_value = db_mo_records
    valid_batch_payload["manufacturing_orders"] = payload_mos

    response = client.post("/batches", json=valid_batch_payload)

    assert response.status_code == expected_status
    assert expected_detail_snippet in response.json()["detail"]


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"batch_number": 12345, "manufacturing_orders": ["MO-1001"]},  # Invalid field type
        {"batch_type": "catalog", "manufacturing_orders": ["MO-1001"]},  # Missing required field
        {"batch_number": "B-1", "manufacturing_orders": "MO-1001"},    # List expected
    ],
    ids=["invalid_type", "missing_field", "string_instead_of_list"]
)
def test_create_batch_schema_validation(invalid_payload):
    """Verify FastAPI/Pydantic schema validation returns 422 for invalid payloads."""
    set_authenticated_user(role="admin")

    response = client.post("/batches", json=invalid_payload)

    assert response.status_code == 422


# =====================================================================
# 4. ERROR HANDLING & BOUNDARY FAILURES
# =====================================================================

@pytest.mark.parametrize(
    "db_exception, expected_status, expected_detail",
    [
        (
            Exception("Database connection failure"),
            500,
            "Failed to create batch"
        ),
        (
            HTTPException(status_code=500, detail="Database connection failed: Timeout"),
            500,
            "Database connection failed: Timeout"
        ),
    ],
    ids=["unexpected_db_error", "http_db_connection_error"]
)
def test_create_batch_database_failures(
    mock_db, valid_batch_payload, db_exception, expected_status, expected_detail
):
    """Verify database infrastructure failures produce appropriate 500 error responses and roll back state."""
    set_authenticated_user(role="admin")
    
    with patch("manufacturing_orders_module.get_db_connection", side_effect=db_exception):
        response = client.post("/batches", json=valid_batch_payload)

    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]


def test_create_batch_transaction_rollback_on_query_error(mock_db, valid_batch_payload):
    """Verify that a database query error during batch execution prevents committing changes."""
    set_authenticated_user(role="admin")
    cursor, conn = mock_db
    
    # Validation query passes, but update query fails
    cursor.fetchall.return_value = [("MO-1001", "created"), ("MO-1002", "created")]
    cursor.execute.side_effect = [None, Exception("Deadlock encountered")]

    response = client.post("/batches", json=valid_batch_payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to create batch"
    conn.commit.assert_not_called()