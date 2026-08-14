import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock, patch

from modules.manufacturing_orders_module import (
    get_current_user,
    mo_router,
)


# ============================================================================
# FIXTURES & SETUP
# ============================================================================

@pytest.fixture
def fastapi_app():
    """Constructs a clean FastAPI app instance with the MO router mounted."""
    test_app = FastAPI()
    test_app.include_router(mo_router)
    return test_app


@pytest.fixture
def mock_db():
    """Mocks the database connection boundary context manager."""
    with patch("modules.manufacturing_orders_module.get_db_connection") as mock_conn_ctx:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn_ctx.return_value.__enter__.return_value = mock_conn
        yield mock_cursor


@pytest.fixture
def valid_batch_payload():
    """Provides a valid BatchCreate dictionary payload for API endpoints."""
    return {
        "batch_number": "BATCH-2026-001",
        "batch_type": "catalog",
        "manufacturing_orders": ["MO-001", "MO-002"],
        "device_types": {"SENSOR_A": 10},
        "created_by": "admin_user",
    }


# ============================================================================
# BEHAVIOR TESTS: MO Management Authorization & Creation
# Tested via PUBLIC Endpoint: POST /manufacturing-orders
# Covers both `can_manage_mo` and `is_team_lead` behaviorally
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_context, team_lead_query_result, expected_status",
    [
        # Admin access - directly authorized regardless of team lead status
        ({"user_id": 1, "username": "admin_user", "role": "admin"}, None, 200),
        # Operator who IS a designated team lead -> Authorized
        ({"user_id": 10, "username": "lead_user", "role": "operator"}, (1,), 200),
        # Operator who IS NOT a team lead -> Forbidden
        ({"user_id": 20, "username": "standard_op", "role": "operator"}, None, 403),
        # Viewer role (not team lead) -> Forbidden
        ({"user_id": 30, "username": "viewer_user", "role": "viewer"}, None, 403),
        # Missing user ID / unauthenticated dict context -> Forbidden
        ({"user_id": None, "role": "operator"}, None, 403),
    ],
)
async def test_create_manufacturing_order_authorization(
    fastapi_app,
    mock_db,
    user_context,
    team_lead_query_result,
    expected_status,
):
    """
    Behaviorally verifies MO management access control via POST /manufacturing-orders.
    Validates that both Admins and designated Team Leads can manage MOs, while standard users cannot.
    """
    fastapi_app.dependency_overrides[get_current_user] = lambda: user_context

    # DB mocks for successful creation flow when authorized
    mock_db.fetchone.side_effect = [
        team_lead_query_result,  # Team lead check query (is_team_lead)
        None,                    # Duplicate MO check query (does not exist)
    ]
    mock_db.fetchall.return_value = [("SENSOR_A",)]  # Device type validation query

    form_data = {
        "manufacturing_order_number": "MO-2026-999",
        "customer_name": "Acme Corp",
        "product_line": "LINE_A",
        "device_details": '[{"device_type": "SENSOR_A", "quantity": 5}]',
        "priority": "medium",
    }

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as client:
        response = await client.post("/manufacturing-orders", data=form_data)

    assert response.status_code == expected_status
    if expected_status == 403:
        assert "Admin or team-lead access required" in response.json()["detail"]
    else:
        assert response.json()["success"] is True
        assert response.json()["manufacturing_order_number"] == "MO-2026-999"


# ============================================================================
# BEHAVIOR TESTS: Batch Creation (POST /batches)
# ============================================================================

@pytest.mark.asyncio
async def test_create_batch_success(fastapi_app, mock_db, valid_batch_payload):
    """Verifies successful batch creation response contract and state update execution."""
    mock_db.fetchall.return_value = [("MO-001", "created"), ("MO-002", "created")]
    mock_db.rowcount = 1  # 1 device row updated per iteration

    fastapi_app.dependency_overrides[get_current_user] = lambda: {
        "user_id": 1,
        "username": "admin_user",
        "role": "admin",
    }

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as client:
        response = await client.post("/batches", json=valid_batch_payload)

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "Batch BATCH-2026-001 created successfully",
        "batch_number": "BATCH-2026-001",
        "manufacturing_orders_count": 2,
        "devices_updated": 2,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_role, expected_status, expected_detail",
    [
        ("operator", 403, "Admin access required"),
        ("viewer", 403, "Admin access required"),
    ],
)
async def test_create_batch_role_authorization(
    fastapi_app, valid_batch_payload, user_role, expected_status, expected_detail
):
    """Ensures non-admin roles are rejected when attempting batch creation."""
    fastapi_app.dependency_overrides[get_current_user] = lambda: {
        "user_id": 2,
        "username": "standard_user",
        "role": user_role,
    }

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as client:
        response = await client.post("/batches", json=valid_batch_payload)

    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "db_mo_records, expected_status, expected_detail_snippet",
    [
        # Missing MOs from Database
        ([("MO-001", "created")], 400, "One or more manufacturing orders not found"),
        # MO in invalid status
        ([("MO-001", "created"), ("MO-002", "in_progress")], 400, "MOs not in 'created' status: MO-002"),
        # Multiple MOs in invalid status
        ([("MO-001", "completed"), ("MO-002", "in_progress")], 400, "MOs not in 'created' status"),
    ],
)
async def test_create_batch_validation_errors(
    fastapi_app, mock_db, valid_batch_payload, db_mo_records, expected_status, expected_detail_snippet
):
    """Validates Bad Request responses when target MOs fail business pre-conditions."""
    mock_db.fetchall.return_value = db_mo_records

    fastapi_app.dependency_overrides[get_current_user] = lambda: {
        "user_id": 1,
        "username": "admin_user",
        "role": "admin",
    }

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as client:
        response = await client.post("/batches", json=valid_batch_payload)

    assert response.status_code == expected_status
    assert expected_detail_snippet in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_batch_database_failure_handling(fastapi_app, mock_db, valid_batch_payload):
    """Ensures unhandled database exceptions yield a standard 500 error response."""
    mock_db.execute.side_effect = Exception("Database connection dropped")

    fastapi_app.dependency_overrides[get_current_user] = lambda: {
        "user_id": 1,
        "username": "admin_user",
        "role": "admin",
    }

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as client:
        response = await client.post("/batches", json=valid_batch_payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to create batch"