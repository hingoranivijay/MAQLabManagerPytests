import datetime
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfReader

from manufacturing_orders_module import (
    get_current_user,
    mo_router,
)

pytest_plugins = ["pytest_asyncio"]


# ==============================================================================
# FIXTURES & ISOLATED BOUNDARIES
# ==============================================================================

@pytest.fixture
def mock_db():
    """
    Creates a mock PostgreSQL database connection and cursor that operates correctly 
    when used as a context manager (`with get_db_connection() as conn:`).
    """
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__.return_value = conn
    return conn, cursor


@pytest.fixture
def fastapi_app(mock_db):
    """
    Instantiates the FastAPI application with mo_router.
    Applies a patch on get_db_connection to intercept database context manager calls.
    """
    app_instance = FastAPI()
    app_instance.include_router(mo_router)
    return app_instance


@pytest.fixture
def client(fastapi_app):
    """TestClient instance for making API requests."""
    return TestClient(fastapi_app)


def set_user_override(fastapi_app: FastAPI, role: str = "admin", user_id: int = 101, username: str = "test_user"):
    """Helper to dynamically set the current authenticated user dependency override."""
    fastapi_app.dependency_overrides[get_current_user] = lambda: {
        "user_id": user_id,
        "username": username,
        "role": role,
    }


@pytest.fixture(autouse=True)
def isolate_filesystem_and_logs(monkeypatch):
    """Isolates external storage side-effects and action logging."""
    monkeypatch.setattr("manufacturing_orders_module.log_action", MagicMock())
    monkeypatch.setattr("pathlib.Path.mkdir", MagicMock())


# ==============================================================================
# BEHAVIORAL TESTS: GENERATE MANUFACTURING ORDER PDF
# ==============================================================================

class TestGenerateManufacturingOrderPDF:
    
    @pytest.mark.parametrize(
        "role, expected_status",
        [
            ("admin", 200),
            ("operator", 200),
            ("viewer", 403),
            ("guest", 403),
            ("quality_inspector", 403),
            ("", 403),
        ],
    )
    def test_pdf_generation_rbac(self, fastapi_app, client, mock_db, role, expected_status):
        """
        Behavioral Test: Ensures PDF generation adheres strictly to Role-Based Access Control rules.
        """
        set_user_override(fastapi_app, role=role)
        conn, cursor = mock_db

        cursor.fetchone.return_value = {
            "customer_name": "ACME Corp",
            "product_line": "Robotics",
            "priority": "high",
            "due_date": datetime.date(2026, 12, 31),
            "created_at": datetime.datetime(2026, 1, 1, 10, 0),
            "notes": "Urgent shipment",
        }
        cursor.fetchall.return_value = [
            {"device_type": "Actuator", "quantity": 5, "description": "Motor Drive"}
        ]

        with patch("manufacturing_orders_module.get_db_connection") as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = conn
            response = client.post("/manufacturing-orders/MO-2026-001/generate-pdf")

        assert response.status_code == expected_status

        if expected_status == 200:
            assert response.headers["content-type"] == "application/pdf"
            assert 'filename="MO-2026-001.pdf"' in response.headers["content-disposition"]
            
            # Extract PDF text and verify expected domain values
            pdf_reader = PdfReader(io.BytesIO(response.content))
            page_text = pdf_reader.pages[0].extract_text()
            assert "ACME Corp" in page_text
            assert "Robotics" in page_text

    @pytest.mark.parametrize(
        "mo_number, db_order, db_devices, expected_status, expected_detail",
        [
            (
                "MO-100",
                {
                    "customer_name": "Initech",
                    "product_line": "Hardware",
                    "priority": "low",
                    "due_date": None,
                    "created_at": None,
                    "notes": None,
                },
                [],
                200,
                None,
            ),
            (
                "MO-999-NOTFOUND",
                None,
                [],
                404,
                "Manufacturing order not found",
            ),
        ],
    )
    def test_pdf_generation_resource_outcomes(
        self, fastapi_app, client, mock_db, mo_number, db_order, db_devices, expected_status, expected_detail
    ):
        """
        Behavioral Test: Validates API contract responses for existing and missing orders.
        """
        set_user_override(fastapi_app, role="admin")
        conn, cursor = mock_db
        cursor.fetchone.return_value = db_order
        cursor.fetchall.return_value = db_devices

        with patch("manufacturing_orders_module.get_db_connection") as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = conn
            response = client.post(f"/manufacturing-orders/{mo_number}/generate-pdf")

        assert response.status_code == expected_status
        if expected_detail:
            assert response.json()["detail"] == expected_detail

    def test_pdf_generation_graceful_degradation_on_barcode_error(self, fastapi_app, client, mock_db):
        """
        Resilience Test: Validates that document generation succeeds even if the barcode engine raises an error.
        """
        set_user_override(fastapi_app, role="admin")
        conn, cursor = mock_db

        cursor.fetchone.return_value = {
            "customer_name": "Cyberdyne",
            "product_line": "Defense",
            "priority": "medium",
            "due_date": None,
            "created_at": None,
            "notes": "",
        }
        cursor.fetchall.return_value = []

        with patch("manufacturing_orders_module.get_db_connection") as mock_get_db, \
             patch("barcode.Code128", side_effect=RuntimeError("Barcode engine failure")):
            mock_get_db.return_value.__enter__.return_value = conn
            response = client.post("/manufacturing-orders/MO-RESILIENT/generate-pdf")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")


# ==============================================================================
# BEHAVIORAL TESTS: ORDER CREATION & BATCH MANAGEMENT
# ==============================================================================

class TestManufacturingOrderLifecycle:

    @pytest.mark.parametrize(
        "role, is_lead, expected_status",
        [
            ("admin", False, 200),
            ("operator", True, 200),   # Team Leads permitted
            ("operator", False, 403),  # Standard Operators denied creation
        ],
    )
    def test_create_order_permissions(self, fastapi_app, client, mock_db, role, is_lead, expected_status):
        """
        Behavioral Test: Validates authorization rules governing MO creation boundaries.
        """
        set_user_override(fastapi_app, role=role, user_id=42)
        conn, cursor = mock_db
        
        # Uniqueness check: returns None because the order does not exist yet
        cursor.fetchone.return_value = None

        payload = {
            "manufacturing_order_number": "MO-CREATE-2026",
            "customer_name": "Aperture Science",
            "product_line": "Testing",
            "priority": "high",
        }

        with patch("manufacturing_orders_module.get_db_connection") as mock_get_db, \
             patch("manufacturing_orders_module.is_team_lead", return_value=is_lead), \
             patch("manufacturing_orders_module.validate_device_types_for_product_line", return_value=True):
            mock_get_db.return_value.__enter__.return_value = conn
            response = client.post("/manufacturing-orders", data=payload)

        assert response.status_code == expected_status
        if expected_status == 200:
            result = response.json()
            assert result["success"] is True
            assert result["manufacturing_order_number"] == "MO-CREATE-2026"

    @pytest.mark.parametrize(
        "mo_states, expected_status, expected_error_msg",
        [
            (
                [("MO-101", "created"), ("MO-102", "created")],
                200,
                None,
            ),
            (
                [("MO-101", "created"), ("MO-102", "in_progress")],
                400,
                "MOs not in 'created' status: MO-102",
            ),
        ],
    )
    def test_batch_creation_state_transitions(
        self, fastapi_app, client, mock_db, mo_states, expected_status, expected_error_msg
    ):
        """
        Behavioral Test: Validates state machine transition rules for batching manufacturing orders.
        """
        set_user_override(fastapi_app, role="admin")
        conn, cursor = mock_db
        
        cursor.fetchall.return_value = mo_states
        cursor.fetchone.return_value = [10]  # Device count return value

        batch_payload = {
            "batch_number": "BATCH-2026-01",
            "batch_type": "catalog",
            "manufacturing_orders": [mo for mo, _ in mo_states],
            "device_types": {"Sensor": 10},
            "created_by": "test_operator",
        }

        with patch("manufacturing_orders_module.get_db_connection") as mock_get_db:
            mock_get_db.return_value.__enter__.return_value = conn
            response = client.post("/batches/simple", json=batch_payload)

        assert response.status_code == expected_status
        if expected_status == 200:
            assert response.json()["success"] is True
        else:
            assert expected_error_msg in response.json()["detail"]