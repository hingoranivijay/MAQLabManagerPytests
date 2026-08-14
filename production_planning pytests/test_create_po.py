import io
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.production_planning_module import get_current_user, pp_router


# ── Query-Aware DB Mock Infrastructure ────────────────────────────────────────

class QueryAwareCursor:
    """Mock database cursor that returns responses based on SQL patterns rather

    than fragile positional side-effect sequences.
    """

    def __init__(self):
        self.last_query = ""
        self.last_params = ()
        self.custom_handlers = []

    def execute(self, query, params=None):
        self.last_query = query
        self.last_params = params or ()

    def fetchone(self):
        query = self.last_query.lower()
        for handler in self.custom_handlers:
            res = handler("fetchone", query, self.last_params)
            if res is not None:
                return res

        # Default query responses based on query signatures
        if "select 1 from purchase_orders" in query:
            return None  # Default: PO does not exist yet
        if "insert into purchase_orders" in query:
            return (101,)  # Default created PO ID
        if "select team_id from users" in query:
            return (1,)  # Default user's team ID
        if "select id from teams where team_lead_id" in query:
            return None
        if "select status from purchase_orders" in query:
            return ("open",)
        if "select assigned_team_id" in query:
            return (1, "open")
        if "count(*) filter" in query:
            return (1, 0)  # Default: 1 line item, 0 unfilled
        if "select id, name from teams" in query:
            return (1, "Assembly Team")
        return (1,)

    def fetchall(self):
        query = self.last_query.lower()
        for handler in self.custom_handlers:
            res = handler("fetchall", query, self.last_params)
            if res is not None:
                return res

        if "select id from teams where team_lead_id" in query:
            return [(10,)]
        if "select status, count(*)" in query:
            return [("open", 5), ("assigned", 3), ("converted", 2)]
        if "purchase_order_line_items" in query:
            return [(10, 10)]  # Default line item ID 10 with 10 remaining
        return []


@pytest.fixture(autouse=True)
def mock_db():
    """Provides an isolated query-aware database connection boundary for all tests."""
    cursor = QueryAwareCursor()
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("modules.production_planning_module.get_db_connection") as mock_conn_ctx:
        mock_conn_ctx.return_value.__enter__.return_value = conn
        yield {"conn": conn, "cursor": cursor}


# ── Test Client & Auth Fixtures ───────────────────────────────────────────────

@pytest.fixture
def fastapi_app():
    application = FastAPI()
    application.include_router(pp_router)
    return application


@pytest.fixture
def pm_user():
    return {"user_id": 1, "role": "admin"}


@pytest.fixture
def operator_user():
    return {"user_id": 2, "role": "operator"}


@pytest.fixture
def client(fastapi_app, pm_user):
    fastapi_app.dependency_overrides[get_current_user] = lambda: pm_user
    with TestClient(fastapi_app) as test_client:
        yield test_client
    fastapi_app.dependency_overrides.clear()


# ── Behavioral Tests: User Capabilities & Dashboard Summary ──────────────────

class TestUserPermissionsAndSummary:

    def test_my_permissions_pm_and_team_lead(self, client, mock_db):
        """Verifies capability flags returned for a PM user who leads a team."""
        response = client.get("/my-permissions")

        assert response.status_code == 200
        assert response.json() == {
            "is_pm": True,
            "is_team_lead": True,
            "led_team_ids": [10],
            "can_manage_mo": True,
        }

    def test_summary_status_counts(self, client, mock_db):
        """Validates dashboard aggregation of purchase order status counts."""
        response = client.get("/summary")

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 10
        assert payload["by_status"]["open"] == 5
        assert payload["by_status"]["assigned"] == 3
        assert payload["by_status"]["converted"] == 2

    def test_summary_forbidden_for_non_pm(self, fastapi_app, operator_user, mock_db):
        """Ensures summary endpoint rejects non-PM roles with 403 Forbidden."""
        fastapi_app.dependency_overrides[get_current_user] = lambda: operator_user
        with TestClient(fastapi_app) as unauth_client:
            response = unauth_client.get("/summary")

        assert response.status_code == 403
        assert response.json()["detail"] == "Production manager access required"


# ── Behavioral Tests: Purchase Order Creation (`create_po`) ───────────────────

class TestPOCreation:

    @pytest.mark.parametrize(
        "form_data, expected_id",
        [
            (
                {"po_number": "PO-1001", "customer_name": "Acme Corp"},
                101,
            ),
            (
                {
                    "po_number": "PO-1002",
                    "customer_name": "Stark Industries",
                    "product_line": "Robotics",
                    "requested_date": "2026-11-01",
                    "priority": "high",
                    "notes": "Rush shipment required",
                    "line_items": json.dumps([
                        {"device_type": "Actuator-X", "quantity": 10, "description": "12V Actuator"},
                        {"device_type": "Sensor-Y", "quantity": 20, "description": "Temp Sensor"},
                    ]),
                },
                101,
            ),
        ],
        ids=["minimal_po", "full_po_with_line_items"],
    )
    def test_create_po_success(self, client, mock_db, form_data, expected_id):
        """Verifies successful PO creation returns HTTP 200 and the generated PO ID."""
        response = client.post("/purchase-orders", data=form_data)

        assert response.status_code == 200
        assert response.json() == {"success": True, "id": expected_id}

    def test_create_po_file_upload_persists_to_disk(self, client, mock_db, tmp_path):
        """Validates that uploaded PO attachment files are physically persisted to disk."""
        file_content = b"%PDF-1.4 customer purchase order binary data"
        files = {"file": ("po_doc.pdf", io.BytesIO(file_content), "application/pdf")}
        form_data = {"po_number": "PO-FILE-01", "customer_name": "Wayne Enterprises"}

        with patch("modules.production_planning_module.UPLOAD_DIR", tmp_path):
            response = client.post("/purchase-orders", data=form_data, files=files)

        assert response.status_code == 200
        assert response.json() == {"success": True, "id": 101}

        saved_files = list(tmp_path.glob("PO-FILE-01_*"))
        assert len(saved_files) == 1
        assert saved_files[0].read_bytes() == file_content

    @pytest.mark.parametrize(
        "form_data, db_duplicate, expected_status, expected_detail",
        [
            (
                {"po_number": "PO-DUP-01", "customer_name": "Acme Corp"},
                True,
                409,
                "A PO with this number already exists",
            ),
            (
                {"po_number": "PO-BAD-JSON", "customer_name": "Globex", "line_items": "{malformed_json:"},
                False,
                400,
                "Invalid line_items payload",
            ),
            (
                {"customer_name": "No PONumber Corp"},
                False,
                422,
                None,
            ),
        ],
        ids=["duplicate_po_number", "malformed_json", "missing_required_field"],
    )
    def test_create_po_validation_and_conflicts(
        self, client, mock_db, form_data, db_duplicate, expected_status, expected_detail
    ):
        """Consolidates input validation, payload parsing, and duplicate handling."""
        if db_duplicate:
            def handler(mode, query, params):
                if "select 1 from purchase_orders" in query:
                    return (1,)  # Signal existing record
                return None
            mock_db["cursor"].custom_handlers.append(handler)

        response = client.post("/purchase-orders", data=form_data)

        assert response.status_code == expected_status
        if expected_detail:
            assert response.json()["detail"] == expected_detail

    @pytest.mark.parametrize(
        "role, expected_status",
        [
            ("admin", 200),
            ("super_user", 200),
            ("management", 200),
            ("operator", 403),
            ("viewer", 403),
        ],
        ids=["admin_allowed", "superuser_allowed", "management_allowed", "operator_forbidden", "viewer_forbidden"],
    )
    def test_create_po_role_access_control(self, fastapi_app, mock_db, role, expected_status):
        """Enforces role-based authorization rules for PO creation."""
        fastapi_app.dependency_overrides[get_current_user] = lambda: {"user_id": 9, "role": role}
        with TestClient(fastapi_app) as auth_client:
            response = auth_client.post(
                "/purchase-orders",
                data={"po_number": f"PO-ROLE-{role}", "customer_name": "Role Test Corp"},
            )

        assert response.status_code == expected_status
        if expected_status == 403:
            assert response.json()["detail"] == "Production manager access required"


# ── Behavioral Tests: Document Parsing (`parse_po_document`) ───────────────

class TestPODocumentParsing:

    def test_parse_po_empty_file_rejected(self, client):
        """Rejects empty uploaded files with HTTP 400 Bad Request."""
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
        response = client.post("/purchase-orders/parse", files=files)

        assert response.status_code == 400
        assert response.json()["detail"] == "Empty file"

    def test_parse_po_unextractable_document_returns_422(self, client):
        """Validates handling when text extraction yields no parseable content."""
        minimal_pdf = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 300 144]>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n"
            b"0000000058 00000 n\n0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )
        files = {"file": ("scanned_blank.pdf", io.BytesIO(minimal_pdf), "application/pdf")}

        response = client.post("/purchase-orders/parse", files=files)

        assert response.status_code == 422
        assert "Couldn't read any text" in response.json()["detail"]

    def test_parse_po_forbidden_for_non_pm(self, fastapi_app, operator_user):
        """Restricts document parsing endpoints to authorized Production Managers."""
        fastapi_app.dependency_overrides[get_current_user] = lambda: operator_user
        files = {"file": ("doc.pdf", io.BytesIO(b"dummy bytes"), "application/pdf")}

        with TestClient(fastapi_app) as unauth_client:
            response = unauth_client.post("/purchase-orders/parse", files=files)

        assert response.status_code == 403


# ── Behavioral Tests: Updates & Assignment ───────────────────────────────────

class TestPOUpdatesAndAssignment:

    def test_update_po_prevents_editing_converted(self, client, mock_db):
        """Rejects edits to converted POs with 400 Bad Request."""
        def handler(mode, query, params):
            if "select status from purchase_orders" in query:
                return ("converted",)
            return None
        mock_db["cursor"].custom_handlers.append(handler)

        response = client.put("/purchase-orders/10", json={"customer_name": "New Name"})

        assert response.status_code == 400
        assert response.json()["detail"] == "Cannot edit a converted PO"

    def test_assign_po_success(self, client, mock_db):
        """Assigns an open PO to a target team successfully."""
        response = client.post("/purchase-orders/10/assign", json={"assigned_team_id": 5})

        assert response.status_code == 200
        assert response.json() == {"success": True}


# ── Behavioral Tests: MO Linking & Allocation State Transitions ─────────────

class TestMOLinkingAndAllocations:

    @pytest.mark.parametrize(
        "allocations, remaining_qty, unfilled_count, expected_status, expected_code, expected_detail",
        [
            (
                [{"po_line_item_id": 10, "quantity": 5}],
                10,
                1,
                "in_progress",
                200,
                None,
            ),
            (
                [{"po_line_item_id": 10, "quantity": 10}],
                10,
                0,
                "converted",
                200,
                None,
            ),
            (
                [{"po_line_item_id": 10, "quantity": 15}],
                10,
                1,
                None,
                400,
                "Cannot allocate 15; only 10 remaining for that line item",
            ),
            (
                [{"po_line_item_id": 10, "quantity": -2}],
                10,
                1,
                None,
                400,
                "Allocation quantity cannot be negative",
            ),
            (
                [{"po_line_item_id": 99, "quantity": 5}],
                10,
                1,
                None,
                400,
                "Allocation references a line item that is not part of this PO",
            ),
        ],
        ids=[
            "partial_allocation_in_progress",
            "full_allocation_converted",
            "over_allocation_error",
            "negative_allocation_error",
            "invalid_line_item_error",
        ],
    )
    def test_link_mo_allocation_outcomes(
        self,
        client,
        mock_db,
        allocations,
        remaining_qty,
        unfilled_count,
        expected_status,
        expected_code,
        expected_detail,
    ):
        """Validates MO allocations, line item limits, and automatic state transitions."""
        def handler(mode, query, params):
            if "purchase_order_line_items" in query and mode == "fetchall":
                return [(10, remaining_qty)]
            if "count(*) filter" in query and mode == "fetchone":
                return (1, unfilled_count)
            return None

        mock_db["cursor"].custom_handlers.append(handler)

        payload = {
            "manufacturing_order_number": "MO-8000",
            "allocations": allocations,
        }

        response = client.post("/purchase-orders/50/link-mo", json=payload)

        assert response.status_code == expected_code
        if expected_code == 200:
            assert response.json()["status"] == expected_status
        elif expected_detail:
            assert response.json()["detail"] == expected_detail

    @pytest.mark.parametrize(
        "mark_converted, initial_status, expected_status",
        [
            (True, "assigned", "converted"),
            (False, "open", "in_progress"),
            (False, "converted", "converted"),
        ],
        ids=["legacy_mark_converted_true", "legacy_auto_in_progress", "legacy_preserve_converted"],
    )
    def test_link_mo_legacy_path(
        self, client, mock_db, mark_converted, initial_status, expected_status
    ):
        """Tests status transition rules for unallocated (legacy) MO linking."""
        def handler(mode, query, params):
            if "select assigned_team_id, status" in query:
                return (1, initial_status)
            return None

        mock_db["cursor"].custom_handlers.append(handler)

        payload = {
            "manufacturing_order_number": "MO-LEGACY-01",
            "mark_converted": mark_converted,
            "allocations": None,
        }

        response = client.post("/purchase-orders/50/link-mo", json=payload)

        assert response.status_code == 200
        assert response.json()["status"] == expected_status


# ── Behavioral Tests: Status Updates & File Download Boundary ───────────────

class TestPOStatusAndFileDownloads:

    @pytest.mark.parametrize(
        "new_status, expected_code",
        [
            ("in_progress", 200),
            ("cancelled", 200),
            ("invalid_status_enum", 400),
        ],
        ids=["status_in_progress", "status_cancelled", "invalid_enum_rejected"],
    )
    def test_update_po_status_domain_rules(self, client, mock_db, new_status, expected_code):
        """Enforces domain constraints for PO status transitions."""
        response = client.put("/purchase-orders/10/status", json={"status": new_status})
        assert response.status_code == expected_code

    def test_download_po_file_not_found_on_disk(self, client, mock_db):
        """Returns HTTP 404 when database references an attachment missing from disk."""
        def handler(mode, query, params):
            if "select assigned_team_id, file_path" in query:
                return (1, "/nonexistent/path/doc.pdf", "invoice.pdf")
            return None

        mock_db["cursor"].custom_handlers.append(handler)

        response = client.get("/purchase-orders/10/file")

        assert response.status_code == 404
        assert response.json()["detail"] == "No file attached to this PO"

    def test_download_po_file_success(self, client, mock_db, tmp_path):
        """Delivers file payload when attached document is present on disk."""
        target_file = tmp_path / "existing_doc.pdf"
        target_file.write_bytes(b"%PDF document payload")

        def handler(mode, query, params):
            if "select assigned_team_id, file_path" in query:
                return (1, str(target_file), "original_invoice.pdf")
            return None

        mock_db["cursor"].custom_handlers.append(handler)

        response = client.get("/purchase-orders/10/file")

        assert response.status_code == 200
        assert response.content == b"%PDF document payload"