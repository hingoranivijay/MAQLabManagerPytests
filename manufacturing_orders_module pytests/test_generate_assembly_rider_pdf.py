import datetime
import json
import sys
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
from fastapi import HTTPException

from manufacturing_orders_module import generate_assembly_rider_pdf


# ============================================================================
# Core Fixtures & Test Helpers
# ============================================================================

class BehavioralDBContext:
    """
    Behavioral Database Context: Simulates database responses based on query signatures
    rather than imposing rigid call order or internal SQL matching.
    """
    def __init__(self, batch_info, mos, device_summary, test_sequences, test_definitions):
        self.batch_info = batch_info
        self.mos = mos
        self.device_summary = device_summary
        self.test_sequences = test_sequences
        self.test_definitions = test_definitions
        self.cursor = MagicMock()
        self.cursor.fetchone.side_effect = self._fetchone
        self.cursor.fetchall.side_effect = self._fetchall

    def _fetchone(self):
        call_args = self.cursor.execute.call_args
        if not call_args:
            return None
        query, params = call_args[0][0], call_args[0][1] if len(call_args[0]) > 1 else ()

        if "FROM manufacturing_order_devices" in query and "LIMIT 1" in query:
            return self.batch_info
        elif "FROM device_test_sequences" in query:
            dev_type = params[0]
            seq = self.test_sequences.get(dev_type)
            return {"test_sequence": seq} if seq is not None else None
        return None

    def _fetchall(self):
        call_args = self.cursor.execute.call_args
        if not call_args:
            return []
        query, params = call_args[0][0], call_args[0][1] if len(call_args[0]) > 1 else ()

        if "JOIN manufacturing_orders mo" in query:
            return self.mos
        elif "GROUP BY mod.device_type" in query:
            return self.device_summary
        elif "FROM test_definitions" in query:
            test_ids = params
            return [t for t in self.test_definitions if t["test_id"] in test_ids]
        return []


@pytest.fixture
def default_db_data():
    """Provides standard valid dataset for batch operations."""
    return {
        "batch_info": {
            "batch_number": "BATCH-100",
            "batch_type": "standard",
            "batch_created_at": datetime.datetime(2026, 8, 10, 10, 0, 0),
            "batch_created_by": "john_doe",
        },
        "mos": [
            {"manufacturing_order_number": "MO-001", "customer_name": "Acme Corp", "product_line": "Line A"},
            {"manufacturing_order_number": "MO-002", "customer_name": "Beta LLC", "product_line": "Line B"},
        ],
        "device_summary": [
            {"device_type": "SensorNode", "total_quantity": 10},
            {"device_type": "Gateway", "total_quantity": 2},
        ],
        "test_sequences": {
            "SensorNode": [{"test_id": "T1", "is_required": True, "sequence_order": 1}],
            "Gateway": json.dumps([{"test_id": "T2", "is_required": True, "sequence_order": 1}]),
        },
        "test_definitions": [
            {"test_id": "T1", "test_name": "Voltage Test"},
            {"test_id": "T2", "test_name": "Connectivity Test"},
        ],
    }


@pytest.fixture
def setup_mock_db(default_db_data):
    """Mocks database connection boundary dynamically based on input dataset."""
    def _setup(custom_data=None):
        data = {**default_db_data, **(custom_data or {})}
        db_context = BehavioralDBContext(**data)
        mock_get_db = patch("manufacturing_orders_module.get_db_connection").start()
        mock_get_db.return_value.__enter__.return_value.cursor.return_value = db_context.cursor
        return db_context
    yield _setup
    patch.stopall()


@pytest.fixture(autouse=True)
def mock_audit_log():
    """Mocks system audit logging boundary across all tests."""
    with patch("manufacturing_orders_module.log_action") as mock_log:
        yield mock_log


# ============================================================================
# Role & Authorization Tests
# ============================================================================

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
@pytest.mark.asyncio
async def test_generate_assembly_rider_authorization_matrix(
    role, expected_status, setup_mock_db, mock_audit_log, tmp_path
):
    """
    Behavioral Test: Validates access control matrix across user roles.
    Ensures authorized users receive a valid PDF (200 OK) while unauthorized
    users are denied with HTTP 403.
    """
    setup_mock_db()
    current_user = {"user_id": 1, "username": "test_user", "role": role}

    with patch("manufacturing_orders_module.Path") as mock_path:
        mock_path.side_effect = lambda *args: Path(tmp_path, *args) if args else Path(tmp_path)

        if expected_status == 403:
            with pytest.raises(HTTPException) as exc_info:
                await generate_assembly_rider_pdf(batch_number="BATCH-100", current_user=current_user)
            assert exc_info.value.status_code == 403
            assert exc_info.value.detail == "Insufficient permissions"
            mock_audit_log.assert_not_called()
        else:
            response = await generate_assembly_rider_pdf(batch_number="BATCH-100", current_user=current_user)
            assert response.status_code == 200
            assert response.media_type == "application/pdf"
            assert response.body.startswith(b"%PDF")
            mock_audit_log.assert_called_once_with(
                1, "generate_assembly_rider", "batch_management", ANY
            )


# ============================================================================
# Functional Behavior & Edge Cases
# ============================================================================

@pytest.mark.parametrize(
    "sequence_format_desc, custom_test_seq",
    [
        ("parsed_dict_list", {"SensorNode": [{"test_id": "T1", "is_required": True, "sequence_order": 1}]}),
        ("stringified_json", {"SensorNode": json.dumps([{"test_id": "T1", "is_required": True, "sequence_order": 1}])}),
        ("missing_test_seq", {"SensorNode": None}),
        ("empty_test_seq", {"SensorNode": []}),
    ],
)
@pytest.mark.asyncio
async def test_generate_assembly_rider_handles_varying_sequence_formats(
    sequence_format_desc, custom_test_seq, setup_mock_db, tmp_path
):
    """
    Behavioral Test: Ensures endpoint handles diverse database formats for test sequences
    (dicts, stringified JSON, missing rows, or empty sequence arrays) gracefully.
    """
    setup_mock_db({"test_sequences": custom_test_seq})
    current_user = {"user_id": 1, "username": "admin_user", "role": "admin"}

    with patch("manufacturing_orders_module.Path") as mock_path:
        mock_path.side_effect = lambda *args: Path(tmp_path, *args) if args else Path(tmp_path)
        response = await generate_assembly_rider_pdf(batch_number="BATCH-100", current_user=current_user)

    assert response.status_code == 200
    assert response.body.startswith(b"%PDF")


@pytest.mark.asyncio
async def test_generate_assembly_rider_batch_not_found_raises_404(setup_mock_db):
    """Behavioral Test: Requesting a non-existent batch number must return HTTP 404."""
    setup_mock_db({"batch_info": None})
    current_user = {"user_id": 1, "username": "operator_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await generate_assembly_rider_pdf(batch_number="NON-EXISTENT", current_user=current_user)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Batch not found"


# ============================================================================
# Resilience & External Dependency Fallbacks
# ============================================================================

@pytest.mark.asyncio
async def test_generate_assembly_rider_degrades_gracefully_without_barcode_lib(
    setup_mock_db, tmp_path
):
    """
    Resilience Test: Validates that if the external barcode generation library is missing,
    the endpoint falls back to text rendering without raising internal errors.
    """
    setup_mock_db()
    current_user = {"user_id": 1, "username": "admin_user", "role": "admin"}

    with patch.dict(sys.modules, {"barcode": None}), patch("manufacturing_orders_module.Path") as mock_path:
        mock_path.side_effect = lambda *args: Path(tmp_path, *args) if args else Path(tmp_path)

        response = await generate_assembly_rider_pdf(batch_number="BATCH-100", current_user=current_user)

    assert response.status_code == 200
    assert response.body.startswith(b"%PDF")


@pytest.mark.asyncio
async def test_generate_assembly_rider_resilient_to_filesystem_write_failures(
    setup_mock_db, tmp_path
):
    """
    Resilience Test: Disk or network storage failures during persistent copy saving
    must not fail the HTTP request—the generated PDF is still delivered to the client.
    """
    setup_mock_db()
    current_user = {"user_id": 1, "username": "admin_user", "role": "admin"}
    real_open = open

    def open_side_effect(file, *args, **kwargs):
        if str(file).endswith(".pdf"):
            raise PermissionError("Network drive unwritable")
        return real_open(file, *args, **kwargs)

    with patch("manufacturing_orders_module.Path") as mock_path, patch("builtins.open", side_effect=open_side_effect):
        mock_path.side_effect = lambda *args: Path(tmp_path, *args) if args else Path(tmp_path)
        response = await generate_assembly_rider_pdf(batch_number="BATCH-100", current_user=current_user)

    assert response.status_code == 200
    assert response.body.startswith(b"%PDF")


@pytest.mark.asyncio
async def test_generate_assembly_rider_handles_unhandled_database_failures():
    """Resilience Test: Database system failure triggers top-level HTTP 500 error."""
    with patch("manufacturing_orders_module.get_db_connection") as mock_get_db:
        mock_get_db.side_effect = Exception("Fatal database connection error")
        current_user = {"user_id": 1, "username": "admin_user", "role": "admin"}

        with pytest.raises(HTTPException) as exc_info:
            await generate_assembly_rider_pdf(batch_number="BATCH-100", current_user=current_user)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to generate Assembly Rider"