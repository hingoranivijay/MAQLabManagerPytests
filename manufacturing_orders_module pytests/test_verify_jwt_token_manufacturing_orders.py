import datetime
from datetime import timezone, timedelta
import jwt
import pytest
from fastapi import HTTPException

from modules.manufacturing_orders_module import SECRET_KEY, verify_jwt_token


def create_token(payload: dict, secret: str = SECRET_KEY, algorithm: str = "HS256") -> str:
    """Helper utility to construct test JWTs with custom claims and signing options."""
    return jwt.encode(payload, secret, algorithm=algorithm)


# ============================================================================
# BEHAVIOR: Decoding Valid Tokens
# ============================================================================

@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": 42, "username": "johndoe", "role": "operator"},
        {
            "user_id": 10,
            "role": "admin",
            "exp": datetime.datetime.now(timezone.utc) + timedelta(hours=1),
        },
    ],
    ids=["valid-claims-without-exp", "valid-claims-with-future-exp"],
)
def test_verify_jwt_token_returns_decoded_payload_for_valid_tokens(payload: dict):
    """
    Behavior: Valid, unexpired tokens signed with the expected secret
    must decode successfully and return the full claims payload.
    """
    token = create_token(payload)

    result = verify_jwt_token(token)

    assert result["user_id"] == payload["user_id"]
    assert result["role"] == payload["role"]
    if "username" in payload:
        assert result["username"] == payload["username"]


# ============================================================================
# BEHAVIOR: Rejecting Expired Tokens
# ============================================================================

def test_verify_jwt_token_raises_401_for_expired_token():
    """
    Behavior: Tokens whose 'exp' claim is in the past must be rejected with an
    HTTP 401 response and an explicit expiration message.
    """
    expired_time = datetime.datetime.now(timezone.utc) - timedelta(seconds=10)
    token = create_token({"user_id": 1, "role": "operator", "exp": expired_time})

    with pytest.raises(HTTPException) as exc_info:
        verify_jwt_token(token)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Token expired. Please login again."


# ============================================================================
# BEHAVIOR: Rejecting Invalid, Tampered, or Malformed Tokens
# ============================================================================

@pytest.mark.parametrize(
    "token",
    [
        # Untrusted secret signature
        create_token({"user_id": 99, "role": "admin"}, secret="wrong-secret-key"),
        # Algorithm mismatch (server expects HS256)
        create_token({"user_id": 88, "role": "admin"}, algorithm="HS512"),
        # Structural/Format issues
        "",
        "invalid_token_string",
        "header.payload",  # missing signature
        "header.payload.signature.extra",
        "eyJhbGciOiJIUzI1NiJ9.invalid.signature",  # corrupted base64 payload
    ],
    ids=[
        "wrong-secret",
        "unsupported-algorithm",
        "empty-string",
        "random-string",
        "missing-signature-segment",
        "too-many-segments",
        "corrupted-base64",
    ],
)
def test_verify_jwt_token_raises_401_for_invalid_or_tampered_tokens(token: str):
    """
    Behavior: Any token that fails cryptographic validation or fails structural
    parsing must raise an HTTP 401 response stating the token is invalid.
    """
    with pytest.raises(HTTPException) as exc_info:
        verify_jwt_token(token)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid token. Please login again."