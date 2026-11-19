"""Authentication and JWT Token API Routes (Task 30).

Provides secure token issuance, verification, and session credential exchange.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.security import (
    TokenClaims,
    UserRole,
    get_current_user_claims,
    get_jwt_manager,
    get_access_controller,
)
from src.core.logging import setup_logger

logger = setup_logger("AuthRoutes")

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    """Credentials / handshake request for acquiring an access token."""
    user_id: str = Field(..., description="Unique subject or surveyor user identifier")
    username: Optional[str] = Field(default=None, description="Display handle")
    role: Optional[str] = Field(default="SURVEYOR", description="User role: ADMIN, SURVEYOR, NAVIGATOR, VIEWER")
    allowed_sessions: Optional[List[str]] = Field(default=None, description="Explicit list of permitted sessions (or ['*'] for all)")
    device_id: Optional[str] = Field(default=None, description="Client hardware identifier")
    expires_in_seconds: Optional[int] = Field(default=3600, description="Token lifetime in seconds (default: 1 hour)")


class TokenResponse(BaseModel):
    """JWT Token bearer response."""
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user_id: str
    role: str
    allowed_sessions: List[str]


@router.post("/token", response_model=TokenResponse, summary="Issue Authentication Token")
def issue_access_token(req: TokenRequest) -> TokenResponse:
    """Issues a cryptographically signed HMAC-SHA256 JWT access token."""
    raw_role = (req.role or "SURVEYOR").upper()
    try:
        role = UserRole(raw_role)
    except ValueError:
        role = UserRole.SURVEYOR

    jwt_mgr = get_jwt_manager()
    token = jwt_mgr.create_token(
        user_id=req.user_id,
        username=req.username or req.user_id,
        role=role,
        allowed_sessions=req.allowed_sessions,
        device_id=req.device_id,
        expires_in_seconds=req.expires_in_seconds or 3600,
    )

    # Register allowed sessions in access controller if specific sessions requested
    if req.allowed_sessions and "*" not in req.allowed_sessions:
        controller = get_access_controller()
        for sid in req.allowed_sessions:
            controller.register_session(session_id=sid, owner_user_id=req.user_id, device_id=req.device_id)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_seconds=req.expires_in_seconds or 3600,
        user_id=req.user_id,
        role=role.value,
        allowed_sessions=req.allowed_sessions or ["*"],
    )


@router.get("/verify", summary="Verify Authenticated Token")
def verify_token(claims: TokenClaims = Depends(get_current_user_claims)) -> Dict[str, Any]:
    """Inspects and returns decoded claims for the current authenticated token."""
    return {
        "status": "valid",
        "claims": claims.to_dict(),
    }


@router.post("/dev-token", response_model=TokenResponse, summary="Generate Quick Developer / Test Token")
def issue_dev_token(
    user_id: str = "test_surveyor_01",
    session_id: str = "TEST_SESSION_01",
    role: str = "SURVEYOR",
) -> TokenResponse:
    """Convenience helper for automated test suites and development environments."""
    req = TokenRequest(
        user_id=user_id,
        username=user_id,
        role=role,
        allowed_sessions=[session_id],
        expires_in_seconds=7200,
    )
    return issue_access_token(req)
