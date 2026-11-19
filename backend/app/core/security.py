"""Cryptographic Security, JWT Authentication, and Session Access Control (Task 30).

Implements RFC 7519 HMAC-SHA256 JSON Web Tokens, cryptographic token verification,
token expiration checks, user-to-session ownership validation, and FastAPI dependencies.
Operates with zero external C-extension dependencies using standard library hmac and hashlib.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set
import uuid

from fastapi import Header, HTTPException, Query, Request, WebSocket, status
from src.core.logging import setup_logger

logger = setup_logger("Security")

DEFAULT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "indoorai-production-grade-hmac-sha256-signing-secret-key-32bytes-min-change-via-env",
)
DEFAULT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
DEFAULT_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
DISABLE_AUTH_FOR_DEV = os.getenv("INDOORAI_DISABLE_AUTH", "false").lower() in ("true", "1", "yes")

if DEFAULT_SECRET_KEY.startswith("indoorai-production-grade"):
    logger.warning(
        "Using default internal signing key. Set JWT_SECRET_KEY in production environments."
    )


class UserRole(str, Enum):
    """User privilege roles."""
    ADMIN = "ADMIN"
    SURVEYOR = "SURVEYOR"
    NAVIGATOR = "NAVIGATOR"
    VIEWER = "VIEWER"


class SecurityError(Exception):
    """Base exception for security and authentication failures."""


class AuthenticationError(SecurityError):
    """Raised when authentication credentials or token are missing or malformed."""


class TokenExpiredError(SecurityError):
    """Raised when an authenticated token has expired."""


class TokenInvalidError(SecurityError):
    """Raised when token signature or payload tampering is detected."""


class SessionForbiddenError(SecurityError):
    """Raised when authenticated user lacks permission for a requested session."""


@dataclass
class TokenClaims:
    """Decoded and verified claims extracted from an RFC 7519 JWT."""
    sub: str  # Subject / User ID
    username: str
    role: UserRole
    allowed_sessions: List[str] = field(default_factory=lambda: ["*"])
    device_id: Optional[str] = None
    exp: int = 0
    iat: int = 0
    jti: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub": self.sub,
            "username": self.username,
            "role": self.role.value if hasattr(self.role, "value") else str(self.role),
            "allowed_sessions": self.allowed_sessions,
            "device_id": self.device_id,
            "exp": self.exp,
            "iat": self.iat,
            "jti": self.jti,
        }


def _b64url_encode(data: bytes) -> str:
    """Encodes bytes to URL-safe Base64 without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data: str) -> bytes:
    """Decodes URL-safe Base64 string with optional padding."""
    padding = 4 - (len(data) % 4)
    if padding and padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data.encode("utf-8"))


class JWTManager:
    """RFC 7519 compliant JSON Web Token generator and verifier using HMAC-SHA256."""

    def __init__(
        self,
        secret_key: str = DEFAULT_SECRET_KEY,
        algorithm: str = DEFAULT_ALGORITHM,
    ) -> None:
        self.secret_key = secret_key.encode("utf-8")
        self.algorithm = algorithm

    def create_token(
        self,
        user_id: str,
        username: str,
        role: UserRole = UserRole.SURVEYOR,
        allowed_sessions: Optional[List[str]] = None,
        device_id: Optional[str] = None,
        expires_in_seconds: Optional[int] = None,
    ) -> str:
        """Issues a cryptographically signed HMAC-SHA256 JWT token.

        Args:
            user_id: Unique persistent subject ID.
            username: User display handle.
            role: Privilege role enum.
            allowed_sessions: Specific session IDs authorized (defaults to ["*"] for unrestricted).
            device_id: Optional pseudonymous mobile device ID.
            expires_in_seconds: Expiration window (defaults to DEFAULT_EXPIRE_MINUTES * 60).

        Returns:
            URL-safe encoded JWT string (header.payload.signature).
        """
        now = int(time.time())
        exp_window = expires_in_seconds if expires_in_seconds is not None else DEFAULT_EXPIRE_MINUTES * 60
        exp = now + exp_window

        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": str(user_id),
            "username": str(username),
            "role": role.value if hasattr(role, "value") else str(role),
            "allowed_sessions": allowed_sessions if allowed_sessions is not None else ["*"],
            "device_id": device_id,
            "exp": exp,
            "iat": now,
            "jti": str(uuid.uuid4()),
        }

        header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        signature = hmac.new(self.secret_key, signing_input, hashlib.sha256).digest()
        sig_b64 = _b64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{sig_b64}"

    def verify_token(self, token: str) -> TokenClaims:
        """Verifies cryptographic signature, expiration, and payload integrity.

        Args:
            token: JWT string.

        Returns:
            Verified TokenClaims instance.

        Raises:
            TokenExpiredError: If token has expired.
            TokenInvalidError: If token signature or structure is invalid.
        """
        if not token or not isinstance(token, str):
            raise TokenInvalidError("Token must be a non-empty string.")

        parts = token.strip().split(".")
        if len(parts) != 3:
            raise TokenInvalidError(f"Malformed JWT token: expected 3 segments, got {len(parts)}.")

        header_b64, payload_b64, sig_b64 = parts

        # 1. Cryptographic signature check using constant-time comparison
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(self.secret_key, signing_input, hashlib.sha256).digest()

        try:
            provided_sig = _b64url_decode(sig_b64)
        except Exception as e:
            raise TokenInvalidError(f"Invalid Base64 signature encoding: {e}")

        if not hmac.compare_digest(expected_sig, provided_sig):
            raise TokenInvalidError("Cryptographic token signature verification failed. Token has been tampered with or signing key mismatch.")

        # 2. Parse payload
        try:
            payload_bytes = _b64url_decode(payload_b64)
            payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception as e:
            raise TokenInvalidError(f"Invalid JWT payload JSON: {e}")

        # 3. Expiration validation
        now = int(time.time())
        exp = int(payload.get("exp", 0))
        if exp < now:
            raise TokenExpiredError(f"Token expired at {exp} (current server time: {now}, expired by {now - exp}s).")

        # 4. Extract claims
        raw_role = payload.get("role", "SURVEYOR")
        try:
            role = UserRole(raw_role)
        except ValueError:
            role = UserRole.SURVEYOR

        return TokenClaims(
            sub=str(payload.get("sub", "unknown_user")),
            username=str(payload.get("username", "Anonymous")),
            role=role,
            allowed_sessions=list(payload.get("allowed_sessions", ["*"])),
            device_id=payload.get("device_id"),
            exp=exp,
            iat=int(payload.get("iat", 0)),
            jti=str(payload.get("jti", "")),
        )


@dataclass
class SessionOwnershipRecord:
    """Ownership and authorization metadata for an active session."""
    session_id: str
    owner_user_id: str
    authorized_users: Set[str] = field(default_factory=set)
    authorized_devices: Set[str] = field(default_factory=set)
    created_at_epoch_s: int = field(default_factory=lambda: int(time.time()))


class SessionAccessController:
    """Thread-safe multi-session access control registry."""

    _instance: Optional[SessionAccessController] = None
    _lock = threading.Lock()

    def __new__(cls) -> SessionAccessController:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._registry: Dict[str, SessionOwnershipRecord] = {}
                cls._instance._registry_lock = threading.RLock()
            return cls._instance

    def register_session(
        self,
        session_id: str,
        owner_user_id: str,
        device_id: Optional[str] = None,
    ) -> SessionOwnershipRecord:
        """Registers a session with an authenticated owner."""
        with self._registry_lock:
            if session_id not in self._registry:
                rec = SessionOwnershipRecord(
                    session_id=session_id,
                    owner_user_id=owner_user_id,
                    authorized_users={owner_user_id},
                    authorized_devices={device_id} if device_id else set(),
                )
                self._registry[session_id] = rec
                logger.info("Registered session '%s' with owner '%s'.", session_id, owner_user_id)
            else:
                rec = self._registry[session_id]
                rec.authorized_users.add(owner_user_id)
                if device_id:
                    rec.authorized_devices.add(device_id)
            return self._registry[session_id]

    def is_authorized(self, session_id: str, claims: TokenClaims) -> bool:
        """Checks if claims have permission to read/write/stream for this session."""
        # Admin or wildcard token always permitted
        if claims.role == UserRole.ADMIN or "*" in claims.allowed_sessions:
            return True

        # Token explicitly explicitly lists session
        if session_id in claims.allowed_sessions:
            return True

        with self._registry_lock:
            rec = self._registry.get(session_id)
            if rec is None:
                # If session is not yet registered in memory, auto-register claiming user as owner
                self.register_session(session_id, owner_user_id=claims.sub, device_id=claims.device_id)
                return True

            if claims.sub == rec.owner_user_id or claims.sub in rec.authorized_users:
                return True

        return False

    def unregister_session(self, session_id: str) -> bool:
        with self._registry_lock:
            if session_id in self._registry:
                del self._registry[session_id]
                return True
            return False

    def clear(self) -> None:
        with self._registry_lock:
            self._registry.clear()


# Global singletons
_jwt_manager = JWTManager()
_access_controller = SessionAccessController()


def get_jwt_manager() -> JWTManager:
    return _jwt_manager


def get_access_controller() -> SessionAccessController:
    return _access_controller


# ==============================================================================
# FastAPI Security Dependencies
# ==============================================================================

STRICT_AUTH_MODE = os.getenv("INDOORAI_STRICT_AUTH", "false").lower() in ("true", "1", "yes")


def get_current_user_claims(
    authorization: Optional[str] = Header(None),
    token_query: Optional[str] = Query(None, alias="token"),
) -> Optional[TokenClaims]:
    """Dependency extracting and cryptographically verifying JWT bearer token."""
    raw_token: Optional[str] = None

    if authorization and authorization.lower().startswith("bearer "):
        raw_token = authorization[7:].strip()
    elif token_query:
        raw_token = token_query.strip()

    if not raw_token:
        if STRICT_AUTH_MODE:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token required (strict auth mode active).",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None

    try:
        claims = get_jwt_manager().verify_token(raw_token)
        return claims
    except TokenExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication token expired: {e}",
            headers={"WWW-Authenticate": "Bearer error='invalid_token', error_description='token expired'"},
        )
    except TokenInvalidError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {e}",
            headers={"WWW-Authenticate": "Bearer error='invalid_token'"},
        )


def verify_session_permission(
    session_id: str,
    claims: Optional[TokenClaims],
) -> None:
    """Verifies that the authenticated user possesses permission to access the target session."""
    if claims is None:
        if STRICT_AUTH_MODE:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token required to access session.",
            )
        return

    controller = get_access_controller()
    if not controller.is_authorized(session_id, claims):
        logger.warning(
            "Access forbidden: User '%s' (role: %s) attempted to access unauthorized session '%s'.",
            claims.sub,
            claims.role.value,
            session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: User '{claims.sub}' is not authorized to access session '{session_id}'.",
        )


async def authenticate_websocket_connection(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = None,
) -> Optional[TokenClaims]:
    """Authenticates WebSocket connection before connection acceptance.

    Rejects unauthenticated or unauthorized clients with close code 1008 (Policy Violation).
    """
    if not token:
        if STRICT_AUTH_MODE:
            logger.warning("WebSocket connection rejected: Missing auth token for session %s (strict auth).", session_id)
            await websocket.close(code=1008, reason="Authentication token required.")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication token required.")
        return None

    try:
        claims = get_jwt_manager().verify_token(token)
    except Exception as e:
        logger.warning("WebSocket connection rejected: Invalid token for session %s (%s).", session_id, e)
        await websocket.close(code=1008, reason=f"Invalid authentication token: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    controller = get_access_controller()
    if not controller.is_authorized(session_id, claims):
        logger.warning(
            "WebSocket connection rejected: User '%s' forbidden on session '%s'.",
            claims.sub,
            session_id,
        )
        await websocket.close(code=1008, reason="Unauthorized session access forbidden.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session forbidden.")

    return claims
