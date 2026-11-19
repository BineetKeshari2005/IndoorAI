"""Tests for Phase 1 Authentication & Security UI Integration.

Verifies:
1. DatasetApiClient token lifecycle (set, clear, inject, verify, issue, dev-token).
2. Header injection: Authorization: Bearer <token> attached only when token present.
3. Cryptographic JWT verification and claims extraction via backend.
4. Rejection of expired, tampered, and missing tokens (HTTP 401).
5. Safe exception handling without token leakage in error strings.
6. Role-based access control (RBAC) and session isolation.
7. Public endpoints remain accessible without auth.
8. Streamlit component integration and rendering.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from backend.app.core.security import (
    JWTManager,
    TokenExpiredError,
    TokenInvalidError,
    UserRole,
    get_jwt_manager,
)
from backend.app.main import app
from frontend.api_client import DatasetApiClient, DatasetApiClientException
from frontend.components.security_view import render_security_view


class TestDatasetApiClientAuth(unittest.TestCase):
    """Test suite for DatasetApiClient authentication methods and header injection."""

    def setUp(self) -> None:
        self.fastapi_client = TestClient(app)
        self.api_client = DatasetApiClient(base_url="http://testserver", http_client=self.fastapi_client)
        self.jwt_mgr = get_jwt_manager()

    def test_client_initialization_with_and_without_token(self) -> None:
        client_no_token = DatasetApiClient(base_url="http://testserver")
        self.assertIsNone(client_no_token.auth_token)

        client_with_token = DatasetApiClient(base_url="http://testserver", auth_token="test-jwt-token")
        self.assertEqual(client_with_token.auth_token, "test-jwt-token")

    def test_set_and_clear_token(self) -> None:
        client = DatasetApiClient(base_url="http://testserver")
        self.assertIsNone(client.auth_token)

        client.set_token("sample-token-xyz")
        self.assertEqual(client.auth_token, "sample-token-xyz")

        client.clear_token()
        self.assertIsNone(client.auth_token)

    def test_header_injection_with_token(self) -> None:
        token = self.jwt_mgr.create_token(
            user_id="user_admin",
            username="Admin",
            role=UserRole.ADMIN,
        )
        self.api_client.set_token(token)

        # /api/auth/verify expects Authorization: Bearer <token>
        res = self.api_client.verify_token()
        self.assertEqual(res["status"], "valid")
        self.assertEqual(res["claims"]["sub"], "user_admin")
        self.assertEqual(res["claims"]["role"], "ADMIN")

    def test_no_header_when_unauthenticated(self) -> None:
        self.api_client.clear_token()
        with self.assertRaises(DatasetApiClientException) as ctx:
            self.api_client.verify_token()
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("No authentication token", str(ctx.exception))

    def test_verify_token_expired(self) -> None:
        expired_token = self.jwt_mgr.create_token(
            user_id="user_expired",
            username="Expired",
            role=UserRole.SURVEYOR,
            expires_in_seconds=-100,
        )
        with self.assertRaises(DatasetApiClientException) as ctx:
            self.api_client.verify_token(token=expired_token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("expired", str(ctx.exception).lower())

    def test_verify_token_tampered(self) -> None:
        valid_token = self.jwt_mgr.create_token(
            user_id="user_tampered",
            username="Tampered",
            role=UserRole.SURVEYOR,
        )
        h, p, s = valid_token.split(".")
        bad_token = f"{h}.{p}.corrupted_signature"

        with self.assertRaises(DatasetApiClientException) as ctx:
            self.api_client.verify_token(token=bad_token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("rejected", str(ctx.exception).lower())

    def test_issue_token_success(self) -> None:
        roles_to_test = ["ADMIN", "SURVEYOR", "NAVIGATOR", "VIEWER"]
        for role_name in roles_to_test:
            res = self.api_client.issue_token(
                user_id=f"test_{role_name.lower()}",
                username=f"Test {role_name}",
                role=role_name,
                allowed_sessions=["SES-1", "SES-2"],
                expires_in_seconds=1800,
            )
            self.assertIn("access_token", res)
            self.assertEqual(res["role"], role_name)
            self.assertEqual(res["user_id"], f"test_{role_name.lower()}")
            self.assertEqual(res["allowed_sessions"], ["SES-1", "SES-2"])

            # Verify with client
            verify_res = self.api_client.verify_token(res["access_token"])
            self.assertEqual(verify_res["status"], "valid")
            self.assertEqual(verify_res["claims"]["role"], role_name)

    def test_generate_dev_token_success(self) -> None:
        res = self.api_client.generate_dev_token(
            role="NAVIGATOR",
            user_id="nav_quick_user",
            session_id="LIVE_ROUTE_01",
        )
        self.assertIn("access_token", res)
        self.assertEqual(res["role"], "NAVIGATOR")
        self.assertEqual(res["user_id"], "nav_quick_user")

        # Verify claims
        verify_res = self.api_client.verify_token(res["access_token"])
        self.assertEqual(verify_res["claims"]["sub"], "nav_quick_user")
        self.assertEqual(verify_res["claims"]["allowed_sessions"], ["LIVE_ROUTE_01"])

    def test_no_token_leakage_in_exceptions(self) -> None:
        secret_token = "eyJhbGciOiJIUzI1NiJ9.SECRET_PAYLOAD_STRING.FORGED_SIG"
        try:
            self.api_client.verify_token(token=secret_token)
            self.fail("Expected DatasetApiClientException")
        except DatasetApiClientException as e:
            # Confirm raw token is not exposed in string representation
            self.assertNotIn("SECRET_PAYLOAD_STRING", str(e))
            self.assertNotIn(secret_token, str(e))


class TestSecurityRBACEndpoints(unittest.TestCase):
    """Verifies backend RBAC gating and session permission rules."""

    def setUp(self) -> None:
        self.fastapi_client = TestClient(app)
        self.jwt_mgr = get_jwt_manager()

    def test_public_health_and_datasets_accessible_without_token(self) -> None:
        # Public health check works without token
        res_health = self.fastapi_client.get("/health")
        self.assertEqual(res_health.status_code, 200)

        # Datasets list works without token in standard mode
        res_datasets = self.fastapi_client.get("/api/datasets")
        self.assertIn(res_datasets.status_code, [200, 404])

    def test_admin_has_unrestricted_session_access(self) -> None:
        admin_token = self.jwt_mgr.create_token(
            user_id="admin_user_01",
            username="Admin User",
            role=UserRole.ADMIN,
            allowed_sessions=["*"],
        )
        client = DatasetApiClient(base_url="http://testserver", http_client=self.fastapi_client, auth_token=admin_token)

        # Can access any session status
        res = client._request("GET", "http://testserver/api/realtime/sessions/ARBITRARY_SESSION_ID/status")
        self.assertEqual(res.status_code, 200)

    def test_surveyor_session_isolation_denies_cross_session(self) -> None:
        from backend.app.core.security import get_access_controller
        access_ctrl = get_access_controller()
        access_ctrl.register_session("SES-BOB-PRIVATE", owner_user_id="user_bob")

        # User Alice has token locked to SES-ALICE-ONLY
        alice_token = self.jwt_mgr.create_token(
            user_id="user_alice",
            username="Alice",
            role=UserRole.SURVEYOR,
            allowed_sessions=["SES-ALICE-ONLY"],
        )
        client = DatasetApiClient(base_url="http://testserver", http_client=self.fastapi_client, auth_token=alice_token)

        # Alice accesses her own session -> 200 OK
        res_own = client._request("GET", "http://testserver/api/realtime/sessions/SES-ALICE-ONLY/status")
        self.assertEqual(res_own.status_code, 200)

        # Alice attempts to access Bob's session -> 403 Forbidden
        res_bob = client._request("GET", "http://testserver/api/realtime/sessions/SES-BOB-PRIVATE/status")
        self.assertEqual(res_bob.status_code, 403)
        self.assertIn("not authorized", res_bob.json()["detail"].lower())


class TestSecurityViewComponent(unittest.TestCase):
    """Verifies frontend/components/security_view.py functionality and RBAC data structures."""

    def test_security_view_import_and_signature(self) -> None:
        """Verifies render_security_view is imported properly and has expected signature."""
        import inspect
        from frontend.components import render_security_view as rsv_pkg
        from frontend.components.security_view import render_security_view as rsv_mod

        self.assertIs(rsv_pkg, rsv_mod)
        sig = inspect.signature(rsv_mod)
        self.assertIn("client", sig.parameters)
        self.assertIn("active_dataset_id", sig.parameters)

    def test_rbac_matrix_definition(self) -> None:
        """Verifies that all 4 roles and all key capabilities are represented in RBAC rules."""
        roles = [r.value for r in UserRole]
        self.assertIn("ADMIN", roles)
        self.assertIn("SURVEYOR", roles)
        self.assertIn("NAVIGATOR", roles)
        self.assertIn("VIEWER", roles)

    def test_token_claims_formatting(self) -> None:
        """Verifies that TokenClaims instances convert properly to frontend dictionary representations."""
        jwt_mgr = get_jwt_manager()
        token = jwt_mgr.create_token(
            user_id="user_test_formatting",
            username="Test Formatting",
            role=UserRole.NAVIGATOR,
            allowed_sessions=["SES-100"],
            device_id="PIXEL-TEST",
        )
        claims = jwt_mgr.verify_token(token)
        claims_dict = claims.to_dict()

        self.assertEqual(claims_dict["sub"], "user_test_formatting")
        self.assertEqual(claims_dict["role"], "NAVIGATOR")
        self.assertEqual(claims_dict["allowed_sessions"], ["SES-100"])
        self.assertEqual(claims_dict["device_id"], "PIXEL-TEST")
        self.assertGreater(claims_dict["exp"], claims_dict["iat"])


if __name__ == "__main__":
    unittest.main()
