"""Security & Role-Based Access Control (RBAC) Dashboard View (Task 30).

Provides interactive inspection of cryptographic authentication status, decoded JWT claims,
role-to-permission matrices, live endpoint access-control probes, and scoped developer token issuance.
"""

from __future__ import annotations

import datetime
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from frontend.api_client import DatasetApiClient, DatasetApiClientException


def render_security_view(client: DatasetApiClient, active_dataset_id: Optional[str] = None) -> None:
    """Renders the complete Security & Access Control Dashboard View."""
    st.markdown("## 🔐 20. Security & Role-Based Access Control (RBAC)")
    st.markdown(
        "Inspect cryptographic authentication state, verify RFC 7519 HMAC-SHA256 token signatures, "
        "review role privilege matrices, run live endpoint access-control probes, and issue "
        "scoped credentials for authorized testing (Task 30)."
    )

    current_token: Optional[str] = st.session_state.get("auth_token")

    # -------------------------------------------------------------------------
    # 1. Active Authentication Status & Token Details
    # -------------------------------------------------------------------------
    st.markdown("### 1. Active Authentication Status & Token Claims")

    claims_data: Optional[Dict[str, Any]] = None
    is_authenticated = False
    verification_error: Optional[str] = None

    if current_token:
        try:
            res = client.verify_token(current_token)
            if res.get("status") == "valid" and "claims" in res:
                claims_data = res["claims"]
                is_authenticated = True
                # Cache claims in session_state
                st.session_state["auth_claims"] = claims_data
                st.session_state["auth_user"] = claims_data.get("sub", "Unknown")
                st.session_state["auth_role"] = claims_data.get("role", "VIEWER")
        except DatasetApiClientException as e:
            verification_error = str(e)
            is_authenticated = False
            st.session_state["auth_claims"] = None
        except Exception as e:
            verification_error = f"Verification error: {e}"
            is_authenticated = False
            st.session_state["auth_claims"] = None
    else:
        st.session_state["auth_claims"] = None

    # Status Cards Grid
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        if is_authenticated:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-value" style="color: #16A34A; font-size: 1.5rem;">🟢 Active</div>'
                '<div class="metric-label">Authentication Status</div>'
                '<p style="margin-top: 4px; font-size: 0.85rem; color: #166534;">Verified JWT Bearer</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-value" style="color: #64748B; font-size: 1.5rem;">⚪ Guest</div>'
                '<div class="metric-label">Authentication Status</div>'
                '<p style="margin-top: 4px; font-size: 0.85rem; color: #475569;">Unauthenticated Mode</p>'
                '</div>',
                unsafe_allow_html=True,
            )

    with col_stat2:
        role_label = claims_data.get("role", "VIEWER") if claims_data else "GUEST"
        role_colors = {
            "ADMIN": ("#DC2626", "#FEE2E2", "#991B1B"),
            "SURVEYOR": ("#2563EB", "#DBEAFE", "#1E40AF"),
            "NAVIGATOR": ("#059669", "#D1FAE5", "#065F46"),
            "VIEWER": ("#475569", "#F1F5F9", "#334155"),
            "GUEST": ("#64748B", "#F8FAFC", "#475569"),
        }
        fg_col, bg_col, text_col = role_colors.get(role_label, ("#475569", "#F1F5F9", "#334155"))
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color: {fg_col}; font-size: 1.5rem;">{role_label}</div>'
            f'<div class="metric-label">Authenticated Role</div>'
            f'<p style="margin-top: 4px; font-size: 0.85rem; color: {text_col};">Privilege Scope</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_stat3:
        sub_label = claims_data.get("sub", "Anonymous") if claims_data else "None"
        display_sub = sub_label[:14] + "..." if len(sub_label) > 16 else sub_label
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color: #0F172A; font-size: 1.35rem;">{display_sub}</div>'
            f'<div class="metric-label">Subject / User ID</div>'
            f'<p style="margin-top: 4px; font-size: 0.85rem; color: #64748B;">Client Identity</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_stat4:
        if claims_data and "exp" in claims_data:
            exp_time = int(claims_data["exp"])
            now = int(time.time())
            diff = exp_time - now
            if diff > 0:
                mins = diff // 60
                secs = diff % 60
                time_str = f"{mins}m {secs}s"
                time_color = "#16A34A" if mins >= 10 else "#D97706"
                time_note = "Valid Window"
            else:
                time_str = "Expired"
                time_color = "#DC2626"
                time_note = "Token Needs Refresh"
        else:
            time_str = "N/A"
            time_color = "#64748B"
            time_note = "No Active Token"

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color: {time_color}; font-size: 1.5rem;">{time_str}</div>'
            f'<div class="metric-label">Time to Expiration</div>'
            f'<p style="margin-top: 4px; font-size: 0.85rem; color: #64748B;">{time_note}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Detailed Claims Card (if authenticated)
    if is_authenticated and claims_data:
        st.markdown("#### 📋 Decoded Token Claims (RFC 7519)")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown(f"**Username / Display:** `{claims_data.get('username', 'N/A')}`")
            st.markdown(f"**Subject ID (`sub`):** `{claims_data.get('sub')}`")
            st.markdown(f"**Assigned Role (`role`):** `{claims_data.get('role')}`")
            st.markdown(f"**Client Device (`device_id`):** `{claims_data.get('device_id') or 'Unspecified (Web / CLI)'}`")
        with c_col2:
            iat_dt = datetime.datetime.fromtimestamp(claims_data.get("iat", 0)).strftime("%Y-%m-%d %H:%M:%S UTC")
            exp_dt = datetime.datetime.fromtimestamp(claims_data.get("exp", 0)).strftime("%Y-%m-%d %H:%M:%S UTC")
            st.markdown(f"**Issued At (`iat`):** `{iat_dt}`")
            st.markdown(f"**Expires At (`exp`):** `{exp_dt}`")
            st.markdown(f"**Token ID (`jti`):** `{claims_data.get('jti', 'N/A')}`")
            allowed_sess = claims_data.get("allowed_sessions", ["*"])
            sess_str = "Unrestricted (*)" if "*" in allowed_sess else ", ".join(allowed_sess)
            st.markdown(f"**Permitted Sessions:** `{sess_str}`")

        # Masked Token String Box
        with st.expander("👁️ View Raw Authenticated Bearer Token"):
            masked_token = (
                f"{current_token[:20]}...[MASKED]...{current_token[-16:]}"
                if len(current_token) > 40
                else current_token
            )
            st.markdown(f"**Masked Token Preview:** `{masked_token}`")
            st.code(current_token, language="text")
            st.caption("Keep this token confidential. Never share or log full production bearer tokens.")

        # Session Action Button
        if st.button("🚪 Logout / Clear Active Token", type="secondary"):
            st.session_state["auth_token"] = None
            st.session_state["auth_user"] = None
            st.session_state["auth_role"] = None
            st.session_state["auth_claims"] = None
            client.clear_token()
            st.success("Authentication token cleared. Session returned to guest mode.")
            st.rerun()

    elif current_token and verification_error:
        st.error(f"⚠️ Token Verification Failed: {verification_error}")
        st.caption("The stored token is invalid, expired, or was signed with a different key.")
        if st.button("Clear Invalid Token", type="primary"):
            st.session_state["auth_token"] = None
            client.clear_token()
            st.rerun()
    else:
        st.info("💡 You are currently browsing in Guest mode. Public endpoints (such as UJIIndoorLoc benchmarks and catalog exploration) are accessible. Use the token generator or sidebar to log in.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Role Permissions Matrix (RBAC Definition)
    # -------------------------------------------------------------------------
    st.markdown("### 2. Role-Based Access Control (RBAC) Permissions Matrix")
    st.markdown(
        "IndoorAI enforces cryptographically verified role-based access control. "
        "User roles cannot be elevated on the client: the backend validates each request against "
        "the cryptographically signed token claims."
    )

    rbac_records = [
        {
            "Capability / Operation": "UJIIndoorLoc ML Benchmark & Inference",
            "ADMIN": "✅ Full",
            "SURVEYOR": "✅ Full",
            "NAVIGATOR": "✅ Full",
            "VIEWER": "✅ Full (Public)",
        },
        {
            "Capability / Operation": "Survey Catalog & Quality Reports Inspection",
            "ADMIN": "✅ Full",
            "SURVEYOR": "✅ Full",
            "NAVIGATOR": "✅ Full",
            "VIEWER": "✅ Full (Public)",
        },
        {
            "Capability / Operation": "Dataset Upload & Multi-modal Ingestion",
            "ADMIN": "✅ Permitted",
            "SURVEYOR": "✅ Permitted",
            "NAVIGATOR": "❌ Denied",
            "VIEWER": "❌ Denied",
        },
        {
            "Capability / Operation": "PDR Calibration & Step Parameter Optimization",
            "ADMIN": "✅ Permitted",
            "SURVEYOR": "✅ Permitted",
            "NAVIGATOR": "❌ Denied",
            "VIEWER": "❌ Denied",
        },
        {
            "Capability / Operation": "Wi-Fi Radio Map Compilation & Landmarks",
            "ADMIN": "✅ Permitted",
            "SURVEYOR": "✅ Permitted",
            "NAVIGATOR": "❌ Denied",
            "VIEWER": "❌ Denied",
        },
        {
            "Capability / Operation": "Live WebSocket Ingestion (Real-Time Streams)",
            "ADMIN": "✅ Any Session",
            "SURVEYOR": "✅ Owned Sessions",
            "NAVIGATOR": "❌ Denied",
            "VIEWER": "❌ Denied",
        },
        {
            "Capability / Operation": "Indoor Route Planning & Turn-by-Turn Guidance",
            "ADMIN": "✅ Permitted",
            "SURVEYOR": "✅ Permitted",
            "NAVIGATOR": "✅ Primary Role",
            "VIEWER": "❌ Denied",
        },
        {
            "Capability / Operation": "Floor Plan Editing & Topological Publishing",
            "ADMIN": "✅ Permitted",
            "SURVEYOR": "❌ Restricted",
            "NAVIGATOR": "❌ Denied",
            "VIEWER": "❌ Denied",
        },
        {
            "Capability / Operation": "Dataset Purge & Permanent Deletion",
            "ADMIN": "✅ Permitted",
            "SURVEYOR": "❌ Restricted",
            "NAVIGATOR": "❌ Denied",
            "VIEWER": "❌ Denied",
        },
    ]

    rbac_df = pd.DataFrame(rbac_records)
    st.dataframe(rbac_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-top: 8px; margin-bottom: 16px;">
            <p style="margin: 0; font-size: 0.88rem; color: #475569;">
                <strong>🛡️ Cryptographic Invariant:</strong> Tokens are signed using HMAC-SHA256 with constant-time signature comparison (<code>hmac.compare_digest</code>). Tokens cannot be forged, manipulated, or transferred across disjoint sessions without failing server-side signature validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3. Interactive Access Control Tester
    # -------------------------------------------------------------------------
    st.markdown("### 3. Interactive Endpoint Access-Control Probe")
    st.markdown(
        "Execute live diagnostic probes against backend endpoints using the currently active token "
        "to test authentication gating, status codes, and cross-session authorization enforcement."
    )

    probe_col1, probe_col2 = st.columns([3, 2])

    with probe_col1:
        probe_target = st.selectbox(
            "Select Probe Target Endpoint:",
            [
                "GET /api/auth/verify (Verify Token & Identity)",
                "GET /api/realtime/sessions (List Authorized Sessions)",
                "GET /api/realtime/sessions/{session_id}/status (Session Telemetry Status)",
                "POST /api/realtime/sessions/{session_id}/control (Session Stream Control)",
                "GET /api/datasets (Catalog Inspection)",
            ],
            index=0,
        )

        test_session_id = st.text_input(
            "Target Session ID (for session endpoints):",
            value="LIVE_SURVEY_01",
            help="Session identifier to probe. Test authorized vs unauthorized session access.",
        )

    with probe_col2:
        st.markdown("#### Probe Parameters")
        st.caption(f"Active Token: `{'Present (' + role_label + ')' if is_authenticated else 'None (Guest)'}`")
        execute_probe = st.button("⚡ Execute Security Probe", type="primary", use_container_width=True)

    if execute_probe:
        st.markdown("#### Probe Results")
        probe_url = ""
        probe_method = "GET"
        probe_json: Optional[Dict[str, Any]] = None
        probe_params: Optional[Dict[str, Any]] = None

        if "GET /api/auth/verify" in probe_target:
            probe_url = f"{client.base_url}/api/auth/verify"
            probe_method = "GET"
        elif "GET /api/realtime/sessions/{session_id}/status" in probe_target:
            probe_url = f"{client.base_url}/api/realtime/sessions/{test_session_id}/status"
            probe_method = "GET"
        elif "POST /api/realtime/sessions/{session_id}/control" in probe_target:
            probe_url = f"{client.base_url}/api/realtime/sessions/{test_session_id}/control"
            probe_method = "POST"
            probe_params = {"action": "START"}
        elif "GET /api/realtime/sessions" in probe_target:
            probe_url = f"{client.base_url}/api/realtime/sessions"
            probe_method = "GET"
        elif "GET /api/datasets" in probe_target:
            probe_url = f"{client.base_url}/api/datasets"
            probe_method = "GET"

        start_t = time.perf_counter()
        status_code = 0
        response_body: Any = None
        error_detail = ""

        try:
            res = client._request(
                method=probe_method,
                url=probe_url,
                params=probe_params,
                json=probe_json,
                timeout=5.0,
            )
            status_code = res.status_code
            try:
                response_body = res.json()
            except Exception:
                response_body = res.text
        except Exception as e:
            error_detail = str(e)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            if status_code == 200:
                st.success(f"HTTP Status: **{status_code} OK**")
            elif status_code == 401:
                st.error(f"HTTP Status: **{status_code} Unauthorized**")
            elif status_code == 403:
                st.warning(f"HTTP Status: **{status_code} Forbidden**")
            else:
                st.info(f"HTTP Status: **{status_code}**")

        with res_col2:
            st.metric("Roundtrip Latency", f"{elapsed_ms:.1f} ms")

        with res_col3:
            if status_code == 200:
                decision = "🟢 Access Granted"
            elif status_code == 401:
                decision = "🔴 Denied (Unauthenticated)"
            elif status_code == 403:
                decision = "🟠 Denied (Forbidden Scope)"
            else:
                decision = f"⚪ Code {status_code}"
            st.metric("Authorization Decision", decision)

        st.markdown("**Target Request:**")
        st.code(f"{probe_method} {probe_url}", language="http")

        st.markdown("**Server Response Payload:**")
        if response_body:
            st.json(response_body)
        elif error_detail:
            st.error(f"Client request error: {error_detail}")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 4. Developer Token Generator (Sandbox & Test Mode)
    # -------------------------------------------------------------------------
    st.markdown("### 4. Developer & Testing Token Generator")
    st.markdown(
        "Issue cryptographically signed JWT credentials for development, integration testing, "
        "and demonstration purposes. In production, tokens are issued by external identity providers (OAuth 2.0 / OIDC)."
    )

    with st.expander("🛠️ Open Developer Token Issuance Form", expanded=not is_authenticated):
        st.caption("Generate a signed token for any role to test RBAC and session isolation in real-time.")

        dev_col1, dev_col2 = st.columns(2)
        with dev_col1:
            gen_role = st.selectbox(
                "Privilege Role:",
                ["SURVEYOR", "ADMIN", "NAVIGATOR", "VIEWER"],
                index=0,
                help="Select the privilege level to embed in the JWT claims.",
            )
            gen_user_id = st.text_input("User / Subject Identifier (`sub`):", value=f"dev_{gen_role.lower()}_01")
            gen_username = st.text_input("Display Name (`username`):", value=f"Dev {gen_role.title()} User")

        with dev_col2:
            gen_sessions_str = st.text_input(
                "Permitted Session IDs (comma-separated or '*' for all):",
                value="*",
                help="Specify '*' for unrestricted access, or list explicit session IDs to test cross-session restrictions.",
            )
            gen_device_id = st.text_input("Device ID (Optional):", value="DEV-WORKSTATION-01")
            gen_exp_mins = st.slider("Token Validity (Minutes):", min_value=5, max_value=1440, value=60, step=15)

        gen_button_col1, gen_button_col2 = st.columns([2, 3])
        with gen_button_col1:
            if st.button("🚀 Issue & Activate Token", type="primary", use_container_width=True):
                allowed_sessions = (
                    ["*"]
                    if gen_sessions_str.strip() == "*"
                    else [s.strip() for s in gen_sessions_str.split(",") if s.strip()]
                )
                try:
                    issued = client.issue_token(
                        user_id=gen_user_id.strip() or "dev_user",
                        username=gen_username.strip() or gen_user_id.strip(),
                        role=gen_role,
                        allowed_sessions=allowed_sessions,
                        device_id=gen_device_id.strip() or None,
                        expires_in_seconds=gen_exp_mins * 60,
                    )
                    new_token = issued.get("access_token")
                    if new_token:
                        st.session_state["auth_token"] = new_token
                        client.set_token(new_token)
                        # Verify immediately
                        verified = client.verify_token(new_token)
                        st.session_state["auth_claims"] = verified.get("claims")
                        st.session_state["auth_user"] = gen_user_id
                        st.session_state["auth_role"] = gen_role
                        st.success(f"✅ Issued and activated {gen_role} token for '{gen_user_id}'!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to issue token: {e}")

        # Quick Presets
        st.markdown("##### Quick Role Presets (1-Click)")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            if st.button("🔑 Admin (All)", use_container_width=True):
                try:
                    tok = client.generate_dev_token(role="ADMIN", user_id="admin_superuser", session_id="*")
                    t_str = tok.get("access_token")
                    st.session_state["auth_token"] = t_str
                    client.set_token(t_str)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        with p_col2:
            if st.button("📐 Surveyor", use_container_width=True):
                try:
                    tok = client.generate_dev_token(role="SURVEYOR", user_id="surveyor_alice", session_id="LIVE_SURVEY_01")
                    t_str = tok.get("access_token")
                    st.session_state["auth_token"] = t_str
                    client.set_token(t_str)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        with p_col3:
            if st.button("🧭 Navigator", use_container_width=True):
                try:
                    tok = client.generate_dev_token(role="NAVIGATOR", user_id="navigator_bob", session_id="LIVE_SURVEY_01")
                    t_str = tok.get("access_token")
                    st.session_state["auth_token"] = t_str
                    client.set_token(t_str)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        with p_col4:
            if st.button("👁️ Viewer", use_container_width=True):
                try:
                    tok = client.generate_dev_token(role="VIEWER", user_id="viewer_guest", session_id="*")
                    t_str = tok.get("access_token")
                    st.session_state["auth_token"] = t_str
                    client.set_token(t_str)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 5. Security Architecture & Diagnostics Summary
    # -------------------------------------------------------------------------
    st.markdown("### 5. Security Architecture & System Diagnostics")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.markdown("#### Cryptographic Specifications")
        st.markdown(
            """
            - **Standard:** RFC 7519 JSON Web Token (JWT)
            - **Signing Algorithm:** HMAC-SHA256 (`HS256`)
            - **Constant-Time Verification:** Python `hmac.compare_digest`
            - **Zero C-Extension Reliance:** Standard library implementation
            - **Clock Skew Tolerance:** Exact server epoch seconds
            - **Header Specification:** `Authorization: Bearer <token>`
            """
        )

    with d_col2:
        st.markdown("#### Defense-in-Depth Protections")
        st.markdown(
            """
            - **Tamper Resistance:** Payload alterations invalidate HMAC signature
            - **Session Isolation:** Cross-session spoofing prohibited by `SessionAccessController`
            - **Strict Auth Guard:** Configurable `INDOORAI_STRICT_AUTH` production mode
            - **Secret Hygiene:** Custom signing secret support via `JWT_SECRET_KEY`
            - **Token Expiry:** Configurable lifetime window with automatic rejection
            - **WebSocket Protection:** Handshake verification with policy violation close (1008)
            """
        )
