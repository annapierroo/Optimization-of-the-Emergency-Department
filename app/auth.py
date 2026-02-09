"""Simple role-based authentication helpers for Streamlit."""
from __future__ import annotations

import streamlit as st


_USERS = {
    "admin": "admin",
    "user": "user",
}


def _set_authenticated(username: str) -> None:
    st.session_state["authenticated"] = True
    st.session_state["username"] = username
    st.session_state["role"] = username


def _clear_session() -> None:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None


def ensure_login() -> None:
    """Render login form and stop page execution until authenticated."""
    if st.session_state.get("authenticated", False):
        return

    hide_sidebar_navigation()
    st.title("Emergency Department Dashboard Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if not submitted:
        st.stop()

    expected_password = _USERS.get(username)
    if expected_password is None or password != expected_password:
        st.error("Invalid username or password.")
        st.stop()

    _set_authenticated(username)
    st.rerun()


def get_role() -> str | None:
    return st.session_state.get("role")


def is_admin() -> bool:
    return get_role() == "admin"


def require_user_role() -> None:
    """Allow only the prediction user role to access a page."""
    role = get_role()
    if role != "user":
        st.switch_page("streamlit_app.py")
        st.stop()


def require_admin_role() -> None:
    """Allow only admin role to access a page."""
    if not is_admin():
        st.switch_page("streamlit_app.py")
        st.stop()


def hide_sidebar_navigation() -> None:
    """Hide Streamlit multipage navigation block in the sidebar."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hide_data_drift_for_user() -> None:
    """Hide Data Drift page entry from sidebar navigation for regular users."""
    if get_role() != "user":
        return
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] a[href*="4_Data_Drift"],
        [data-testid="stSidebarNav"] a[href*="Data_Drift"],
        [data-testid="stSidebarNav"] a[href*="data_drift"] {
            display: none !important;
        }
        [data-testid="stSidebarNav"] li:has(a[href*="4_Data_Drift"]),
        [data-testid="stSidebarNav"] li:has(a[href*="Data_Drift"]),
        [data-testid="stSidebarNav"] li:has(a[href*="data_drift"]) {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_session_panel() -> None:
    """Render user/session controls in the sidebar."""
    hide_data_drift_for_user()
    st.sidebar.caption(f"Logged in as: {st.session_state.get('username')}")
    if st.sidebar.button("Logout"):
        _clear_session()
        st.rerun()
