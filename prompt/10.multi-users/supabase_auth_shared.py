"""multi-users-ref.py · pages/회원가입.py 공통 — Supabase 클라이언트·회원가입 로직."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import streamlit as st
from supabase import Client, create_client


def streamlit_secrets_into_environ() -> None:
    """Streamlit Cloud Secrets가 os.getenv로 보이도록 동기화."""
    try:
        if not hasattr(st, "secrets"):
            return
        # secrets.toml(또는 Cloud에서 마운트된 secrets)이 없을 때 `st.secrets[key]`를
        # 건드리면 Streamlit이 'No secrets found. Valid paths...' 를 UI에 띄운다.
        # 디스크/마운트에 toml이 실제로 파싱될 때만 이후 접근을 한다.
        try:
            from streamlit.runtime.secrets import secrets_singleton
        except ImportError:
            secrets_singleton = None
        if secrets_singleton is not None and not secrets_singleton.load_if_toml_exists():
            return
        for key in (
            "SUPABASE_URL",
            "SUPABASE_ANON_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
            "SUPABASE_EMAIL_REDIRECT_URL",
        ):
            if os.getenv(key):
                continue
            try:
                val = st.secrets[key]  # type: ignore[index]
            except Exception:
                continue
            if val:
                os.environ[key] = str(val)
    except Exception:
        pass


def supabase_client_key() -> tuple[str, str]:
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    return url, key


@st.cache_resource
def init_supabase(url: str, key: str) -> Optional[Client]:
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"Supabase 연결 실패: {e}")
        return None


def _sign_up_credentials(email: str, password: str) -> Dict[str, Any]:
    """sign_up 페이로드. 확인 메일의 링크가 돌아올 URL은 대시보드 Redirect URLs에도 등록해야 함."""
    em = email.strip().lower()
    creds: Dict[str, Any] = {"email": em, "password": password}
    redirect = (os.getenv("SUPABASE_EMAIL_REDIRECT_URL") or "").strip()
    if redirect:
        creds["options"] = {"email_redirect_to": redirect}
    return creds


def try_sign_up(client: Client, email: str, password: str) -> Dict[str, Any]:
    """
    Returns dict keys:
      status: "auto_login" | "need_confirm" | "weak_password" | "fail" | "error"
      user_email, user_id, access_token, refresh_token (when auto_login)
      message (human), raw_error (optional)
    """
    em = email.strip().lower()
    creds = _sign_up_credentials(email, password)
    if len(password) < 6:
        return {
            "status": "weak_password",
            "user_email": em,
            "user_id": None,
            "access_token": None,
            "refresh_token": None,
            "message": "비밀번호는 Supabase 기본 정책상 최소 6자 이상이어야 합니다.",
            "raw_error": None,
        }
    try:
        res = client.auth.sign_up(creds)
        if res.user and res.session:
            return {
                "status": "auto_login",
                "user_email": em,
                "user_id": res.user.id,
                "access_token": res.session.access_token,
                "refresh_token": res.session.refresh_token,
                "message": "",
                "raw_error": None,
                # Confirm email 꺼짐 → 확인 메일 자체를 보내지 않음
                "confirm_email_disabled": True,
            }
        if res.user:
            return {
                "status": "need_confirm",
                "user_email": em,
                "user_id": res.user.id,
                "access_token": None,
                "refresh_token": None,
                "message": "가입 요청이 접수되었습니다. 받은편지함·스팸함을 확인한 뒤 메일의 링크를 누르고 로그인하세요.",
                "raw_error": None,
                "confirm_email_disabled": False,
            }
        return {
            "status": "fail",
            "user_email": em,
            "user_id": None,
            "access_token": None,
            "refresh_token": None,
            "message": "회원가입에 실패했습니다.",
            "raw_error": None,
        }
    except Exception as e:
        raw = str(e)
        return {
            "status": "error",
            "user_email": em,
            "user_id": None,
            "access_token": None,
            "refresh_token": None,
            "message": raw,
            "raw_error": raw,
        }
