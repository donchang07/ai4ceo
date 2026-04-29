"""
회원가입 전용 페이지.
실행: 상위 폴더에서 `streamlit run multi-users-ref.py` 후 사이드바에서 본 페이지로 이동.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

from supabase_auth_shared import (  # noqa: E402
    init_supabase,
    streamlit_secrets_into_environ,
    supabase_client_key,
    try_sign_up,
)

st.set_page_config(page_title="회원가입", page_icon="✉️", layout="centered")

streamlit_secrets_into_environ()
_u, _k = supabase_client_key()
supabase = init_supabase(_u, _k)

st.markdown(
    """
<style>
h1 { color: #1f77b4 !important; font-size: 1.5rem !important; }
.stButton > button {
  background-color: #ff69b4 !important; color: white !important; border: none !important;
  border-radius: 5px !important; font-weight: bold !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("회원가입")
st.caption("이 Supabase 프로젝트에 계정을 만듭니다. 확인 메일을 켠 경우 메일함의 링크를 누른 뒤 챗봇에서 로그인하세요.")

with st.expander("확인 메일이 오지 않을 때 (Supabase 설정)", expanded=False):
    st.markdown(
        """
1. **가입 직후 바로 로그인되는 경우**  
   Dashboard → **Authentication → Providers → Email** 에서 **Confirm email** 이 **꺼져** 있으면  
   Supabase는 확인 메일을 **보내지 않고** 세션만 줍니다. 메일을 받으려면 **Confirm email** 을 **켜세요**.

2. **Confirm email 을 켰는데도 메일이 없을 때**  
   - 스팸·프로모션함 확인  
   - [Authentication → Users](https://supabase.com/dashboard) → 해당 사용자 **Last sign in** / 로그  
   - 무료 프로젝트는 **시간당 Auth 이메일 할당량**이 있음 (잠시 후 재시도)  
   - **Authentication → Emails** 에서 커스텀 SMTP 설정(운영 시 권장)

3. **확인 링크 클릭 후 오류**  
   Dashboard → **Authentication → URL Configuration** 에 **Site URL** 과 **Redirect URLs** 에  
   앱 주소(예: `https://xxx.streamlit.app`)와, Secrets에 넣은  
   `SUPABASE_EMAIL_REDIRECT_URL` 과 **동일한** URL이 허용 목록에 있어야 합니다.
        """
    )
    redir = os.getenv("SUPABASE_EMAIL_REDIRECT_URL")
    if redir:
        st.caption(f"현재 `SUPABASE_EMAIL_REDIRECT_URL`: `{redir}`")
    else:
        st.caption(
            "선택: Streamlit Cloud Secrets 또는 `.env`에 `SUPABASE_EMAIL_REDIRECT_URL` "
            "(예: 배포된 앱의 `https://...streamlit.app`)을 넣으면 확인 메일 링크가 그 주소로 돌아옵니다."
        )

if not supabase:
    st.error("Supabase에 연결할 수 없습니다. `SUPABASE_URL` / `SUPABASE_ANON_KEY`(또는 서비스 롤 키)를 설정하세요.")
else:
    reg_email = st.text_input("이메일", key="page_signup_email", placeholder="you@example.com")
    reg_pw = st.text_input("비밀번호 (6자 이상)", type="password", key="page_signup_pw")
    if st.button("회원가입", use_container_width=True):
        if not reg_email or not reg_pw:
            st.warning("이메일과 비밀번호를 입력하세요.")
        else:
            r = try_sign_up(supabase, reg_email, reg_pw)
            if r["status"] == "weak_password":
                st.warning(r["message"])
            elif r["status"] == "auto_login":
                st.session_state.user_email = r["user_email"]
                st.session_state.user_id = r["user_id"]
                st.session_state.sb_access_token = r["access_token"]
                st.session_state.sb_refresh_token = r["refresh_token"]
                st.session_state.sessions_bootstrapped = False
                st.success(
                    "가입되었습니다. 챗봇으로 이동합니다.\n\n"
                    "※ **Confirm email** 이 꺼져 있으면 확인 메일은 **발송되지 않고** 바로 로그인됩니다. "
                    "메일을 받으려면 Supabase → Authentication → Providers → Email 에서 **Confirm email** 을 켜세요."
                )
                st.switch_page("multi-users-ref.py")
            elif r["status"] == "need_confirm":
                st.success(r["message"])
                st.info(
                    "메일이 5~10분 안에도 없으면 위 **「확인 메일이 오지 않을 때」** 절차와 "
                    "Supabase **Logs → Auth** 를 확인하세요."
                )
            elif r["status"] == "fail":
                st.error(r["message"])
            else:
                st.error(f"회원가입 오류: {r['message']}")
                raw = r.get("raw_error") or ""
                if "already been registered" in raw.lower() or "user already registered" in raw.lower():
                    st.info("이미 등록된 이메일입니다. 챗봇에서 로그인을 시도하세요.")

st.markdown("---")
if hasattr(st, "page_link"):
    st.page_link("multi-users-ref.py", label="← PDF 챗봇 로그인 화면으로", icon="🏠")
elif hasattr(st, "switch_page"):
    if st.button("로그인 화면으로 돌아가기", use_container_width=True):
        st.switch_page("multi-users-ref.py")
else:
    st.caption("챗봇 메인: `streamlit run multi-users-ref.py`")
