"""
PDF 기반 멀티유저 멀티세션 RAG 챗봇
- Supabase 인증/세션 저장/로드
- OpenAI/Anthropic/Gemini 키를 사이드바에서 입력
- Streamlit Cloud 호환
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import tempfile
from dotenv import load_dotenv
from supabase import Client, create_client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field, PrivateAttr
import re

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 환경 변수 로드 (Supabase URL/KEY 용)
load_dotenv()


def _load_streamlit_secrets_to_env():
    """Streamlit Cloud secrets를 환경변수로 주입 (배포 시 사용)."""
    if not hasattr(st, "secrets"):
        return
    for key in ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY"]:
        if key in st.secrets and key not in os.environ:
            os.environ[key] = str(st.secrets[key])


_load_streamlit_secrets_to_env()

# 페이지 설정
st.set_page_config(
    page_title="PDF 기반 멀티유저 멀티세션 RAG 챗봇",
    page_icon="📚",
    layout="wide"
)


def sanitize_text(text: Optional[str]) -> str:
    """제어문자를 제거해 DB 저장 시 오류를 최소화."""
    if text is None:
        return ""
    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned


@st.cache_resource
def init_supabase() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        st.session_state.supabase_error = f"URL 또는 KEY가 없습니다. URL: {bool(url)}, KEY: {bool(key)}"
        return None
    try:
        client = create_client(url, key)
        # 연결 성공 시 에러 정보 초기화
        if "supabase_error" in st.session_state:
            del st.session_state.supabase_error
        return client
    except Exception as e:
        # 에러 정보를 session_state에 저장 (디버깅용)
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        st.session_state.supabase_error = error_detail
        st.error(f"Supabase 연결 실패: {e}")
        return None


supabase = init_supabase()


def ensure_api_keys(openai_key: str, anthropic_key: str, gemini_key: str):
    """사이드바 입력값을 환경 변수에 반영."""
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key.strip()
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key.strip()
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key.strip()


def get_supabase_status() -> Dict[str, Any]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    status: Dict[str, Any] = {
        "has_url": bool(url),
        "has_key": bool(key),
        "connected": supabase is not None,
        "auth": None,
        "error": None,
    }
    if supabase:
        try:
            status["auth"] = supabase.auth.get_session()
        except Exception as e:
            status["error"] = str(e)
    else:
        # 연결 실패 시 에러 정보 추가
        if hasattr(st.session_state, "supabase_error"):
            status["error"] = st.session_state.supabase_error
        elif not url or not key:
            status["error"] = "URL 또는 KEY가 설정되지 않았습니다."
    return status


def sign_in(email: str, password: str) -> bool:
    """Supabase 이메일/패스워드 로그인."""
    if not supabase:
        st.error("Supabase 설정을 확인해주세요.")
        return False
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if res and res.session:
            st.session_state.user_email = email
            st.session_state.user_id = res.user.id
            st.session_state.sb_session = res.session
            return True
        st.error("로그인에 실패했습니다.")
        return False
    except Exception as e:
        st.error(f"로그인 오류: {e}")
        return False


def sign_out():
    if supabase:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
    st.session_state.user_email = None
    st.session_state.user_id = None
    st.session_state.sb_session = None


class SessionRetriever(BaseRetriever):
    """세션/사용자 단위 Supabase RPC 기반 검색기."""

    k: int = Field(default=8, description="검색 문서 수")
    _supabase: Client = PrivateAttr()
    _embeddings: OpenAIEmbeddings = PrivateAttr()
    _session_id: Optional[str] = PrivateAttr()
    _user_id: Optional[str] = PrivateAttr()

    def __init__(
        self,
        supabase_client: Client,
        embeddings: OpenAIEmbeddings,
        session_id: Optional[str],
        user_id: Optional[str],
        k: int = 8,
    ):
        super().__init__(k=k)
        self._supabase = supabase_client
        self._embeddings = embeddings
        self._session_id = session_id
        self._user_id = user_id

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            q_emb = self._embeddings.embed_query(query)
            params = {
                "query_embedding": q_emb,
                "match_threshold": 0.7,
                "match_count": self.k * 2,
                "filter_user_id": self._user_id,
            }
            result = self._supabase.rpc("match_documents", params).execute()
            docs: List[Document] = []
            if result.data:
                for item in result.data:
                    meta = item.get("metadata", {}) or {}
                    sid = meta.get("session_id")
                    if self._session_id and sid != self._session_id:
                        continue
                    docs.append(
                        Document(
                            page_content=item.get("content", ""),
                            metadata=meta,
                        )
                    )
                    if len(docs) >= self.k:
                        break
            return docs
        except Exception as e:
            st.warning(f"문서 검색 오류: {e}")
            return []


def create_session() -> Optional[str]:
    """새 세션 생성 (사용자별)."""
    if not supabase or not st.session_state.user_id:
        st.warning("로그인 후 세션을 생성하세요.")
        return None
    sid = str(uuid.uuid4())
    payload = {
        "id": sid,
        "session_id": sid,
        "user_id": st.session_state.user_id,
        "title": "New Chat",
    }
    try:
        res = supabase.table("sessions").insert(payload).execute()
        if res.data:
            return res.data[0].get("id", sid)
    except Exception as e:
        st.error(f"세션 생성 실패: {e}")
    return None


def get_sessions() -> List[Dict[str, Any]]:
    if not supabase or not st.session_state.user_id:
        return []
    try:
        res = (
            supabase.table("sessions")
            .select("id, title, created_at, updated_at, session_id")
            .eq("user_id", st.session_state.user_id)
            .order("updated_at", desc=True)
            .limit(100)
            .execute()
        )
        return res.data or []
    except Exception as e:
        st.error(f"세션 목록 조회 실패: {e}")
        return []


def _generate_title() -> str:
    """간단 제목 생성기 (OpenAI 우선, 없으면 첫 질문 사용)."""
    try:
        user_msg = next(
            (m["content"] for m in st.session_state.chat_history if m["role"] == "user"),
            "",
        )
        ai_msg = next(
            (m["content"] for m in st.session_state.chat_history if m["role"] in ["assistant", "ai"]),
            "",
        )
        if not user_msg:
            return "New Chat"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not ai_msg:
            return user_msg[:30] + ("..." if len(user_msg) > 30 else "")
        llm = ChatOpenAI(model="gpt-5.1", temperature=0.6, openai_api_key=api_key)
        prompt = f"질문: {user_msg}\n답변: {ai_msg}\n15자 이내 한국어 제목:"
        title = llm.invoke(prompt).content.strip().strip('"').strip("'")
        if not title:
            return user_msg[:30]
        return title[:30]
    except Exception:
        return "New Chat"


def save_session(session_id: str) -> bool:
    """세션 및 메시지 저장."""
    if not supabase or not st.session_state.user_id:
        st.warning("로그인 후 저장할 수 있습니다.")
        return False
    try:
        title = _generate_title()
        session_payload = {
            "title": title,
            "user_id": st.session_state.user_id,
            "session_id": session_id,
        }
        existing = (
            supabase.table("sessions")
            .select("id")
            .eq("id", session_id)
            .eq("user_id", st.session_state.user_id)
            .execute()
        )
        if existing.data:
            supabase.table("sessions").update(session_payload).eq("id", session_id).execute()
        else:
            session_payload["id"] = session_id
            supabase.table("sessions").insert(session_payload).execute()

        for msg in st.session_state.chat_history:
            role = "ai" if msg.get("role") == "assistant" else msg.get("role")
            content = sanitize_text(str(msg.get("content", "")))
            if not content.strip():
                continue
            payload = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "user_id": st.session_state.user_id,
            }
            try:
                supabase.table("messages").insert(payload).execute()
            except Exception:
                # messages 테이블에 user_id가 없는 경우 fallback
                payload.pop("user_id", None)
                supabase.table("messages").insert(payload).execute()
        st.success("세션이 저장되었습니다.")
        return True
    except Exception as e:
        st.error(f"세션 저장 실패: {e}")
        return False


def load_session(session_id: str) -> bool:
    """세션 로드."""
    if not supabase or not st.session_state.user_id:
        st.warning("로그인 후 로드할 수 있습니다.")
        return False
    try:
        res = (
            supabase.table("messages")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        data = res.data or []
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        for msg in data:
            role = msg.get("role", "")
            display_role = "assistant" if role == "ai" else role
            content = msg.get("content", "")
            st.session_state.chat_history.append({"role": display_role, "content": content})
            if display_role == "user":
                st.session_state.conversation_memory.append(f"사용자: {content}")
            elif display_role == "assistant":
                st.session_state.conversation_memory.append(f"AI: {content}")

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            st.session_state.retriever = SessionRetriever(
                supabase, embeddings, session_id, st.session_state.user_id, k=8
            )
        else:
            st.session_state.retriever = None
        return True
    except Exception as e:
        st.error(f"세션 로드 실패: {e}")
        return False


def delete_session(session_id: str) -> bool:
    if not supabase or not st.session_state.user_id:
        return False
    try:
        # 관련 문서 삭제
        try:
            docs = supabase.table("documents").select("id, metadata").execute()
            if docs.data:
                for doc in docs.data:
                    meta = doc.get("metadata", {}) or {}
                    if meta.get("session_id") == session_id and meta.get("user_id") == st.session_state.user_id:
                        supabase.table("documents").delete().eq("id", doc["id"]).execute()
        except Exception:
            pass
        supabase.table("sessions").delete().eq("id", session_id).eq("user_id", st.session_state.user_id).execute()
        return True
    except Exception as e:
        st.error(f"세션 삭제 실패: {e}")
        return False


def save_documents_to_supabase(chunks: List[Any], embeddings: OpenAIEmbeddings, session_id: str) -> bool:
    """문서 임베딩을 Supabase documents 테이블에 저장."""
    if not supabase or not st.session_state.user_id:
        st.warning("로그인 후 파일을 처리하세요.")
        return False


# ---- 초기 상태 ----
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-5.1"
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ---- 스타일 ----
st.markdown(
    """
<style>
h1 {font-size: 1.4rem !important; font-weight: 600 !important; color: #ff69b4 !important;}
h2 {font-size: 1.2rem !important; font-weight: 600 !important; color: #ffd700 !important;}
h3 {font-size: 1.1rem !important; font-weight: 600 !important; color: #1f77b4 !important;}
.stChatMessage {font-size: 0.95rem !important; line-height: 1.5 !important;}
.stChatMessage p {font-size: 0.95rem !important; line-height: 1.5 !important; margin: 0.5rem 0 !important;}
.stChatMessage * {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;}
.stButton > button {background-color: #ff69b4 !important; color: white !important; border: none !important; border-radius: 5px !important; padding: 0.5rem 1rem !important; font-weight: bold !important;}
.stButton > button:hover {background-color: #ff1493 !important;}
.stSidebar .stButton > button {font-size: 0.75rem !important; padding: 0.35rem 0.7rem !important;}
</style>
""",
    unsafe_allow_html=True,
)

# ---- 제목 ----
st.markdown(
    """
<div style="text-align: center; margin-top: -3.5rem; margin-bottom: 0.5rem;">
    <h1 style="font-size: 2.4rem; font-weight: bold; margin: 0;">
        <span style="color: #1f77b4;">PDF</span>
        <span style="color: #ffffff; font-size: 0.7em;">기반</span>
        <span style="color: #9b59b6;">멀티유저</span>
        <span style="color: #ffd700;">멀티세션</span>
        <span style="color: #d62728; font-size: 0.7em;">RAG 챗봇</span>
    </h1>
</div>
""",
    unsafe_allow_html=True,
)
st.caption("Supabase 기반 세션 저장 · 로그인, 사이드바에서 키 입력")


def build_llm(model_name: str):
    """모델명에 따라 LLM 인스턴스 생성."""
    if model_name == "gpt-5.1":
        return ChatOpenAI(model="gpt-5.1", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
    if model_name == "claude-4-sonnet-latest":
        return ChatAnthropic(model="claude-4-sonnet-latest", temperature=0.7, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    if model_name == "gemini-1.5-pro-latest":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return ChatOpenAI(model="gpt-5.1", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))


# ---- 사이드바 ----
with st.sidebar:
    st.markdown('<h2 style="color:#1f77b4;">API 키</h2>', unsafe_allow_html=True)
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="sb_openai_key")
    anthropic_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...", key="sb_anthropic_key")
    gemini_key = st.text_input("Google (Gemini) API Key", type="password", placeholder="AIza...", key="sb_gemini_key")
    ensure_api_keys(openai_key, anthropic_key, gemini_key)

    st.markdown('<h2 style="color:#9b59b6;">Supabase 로그인</h2>', unsafe_allow_html=True)
    login_id = st.text_input("Login ID (이메일)", key="sb_login_id")
    login_pw = st.text_input("Password", type="password", key="sb_login_pw")
    col_login, col_logout = st.columns(2)
    with col_login:
        if st.button("로그인", use_container_width=True):
            if login_id and login_pw:
                if sign_in(login_id, login_pw):
                    st.success("로그인되었습니다.")
                    if not st.session_state.current_session_id:
                        st.session_state.current_session_id = create_session()
                    st.rerun()
            else:
                st.warning("이메일과 비밀번호를 입력하세요.")
    with col_logout:
        if st.button("로그아웃", use_container_width=True):
            sign_out()
            st.success("로그아웃 완료")
            st.rerun()

    if st.session_state.user_email:
        st.info(f"현재 사용자: {st.session_state.user_email}")
    else:
        st.warning("로그인 후 세션/저장 기능을 사용할 수 있습니다.")

    st.markdown('<h2 style="color:#1f77b4;">모델 선택</h2>', unsafe_allow_html=True)
    st.session_state.selected_model = st.selectbox(
        "LLM 선택",
        options=["gpt-5.1", "claude-4-sonnet-latest", "gemini-1.5-pro-latest"],
        index=["gpt-5.1", "claude-4-sonnet-latest", "gemini-1.5-pro-latest"].index(
            st.session_state.selected_model if st.session_state.selected_model in ["gpt-5.1", "claude-4-sonnet-latest", "gemini-1.5-pro-latest"] else "gpt-5.1"
        ),
    )

    st.markdown('<h2 style="color:#ffd700;">Supabase 상태</h2>', unsafe_allow_html=True)
    sb_status = get_supabase_status()
    st.write(f"URL: {'✅' if sb_status['has_url'] else '❌'} / KEY: {'✅' if sb_status['has_key'] else '❌'} / 연결: {'✅' if sb_status['connected'] else '❌'}")
    if sb_status.get("error"):
        st.warning(sb_status["error"])
    
    # 디버그 정보 (개발용)
    with st.expander("🔍 디버그 정보 (개발용)", expanded=False):
        st.write("**환경변수 확인:**")
        url_val = os.getenv("SUPABASE_URL")
        key_val = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        st.write(f"- SUPABASE_URL: {'설정됨' if url_val else '❌ 없음'}")
        if url_val:
            st.code(url_val[:50] + "..." if len(url_val) > 50 else url_val, language=None)
        st.write(f"- SUPABASE_KEY: {'설정됨' if key_val else '❌ 없음'}")
        if key_val:
            st.code(key_val[:30] + "..." if len(key_val) > 30 else key_val, language=None)
        
        st.write("**연결 상태:**")
        st.write(f"- supabase 객체: {'✅ 생성됨' if supabase is not None else '❌ None'}")
        
        if hasattr(st.session_state, "supabase_error"):
            st.write("**에러 정보:**")
            st.error(st.session_state.supabase_error)
        
        # Streamlit secrets 확인
        try:
            if hasattr(st, "secrets") and st.secrets:
                st.write("**Streamlit Secrets 확인:**")
                secrets_keys = list(st.secrets.keys())
                st.write(f"- Secrets 키 개수: {len(secrets_keys)}")
                if "SUPABASE_URL" in secrets_keys:
                    st.write("  - SUPABASE_URL: ✅")
                if "SUPABASE_ANON_KEY" in secrets_keys:
                    st.write("  - SUPABASE_ANON_KEY: ✅")
                if "SUPABASE_SERVICE_ROLE_KEY" in secrets_keys:
                    st.write("  - SUPABASE_SERVICE_ROLE_KEY: ✅")
        except Exception:
            st.write("- Streamlit Secrets: 확인 불가")

    st.markdown('<h2 style="color:#1f77b4;">세션 관리</h2>', unsafe_allow_html=True)
    if supabase and st.session_state.user_id:
        sessions = get_sessions()
        options = ["새 세션"] + [s.get("title") or "New Chat" for s in sessions]
        session_map = {s.get("title") or "New Chat": s.get("id") for s in sessions}
        current_idx = 0
        if st.session_state.current_session_id:
            for idx, s in enumerate(sessions, start=1):
                if s.get("id") == st.session_state.current_session_id:
                    current_idx = idx
                    break
        selected_display = st.selectbox("세션 선택", options=options, index=current_idx, key="sb_session_sel")
        selected_id = session_map.get(selected_display) if selected_display != "새 세션" else None

        col_load, col_new = st.columns(2)
        with col_load:
            if st.button("📂 세션 로드", use_container_width=True, disabled=selected_id is None):
                if selected_id:
                    if st.session_state.current_session_id and st.session_state.current_session_id != selected_id:
                        save_session(st.session_state.current_session_id)
                    if load_session(selected_id):
                        st.session_state.current_session_id = selected_id
                        st.success("세션 로드 완료")
                        st.rerun()
        with col_new:
            if st.button("➕ 새 세션", use_container_width=True):
                if st.session_state.current_session_id:
                    save_session(st.session_state.current_session_id)
                new_id = create_session()
                if new_id:
                    st.session_state.current_session_id = new_id
                    st.session_state.chat_history = []
                    st.session_state.conversation_memory = []
                    st.session_state.processed_files = []
                    st.session_state.retriever = None
                    st.success("새 세션 생성")
                    st.rerun()

        col_save, col_del = st.columns(2)
        with col_save:
            if st.button("💾 세션 저장", use_container_width=True):
                if st.session_state.current_session_id:
                    save_session(st.session_state.current_session_id)
        with col_del:
            if st.button("🗑️ 세션 삭제", use_container_width=True, type="secondary", disabled=selected_id is None):
                if selected_id and delete_session(selected_id):
                    st.success("세션 삭제 완료")
                    if selected_id == st.session_state.current_session_id:
                        st.session_state.current_session_id = create_session()
                    st.rerun()

        if st.button("🔄 화면 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.session_state.processed_files = []
            st.session_state.retriever = None
            st.success("화면을 초기화했습니다.")
            st.rerun()

        if st.button("🗂️ vectordb", use_container_width=True):
            sources = set()
            try:
                doc_res = supabase.table("documents").select("metadata").execute()
                if doc_res.data:
                    for d in doc_res.data:
                        meta = d.get("metadata", {}) or {}
                        if meta.get("session_id") == st.session_state.current_session_id and meta.get("user_id") == st.session_state.user_id:
                            src = meta.get("source")
                            if src:
                                sources.add(str(src))
            except Exception:
                pass
            if st.session_state.processed_files:
                sources.update([str(f) for f in st.session_state.processed_files])
            if sources:
                st.info("현재 세션 파일:\n" + "\n".join(sorted(sources)))
            else:
                st.warning("저장된 파일이 없습니다.")
    else:
        st.info("로그인하면 세션 관리가 활성화됩니다.")

    st.markdown("---")
    st.markdown('<h2 style="color:#1f77b4;">PDF 업로드</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PDF를 선택하세요", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if not st.session_state.user_id:
            st.warning("로그인 후 처리할 수 있습니다.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.warning("OpenAI API 키를 입력하세요.")
        else:
            if st.button("파일 처리하기"):
                with st.spinner("PDF 처리 중..."):
                    try:
                        temp_dir = tempfile.TemporaryDirectory()
                        docs = []
                        new_files = []
                        for up in uploaded_files:
                            if up.name in st.session_state.processed_files:
                                continue
                            path = os.path.join(temp_dir.name, up.name)
                            with open(path, "wb") as f:
                                f.write(up.getbuffer())
                            loader = PyPDFLoader(path)
                            loaded = loader.load()
                            for d in loaded:
                                d.metadata["source"] = up.name
                            docs.extend(loaded)
                            new_files.append(up.name)
                        if docs:
                            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                            chunks = splitter.split_documents(docs)
                            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
                            save_documents_to_supabase(chunks, embeddings, st.session_state.current_session_id or create_session())
                            st.session_state.retriever = SessionRetriever(
                                supabase,
                                embeddings,
                                st.session_state.current_session_id,
                                st.session_state.user_id,
                                k=8,
                            )
                            st.session_state.processed_files.extend(new_files)
                            save_session(st.session_state.current_session_id)
                            st.success("파일 처리 및 세션 저장 완료")
                        else:
                            st.info("새롭게 처리할 파일이 없습니다.")
                    except Exception as e:
                        st.error(f"파일 처리 실패: {e}")

    if st.session_state.processed_files:
        st.markdown('<h3 style="color:#ffd700;">처리된 파일</h3>', unsafe_allow_html=True)
        for f in st.session_state.processed_files:
            st.write(f"- {f}")


# ---- 메인 채팅 영역 ----
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("질문을 입력하세요")
if prompt:
    if not st.session_state.user_id:
        st.warning("로그인 후 질문할 수 있습니다.")
        st.stop()
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            st.write("먼저 PDF를 업로드하고 처리해주세요.")
        st.session_state.chat_history.append({"role": "assistant", "content": "먼저 PDF를 업로드하고 처리해주세요."})
    else:
        with st.spinner("답변 생성 중..."):
            try:
                docs = st.session_state.retriever.invoke(prompt)
                top_docs = docs[:3] if docs else []
                context_parts = []
                for idx, doc in enumerate(top_docs):
                    context_parts.append(f"[문서 {idx+1}]\n{doc.page_content}\n")
                context_text = "\n".join(context_parts)
                conv_context = ""
                if st.session_state.conversation_memory:
                    recent = st.session_state.conversation_memory[-40:]
                    conv_context = "\n".join(recent)
                system_prompt = f"""
질문: {prompt}

관련 문서:
{context_text}

이전 대화:
{conv_context}

위 정보를 종합하여 한국어 존댓말로 구조화된 답변을 작성하세요.
- 헤딩(#, ##, ###)을 적절히 사용
- 출처 표기나 (문서1) 형태 참조는 넣지 않음
"""
                llm = build_llm(st.session_state.selected_model)
                answer = llm.invoke(system_prompt).content
                with st.chat_message("assistant"):
                    st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.conversation_memory.append(f"사용자: {prompt}")
                st.session_state.conversation_memory.append(f"AI: {answer}")
                if len(st.session_state.conversation_memory) > 120:
                    st.session_state.conversation_memory = st.session_state.conversation_memory[-120:]
                save_session(st.session_state.current_session_id or create_session())
            except Exception as e:
                with st.chat_message("assistant"):
                    st.write(f"오류가 발생했습니다: {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"오류가 발생했습니다: {e}"})
