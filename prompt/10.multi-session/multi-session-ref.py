"""
PDF 기반 멀티세션 RAG 챗봇
- Supabase를 활용한 세션 저장/로드 기능
- 자동 저장: 파일 처리 및 대화 후 자동 세션 저장
- 세션 제목 자동 생성: LLM 기반 제목 생성
- 멀티세션 관리: 세션 선택, 삭제 기능
- Vector database: Supabase 사용 (이미 embedding된 내용 재사용)
"""

import os
import sys
import streamlit as st
import tempfile
import uuid
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field, PrivateAttr
from typing import Optional, Dict, List, Any
import re

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 환경 변수 로드
load_dotenv()

# 텍스트 정제 (null 문자 등 문제되는 제어문자 제거)
def sanitize_text(text: str) -> str:
    if text is None:
        return ""
    # null 문자 제거
    cleaned = text.replace("\x00", "")
    # 기타 비인쇄 제어문자 제거 (개행/탭 제외)
    cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned

# Supabase 클라이언트 초기화
@st.cache_resource
def init_supabase() -> Client:
    """Supabase 클라이언트 초기화"""
    supabase_url = os.getenv("SUPABASE_URL")
    # 서비스 롤 키가 있으면 우선 사용 (서버 사이드에서만 보관)
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        return None
    
    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Supabase 연결 실패: {str(e)}")
        return None

supabase = init_supabase()


def get_supabase_status() -> Dict[str, Any]:
    """Supabase 연결/환경 상태 점검"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    status: Dict[str, Any] = {
        "has_url": bool(url),
        "has_key": bool(key),
        "connected": supabase is not None,
        "query_ok": False,
        "error": None,
    }
    if supabase:
        try:
            supabase.table("sessions").select("id").limit(1).execute()
            status["query_ok"] = True
        except Exception as e:
            status["error"] = str(e)
    return status


class SessionRetriever(BaseRetriever):
    """세션별 벡터 검색 Retriever"""
    
    k: int = Field(default=10, description="검색할 문서 수")
    
    _supabase: Client = PrivateAttr()
    _embeddings: OpenAIEmbeddings = PrivateAttr()
    _session_id: Optional[str] = PrivateAttr()
    
    def __init__(self, supabase_client: Client, embeddings: OpenAIEmbeddings, session_id: Optional[str] = None, k: int = 10):
        super().__init__(k=k)
        self._supabase = supabase_client
        self._embeddings = embeddings
        self._session_id = session_id
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """쿼리에 대한 유사 문서 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self._embeddings.embed_query(query)
            
            # Supabase RPC 함수 호출 (filter_user_id를 None으로 전달하여 모든 문서 검색)
            rpc_params = {
                'query_embedding': query_embedding,
                'match_threshold': 0.7,
                'match_count': self.k * 2,  # session_id 필터링을 위해 더 많이 가져옴
                'filter_user_id': None  # user_id 필터링 없이 모든 문서 검색
            }
            
            result = self._supabase.rpc('match_documents', rpc_params).execute()
            
            # 결과를 Document 형식으로 변환하고 session_id로 필터링
            documents = []
            if result.data:
                for item in result.data:
                    # metadata에서 session_id 확인
                    metadata = item.get('metadata', {})
                    item_session_id = metadata.get('session_id') if isinstance(metadata, dict) else None
                    
                    # session_id가 일치하거나 None인 경우만 포함 (세션별 필터링)
                    if self._session_id is None or item_session_id == self._session_id:
                        doc = Document(
                            page_content=item.get('content', ''),
                            metadata=metadata if isinstance(metadata, dict) else {}
                        )
                        documents.append(doc)
                    
                    # 필요한 개수만큼 모으면 중단
                    if len(documents) >= self.k:
                        break
            
            return documents
        
        except Exception as e:
            import traceback
            st.error(f"Retriever 오류: {str(e)}")
            st.error(traceback.format_exc())
            return []


# 페이지 설정
st.set_page_config(
    page_title="PDF 기반 멀티세션 RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

# 초기 상태 설정
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-5.1"

if "sessions_loaded" not in st.session_state:
    st.session_state.sessions_loaded = False

# CSS 스타일 (ref.py 참고)
st.markdown("""
<style>
/* 헤딩 스타일 */
h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ff69b4 !important; /* 분홍색 */
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important; /* 노랑색 */
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important; /* 청색 */
}
h4 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
h5 {
    font-size: 1rem !important;
    font-weight: 600 !important;
}
h6 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 채팅 메시지 스타일 */
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

/* 답변 내용 스타일 */
.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

/* 리스트 스타일 */
.stChatMessage ul, .stChatMessage ol {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

.stChatMessage li {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.3rem 0 !important;
}

/* 강조 텍스트 스타일 */
.stChatMessage strong, .stChatMessage b {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 인용문 스타일 */
.stChatMessage blockquote {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    padding-left: 1rem !important;
    border-left: 3px solid #e0e0e0 !important;
}

/* 코드 스타일 */
.stChatMessage code {
    font-size: 0.9rem !important;
    background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
}

/* 전체 텍스트 일관성 */
.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* 버튼 스타일 */
.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
}

.stButton > button:hover {
    background-color: #ff1493 !important;
}

/* 사이드바 버튼 폰트 크기 축소 */
.stSidebar .stButton > button {
    font-size: 0.7rem !important;
    padding: 0.3rem 0.65rem !important;
}
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown("""
<div style="text-align: center; margin-top: -4rem; margin-bottom: 0.5rem;">
    <h1 style="font-size: 2.5rem; font-weight: bold; margin: 0;">
        <span style="color: #1f77b4;">PDF</span> 
        <span style="color: #ffffff; font-size: 0.7em;">기반</span> 
        <span style="color: #9b59b6;">멀티세션</span> 
        <span style="color: #ffd700;">RAG</span> 
        <span style="color: #d62728; font-size: 0.7em;">챗봇</span>
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("PDF 파일을 업로드하고 내용에 관해 질문해보세요!")

# 세션 관리 함수들
def get_sessions() -> List[Dict]:
    """Supabase에서 세션 목록 가져오기"""
    if not supabase:
        return []
    
    try:
        # sessions 테이블에서 세션 목록 조회 (updated_at 내림차순)
        # limit을 늘려서 더 많은 세션을 가져올 수 있도록 함
        result = supabase.table("sessions").select(
            "id, title, created_at, updated_at, session_id"
        ).order("updated_at", desc=True).limit(100).execute()
        
        if result.data:
            # 디버깅: 세션 개수 확인
            st.session_state.debug_session_count = len(result.data)
            return result.data
        else:
            return []
    except Exception as e:
        # 에러 발생 시 상세 정보 표시
        st.error(f"세션 목록 조회 실패: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []

def create_session() -> Optional[str]:
    """새 세션 생성"""
    if not supabase:
        return None
    
    try:
        session_id = str(uuid.uuid4())
        result = supabase.table("sessions").insert({
            "id": session_id,
            "session_id": session_id,  # session_id 컬럼도 설정 (스키마에 따라 필요할 수 있음)
            "title": "New Chat"
        }).execute()
        
        if result.data:
            return result.data[0]["id"]
        return None
    except Exception as e:
        st.error(f"세션 생성 실패: {str(e)}")
        return None

def save_session(session_id: str):
    """현재 세션을 Supabase에 저장"""
    if not supabase:
        st.error("Supabase 연결이 없습니다.")
        return False
    
    try:
        # 세션 데이터 준비
        session_data = {
            "id": session_id,
            "session_id": session_id,  # session_id 컬럼도 설정 (스키마에 따라 필요할 수 있음)
            "title": "New Chat"  # 기본값, 나중에 업데이트됨
        }
        
        # 세션 제목 자동 생성 (첫 질문과 답변이 있을 때)
        if len(st.session_state.chat_history) >= 2:
            # 첫 사용자 메시지와 AI 응답 추출
            user_msg = next((msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"), "")
            # 'assistant' 또는 'ai' 모두 확인
            ai_msg = next((msg["content"] for msg in st.session_state.chat_history if msg["role"] in ["assistant", "ai"]), "")
            
            if user_msg and ai_msg:
                try:
                    # LLM으로 제목 생성
                    title = generate_session_title(user_msg, ai_msg)
                    session_data["title"] = title
                except Exception as e:
                    # 제목 생성 실패해도 계속 진행
                    st.warning(f"제목 생성 실패: {str(e)}")
        
        # 기존 세션이 있는지 확인
        try:
            existing = supabase.table("sessions").select("id").eq("id", session_id).execute()
        except Exception as e:
            st.error(f"세션 조회 실패: {str(e)}")
            return False
        
        if existing.data:
            # 업데이트 (기존 세션 정보는 유지하고 title만 업데이트)
            try:
                # 기존 세션의 정보를 가져와서 유지
                existing_session = supabase.table("sessions").select("*").eq("id", session_id).execute()
                if existing_session.data:
                    # 기존 세션의 모든 필드를 유지하고 title만 업데이트
                    update_data = {
                        "title": session_data.get("title", "New Chat")
                        # id와 session_id는 업데이트하지 않음 (기존 값 유지)
                    }
                    supabase.table("sessions").update(update_data).eq("id", session_id).execute()
            except Exception as e:
                st.error(f"세션 업데이트 실패: {str(e)}")
                return False
        else:
            # 새로 생성
            try:
                supabase.table("sessions").insert(session_data).execute()
            except Exception as e:
                st.error(f"세션 생성 실패: {str(e)}")
                return False
        
        # 메시지 저장 (role 변환: 'assistant' -> 'ai')
        # 기존 메시지 목록 가져오기 (중복 체크용)
        existing_messages = []
        try:
            messages_result = supabase.table("messages").select("id, role, content").eq("session_id", session_id).execute()
            if messages_result.data:
                # 기존 메시지를 (role, content) 튜플로 저장하여 빠른 중복 체크
                existing_messages = [(msg.get("role"), msg.get("content", "")[:1000]) for msg in messages_result.data]
        except Exception as e:
            st.warning(f"기존 메시지 조회 실패 (계속 진행): {str(e)}")
        
        # 메시지 저장 (중복 체크 개선 - 기존 메시지는 유지하고 새 메시지만 추가)
        saved_count = 0
        skipped_count = 0
        for msg in st.session_state.chat_history:
            try:
                role = msg.get("role")
                if not role:
                    continue
                
                # role 변환: 'assistant' -> 'ai'
                if role == "assistant":
                    role = "ai"
                
                # content 검증 및 처리
                msg_content = msg.get("content", "")
                if msg_content is None:
                    msg_content = ""
                
                # 문자열로 변환
                msg_content = str(msg_content)
                
                # 크기 제한 (PostgreSQL TEXT는 최대 1GB이지만, Supabase API는 더 작은 제한이 있을 수 있음)
                # 1MB로 제한 (약 1,000,000 문자)
                MAX_CONTENT_LENGTH = 1000000
                if len(msg_content) > MAX_CONTENT_LENGTH:
                    msg_content = msg_content[:MAX_CONTENT_LENGTH] + "\n\n[메시지가 너무 길어 일부가 잘렸습니다...]"
                    st.warning(f"메시지가 너무 길어 일부가 잘렸습니다. (원본 길이: {len(msg_content)} 문자)")
                
                # 빈 메시지는 건너뛰기
                if not msg_content.strip():
                    continue
                
                # 중복 체크: 기존 메시지 목록에서 빠르게 확인
                # content의 처음 1000자만 비교 (성능 최적화)
                msg_content_preview = msg_content[:1000]
                is_duplicate = False
                
                # 기존 메시지 목록에서 확인
                for existing_role, existing_content_preview in existing_messages:
                    if existing_role == role and existing_content_preview == msg_content_preview:
                        # 전체 content도 확인 (정확한 중복 체크)
                        if len(msg_content) < 10000:  # 10KB 미만일 때만 전체 비교
                            try:
                                existing_msg = supabase.table("messages").select("id").eq("session_id", session_id).eq("role", role).eq("content", msg_content).limit(1).execute()
                                if existing_msg.data:
                                    is_duplicate = True
                                    skipped_count += 1
                                    break
                            except Exception:
                                pass
                        else:
                            # 긴 메시지는 preview만으로 판단
                            is_duplicate = True
                            skipped_count += 1
                            break
                
                if is_duplicate:
                    continue  # 이미 존재하면 건너뛰기
                
                # session_id가 유효한 UUID인지 확인
                try:
                    uuid.UUID(session_id)
                except (ValueError, AttributeError):
                    st.error(f"유효하지 않은 session_id 형식: {session_id}")
                    continue
                
                # 메시지 저장 (안전한 데이터 타입 보장)
                try:
                    result = supabase.table("messages").insert({
                        "session_id": str(session_id),  # 명시적으로 문자열로 변환
                        "role": str(role),  # 명시적으로 문자열로 변환
                        "content": str(msg_content)  # 명시적으로 문자열로 변환
                    }).execute()
                    
                    if result.data:
                        saved_count += 1
                        # 새로 저장된 메시지를 기존 메시지 목록에 추가 (중복 방지)
                        existing_messages.append((role, msg_content_preview))
                    else:
                        st.warning(f"메시지 저장 결과가 비어있습니다. (role: {role})")
                except Exception as insert_error:
                    # 더 상세한 에러 정보
                    error_detail = str(insert_error)
                    if hasattr(insert_error, 'message'):
                        error_detail = insert_error.message
                    elif isinstance(insert_error, dict):
                        error_detail = str(insert_error)
                    st.warning(f"메시지 저장 실패 (role: {role}, 길이: {len(msg_content)}): {error_detail}")
                    raise  # 상위 except로 전달
                
            except Exception as e:
                # 개별 메시지 저장 실패해도 계속 진행
                error_msg = str(e)
                # 에러 메시지가 너무 길면 축약
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "..."
                st.warning(f"메시지 저장 실패 (role: {role if 'role' in locals() else 'unknown'}): {error_msg}")
                continue
        
        # 저장 결과 요약 (디버깅용)
        if saved_count > 0:
            st.info(f"✅ {saved_count}개 메시지가 저장되었습니다. (건너뛴 메시지: {skipped_count}개)")
        
        return True
    except Exception as e:
        import traceback
        st.error(f"세션 저장 중 오류 발생: {str(e)}")
        st.error(f"상세 오류: {traceback.format_exc()}")
        return False

def load_session(session_id: str) -> bool:
    """Supabase에서 세션 로드"""
    if not supabase:
        st.error("Supabase 연결이 없습니다.")
        return False
    
    try:
        # 세션 메시지 가져오기
        # Supabase Python 클라이언트는 asc 파라미터를 지원하지 않으므로
        # order()를 제거하고 Python에서 정렬
        result = supabase.table("messages").select(
            "id, role, content, created_at"
        ).eq("session_id", session_id).execute()
        
        # Python에서 created_at 기준으로 오름차순 정렬
        if result.data:
            from datetime import datetime
            result.data.sort(key=lambda x: x.get("created_at", ""))
        
        if result.data and len(result.data) > 0:
            # 상태 초기화
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            # processed_files는 유지 (세션별로 다를 수 있음)
            
            # 메시지 복원 (role 변환: 'ai' -> 'assistant')
            loaded_count = 0
            for msg in result.data:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if not role or not content:
                    continue
                
                # role 변환: 'ai' -> 'assistant' (Streamlit chat_message는 'assistant' 사용)
                display_role = "assistant" if role == "ai" else role
                
                st.session_state.chat_history.append({
                    "role": display_role,
                    "content": content
                })
                
                # conversation_memory 복원
                if role == "user":
                    st.session_state.conversation_memory.append(f"사용자: {content}")
                elif role == "ai":
                    st.session_state.conversation_memory.append(f"AI: {content}")
                
                loaded_count += 1
            
            # 세션의 문서가 있는지 확인하고 retriever 복원
            try:
                # 해당 세션의 문서가 있는지 확인 (metadata에 session_id가 있는 문서)
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                    st.session_state.retriever = SessionRetriever(
                        supabase,
                        embeddings,
                        session_id,
                        k=10
                    )
                else:
                    st.session_state.retriever = None
            except Exception as e:
                # retriever 복원 실패해도 세션 로드는 성공
                st.session_state.retriever = None
                st.warning(f"Retriever 복원 실패: {str(e)}")
            
            if loaded_count > 0:
                return True
            else:
                st.warning("세션에 메시지가 없습니다.")
                return False
        else:
            # 메시지가 없는 경우도 세션 로드는 성공 (빈 세션)
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.session_state.retriever = None
            return True
    except Exception as e:
        st.error(f"세션 로드 실패: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def delete_session(session_id: str) -> bool:
    """세션 삭제 (세션, 메시지, 문서 모두 삭제)"""
    if not supabase:
        return False
    
    try:
        # 1. 해당 세션의 문서 삭제 (metadata에 session_id가 있는 문서들)
        try:
            # Supabase PostgREST는 JSONB 필드 쿼리를 지원하지만, 
            # 직접 쿼리가 안 될 수 있으므로 모든 문서를 가져와서 필터링
            all_docs = supabase.table("documents").select("id, metadata").execute()
            
            if all_docs.data:
                deleted_count = 0
                for doc in all_docs.data:
                    metadata = doc.get("metadata", {})
                    # metadata가 dict이고 session_id가 일치하는 경우 삭제
                    if isinstance(metadata, dict) and metadata.get("session_id") == session_id:
                        try:
                            supabase.table("documents").delete().eq("id", doc["id"]).execute()
                            deleted_count += 1
                        except Exception as e:
                            # 개별 문서 삭제 실패해도 계속 진행
                            continue
                
                if deleted_count > 0:
                    st.info(f"세션 관련 문서 {deleted_count}개가 삭제되었습니다.")
        except Exception as e:
            # 문서 삭제 실패해도 계속 진행 (세션 삭제는 계속)
            st.warning(f"문서 삭제 중 일부 오류 발생: {str(e)}")
        
        # 2. 세션 삭제 (CASCADE로 메시지도 자동 삭제됨)
        supabase.table("sessions").delete().eq("id", session_id).execute()
        
        return True
    except Exception as e:
        import traceback
        st.error(f"세션 삭제 중 오류 발생: {str(e)}")
        st.error(traceback.format_exc())
        return False

def generate_session_title(user_question: str, ai_response: str) -> str:
    """세션 제목 자동 생성 (LLM 기반)"""
    try:
        # OpenAI API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # API 키가 없으면 첫 질문의 일부를 제목으로 사용
            return user_question[:30] + "..." if len(user_question) > 30 else user_question
        
        # LLM으로 제목 생성
        llm = ChatOpenAI(model="gpt-5.1", temperature=0.7, openai_api_key=api_key)
        
        prompt = f"""다음 질문과 답변을 기반으로 핵심 키워드 2-3개를 추출하여 간결한 세션 제목을 생성해주세요.

사용자 질문: {user_question[:200]}
AI 답변: {ai_response[:300]}

요구사항:
- 핵심 키워드 2-3개를 조합하여 제목 생성
- 15-20자 이내의 간결한 제목
- 한글로 작성
- 따옴표나 특수문자 없이 작성
- 키워드만 조합한 형태로 작성 (예: "인공지능 활용 방안", "마케팅 전략 수립")
- 제목만 출력하세요 (설명 없이)

제목:"""
        
        title = llm.invoke(prompt).content.strip()
        title = title.strip('"').strip("'").strip()
        
        # 제목 길이 제한
        if len(title) > 30:
            title = title[:27] + "..."
        
        return title if title else "New Chat"
    except Exception as e:
        # 오류 발생 시 첫 질문의 일부를 제목으로 사용
        return user_question[:30] + "..." if len(user_question) > 30 else user_question

def generate_followup_questions(user_question: str, ai_response: str, context_text: str) -> List[str]:
    """향후 더 필요한 질문 3개 생성"""
    try:
        # 선택된 모델에 따라 LLM 생성
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return []
        
        model_name = st.session_state.selected_model
        
        if model_name == "gpt-5.1":
            llm = ChatOpenAI(model="gpt-5.1", temperature=1, openai_api_key=api_key)
        elif model_name == "claude-sonnet-4-5":
            claude_key = os.getenv("ANTHROPIC_API_KEY")
            if not claude_key:
                return []
            llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=1, anthropic_api_key=claude_key)
        elif model_name == "gemini-3-pro-preview":
            gemini_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                return []
            llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=1, google_api_key=gemini_key)
        else:
            llm = ChatOpenAI(model="gpt-5.1", temperature=1, openai_api_key=api_key)
        
        prompt = f"""다음 질문과 답변을 기반으로, 사용자가 더 깊이 있게 알아볼 수 있는 관련 질문 3개를 생성해주세요.

원래 질문: {user_question}

답변 내용:
{ai_response[:1000]}

관련 문서 컨텍스트:
{context_text[:500]}

요구사항:
- 답변 내용과 관련 문서를 바탕으로 더 깊이 있는 질문 생성
- 각 질문은 한 문장으로 작성
- 질문은 구체적이고 실용적이어야 함
- 질문만 출력 (번호나 설명 없이)
- 각 질문은 줄바꿈으로 구분

예시 형식:
질문 1
질문 2
질문 3

관련 질문:"""
        
        questions_text = llm.invoke(prompt).content.strip()
        
        # 질문들을 리스트로 분리
        questions = []
        for line in questions_text.split('\n'):
            line = line.strip()
            if line:
                # 번호나 불필요한 접두사 제거
                for prefix in ['1.', '2.', '3.', '질문 1:', '질문 2:', '질문 3:', '-', '•']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if line and len(line) > 5:
                    questions.append(line)
        
        # 최대 3개만 반환
        return questions[:3]
    except Exception as e:
        return []

def save_documents_to_supabase(chunks: List[Any], embeddings: OpenAIEmbeddings, session_id: str):
    """문서를 Supabase에 저장 (이미 embedding된 내용 재사용)"""
    if not supabase:
        return False
    
    try:
        # documents 테이블에 저장
        # 이미 존재하는 문서는 재사용 (content 해시로 확인)
        batch_size = 50
        saved_count = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # 배치 임베딩 생성
            texts = []
            cleaned_chunks = []
            for chunk in batch_chunks:
                clean_text = sanitize_text(chunk.page_content)
                if not clean_text.strip():
                    continue
                # 메타데이터도 안전하게 복사/정제
                meta = (chunk.metadata or {}).copy()
                # source 등에 null이 들어있을 수 있으므로 문자열화 후 정제
                for k, v in list(meta.items()):
                    if isinstance(v, str):
                        meta[k] = sanitize_text(v)
                chunk.page_content = clean_text
                chunk.metadata = meta
                texts.append(clean_text)
                cleaned_chunks.append(chunk)
            
            if not texts:
                continue
            
            batch_embeddings = embeddings.embed_documents(texts)
            
            # Supabase에 저장할 데이터 준비
            documents_to_save = []
            for chunk, embedding in zip(cleaned_chunks, batch_embeddings):
                # metadata에 session_id 추가하여 세션별로 문서 분리
                metadata = chunk.metadata.copy() if chunk.metadata else {}
                metadata["session_id"] = session_id
                
                # documents 테이블에 저장 (user_id는 None으로 설정)
                documents_to_save.append({
                    "content": chunk.page_content,
                    "metadata": metadata,
                    "embedding": embedding,
                    # user_id가 NOT NULL이면 아래에서 재시도 시 덮어씀
                    "user_id": None
                })
            
            # 배치로 저장
            if documents_to_save:
                try:
                    result = supabase.table("documents").insert(documents_to_save).execute()
                    if result.data:
                        saved_count += len(result.data)
                except Exception as e:
                    # user_id가 NULL일 수 없는 경우를 대비해 재시도
                    error_msg = str(e).lower()
                    if "row-level security" in error_msg or "rls" in error_msg:
                        st.warning("Supabase RLS 정책으로 documents 삽입이 거부되었습니다. 서비스 롤 키 사용 또는 정책을 확인하세요.")
                    if "null" in error_msg and "user_id" in error_msg:
                        # user_id가 NOT NULL이라면 session_id나 'anon'으로 채워 재시도
                        documents_to_save_no_user = []
                        for doc in documents_to_save:
                            doc_copy = doc.copy()
                            doc_copy["user_id"] = str(session_id) if session_id else "anon"
                            documents_to_save_no_user.append(doc_copy)
                        
                        try:
                            result = supabase.table("documents").insert(documents_to_save_no_user).execute()
                            if result.data:
                                saved_count += len(result.data)
                        except Exception as e2:
                            st.warning(f"문서 저장 중 오류 발생: {str(e2)}")
                    else:
                        st.warning(f"문서 저장 중 일부 오류 발생: {str(e)}")
        
        return saved_count > 0
    except Exception as e:
        st.error(f"문서 저장 중 오류 발생: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

# 사이드바 설정
with st.sidebar:
    st.markdown('<h2 style="color: #1f77b4;">LLM 모델 선택</h2>', unsafe_allow_html=True)
    
    # 모델 선택 (정확한 모델명 사용)
    selected_model = st.selectbox(
        "모델 선택",
        options=["gpt-5.1", "claude-sonnet-4-5", "gemini-3-pro-preview"],
        index=["gpt-5.1", "claude-sonnet-4-5", "gemini-3-pro-preview"].index(st.session_state.selected_model) if st.session_state.selected_model in ["gpt-5.1", "claude-sonnet-4-5", "gemini-3-pro-preview"] else 0,
        key="model_selectbox"
    )
    st.session_state.selected_model = selected_model
    
    st.markdown("---")

    # Supabase 상태 표시
    with st.expander("Supabase 상태", expanded=False):
        sb_status = get_supabase_status()
        st.write(f"URL 설정: {'✅' if sb_status['has_url'] else '❌'}")
        st.write(f"키 설정: {'✅' if sb_status['has_key'] else '❌'}")
        st.write(f"클라이언트 생성: {'✅' if sb_status['connected'] else '❌'}")
        st.write(f"쿼리 테스트: {'✅' if sb_status['query_ok'] else '❌'}")
        if sb_status.get("error"):
            st.warning(f"오류: {sb_status['error']}")
    
    # 세션 관리 섹션
    st.markdown('<h2 style="color: #1f77b4;">세션 관리</h2>', unsafe_allow_html=True)
    
    if supabase:
        # 세션 목록 가져오기
        sessions = get_sessions()
        
        # 세션 옵션 생성 (제목 + 날짜로 고유성 보장)
        session_options = ["새 세션"]
        session_map = {}
        for s in sessions:
            title = s.get("title", "New Chat")
            session_id = s.get("id")
            
            # 날짜 포맷팅 (한국 시간)
            created_at = s.get("created_at", "")
            if created_at:
                try:
                    from datetime import datetime
                    # ISO 형식 문자열을 datetime으로 변환
                    if isinstance(created_at, str):
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        dt = created_at
                    # 한국 시간으로 변환 (UTC+9)
                    from datetime import timezone, timedelta
                    kst = timezone(timedelta(hours=9))
                    dt_kst = dt.astimezone(kst)
                    date_str = dt_kst.strftime("%m/%d %H:%M")
                    display_title = f"{title} ({date_str})"
                except Exception:
                    display_title = title
            else:
                display_title = title
            
            # 같은 제목이 여러 개 있으면 번호 추가하여 고유성 보장
            if display_title in session_map:
                # 이미 같은 제목이 있으면 번호 추가
                counter = 1
                while f"{display_title} #{counter}" in session_map:
                    counter += 1
                display_title = f"{display_title} #{counter}"
            
            # 고유한 키 생성
            unique_key = display_title
            session_options.append(unique_key)
            session_map[unique_key] = session_id
        
        # 세션 선택 (풀다운 메뉴)
        # 이전 선택값 추적
        if "previous_selected_session" not in st.session_state:
            st.session_state.previous_selected_session = "새 세션"
        
        # 현재 세션에 맞는 인덱스 찾기
        current_index = 0
        if st.session_state.current_session_id:
            for idx, s in enumerate(sessions):
                if s["id"] == st.session_state.current_session_id:
                    current_index = idx + 1  # "새 세션"이 0번이므로 +1
                    break
        
        selected_session_display = st.selectbox(
            "세션 선택",
            options=session_options,
            index=current_index,
            key="session_selectbox"
        )
        
        # 선택된 세션 ID 저장 (버튼 클릭 시 사용)
        if selected_session_display != "새 세션":
            selected_session_id = session_map.get(selected_session_display)
        else:
            selected_session_id = None

        # 선택 변경 시 자동 로드 (현재 세션 자동 저장 포함)
        if selected_session_id and selected_session_id != st.session_state.current_session_id:
            # 현재 세션 자동 저장 (가능하면)
            if st.session_state.current_session_id:
                save_session(st.session_state.current_session_id)
            with st.spinner(f"세션 '{selected_session_display}' 자동 로드 중..."):
                if load_session(selected_session_id):
                    st.session_state.current_session_id = selected_session_id
                    st.session_state.previous_selected_session = selected_session_display
                    st.success(f"✅ 세션 '{selected_session_display}'이(가) 로드되었습니다.")
                    st.rerun()
                else:
                    st.error(f"❌ 세션 '{selected_session_display}' 로드에 실패했습니다.")
        
        # 세션 로드 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 세션 로드", use_container_width=True, disabled=(selected_session_display == "새 세션" or selected_session_id is None)):
                if selected_session_id:
                    # 현재 세션 저장 (자동 저장)
                    if st.session_state.current_session_id and st.session_state.current_session_id != selected_session_id:
                        # 다른 세션으로 전환할 때만 현재 세션 저장
                        save_session(st.session_state.current_session_id)
                    
                    # 선택된 세션 로드
                    with st.spinner(f"세션 '{selected_session_display}'을 로드하는 중..."):
                        if load_session(selected_session_id):
                            st.session_state.current_session_id = selected_session_id
                            message_count = len(st.session_state.chat_history)
                            if message_count > 0:
                                st.success(f"✅ 세션 '{selected_session_display}'이 로드되었습니다! (메시지 {message_count}개)")
                            else:
                                st.info(f"세션 '{selected_session_display}'이 로드되었습니다. (빈 세션)")
                            st.rerun()
                        else:
                            st.error(f"❌ 세션 '{selected_session_display}' 로드에 실패했습니다.")
                else:
                    st.error(f"세션 ID를 찾을 수 없습니다: {selected_session_display}")
        
        with col2:
            if st.button("➕ 새 세션", use_container_width=True):
                # 현재 세션 저장
                if st.session_state.current_session_id:
                    save_session(st.session_state.current_session_id)
                
                # 새 세션 생성
                new_session_id = create_session()
                if new_session_id:
                    st.session_state.current_session_id = new_session_id
                    st.session_state.chat_history = []
                    st.session_state.conversation_memory = []
                    st.session_state.processed_files = []
                    st.session_state.retriever = None
                    st.success("✅ 새 세션이 생성되었습니다!")
                    st.rerun()
                else:
                    st.error("❌ 새 세션 생성에 실패했습니다.")
        
        # 세션 관리 버튼들 (한 줄에 2개 배치)
        col_save, col_delete = st.columns(2)
        
        with col_save:
            # 1. 세션 저장 버튼
            if st.button("💾 세션 저장", use_container_width=True):
                if st.session_state.current_session_id:
                    # 세션 저장 (save_session 함수 내에서 제목 생성 및 저장 처리)
                    # 저장만 하고 rerun하지 않음 (이전 세션 목록 유지)
                    if save_session(st.session_state.current_session_id):
                        st.success("✅ 세션이 저장되었습니다!")
                        # rerun하지 않고 성공 메시지만 표시 (세션 목록은 유지됨)
                    else:
                        # save_session 함수 내에서 이미 에러 메시지가 표시됨
                        pass
                else:
                    st.warning("저장할 세션이 없습니다.")
        
        with col_delete:
            # 2. 세션 삭제 버튼
            if selected_session_display != "새 세션" and sessions:
                if st.button("🗑️ 세션 삭제", use_container_width=True, type="secondary"):
                    selected_session_id = session_map.get(selected_session_display)
                    if selected_session_id:
                        session_title = selected_session_display
                        # 확인 메시지
                        with st.spinner("세션을 삭제하는 중..."):
                            if delete_session(selected_session_id):
                                st.success(f"✅ 세션 '{session_title}'이 완전히 삭제되었습니다!")
                                # 현재 세션이 삭제된 세션이면 새 세션 생성
                                if selected_session_id == st.session_state.current_session_id:
                                    new_session_id = create_session()
                                    if new_session_id:
                                        st.session_state.current_session_id = new_session_id
                                        st.session_state.chat_history = []
                                        st.session_state.conversation_memory = []
                                        st.session_state.processed_files = []
                                        st.session_state.retriever = None
                                        st.session_state.previous_selected_session = "새 세션"
                                st.rerun()
                            else:
                                st.error("❌ 세션 삭제에 실패했습니다.")
            else:
                # 세션이 선택되지 않았을 때는 비활성화된 버튼 표시
                st.button("🗑️ 세션 삭제", use_container_width=True, disabled=True, type="secondary")
        
        # 3. 화면 초기화 버튼
        if st.button("🔄 화면 초기화", use_container_width=True):
            # 화면만 clear (Supabase 저장 안 함)
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.session_state.processed_files = []
            st.session_state.retriever = None
            st.success("✅ 화면이 초기화되었습니다!")
            st.rerun()

    # 벡터 DB 파일 목록 보기
    if st.button("🗂️ vectordb", use_container_width=True):
        sources = set()
        if supabase and st.session_state.current_session_id:
            try:
                doc_res = supabase.table("documents").select("metadata").contains(
                    "metadata", {"session_id": st.session_state.current_session_id}
                ).execute()
                if doc_res.data:
                    for d in doc_res.data:
                        meta = d.get("metadata", {}) or {}
                        src = meta.get("source")
                        if src:
                            sources.add(str(src))
            except Exception as e:
                st.error(f"벡터 DB 조회 실패: {str(e)}")
        # 로컬 백업용 processed_files도 병합
        if st.session_state.processed_files:
            for f in st.session_state.processed_files:
                sources.add(str(f))
        if sources:
            st.info("현재 세션에 저장된 파일명:\n" + "\n".join(sorted(sources)))
        else:
            st.warning("현재 세션에서 확인된 파일명이 없습니다.")
    else:
        st.warning("Supabase가 연결되지 않았습니다. 세션 저장 기능을 사용할 수 없습니다.")
    
    st.markdown("---")
    
    # PDF 파일 업로드
    st.markdown('<h2 style="color: #1f77b4;">PDF 파일 업로드</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        process_button = st.button("파일 처리하기")
        
        if process_button:
            with st.spinner("PDF 파일을 처리 중입니다..."):
                try:
                    # 임시 파일 생성 및 처리
                    temp_dir = tempfile.TemporaryDirectory()
                    
                    all_docs = []
                    new_files = []
                    
                    # 각 파일 처리
                    for uploaded_file in uploaded_files:
                        # 이미 처리된 파일 스킵
                        if uploaded_file.name in st.session_state.processed_files:
                            continue
                        
                        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                        
                        # 업로드된 파일을 임시 파일로 저장
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # PDF 로더 생성 및 문서 로드
                        loader = PyPDFLoader(temp_file_path)
                        documents = loader.load()
                        
                        # 메타데이터에 파일 이름 추가
                        for doc in documents:
                            doc.metadata["source"] = uploaded_file.name
                        
                        all_docs.extend(documents)
                        new_files.append(uploaded_file.name)
                    
                    if not all_docs:
                        st.success("모든 파일이 이미 처리되었습니다.")
                    else:
                        # 텍스트 분할
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=100,
                            length_function=len
                        )
                        chunks = text_splitter.split_documents(all_docs)
                        
                        # 모든 청크를 벡터 데이터베이스에 저장
                        total_chunks = len(chunks)
                        st.info(f"총 {total_chunks}개의 청크를 처리합니다.")
                        
                        # 임베딩 생성 (OpenAI 사용)
                        api_key = os.getenv("OPENAI_API_KEY")
                        if not api_key:
                            st.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
                        else:
                            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                            
                            # Retriever 준비 여부 플래그
                            retriever_ready = False
                            retriever_backend = None
                            
                            # Supabase에 문서 저장 (이미 embedding된 내용 재사용)
                            if supabase:
                                save_ok = save_documents_to_supabase(chunks, embeddings, st.session_state.current_session_id)
                                if save_ok:
                                    st.success(f"✅ {total_chunks}개 청크가 저장되었습니다! (Supabase)")
                                    try:
                                        st.session_state.retriever = SessionRetriever(
                                            supabase,
                                            embeddings,
                                            st.session_state.current_session_id,
                                            k=10
                                        )
                                        retriever_ready = True
                                        retriever_backend = "supabase"
                                    except Exception as e:
                                        st.warning(f"Supabase 기반 검색기 초기화 실패: {str(e)}")
                                else:
                                    st.warning("⚠️ Supabase에 문서 저장이 완료되지 않았습니다. 로컬 임시 벡터스토어로 대체합니다.")
                            
                            # Supabase가 없거나 실패한 경우 로컬 FAISS 백업
                            if not retriever_ready:
                                try:
                                    vectorstore = FAISS.from_documents(chunks, embeddings)
                                    st.session_state.vectorstore = vectorstore
                                    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
                                    retriever_ready = True
                                    retriever_backend = "local_faiss"
                                    st.success("✅ 로컬 벡터스토어로 검색이 준비되었습니다.")
                                except Exception as e:
                                    st.error(f"로컬 벡터스토어 생성 실패: {str(e)}")
                            
                            # 검색 준비 상태 안내
                            if retriever_ready:
                                st.info(f"검색 백엔드: {retriever_backend}")
                            else:
                                st.error("검색기가 준비되지 않았습니다. 로그를 확인하세요.")
                            
                            # 처리된 파일 목록 업데이트
                            st.session_state.processed_files.extend(new_files)
                            
                            # 파일 처리 후 자동 세션 저장
                            save_session(st.session_state.current_session_id)
                            st.success("파일이 처리되었고 세션이 자동 저장되었습니다!")
                
                except Exception as e:
                    st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                    st.error("파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.")
    
    # 처리된 파일 목록 표시
    if st.session_state.processed_files:
        st.markdown('<h3 style="color: #ffd700;">처리된 파일 목록</h3>', unsafe_allow_html=True)
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
    
    
    # 메모리 사용량 표시
    if st.session_state.processed_files:
        st.subheader("📊 시스템 상태")
        st.info(f"처리된 파일 수: {len(st.session_state.processed_files)}")
        st.info(f"대화 기록 수: {len(st.session_state.chat_history)}")

# 앱 시작 시 세션 자동 로드 (Streamlit 재시작 후에도 이전 대화 복원)
if supabase and not st.session_state.sessions_loaded:
    # 가장 최근 세션 로드
    sessions = get_sessions()
    if sessions:
        latest_session_id = sessions[0]["id"]
        if load_session(latest_session_id):
            st.session_state.current_session_id = latest_session_id
            st.session_state.sessions_loaded = True

# 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 영역
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            st.write("먼저 PDF 파일을 업로드하고 처리해주세요.")
        st.session_state.chat_history.append({"role": "assistant", "content": "먼저 PDF 파일을 업로드하고 처리해주세요."})
    else:
        try:
            # RAG 검색 (상위 3개 문서만 사용)
            retrieved_docs = st.session_state.retriever.invoke(prompt)
            
            if not retrieved_docs:
                response = f"죄송합니다. '{prompt}'에 대한 관련 문서를 찾을 수 없습니다."
                with st.chat_message("assistant"):
                    st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                # 상위 3개 문서만 사용
                top_docs = retrieved_docs[:3]
                
                # 컨텍스트 구성
                context_text = ""
                max_context_length = 8000
                current_length = 0
                
                for i, doc in enumerate(top_docs):
                    doc_text = f"[문서 {i+1}]\n{doc.page_content}\n\n"
                    if current_length + len(doc_text) > max_context_length:
                        st.warning(f"토큰 제한으로 인해 문서 {i+1}개만 사용합니다.")
                        break
                    context_text += doc_text
                    current_length += len(doc_text)
                
                # 과거 대화 맥락 구성
                conversation_context = ""
                if st.session_state.conversation_memory:
                    conversation_context = "\n\n=== 이전 대화 맥락 ===\n"
                    # 최근 50개 대화 사용
                    recent_conversations = st.session_state.conversation_memory[-50:]
                    for conv in recent_conversations:
                        conversation_context += f"{conv}\n"
                    conversation_context += "=== 대화 맥락 끝 ===\n"
                
                # 시스템 프롬프트 구성
                system_prompt = f"""
질문: {prompt}

관련 문서:
{context_text}{conversation_context}

위 문서 내용과 이전 대화 맥락을 모두 고려하여 질문에 답변해주세요.
이전 대화에서 언급된 내용이 있다면 그것을 참조하여 더 정확하고 맥락적인 답변을 제공하세요.

답변 형식:
- 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
- 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
- 답변이 길거나 복잡한 경우 여러 헤딩을 사용하여 가독성을 높이세요
- 답변은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 (문서1), (문서2) 같은 참조 표시를 하지 마세요
- "참조 문서:", "제공된 문서", "문서 1, 문서 2" 같은 문구를 사용하지 마세요
- 답변은 순수한 내용만 포함하고, 참조 관련 문구는 전혀 포함하지 마세요
- 답변 끝에 참조 정보나 출처 관련 문구를 추가하지 마세요
"""
                
                # 선택된 LLM 모델로 답변 생성 (스트리밍 모드)
                model_name = st.session_state.selected_model
                api_key = os.getenv("OPENAI_API_KEY")
                
                if model_name == "gpt-5.1":
                    llm = ChatOpenAI(model="gpt-5.1", temperature=1, openai_api_key=api_key, streaming=True)
                elif model_name == "claude-sonnet-4-5":
                    claude_key = os.getenv("ANTHROPIC_API_KEY")
                    if not claude_key:
                        st.error("ANTHROPIC_API_KEY가 .env 파일에 설정되지 않았습니다.")
                        st.stop()
                    llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=1, anthropic_api_key=claude_key, streaming=True)
                elif model_name == "gemini-3-pro-preview":
                    gemini_key = os.getenv("GOOGLE_API_KEY")
                    if not gemini_key:
                        st.error("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")
                        st.stop()
                    llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=1, google_api_key=gemini_key, streaming=True)
                else:
                    llm = ChatOpenAI(model="gpt-5.1", temperature=1, openai_api_key=api_key, streaming=True)
                
                # 스트리밍으로 답변 생성 및 표시
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # 스트리밍 응답 처리
                    for chunk in llm.stream(system_prompt):
                        if hasattr(chunk, 'content'):
                            content = chunk.content
                        else:
                            content = str(chunk)
                        
                        if content:
                            full_response += content
                            response_placeholder.write(full_response + "▌")
                    
                    # 관련 질문 3개 생성 (문서를 찾은 경우에만)
                    followup_questions = []
                    if retrieved_docs:
                        followup_questions = generate_followup_questions(prompt, full_response, context_text)
                    
                    # 답변에 관련 질문 추가
                    if followup_questions:
                        full_response += "\n\n---\n\n"
                        full_response += "### 💡 더 알아보기\n\n"
                        full_response += "다음 질문들도 도움이 될 수 있습니다:\n\n"
                        for i, question in enumerate(followup_questions, 1):
                            full_response += f"{i}. {question}\n"
                    
                    # 스트리밍 완료 후 최종 답변 표시 (커서 제거, 관련 질문 포함)
                    response_placeholder.write(full_response)
                
                response = full_response
                
                # 대화 기록에 추가
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # 대화 맥락 메모리에 추가 (최근 50개 대화 유지)
                st.session_state.conversation_memory.append(f"사용자: {prompt}")
                st.session_state.conversation_memory.append(f"AI: {response}")
                if len(st.session_state.conversation_memory) > 100:  # 50개 대화 = 100개 메시지
                    st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]
                
                # 대화 후 자동으로 Supabase에 세션 저장 (제목은 수동 저장 버튼으로만 생성)
                if supabase:
                    save_session(st.session_state.current_session_id)

        except Exception as e:
            with st.chat_message("assistant"):
                st.write(f"오류가 발생했습니다: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})

