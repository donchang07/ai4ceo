"""
통합 앱 (AI Agent 버전)
3개의 앱(시간, 챗봇, RAG)을 AI Agent가 자동으로 선택하여 통합한 메인 앱
LangChain AgentExecutor 사용, Ollama qwen3:latest 모델 사용
"""

import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import time
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# LangChain 관련 임포트
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# OpenAI 임포트 (챗봇용)
from openai import OpenAI

# 환경 변수 로드
load_dotenv()

# LangSmith 추적 비활성화 (API 키가 없을 때 경고 방지)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

# 페이지 설정
st.set_page_config(
    page_title="통합 앱 (AI Agent)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = None

if "tool_history" not in st.session_state:
    st.session_state.tool_history = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 각 앱별 세션 상태 초기화
if "chatbot_messages" not in st.session_state:
    st.session_state.chatbot_messages = []

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

if "rag_vectorstore" not in st.session_state:
    st.session_state.rag_vectorstore = None

if "rag_retriever" not in st.session_state:
    st.session_state.rag_retriever = None

if "rag_processing_complete" not in st.session_state:
    st.session_state.rag_processing_complete = False

if "faiss_db_path" not in st.session_state:
    st.session_state.faiss_db_path = None

# Tool 이름 매핑 (영문 -> 한글)
TOOL_NAME_MAP = {
    "time": "시간",
    "chatbot": "챗봇",
    "rag": "RAG"
}

# ==================== Callback Handler ====================
class ToolSelectionCallback(BaseCallbackHandler):
    """선택된 tool을 추적하는 Callback Handler"""
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Tool이 호출될 때 실행"""
        try:
            tool_name = serialized.get("name", "")
            if tool_name:
                # 영문 tool 이름을 한글로 변환
                korean_name = TOOL_NAME_MAP.get(tool_name, tool_name)
                
                # Streamlit 세션 상태에 안전하게 접근
                try:
                    if hasattr(st, 'session_state'):
                        st.session_state.selected_tool = korean_name
                        if "tool_history" not in st.session_state:
                            st.session_state.tool_history = []
                        if korean_name not in st.session_state.tool_history:
                            st.session_state.tool_history.append(korean_name)
                except (AttributeError, RuntimeError):
                    pass
        except Exception:
            pass

# ==================== Tool 정의 ====================
@tool
def time() -> str:
    """현재 시간과 날짜를 표시하는 tool입니다. 사용자가 시간, 날짜, 현재 시각 등을 물어볼 때 사용합니다."""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y년 %m월 %d일")
    return f"현재 날짜: {current_date}, 현재 시간: {current_time}"

@tool
def chatbot(question: str) -> str:
    """일반적인 질문에 답변하는 챗봇 tool입니다. 
    시간이나 날짜가 아닌 일반적인 대화, 질문, 설명 요청 등에 사용합니다.
    
    Args:
        question: 사용자의 질문이나 메시지
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "OPENAI_API_KEY가 설정되지 않았습니다."
    
    client = OpenAI(api_key=openai_api_key)
    
    # 대화 히스토리 구성
    messages = []
    for msg in st.session_state.chatbot_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # 현재 질문 추가
    messages.append({"role": "user", "content": question})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2048,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        # 세션 상태에 저장
        st.session_state.chatbot_messages.append({"role": "user", "content": question})
        st.session_state.chatbot_messages.append({"role": "assistant", "content": answer})
        
        return answer
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

@tool
def rag(question: str) -> str:
    """PDF 문서에 대해 질문하는 RAG tool입니다. 
    업로드된 PDF 파일의 내용에 대한 질문에 사용합니다.
    
    Args:
        question: PDF 내용에 대한 질문
    """
    if not st.session_state.rag_processing_complete:
        return "먼저 PDF 파일을 업로드하고 처리해주세요."
    
    try:
        # FAISS DB가 로컬에 저장되어 있으면 로드
        if st.session_state.faiss_db_path and not st.session_state.rag_vectorstore:
            embeddings = OllamaEmbeddings(model="bge-m3:latest")
            st.session_state.rag_vectorstore = FAISS.load_local(
                st.session_state.faiss_db_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            vector_retriever = st.session_state.rag_vectorstore.as_retriever(search_kwargs={"k": 4})
            # BM25는 메모리에만 있으므로 재생성 필요 없음
            # 기존 retriever 사용
            if st.session_state.rag_retriever:
                retriever = st.session_state.rag_retriever
            else:
                retriever = vector_retriever
        else:
            retriever = st.session_state.rag_retriever
        
        # 검색 수행
        relevant_docs = retriever.invoke(question)
        
        # 컨텍스트 구성
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # 시스템 메시지
        system_message = """너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. 
그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘."""
        
        # 프롬프트 구성
        prompt_template = f"""다음 컨텍스트를 바탕으로 질문에 답변해주세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        
        # LLM 초기화 (qwen3:latest)
        llm = ChatOllama(model="qwen3:latest", temperature=0.7)
        
        # 메시지 구성 (이전 대화 포함)
        messages = [SystemMessage(content=system_message)]
        
        # 이전 대화 기록 추가 (최근 10개 메시지만 유지)
        recent_messages = st.session_state.rag_messages[-10:]
        for msg in recent_messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # 현재 질문과 컨텍스트 추가
        messages.append(HumanMessage(content=prompt_template))
        
        response = llm.invoke(messages)
        answer = response.content
        
        # 세션 상태에 저장
        st.session_state.rag_messages.append({"role": "user", "content": question})
        st.session_state.rag_messages.append({"role": "assistant", "content": answer})
        
        return answer
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

# ==================== 앱 표시 함수들 ====================
def show_time_app():
    """실시간 시간 표시 앱"""
    st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
    }
    .main .block-container {
        padding: 0;
        max-width: 100%;
        height: 100vh;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding-top: 10vh;
    }
    .time-container {
        background-color: #000000;
        padding: 50px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    .time-display {
        font-family: 'Courier New', monospace;
        font-size: 72px;
        color: #00ff00;
        font-weight: bold;
        margin: 20px 0;
        text-shadow: 0 0 20px #00ff00;
    }
    .date-display {
        font-family: 'Courier New', monospace;
        font-size: 36px;
        color: #ffff00;
        font-weight: bold;
        margin: 20px 0;
        text-shadow: 0 0 10px #ffff00;
    }
    </style>
    """, unsafe_allow_html=True)
    
    placeholder = st.empty()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y년 %m월 %d일")
    
    placeholder.markdown(f"""
    <div class="time-container">
        <div class="date-display">{current_date}</div>
        <div class="time-display">{current_time}</div>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(1)
    st.rerun()

def show_chatbot_app():
    """간단한 챗봇 앱"""
    st.title("💬 My First Chatbot")
    
    for message in st.session_state.chatbot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def show_rag_app():
    """RAG 챗봇 앱"""
    st.title("📚 RAG 챗봇")
    
    if not st.session_state.rag_processing_complete:
        st.info("왼쪽 사이드바에서 PDF 파일을 업로드하고 처리해주세요.")
    
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ==================== 메인 앱 ====================
# 사이드바
with st.sidebar:
    st.title("🤖 통합 앱 (AI Agent)")
    st.markdown("---")
    
    # 선택된 tool 표시
    st.subheader("📌 선택된 Tool")
    if st.session_state.selected_tool:
        st.success(f"✅ {st.session_state.selected_tool}")
    else:
        st.info("질문을 입력하면 AI Agent가 자동으로 tool을 선택합니다.")
    
    st.markdown("---")
    
    # Tool 사용 이력
    if st.session_state.tool_history:
        st.subheader("📋 Tool 사용 이력")
        for tool_name in st.session_state.tool_history[-5:]:  # 최근 5개만 표시
            st.text(f"• {tool_name}")
    
    st.markdown("---")
    
    # RAG 앱의 경우 PDF 업로드 및 처리
    st.subheader("📄 PDF 파일 업로드 (RAG용)")
    uploaded_files = st.file_uploader(
        "PDF 파일을 선택하세요",
        type=["pdf"],
        accept_multiple_files=True,
        key="rag_file_uploader"
    )
    
    if uploaded_files and st.button("📚 PDF 처리하기"):
        with st.spinner("PDF 파일을 처리하는 중..."):
            all_docs = []
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    
                    # 메타데이터에 파일명 추가
                    for doc in docs:
                        doc.metadata['source'] = uploaded_file.name
                    
                    all_docs.extend(docs)
                except Exception as e:
                    st.error(f"파일 {uploaded_file.name} 처리 중 오류: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            if all_docs:
                try:
                    # 텍스트 분할 (chunk size = 1000, overlap = 200)
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(all_docs)
                    
                    # 임베딩 생성 (bge-m3:latest)
                    embeddings = OllamaEmbeddings(model="bge-m3:latest")
                    
                    # FAISS 벡터스토어 생성 및 로컬 파일에 저장 (오프라인)
                    faiss_db_dir = Path("faiss_db")
                    faiss_db_dir.mkdir(exist_ok=True)
                    
                    # 고유한 DB 경로 생성
                    db_id = str(uuid.uuid4())[:8]
                    faiss_db_path = faiss_db_dir / f"faiss_db_{db_id}"
                    
                    # FAISS 벡터스토어 생성
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    
                    # 로컬 파일 시스템에 저장
                    vectorstore.save_local(str(faiss_db_path))
                    
                    # BM25 검색기 생성
                    texts = [doc.page_content for doc in chunks]
                    metadatas = [doc.metadata for doc in chunks]
                    bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
                    bm25_retriever.k = 4
                    
                    # 벡터 검색기 생성
                    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                    
                    # Ensemble Retriever 생성
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, vector_retriever],
                        weights=[0.5, 0.5]
                    )
                    
                    st.session_state.rag_vectorstore = vectorstore
                    st.session_state.rag_retriever = ensemble_retriever
                    st.session_state.faiss_db_path = str(faiss_db_path)
                    st.session_state.rag_processing_complete = True
                    st.success("✅ PDF 파일 처리가 완료되었습니다!")
                    st.info(f"총 {len(all_docs)}개의 문서 페이지가 처리되었습니다.")
                    st.info(f"FAISS DB가 로컬에 저장되었습니다: {faiss_db_path}")
                except Exception as e:
                    st.error(f"벡터 스토어 생성 중 오류가 발생했습니다: {str(e)}")
            else:
                st.error("처리할 문서가 없습니다.")
    
    st.markdown("---")
    
    # 새로시작하기 버튼
    if st.button("🔄 새로시작하기", type="primary", use_container_width=True):
        st.session_state.chatbot_messages = []
        st.session_state.rag_messages = []
        st.session_state.rag_vectorstore = None
        st.session_state.rag_retriever = None
        st.session_state.rag_processing_complete = False
        st.session_state.faiss_db_path = None
        st.session_state.selected_tool = None
        st.session_state.tool_history = []
        st.session_state.agent_messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ 안내")
    st.info("AI Agent가 질문에 따라 자동으로 적절한 tool을 선택합니다.")

# 메인 화면
# Agent 초기화
try:
    # LLM 초기화 (qwen3:latest)
    llm = ChatOllama(model="qwen3:latest", temperature=0.7)
    
    # Tools 리스트
    tools = [time, chatbot, rag]
    
    # 에이전트 프롬프트 설정
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 사용자의 질문을 분석하여 가장 적합한 도구를 선택하는 AI 에이전트입니다.

다음 규칙에 따라 도구를 선택하세요:
1. 현재 날짜나 시간을 물어보면 -> time 도구 사용
2. 일반적인 대화나 질문이면 -> chatbot 도구 사용
3. 업로드된 PDF 문서에 대한 질문이면 -> rag 도구 사용

항상 한국어로 답변해주세요. 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # 에이전트 생성
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Callback Handler
    callback_handler = ToolSelectionCallback()
    
    # AgentExecutor 생성
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=False,
        callbacks=[callback_handler]
    )
    
    # 대화 히스토리 표시
    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력 처리
    if prompt_input := st.chat_input("무엇이든 물어보세요! AI Agent가 자동으로 적절한 tool을 선택합니다."):
        # 사용자 메시지 추가
        st.session_state.agent_messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)
        
        # AI Agent 실행
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤖 AI Agent가 적절한 tool을 선택하고 있습니다...")
            
            try:
                # 대화 히스토리 구성
                chat_history = []
                for msg in st.session_state.agent_messages[:-1]:  # 마지막 메시지 제외
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))
                
                # AgentExecutor 실행
                result = agent_executor.invoke({
                    "input": prompt_input,
                    "chat_history": chat_history
                })
                
                # 응답 추출
                full_response = result.get("output", str(result))
                
                message_placeholder.markdown(full_response)
                st.session_state.agent_messages.append({"role": "assistant", "content": full_response})
                
                # 대화 히스토리 업데이트
                st.session_state.chat_history = chat_history
                st.session_state.chat_history.append(HumanMessage(content=prompt_input))
                st.session_state.chat_history.append(AIMessage(content=full_response))
                
            except Exception as e:
                error_message = f"오류가 발생했습니다: {str(e)}"
                st.error(error_message)
                st.session_state.agent_messages.append({"role": "assistant", "content": error_message})
    
    # 선택된 tool에 따라 해당 앱 표시
    if st.session_state.selected_tool == "시간":
        show_time_app()
    elif st.session_state.selected_tool == "챗봇":
        show_chatbot_app()
    elif st.session_state.selected_tool == "RAG":
        show_rag_app()

except Exception as e:
    st.error(f"초기화 중 오류가 발생했습니다: {str(e)}")
    st.info("Ollama가 실행 중인지 확인하고, qwen3:latest 모델이 설치되어 있는지 확인해주세요.")

