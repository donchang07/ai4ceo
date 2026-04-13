import os

# 오프라인·로컬 전용: LangSmith / LangChain 클라우드 트레이싱 비활성화
# (python-dotenv는 기본적으로 이미 설정된 변수를 덮어쓰지 않으므로 load_dotenv 전에 지정)
# LANGSMITH_TRACING_V2 가 사용자/시스템 환경에만 있어도 켜지므로 LANGSMITH_* 도 명시적으로 끔
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import streamlit as st
import tempfile
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from typing import Any, List
import re

# --- 제목 스타일링 함수 ---
def style_markdown_headers(text: str, font_size: str = "14pt", font_weight: str = "bold") -> str:
    """
    마크다운 헤더(#, ## 등)를 찾아 스타일을 적용합니다.
    적용 시 원본 # 기호는 제거하고 텍스트만 스타일링합니다.
    모든 레벨의 헤더가 동일한 스타일을 갖게 됩니다.
    """
    def replace_header(match):
        header_text = match.group(2).strip() # # 다음의 텍스트
        return f"<span style='font-size:{font_size}; font-weight:{font_weight};'>{header_text}</span>"

    # ^(#{1,6})\s+(.*)$ : 라인 시작부분의 # (1~6개)와 공백문자 이후 모든 문자열
    styled_text = re.sub(r"^(#{1,6})\s+(.*)$", replace_header, text, flags=re.MULTILINE)
    return styled_text

def get_llm(model_name: str, temperature: float = 0.7) -> Any:
    return ChatOllama(model=model_name, temperature=temperature)

# --- 검색 문서 출처 포맷 함수 ---
def format_sources(docs):
    sources = []
    for doc in docs:
        name = doc.metadata.get("source", "알 수 없음")
        page = doc.metadata.get("page")
        if page is not None:
            sources.append(f"- {name} (p.{page})")
        else:
            sources.append(f"- {name}")
    return "출처:\n" + "\n".join(sources) if sources else ""

# --- 검색 문서 chunk별 출처 인라인 포함 함수 ---
def format_context_with_sources(docs):
    context_lines = []
    for doc in docs:
        name = doc.metadata.get("source", "알 수 없음")
        page = doc.metadata.get("page")
        if page is not None:
            source_str = f"(출처: {name} p.{page})"
        else:
            source_str = f"(출처: {name})"
        context_lines.append(f"{doc.page_content.strip()} {source_str}")
    return "\n\n".join(context_lines)

# --- 답변 문단별 출처 후처리 함수 ---
def insert_sources_to_answer(answer: str, docs):
    answer_paragraphs = [p for p in answer.split('\n') if p.strip()]
    sources = []
    for doc in docs:
        name = doc.metadata.get("source")
        page = doc.metadata.get("page")
        if name and name != "알 수 없음":
            if page is not None:
                sources.append(f"(출처: {name} p.{page})")
            else:
                sources.append(f"(출처: {name})")
        else:
            sources.append("")  # 출처가 없으면 빈 문자열
    result = []
    for i, para in enumerate(answer_paragraphs):
        result.append(para.strip())
    # 모든 문단 뒤에 출처 추가
    if sources:
        result.append("\n\n출처:\n" + "\n".join(sources))
    return "\n\n".join(result)

def get_ollama_models() -> tuple[list[str], str | None]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            error_message = result.stderr.strip() or "ollama list 실행에 실패했습니다."
            return [], error_message
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) <= 1:
            return [], "설치된 Ollama 모델이 없습니다."
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        filtered = [m for m in models if "cloud" not in m.lower()]
        return sorted(set(filtered)), None
    except FileNotFoundError:
        return [], "ollama 명령을 찾을 수 없습니다. Ollama가 설치되어 있는지 확인하세요."
    except Exception as e:
        return [], f"ollama list 실행 중 오류가 발생했습니다: {e}"

def check_offline_readiness(required_models: list[str]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            error_message = result.stderr.strip() or "ollama list 실행 실패"
            issues.append(f"Ollama CLI 오류: {error_message}")
            return False, issues
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) <= 1:
            issues.append("설치된 Ollama 모델이 없습니다.")
            return False, issues
        installed = set()
        for line in lines[1:]:
            parts = line.split()
            if parts:
                installed.add(parts[0])
        missing = []
        for model in required_models:
            if model in installed:
                continue
            if ":" not in model:
                has_family = any(inst.startswith(f"{model}:") or inst == model for inst in installed)
                if has_family:
                    continue
            missing.append(model)
        if missing:
            issues.append(f"필수 모델 누락: {', '.join(missing)}")
        return len(issues) == 0, issues
    except FileNotFoundError:
        issues.append("ollama 명령을 찾을 수 없습니다. Ollama 설치/경로를 확인하세요.")
        return False, issues
    except Exception as e:
        issues.append(f"오프라인 점검 중 오류: {e}")
        return False, issues

def build_bm25_retriever(vectorstore, k: int = 4) -> BM25Retriever | None:
    docstore = getattr(vectorstore, "docstore", None)
    docs = []
    if docstore is not None and hasattr(docstore, "_dict"):
        docs = list(docstore._dict.values())
    if not docs:
        return None
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25

def build_ensemble_retriever(vectorstore, llm, k: int = 4) -> EnsembleRetriever:
    vectorstore_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore_retriever
    )
    bm25_retriever = build_bm25_retriever(vectorstore, k=k)
    retrievers = [vectorstore_retriever, compression_retriever]
    weights = [0.5, 0.5]
    if bm25_retriever is not None:
        retrievers.append(bm25_retriever)
        weights = [0.4, 0.4, 0.2]
    return EnsembleRetriever(retrievers=retrievers, weights=weights)

# 환경 변수 로드
load_dotenv()

# .env에 LangSmith 관련 값이 있어도 이 프로세스에서는 원격으로 보내지 않음
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
for _k in (
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_HUB_API_URL",
):
    os.environ.pop(_k, None)
try:
    from langsmith import utils as _ls_utils

    _ls_utils.get_env_var.cache_clear()
except Exception:
    pass

# FAISS 벡터스토어 저장 경로 (오프라인 사용을 위한 로컬 저장)
FAISS_DB_PATH = "./faiss_db"
PROCESSED_FILES_PATH = "./processed_files.json"

# 벡터스토어 디렉토리 생성
Path(FAISS_DB_PATH).mkdir(parents=True, exist_ok=True)

# 페이지 설정
st.set_page_config(
    page_title="AI4CEO 코딩스쿨 RAG",
    page_icon="📚",
    layout="wide"
)

# 초기 상태 설정
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 오프라인 FAISS 벡터스토어 로드 (기존 저장된 벡터스토어가 있으면 로드)
if st.session_state.vectorstore is None:
    faiss_index_path = os.path.join(FAISS_DB_PATH, "index.faiss")
    faiss_store_path = os.path.join(FAISS_DB_PATH, "index.pkl")
    if os.path.exists(FAISS_DB_PATH) and os.path.exists(faiss_index_path) and os.path.exists(faiss_store_path):
        try:
            embeddings = OllamaEmbeddings(model="bge-m3")
            st.session_state.vectorstore = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            # 처리된 파일 목록도 로드
            if os.path.exists(PROCESSED_FILES_PATH):
                with open(PROCESSED_FILES_PATH, "r", encoding="utf-8") as f:
                    st.session_state.processed_files = json.load(f)
            # Retriever 초기화
            if st.session_state.vectorstore:
                llm = get_llm(st.session_state.llm_model if 'llm_model' in st.session_state else "qwen3:latest", temperature=0)
                st.session_state.retriever = build_ensemble_retriever(st.session_state.vectorstore, llm, k=4)
        except Exception as e:
            st.warning(f"기존 벡터스토어 로드 중 오류 발생: {e}. 새로 시작합니다.")
    elif os.path.exists(FAISS_DB_PATH) and os.listdir(FAISS_DB_PATH):
        st.warning("기존 벡터스토어 파일이 불완전합니다. 새로 시작합니다.")

# 제목 및 설명
# st.title("AI4CEO 코딩스쿨 챗봇") # 기존 st.title 제거
st.markdown("""
<h1 style='text-align: center;'>
    <span style='color:red;'>AI</span><span style='color:blue;'>4</span><span style='color:pink;'>CEO</span> <span style='color:yellow;'>코딩</span><span style='color:green;'>스쿨</span> 챗봇
</h1>
""", unsafe_allow_html=True)
st.markdown("PDF 파일을 업로드하고 내용에 관해 질문해보세요!") # 부제는 그대로 둠

# 사이드바 설정
with st.sidebar:
    # 1. LLM 모델 선택
    st.header("LLM 모델 선택")

    # 모델 목록 정의 (Ollama 모델)
    all_models, ollama_error = get_ollama_models()
    if not all_models:
        all_models = [
            "qwen3:latest",
            "gemma3:27b",
            "gemma3:latest",
            "llama4:latest",
            "solar-pro:latest",
            "gpt-oss:20b",
            "deepseek-r1:8b",
            "exaone-deep:latest",
            "qwen2.5vl:latest",
            "deepseek-r1:latest"
        ]
        all_models = [m for m in all_models if "cloud" not in m.lower()]
        if ollama_error:
            st.warning(f"Ollama 모델 목록을 불러오지 못했습니다: {ollama_error}")

    # --- 선택 관리 로직 (단순화) ---
    # st.session_state에 llm_model 초기화
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = all_models[0] # 기본값

    # 현재 선택된 모델의 인덱스 찾기
    try:
        current_index = all_models.index(st.session_state.llm_model)
    except ValueError:
        current_index = 0 # 목록에 없으면 기본값 인덱스

    # --- 단일 모델 라디오 ---
    selected_model = st.radio(
        "사용할 언어모델을 선택하세요", # 다시 원래 레이블 사용
        options=all_models,
        index=current_index,
        key='llm_model_radio' # 단일 키 사용
        # label_visibility="collapsed" 제거 또는 주석 처리하여 레이블 표시
    )

    # 선택 변경 시 session_state 업데이트 (콜백 제거, 직접 할당)
    st.session_state.llm_model = selected_model

    # 오프라인 모드 점검
    if st.button("오프라인 모드 확인"):
        required_models = ["bge-m3", st.session_state.llm_model]
        ok, issues = check_offline_readiness(required_models)
        if ok:
            st.success("오프라인 모드 준비 완료: 필수 모델이 모두 존재합니다.")
        else:
            st.warning("오프라인 모드 점검 실패:")
            for issue in issues:
                st.write(f"- {issue}")

    # 3. PDF 파일 업로드
    st.header("PDF 파일 업로드")
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type="pdf", accept_multiple_files=True)
    # 4. 파일 처리 버튼
    if uploaded_files:
        process_button = st.button("파일 처리하기")
        if process_button:
            try:
                with st.spinner("PDF 파일을 처리 중입니다..."):
                    temp_dir = tempfile.TemporaryDirectory()
                    all_docs = []
                    new_files = []
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name in st.session_state.processed_files:
                            continue
                        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                        with open(temp_file_path, "wb") as f: # Use "wb" for writing bytes
                            f.write(uploaded_file.getbuffer())
                        loader = PDFPlumberLoader(temp_file_path)
                        documents = loader.load()
                        for doc in documents:
                            doc.metadata["source"] = uploaded_file.name
                        all_docs.extend(documents)
                        new_files.append(uploaded_file.name)
                    if not all_docs:
                        st.success("모든 파일이 이미 처리되었습니다.")
                    else:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len
                        )
                        chunks = text_splitter.split_documents(all_docs)
                        # Changed: Use OllamaEmbeddings with bge-m3 model
                        embeddings = OllamaEmbeddings(model="bge-m3")
                        if st.session_state.vectorstore is None:
                            vectorstore = FAISS.from_documents(chunks, embeddings)
                            st.session_state.vectorstore = vectorstore
                        else:
                            st.session_state.vectorstore.add_documents(chunks)
                        
                        # 오프라인 사용을 위해 FAISS 벡터스토어를 디스크에 저장
                        st.session_state.vectorstore.save_local(FAISS_DB_PATH)
                        
                        # Use selected Ollama model for compression retriever
                        llm = get_llm(st.session_state.llm_model, temperature=0)
                        st.session_state.retriever = build_ensemble_retriever(st.session_state.vectorstore, llm, k=4)
                        st.session_state.processed_files.extend(new_files)
                        
                        # 처리된 파일 목록도 디스크에 저장
                        with open(PROCESSED_FILES_PATH, "w", encoding="utf-8") as f:
                            json.dump(st.session_state.processed_files, f, ensure_ascii=False, indent=2)
                        
                        st.success(f"{len(new_files)}개의 PDF 파일 처리가 완료되었습니다! (벡터스토어가 디스크에 저장되었습니다)")
            except Exception as e:
                st.error(f"파일 처리 중 오류 발생: {e}")
                if "pdfplumber" in str(e).lower():
                    st.info(
                        "`pdfplumber`는 **지금 Streamlit이 쓰는 Python**에 설치되어 있어야 합니다. "
                        "프로젝트 루트에서 `uv sync` 후 **`uv run streamlit run ai4ceo-open.py`** 로 실행하거나, "
                        "해당 인터프리터로 `uv pip install pdfplumber`(또는 `pip install pdfplumber`)를 실행하세요."
                    )

    # 5. 처리된 파일 목록
    if st.session_state.processed_files:
        st.subheader("처리된 파일 목록")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")

    # 6. 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str): # content가 문자열인지 확인
            # 이전 대화 기록을 표시할 때도 헤더 스타일링 적용
            styled_content = style_markdown_headers(message["content"])
            st.markdown(styled_content, unsafe_allow_html=True)
        else:
            # 만약 content가 문자열이 아닌 다른 타입일 경우 (예외 처리)
            st.write(message["content"])

# --- Helper function to convert session history to BaseMessages ---
def get_chat_history_messages() -> List[BaseMessage]:
    """Converts session chat history to a list of BaseMessage objects."""
    messages = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages

if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- Main chat logic (RAG or direct LLM) ---
    # Removed internet search conditions, now only handles RAG or direct LLM
    # --- PDF 파일이 없는 경우 (retriever is None) ---
    if st.session_state.retriever is None:
        try:
            llm = get_llm(st.session_state.llm_model)
            chat_history_messages = get_chat_history_messages()
            direct_system_prompt = "당신은 유능한 AI 어시스턴트입니다. 반드시 한국어로 답변해주세요."

            # Simplified prompt template (removed gemini specific handling)
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", direct_system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{question}")
            ])
            chain_input = {"question": prompt, "history": chat_history_messages[:-1]} # Use[:-1] to exclude the current user prompt from history passed to LLM

            chain = (prompt_template | llm | StrOutputParser())

            # Stream mode로 답변 생성
            response_text = "" # 원본 LLM 응답
            with st.chat_message("assistant"):
                stream_placeholder = st.empty()
                try:
                    # 항상 stream mode 사용
                    response_stream = chain.stream(chain_input)
                    for chunk in response_stream:
                        chunk_str = str(chunk)
                        response_text += chunk_str
                        # 스트리밍 중 매번 스타일 적용 및 출력
                        styled_chunk_display = style_markdown_headers(response_text)
                        stream_placeholder.markdown(styled_chunk_display, unsafe_allow_html=True)
                    
                    # 최종적으로 저장되는 내용은 원본 텍스트
                    if response_text:
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    else:
                        raise Exception("응답이 생성되지 않았습니다.")
                except Exception as stream_error:
                    # 스트리밍 중 오류 발생 시 처리
                    error_detail = str(stream_error)
                    if "GGML_ASSERT" in error_detail or "ResponseError" in error_detail:
                        error_message = f"모델 실행 오류가 발생했습니다. 모델 '{st.session_state.llm_model}'이 손상되었거나 호환되지 않을 수 있습니다.\n\n해결 방법:\n1. 다른 모델을 선택해보세요\n2. Ollama를 재시작해보세요\n3. 모델을 다시 다운로드해보세요: `ollama pull {st.session_state.llm_model}`\n\n오류 상세: {error_detail}"
                    else:
                        error_message = f"스트리밍 중 오류 발생: {error_detail}"
                    stream_placeholder.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                    raise

        except Exception as e:
            error_message = f"LLM 생성 중 오류 발생: {e}"
            st.error(error_message)
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] != "assistant":
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})

    # --- PDF 파일이 있는 경우 (retriever is not None) ---
    else:
        try:
            retrieved_docs = st.session_state.retriever.invoke(prompt)
            context = format_context_with_sources(retrieved_docs)
            system_template = """
            답변을 전문적으로 해줘. 반드시 한국어로 답변해주세요.
            사용자의 질문: {question}
            다음 정보를 바탕으로 답변해주세요:
            {context}
            """
            chat_history_messages = get_chat_history_messages()

            # Simplified prompt template (removed gemini specific handling)
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_template),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{question}"),
            ])
            # Use[:-1] to exclude the current user prompt from history passed to LLM
            chain_input = {"context": context, "question": prompt, "history": chat_history_messages[:-1]}

            llm = get_llm(st.session_state.llm_model)
            rag_chain = (prompt_template | llm | StrOutputParser())

            # Stream mode로 답변 생성
            response_text = "" # 원본 LLM 응답
            with st.chat_message("assistant"):
                stream_placeholder = st.empty()
                try:
                    # 항상 stream mode 사용
                    response_stream = rag_chain.stream(chain_input)
                    for chunk in response_stream:
                        chunk_str = str(chunk)
                        response_text += chunk_str
                        # 스트리밍 중 매번 스타일 적용 및 출력 (출처 제외하고 먼저 스타일링)
                        styled_chunk_display = style_markdown_headers(response_text)
                        stream_placeholder.markdown(styled_chunk_display, unsafe_allow_html=True)

                    if not response_text:
                        raise Exception("응답이 생성되지 않았습니다.")

                    # 스트리밍 끝난 후, 원본 LLM 응답에 출처 삽입
                    response_with_sources_raw = insert_sources_to_answer(response_text, retrieved_docs)

                    # 출처 포함된 전체 텍스트에 최종적으로 제목 스타일 적용하여 표시
                    final_styled_text_with_sources = style_markdown_headers(response_with_sources_raw)
                    stream_placeholder.markdown(final_styled_text_with_sources, unsafe_allow_html=True)

                    # 대화 기록에는 출처 포함된 '원본' 텍스트 저장 (스타일링 전)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_with_sources_raw})
                except Exception as stream_error:
                    # 스트리밍 중 오류 발생 시 처리
                    error_detail = str(stream_error)
                    if "GGML_ASSERT" in error_detail or "ResponseError" in error_detail:
                        error_message = f"모델 실행 오류가 발생했습니다. 모델 '{st.session_state.llm_model}'이 손상되었거나 호환되지 않을 수 있습니다.\n\n해결 방법:\n1. 다른 모델을 선택해보세요\n2. Ollama를 재시작해보세요\n3. 모델을 다시 다운로드해보세요: `ollama pull {st.session_state.llm_model}`\n\n오류 상세: {error_detail}"
                    else:
                        error_message = f"RAG 스트리밍 중 오류 발생: {error_detail}"
                    stream_placeholder.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                    raise

        except Exception as e:
            error_message = f"RAG LLM 생성 중 오류 발생: {e}"
            st.error(error_message)
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] != "assistant":
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
