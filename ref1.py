import logging
import os
import re
import tempfile
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"chatbot_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger("standard_chatbot")
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    return logger


LOGGER = setup_logging()


def load_environment() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    return key.strip()


def remove_separators(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"~~(.*?)~~", r"\1", text, flags=re.DOTALL)
    cleaned = re.sub(r"^\s*[-=_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def init_session_state() -> None:
    defaults = {
        "chat_history": [],
        "conversation_memory": [],
        "search_mode": "internet검색",
        "vectorstore": None,
        "retriever": None,
        "processed_files": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_chat_model(streaming: bool = True, temperature: float = 0.7) -> ChatOpenAI:
    api_key = get_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. 프로젝트 루트의 .env 파일을 확인해주세요.")
    return ChatOpenAI(model="gpt-5.2", temperature=temperature, api_key=api_key, streaming=streaming)


def stream_chat_response(prompt: str, context: str = "") -> str:
    llm = get_chat_model(streaming=True, temperature=0.7)
    system_prompt = (
        "당신은 전문적인 AI 어시스턴트입니다.\n"
        "답변은 반드시 마크다운 헤딩 구조를 사용하세요.\n"
        "- 제목은 # 1개\n"
        "- 본문은 ##, ###를 사용해 구조화\n"
        "- 구분선(---, ===, ___)과 취소선(~~ ~~)은 사용하지 마세요.\n"
    )
    user_prompt = f"{context}\n\n질문: {prompt}".strip()

    full_text = ""
    for chunk in llm.stream(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    ):
        piece = getattr(chunk, "content", "") or ""
        full_text += piece
        yield remove_separators(full_text)


def web_search_answer(prompt: str, history: list[dict[str, str]]) -> str:
    api_key = get_api_key()
    if not api_key:
        return "OPENAI_API_KEY가 없어 인터넷 검색을 수행할 수 없습니다."

    context_lines: list[str] = []
    for msg in history[-6:]:
        role = "사용자" if msg["role"] == "user" else "어시스턴트"
        context_lines.append(f"{role}: {msg['content']}")

    context_text = "\n".join(context_lines)
    input_text = (
        "아래 대화 맥락과 현재 질문을 바탕으로, 최신 웹 정보를 반영해 답변하세요.\n"
        "답변은 #, ##, ### 구조를 사용하고 구분선/취소선을 사용하지 마세요.\n\n"
        f"대화 맥락:\n{context_text}\n\n현재 질문: {prompt}"
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-5.2",
            input=input_text,
            tools=[{"type": "web_search", "search_context_size": "high"}],
        )
        return remove_separators(response.output_text)
    except Exception as exc:
        LOGGER.error("web_search 오류: %s", exc)
        return "인터넷 검색 중 오류가 발생했습니다."


def generate_follow_up_questions(question: str, answer: str) -> list[str]:
    try:
        llm = get_chat_model(streaming=False, temperature=0.9)
        prompt = (
            "다음 질문과 답변을 기반으로, 더 상세히 이해하기 위한 후속 질문 3개를 생성하세요.\n"
            "각 줄에 하나씩 질문만 작성하세요.\n\n"
            f"원래 질문: {question}\n\n답변:\n{answer}"
        )
        result = llm.invoke([{"role": "user", "content": prompt}])
        text = getattr(result, "content", str(result))
        items = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
        return items[:3]
    except Exception as exc:
        LOGGER.warning("후속 질문 생성 실패: %s", exc)
        return []


def process_pdf_files(uploaded_files: list[Any]) -> None:
    if not uploaded_files:
        return

    all_docs = []
    new_names = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for uf in uploaded_files:
            if uf.name in st.session_state.processed_files:
                continue
            temp_path = os.path.join(tmpdir, uf.name)
            with open(temp_path, "wb") as f:
                f.write(uf.getbuffer())
            docs = PyPDFLoader(temp_path).load()
            for d in docs:
                d.metadata["source"] = uf.name
            all_docs.extend(docs)
            new_names.append(uf.name)

    if not all_docs:
        st.warning("새로 처리할 PDF 파일이 없습니다.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        st.warning("문서에서 텍스트를 추출하지 못했습니다.")
        return

    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY가 없어 RAG 인덱스를 만들 수 없습니다.")
        return

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        batch_size = 30
        if st.session_state.vectorstore is None:
            vectorstore = None
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
            st.session_state.vectorstore = vectorstore
        else:
            for i in range(0, len(chunks), batch_size):
                st.session_state.vectorstore.add_documents(chunks[i : i + batch_size])

        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )
        st.session_state.processed_files.extend(new_names)
        st.success(f"{len(new_names)}개 파일 처리가 완료되었습니다.")
    except Exception as exc:
        LOGGER.error("PDF 처리 오류: %s", exc)
        st.error("PDF 처리 중 오류가 발생했습니다.")


def build_context_from_memory() -> str:
    memory = st.session_state.conversation_memory[-100:]
    if not memory:
        return ""
    return "이전 대화 맥락:\n" + "\n".join(memory)


def append_and_render_followups(base_answer: str, question: str) -> str:
    followups = generate_follow_up_questions(question, base_answer)
    if not followups:
        return base_answer
    block = ["### 💡 다음에 물어볼 수 있는 질문들", ""]
    for idx, q in enumerate(followups, start=1):
        block.append(f"{idx}. {q}")
    return remove_separators(base_answer + "\n\n" + "\n".join(block))


def render_header() -> None:
    left, center, right = st.columns([1, 4, 1])
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "표준.png")
    with left:
        if os.path.exists(logo_path):
            st.image(logo_path, width=90)
        else:
            st.markdown("## 📚")
    with center:
        st.markdown(
            """
            <div style="text-align:center;">
                <h1 style="font-size:7rem; margin:0;">
                    <span style="color:#1f77b4;">표준</span>
                    <span style="color:#ffd700;">챗봇</span>
                </h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.empty()


def render_styles() -> None:
    st.markdown(
        """
        <style>
        h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
        h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
        h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
        .stChatMessage { border-radius: 10px; padding: 4px; }
        .stButton > button {
            background-color: #ff69b4 !important;
            color: white !important;
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("검색 모드 선택")
        st.session_state.search_mode = st.radio(
            "검색 방식을 선택하세요",
            options=["internet검색", "RAG검색"],
            index=0 if st.session_state.search_mode == "internet검색" else 1,
        )

        uploaded_files = st.file_uploader("PDF 파일 업로드", type="pdf", accept_multiple_files=True)
        if st.button("파일 처리하기"):
            process_pdf_files(uploaded_files or [])

        if st.session_state.processed_files:
            st.write("처리된 파일 목록")
            for name in st.session_state.processed_files:
                st.write(f"- {name}")

        if st.button("대화 초기화"):
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.rerun()

        st.subheader("현재 설정")
        st.text(f"검색 모드: {st.session_state.search_mode}")
        st.text(f"처리된 파일 수: {len(st.session_state.processed_files)}")
        st.text(f"대화 기록 수: {len(st.session_state.chat_history)}")


def run_chat() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        answer = ""
        try:
            if st.session_state.search_mode == "internet검색":
                base = web_search_answer(user_input, st.session_state.chat_history[:-1])
                answer = append_and_render_followups(base, user_input)
                placeholder.markdown(answer)
            else:
                if st.session_state.retriever is None:
                    answer = "RAG검색을 선택하셨습니다. 먼저 PDF를 업로드하고 파일을 처리해주세요."
                    placeholder.markdown(answer)
                else:
                    docs = st.session_state.retriever.invoke(user_input)
                    context_doc = "\n\n".join(doc.page_content for doc in docs[:3]) if docs else ""
                    memory_context = build_context_from_memory()
                    stream_gen = stream_chat_response(user_input, context=f"{memory_context}\n\n문서 컨텍스트:\n{context_doc}")
                    for partial in stream_gen:
                        answer = partial
                        placeholder.markdown(answer)
                    answer = append_and_render_followups(answer, user_input)
                    placeholder.markdown(answer)

        except Exception as exc:
            LOGGER.error("답변 생성 오류: %s", exc)
            answer = "답변 생성 중 오류가 발생했습니다. 설정과 API 키를 확인해주세요."
            placeholder.markdown(answer)

    answer = remove_separators(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.session_state.conversation_memory.append(f"사용자: {user_input}")
    st.session_state.conversation_memory.append(f"어시스턴트: {answer}")
    st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]


def main() -> None:
    load_environment()
    st.set_page_config(page_title="표준 잿봇", page_icon="📚", layout="wide")
    init_session_state()
    render_styles()
    render_header()
    render_sidebar()
    run_chat()


if __name__ == "__main__":
    main()
