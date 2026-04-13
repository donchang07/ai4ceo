from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_perplexity import ChatPerplexity
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 환경 설정 ────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

TOOL_DISPLAY = {
    "show_time": "🕐 시간",
    "general_chatbot": "🤖 챗봇",
    "internet_search": "🌐 인터넷검색",
    "rag_search": "📚 RAG",
}

# ── Tools ────────────────────────────────────────────────────


@tool
def show_time(query: str) -> str:
    """현재 날짜와 시간을 알려줍니다. 사용자가 시간, 날짜, 몇 시, 지금 시간, 오늘 날짜 등을 물어볼 때 이 도구를 사용하세요."""
    now = datetime.now()
    return f"현재 날짜: {now.strftime('%Y년 %m월 %d일 (%A)')}\n현재 시간: {now.strftime('%H시 %M분 %S초')}"


@tool
def general_chatbot(query: str) -> str:
    """일반적인 대화, 질문 답변, 설명, 번역, 요약, 코딩 등 범용 AI 챗봇입니다. 인터넷 검색이나 PDF 문서가 필요 없는 일반적인 질문에 사용하세요."""
    from langchain_core.messages import SystemMessage

    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
    response = llm.invoke([
        SystemMessage(content="You are a helpful and kind assistant."),
        HumanMessage(content=query),
    ])
    return response.content if isinstance(response.content, str) else str(response.content)


@tool
def internet_search(query: str) -> str:
    """인터넷에서 최신 정보를 검색합니다. 최신 뉴스, 실시간 정보, 날씨, 주가, 스포츠 결과, 최근 이벤트 등 인터넷 검색이 필요한 질문에 사용하세요."""
    from langchain_core.messages import SystemMessage

    llm = ChatPerplexity(
        model="sonar-pro",
        pplx_api_key=PERPLEXITY_API_KEY,
        temperature=0,
    )
    response = llm.invoke([
        SystemMessage(
            content="You are a helpful internet research assistant. "
            "Always cite specific numbers, dates, and sources. Answer in Korean."
        ),
        HumanMessage(content=query),
    ])
    if isinstance(response.content, str):
        return response.content.strip()
    chunks: list[str] = []
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    chunks.append(text)
    return "\n".join(chunks).strip() or "답변을 생성하지 못했습니다."


@tool
def rag_search(query: str) -> str:
    """업로드된 PDF 문서를 기반으로 질문에 답변합니다. 사용자가 업로드한 PDF 문서의 내용에 대해 질문할 때 이 도구를 사용하세요."""
    from langchain_core.messages import SystemMessage

    retriever = st.session_state.get("retriever")
    if retriever is None:
        return "먼저 왼쪽 사이드바에서 PDF 파일을 업로드하고 'RAG처리' 버튼을 눌러주세요."

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
    system_prompt = (
        "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
        "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘."
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"질문: {query}\n\n참고 문서:\n{context}\n\n위 참고 문서를 바탕으로 답변해 주세요."
        ),
    ])
    return response.content if isinstance(response.content, str) else str(response.content)


# ── Agent 생성 ───────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = (
    "당신은 사용자의 질문을 분석하여 적절한 도구를 선택하는 AI 에이전트입니다.\n"
    "반드시 하나의 도구를 선택하여 사용자의 질문을 처리하세요.\n"
    "도구 선택 기준:\n"
    "- 시간/날짜 관련 질문 → show_time\n"
    "- 최신 뉴스, 실시간 정보, 인터넷 검색이 필요한 질문 → internet_search\n"
    "- PDF 문서 관련 질문 → rag_search\n"
    "- 그 외 일반적인 질문 → general_chatbot\n\n"
    "도구가 반환한 결과를 그대로 사용자에게 전달하세요."
)


def _create_agent():
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)
    tools = [show_time, general_chatbot, internet_search, rag_search]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


# ── RAG 헬퍼 ────────────────────────────────────────────────


def _load_pdfs(uploaded_files: list) -> list:
    documents = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return documents


def _build_retriever(chunks: list, api_key: str) -> EnsembleRetriever:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vec_ret = vectorstore.as_retriever(search_kwargs={"k": 4})
    bm25_ret = BM25Retriever.from_documents(chunks)
    bm25_ret.k = 4
    return EnsembleRetriever(
        retrievers=[bm25_ret, vec_ret],
        weights=[0.5, 0.5],
    )


# ── 시계 위젯 ───────────────────────────────────────────────


def _render_time_widget() -> None:
    now = datetime.now()
    date_text = now.strftime("%Y-%m-%d (%a)")
    time_text = now.strftime("%H:%M:%S")
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap"
              rel="stylesheet">
        <div style="background-color:#000;padding:1.5rem;border-radius:10px;
                    text-align:center;margin:0.5rem 0;">
            <div style="font-family:'Orbitron',monospace;font-size:1.2rem;
                        color:#ffff00;letter-spacing:0.12em;margin-bottom:0.5rem;">
                {date_text}
            </div>
            <div style="font-family:'Orbitron',monospace;font-size:2.5rem;font-weight:700;
                        color:#00ff00;letter-spacing:0.18em;
                        text-shadow:0 0 12px rgba(0,255,0,0.45);">
                {time_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 메시지에서 사용된 도구 이름 추출 ────────────────────────


def _extract_tool_name(messages: list) -> str | None:
    """Agent 응답 메시지 목록에서 사용된 도구 이름을 추출합니다."""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return msg.tool_calls[0]["name"]
    return None


# ── Streamlit 앱 ────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="AI Agent 통합 앱", page_icon="🤖", layout="wide")

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY를 .env 파일에 설정해 주세요.")
        st.stop()

    # ── Session state 초기화 ──
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "tool_history" not in st.session_state:
        st.session_state.tool_history = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🤖 AI Agent 통합 앱")
        st.caption("질문을 입력하면 AI Agent가 적절한 도구를 자동 선택합니다.")

        st.markdown("---")
        st.markdown("### 📋 도구 목록")
        st.markdown("🕐 **시간** — 현재 시간 표시")
        st.markdown("🤖 **챗봇** — 일반 대화")
        st.markdown("🌐 **인터넷검색** — 최신 정보 검색")
        st.markdown("📚 **RAG** — PDF 문서 Q&A")

        st.markdown("---")
        st.markdown("### 🔧 선택된 도구 이력")
        if st.session_state.tool_history:
            for i, name in enumerate(st.session_state.tool_history, 1):
                display = TOOL_DISPLAY.get(name, name)
                st.text(f"{i}. {display}")
        else:
            st.caption("아직 선택된 도구가 없습니다.")

        st.markdown("---")
        st.subheader("📄 PDF 업로드 (RAG)")
        uploaded_files = st.file_uploader(
            "PDF 파일 선택",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if st.button("RAG처리", use_container_width=True):
            if not uploaded_files:
                st.warning("PDF 파일을 업로드해 주세요.")
            else:
                with st.spinner("PDF 처리 중..."):
                    docs = _load_pdfs(uploaded_files)
                    if not docs:
                        st.warning("PDF에서 텍스트를 추출하지 못했습니다.")
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=200,
                        )
                        chunks = splitter.split_documents(docs)
                        st.session_state.retriever = _build_retriever(
                            chunks, OPENAI_API_KEY,
                        )
                        st.success("저장이 끝났습니다!")

        st.markdown("---")
        if st.button("🔄 새로시작하기", use_container_width=True):
            st.session_state.agent_messages = []
            st.session_state.tool_history = []
            st.session_state.retriever = None
            st.rerun()

    # ── Main 영역 ──
    st.title("AI Agent 통합 앱")

    # 대화 이력 표시
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            if msg.get("show_clock"):
                _render_time_widget()
            st.markdown(msg["content"])

    # 사용자 입력
    user_input = st.chat_input("질문을 입력하세요.")
    if not user_input:
        return

    st.session_state.agent_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("AI Agent가 분석 중..."):
            agent = _create_agent()

            # 대화 이력을 LangChain 메시지로 변환
            lc_messages = []
            for msg in st.session_state.agent_messages[:-1]:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                else:
                    lc_messages.append(AIMessage(content=msg["content"]))
            lc_messages.append(HumanMessage(content=user_input))

            result = agent.invoke({"messages": lc_messages})
            result_messages = result["messages"]

            # 마지막 AI 메시지에서 답변 추출
            answer = ""
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

            if not answer:
                answer = "답변을 생성하지 못했습니다."

            # 사용된 도구 추적
            tool_used = _extract_tool_name(result_messages)
            if tool_used:
                st.session_state.tool_history.append(tool_used)

            # 시간 도구인 경우 시계 위젯도 함께 표시
            show_clock = tool_used == "show_time"
            if show_clock:
                _render_time_widget()

            st.markdown(answer)

    st.session_state.agent_messages.append({
        "role": "assistant",
        "content": answer,
        "show_clock": show_clock,
    })


if __name__ == "__main__":
    main()
