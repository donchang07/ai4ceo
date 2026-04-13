from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


SYSTEM_PROMPT = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘."
)


class AppState(BaseModel):
    messages: list[dict[str, str]] = Field(default_factory=list)
    indexed: bool = False


def _env_path() -> Path:
    return Path(__file__).resolve().parent / ".env"


def _load_documents_from_uploaded_pdfs(uploaded_files: list[Any]) -> list[Document]:
    documents: list[Document] = []
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


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def _build_ensemble_retriever(chunks: list[Document], api_key: str) -> EnsembleRetriever:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


def _ensure_session_state() -> None:
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState().model_dump()
    if "retriever" not in st.session_state:
        st.session_state.retriever = None


def _chat_with_rag(user_query: str, api_key: str) -> str:
    retriever = st.session_state.retriever
    if retriever is None:
        return "먼저 PDF를 업로드하고 RAG처리를 완료해 주세요."

    llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)
    docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"질문: {user_query}\n\n"
                f"참고 문서:\n{context}\n\n"
                "위 참고 문서를 바탕으로 답변해 주세요."
            )
        ),
    ]
    response = llm.invoke(messages)
    return response.content if isinstance(response.content, str) else str(response.content)


def render_app() -> None:
    load_dotenv(dotenv_path=_env_path())
    _ = OpenAI
    _ = ChatOllama

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY를 .env에 설정해 주세요.")
        st.stop()

    _ensure_session_state()
    app_state: dict[str, Any] = st.session_state.app_state
    st.title("RAG Chatbot")
    st.caption("여러 PDF를 업로드한 뒤 RAG처리를 눌러 질문하세요.")

    with st.sidebar:
        st.subheader("PDF 파일 업로드")
        uploaded_files = st.file_uploader(
            "PDF 파일 선택",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if st.button("RAG처리", use_container_width=True):
            if not uploaded_files:
                st.warning("최소 1개 이상의 PDF 파일을 업로드해 주세요.")
            else:
                docs = _load_documents_from_uploaded_pdfs(uploaded_files)
                if not docs:
                    st.warning("PDF에서 텍스트를 추출하지 못했습니다.")
                else:
                    chunks = _split_documents(docs)
                    st.session_state.retriever = _build_ensemble_retriever(chunks, api_key)
                    app_state["indexed"] = True
                    st.success("저장이 끝났습니다.")

    for msg in app_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("질문을 입력하세요.")
    if user_input:
        app_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        answer = _chat_with_rag(user_input, api_key)
        app_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)


def main() -> None:
    st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")
    render_app()


if __name__ == "__main__":
    main()
