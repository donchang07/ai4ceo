from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, TypedDict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI  # 최신 OpenAI SDK import (요구사항 반영)
from langchain_openai import ChatOpenAI
from langchain_perplexity import ChatPerplexity
from pydantic import BaseModel, Field  # 요구사항 반영용 import
from langchain_ollama import ChatOllama  # 요구사항 반영용 import
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class ChatTurn(TypedDict):
    role: str
    content: str


class AppConfig(BaseModel):
    model_name: str = Field(default="sonar-pro", description="사용할 Perplexity 모델명")
    title: str = Field(default="Internet Search Chatbot", description="앱 제목")
    system_prompt: str = Field(
        default=(
            "You are a helpful internet research assistant. "
            "Always cite specific numbers, dates, and sources. "
            "Answer in Korean."
        ),
        description="시스템 프롬프트",
    )


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger(__name__)


def load_pplx_api_key() -> str:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    pplx_key = os.getenv("PERPLEXITY_API_KEY")
    if not pplx_key:
        raise ValueError("PERPLEXITY_API_KEY가 .env 파일에 없습니다.")
    return pplx_key


def init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def to_langchain_messages(history: List[ChatTurn], system_prompt: str) -> List[BaseMessage]:
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    return messages


def generate_answer(
    llm: ChatPerplexity,
    history: List[ChatTurn],
    user_input: str,
    system_prompt: str,
) -> str:
    messages = to_langchain_messages(history, system_prompt)
    messages.append(HumanMessage(content=user_input))
    response = llm.invoke(messages)

    if isinstance(response.content, str):
        return response.content.strip()

    chunks: List[str] = []
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    chunks.append(text)
    return "\n".join(chunks).strip() or "답변을 생성하지 못했습니다."


def render_app() -> None:
    logger = setup_logger()
    config = AppConfig()

    st.title(config.title)

    init_session_state()

    try:
        pplx_key = load_pplx_api_key()
        _ = OpenAI  # SDK import 유지
        _ = ChatOpenAI  # import 유지
        _ = ChatOllama  # 요구 import 유지 목적
    except Exception as exc:
        logger.exception("초기화 실패")
        st.error(f"초기화 중 오류가 발생했습니다: {exc}")
        st.stop()

    llm = ChatPerplexity(
        model=config.model_name,
        pplx_api_key=pplx_key,
        temperature=0,
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("인터넷에서 검색할 질문을 입력하세요.")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("검색 및 답변 생성 중..."):
            try:
                answer = generate_answer(
                    llm=llm,
                    history=st.session_state.chat_history[:-1],
                    user_input=user_input,
                    system_prompt=config.system_prompt,
                )
            except Exception as exc:
                logger.exception("답변 생성 실패")
                answer = f"답변 생성 중 오류가 발생했습니다: {exc}"
            st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})


def main() -> None:
    config = AppConfig()
    st.set_page_config(page_title=config.title, page_icon="🌐", layout="centered")
    render_app()


if __name__ == "__main__":
    main()
