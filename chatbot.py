from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class ChatTurn(TypedDict):
    role: str
    content: str


class ChatbotConfig(BaseModel):
    title: str = Field(default="My First Chatbot")
    model_name: str = Field(default="gpt-4o")
    system_prompt: str = Field(default="You are a helpful and kind assistant.")


def _init_state() -> None:
    if "chatbot_history" not in st.session_state:
        st.session_state.chatbot_history: list[ChatTurn] = []


def _to_messages(history: list[ChatTurn], system_prompt: str) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


def render_app() -> None:
    config = ChatbotConfig()
    st.title(config.title)
    _ = ChatOllama  # 요구 import 유지

    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY를 .env 파일에 설정해 주세요.")
        st.stop()

    _ = OpenAI(api_key=api_key)  # SDK 초기화 확인
    llm = ChatOpenAI(model=config.model_name, api_key=api_key, temperature=0)

    _init_state()

    for msg in st.session_state.chatbot_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("메시지를 입력하세요.")
    if not user_input:
        return

    st.session_state.chatbot_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    messages = _to_messages(st.session_state.chatbot_history[:-1], config.system_prompt)
    messages.append(HumanMessage(content=user_input))
    response = llm.invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)

    st.session_state.chatbot_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)


def main() -> None:
    st.set_page_config(page_title="My First Chatbot", page_icon="🤖", layout="centered")
    render_app()


if __name__ == "__main__":
    main()
