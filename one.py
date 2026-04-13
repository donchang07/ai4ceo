from __future__ import annotations

import importlib.util
from pathlib import Path

import streamlit as st

import chatbot
import internet
import rag

APP_LABELS = ["시간", "챗봇", "인터넷검색", "RAG"]

# --- CSS 스타일 (ref.py 기반) ---
CUSTOM_CSS = """
<style>
/* --- 반응형 기본 설정 --- */
.block-container {
    padding: clamp(1rem, 3vw, 3rem) clamp(1rem, 5vw, 5rem) !important;
    max-width: 100% !important;
}

/* 헤딩 — clamp()로 뷰포트에 비례 */
h1 {
    font-size: clamp(1.1rem, 2.5vw, 1.4rem) !important;
    font-weight: 600 !important;
    color: #ff69b4 !important;
}
h2 {
    font-size: clamp(1rem, 2.2vw, 1.2rem) !important;
    font-weight: 600 !important;
    color: #ffd700 !important;
}
h3 {
    font-size: clamp(0.95rem, 2vw, 1.1rem) !important;
    font-weight: 600 !important;
    color: #1f77b4 !important;
}

/* 채팅 메시지 — 반응형 폰트 */
.stChatMessage,
.stChatMessage p,
.stChatMessage ul, .stChatMessage ol,
.stChatMessage li {
    font-size: clamp(0.85rem, 1.8vw, 0.95rem) !important;
    line-height: 1.5 !important;
}
.stChatMessage p  { margin: 0.5rem 0 !important; }
.stChatMessage ul, .stChatMessage ol { margin: 0.5rem 0 !important; }
.stChatMessage li { margin: 0.3rem 0 !important; }
.stChatMessage blockquote {
    font-size: clamp(0.85rem, 1.8vw, 0.95rem) !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    padding-left: 1rem !important;
    border-left: 3px solid #e0e0e0 !important;
}
.stChatMessage code {
    font-size: clamp(0.8rem, 1.6vw, 0.9rem) !important;
    background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
}
.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* 버튼 */
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

/* 사이드바 — 작은 화면에서 너비 조정 */
[data-testid="stSidebar"] {
    min-width: 200px !important;
}

/* 모바일 (≤768px) */
@media (max-width: 768px) {
    .block-container {
        padding: 0.5rem 0.8rem !important;
    }
    [data-testid="stSidebar"] {
        min-width: 160px !important;
    }
}
</style>
"""

TITLE_HTML = """
<div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
    <h1 style="font-size: clamp(2rem, 8vw, 5rem) !important; font-weight: bold; margin: 0; line-height: 1.2;">
        <span style="color: #1f77b4;">4-in-1</span>
        <span style="color: #ffd700;">AI 앱</span>
    </h1>
</div>
"""

APP_DESCRIPTIONS = {
    "시간": "실시간 디지털 시계를 표시합니다.",
    "챗봇": "OpenAI 모델과 대화합니다.",
    "인터넷검색": "Perplexity를 사용하여 인터넷에서 최신 정보를 검색합니다.",
    "RAG": "PDF 파일을 업로드하여 문서 기반 답변을 받습니다.",
}

APP_ICONS = {
    "시간": "🕐",
    "챗봇": "🤖",
    "인터넷검색": "🌐",
    "RAG": "📚",
}


def _load_local_time_module():
    time_file = Path(__file__).with_name("time.py")
    spec = importlib.util.spec_from_file_location("local_time_app", time_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("time.py 모듈을 불러오지 못했습니다.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_usage_stats() -> None:
    if "usage_stats" not in st.session_state:
        st.session_state.usage_stats = {label: 0 for label in APP_LABELS}


def _reset_all_conversations() -> None:
    keys_to_clear = [
        "chatbot_history",
        "chat_history",
        "app_state",
        "retriever",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="4-in-1 AI 앱", page_icon="🧩", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    _ensure_usage_stats()

    # --- 사이드바 ---
    with st.sidebar:
        st.markdown('<h2>앱 선택</h2>', unsafe_allow_html=True)
        selected = st.radio(
            "메뉴",
            APP_LABELS,
            label_visibility="collapsed",
        )

        icon = APP_ICONS[selected]
        desc = APP_DESCRIPTIONS[selected]
        st.info(f"{icon} {desc}")

        st.markdown("---")
        if st.button("대화 초기화", use_container_width=True):
            _reset_all_conversations()

        # 사용량 통계
        st.markdown("---")
        st.markdown('<h3>사용량 통계</h3>', unsafe_allow_html=True)
        for label in APP_LABELS:
            count = st.session_state.usage_stats[label]
            icon = APP_ICONS[label]
            st.text(f"{icon} {label}: {count}회")

        # 현재 설정
        st.markdown("---")
        st.markdown('<h3>현재 설정</h3>', unsafe_allow_html=True)
        st.text(f"선택된 앱: {selected}")

    # --- 메인 영역 ---
    st.markdown(TITLE_HTML, unsafe_allow_html=True)

    st.session_state.usage_stats[selected] += 1

    if selected == "시간":
        time_app = _load_local_time_module()
        time_app.render_app()
    elif selected == "챗봇":
        chatbot.render_app()
    elif selected == "인터넷검색":
        internet.render_app()
    else:
        rag.render_app()


if __name__ == "__main__":
    main()
