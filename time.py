"""실시간 디지털 시계 (Streamlit).

통합 앱(one.py)은 importlib로 이 모듈을 로드하여 render_app()만 호출합니다.
표준 라이브러리 time과의 충돌을 피하기 위해 time as time_module 로 임포트합니다.
"""

from __future__ import annotations

import time as time_module
from datetime import datetime

import streamlit as st


def render_app() -> None:
    now = datetime.now()
    date_text = now.strftime("%Y년 %m월 %d일 (%a)")
    time_text = now.strftime("%H:%M:%S")

    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700&display=swap" rel="stylesheet">
        <style>
            .stApp {{
                background-color: #000000;
            }}
            .clock-wrap {{
                position: fixed;
                top: 4rem;
                left: 50%;
                transform: translateX(-50%);
                width: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                z-index: 9999;
                pointer-events: none;
            }}
            .clock-date {{
                font-family: 'Share Tech Mono', 'Orbitron', monospace;
                font-size: 2rem;
                color: #ffff00;
                letter-spacing: 0.15em;
                margin-bottom: 1.2rem;
                text-shadow: 0 0 8px rgba(255, 255, 0, 0.4);
            }}
            .clock-time {{
                font-family: 'Share Tech Mono', 'Orbitron', monospace;
                font-size: clamp(3rem, 9vw, 6rem);
                font-weight: 700;
                color: #00ff00;
                letter-spacing: 0.2em;
                text-shadow: 0 0 18px rgba(0, 255, 0, 0.6);
            }}
        </style>
        <div class="clock-wrap">
            <div class="clock-date">{date_text}</div>
            <div class="clock-time">{time_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    time_module.sleep(1)
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="실시간 시계", page_icon="🕐", layout="centered")
    render_app()


if __name__ == "__main__":
    main()
