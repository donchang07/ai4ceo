"""실시간 디지털 시계 (Streamlit). 통합 앱(one.py)에서는 render_app()만 호출됩니다."""

from __future__ import annotations

import time as time_module
from datetime import datetime

import streamlit as st


def render_app() -> None:
    now = datetime.now()
    date_text = now.strftime("%Y-%m-%d (%a)")
    time_text = now.strftime("%H:%M:%S")

    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap" rel="stylesheet">
        <style>
            .clock-root {
                background-color: #000000;
                min-height: 320px;
                width: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-start;
                padding-top: 2.5rem;
                box-sizing: border-box;
            }
            .clock-date {
                font-family: 'Orbitron', monospace;
                font-size: 1.35rem;
                color: #ffff00;
                letter-spacing: 0.12em;
                margin-bottom: 1rem;
                text-align: center;
            }
            .clock-time {
                font-family: 'Orbitron', monospace;
                font-size: clamp(2rem, 6vw, 3.5rem);
                font-weight: 700;
                color: #00ff00;
                letter-spacing: 0.18em;
                text-align: center;
                text-shadow: 0 0 12px rgba(0, 255, 0, 0.45);
            }
        </style>
        <div class="clock-root">
            <div class="clock-date">"""
        + date_text
        + """</div>
            <div class="clock-time">"""
        + time_text
        + """</div>
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
