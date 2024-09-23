#
# 이것은 아주 간단한 Streamlit 프로그램을 만들어보는 예제입니다.

import streamlit as st

# 타이틀 적용 예시
st.title("Langchain과 LLM을 활용한 application 개발 :100:")

# 특수 이모티콘 삽입 예시
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# caption은 설명을 추가할 때 사용

st.caption(
    """
    쉽지는 않겠지만, 함께 노력하면 가능할 거에요 :smile:
    여유있는 커피 한 잔~~ :coffee:과 함께
           앞으로 화이팅!!! :muscle:
           """
)

# Header 적용
st.header("1. API key setting :sparkles:")

# Subheader 적용
st.subheader("1.1. OpenAI API key setting")


# 코드 표시
sample_code = """
# OpenAI API 테스트

from openai import OpenAI
client = OpenAI()
try:
    client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {
            "role": "system",
            "content": "당신은 친절한 AI Assistance 입니다.",
        },
        {
            "role": "user",
            "content": "say this is test",
        },
        ],
        max_tokens=2024
    )
    print("OpenAI API 성공")
except Exception:
    print("OpenAI API 실패")
"""
st.code(sample_code, language="python")

# 일반 텍스트
st.text(
    """
API 키 보안: API 키는 민감한 정보이므로 코드에 하드코딩하지 말고 환경 변수나 비밀 관리 시스템을 사용하여 관리하세요.
요금제 확인: API 사용량에 따라 비용이 발생할 수 있으므로, OpenAI 계정의 요금제와 사용량을 정기적으로 확인하세요..
"""
)

# 마크다운 문법 지원
st.markdown(
    """
# 헤드라인1 (마크다운 문법) 
## 헤드라인2
### 헤드라인3
- bullet1
- bullet2

    """
)


# 컬러코드: blue, green, orange, red, violet
st.markdown(
    "텍스트의 색상을 :red[빨간색]으로, 그리고 **:blue[파란색]** 볼드체로 설정할 수 있습니다."
)

st.markdown(
    ":green[$\sqrt{x^2+y^2}=1$] 와 같이 latex 문법의 수식 표현도 가능합니다 :pencil:"
)


