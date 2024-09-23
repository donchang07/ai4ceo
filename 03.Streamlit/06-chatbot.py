#
# 화면에 간단한 챗봇을 만들어 봅시다
# 그리고 이 챗봇의 문제점을 알아봅시다
#

from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.title("간단한 챗봇 만들기")
input = st.text_input("질문을 해보세요", "")

if len(input) > 0:
    st.write(f"입력: {input}")
# 
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "당신은 친절한 여행 전문가입니다.",
            },
            {
                "role": "user",
                "content": input,
            },
        ],
        max_tokens=2048,
    )
    st.write(response.choices[0].message.content)