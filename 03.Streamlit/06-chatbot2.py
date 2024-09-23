#
# 화면에 간단한 챗봇을 만들어 봅시다
# 그리고 이 챗봇의 문제점을 알아봅시다
#

from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Streamlit 앱 제목 설정
st.title("간단한 챗봇 만들기")
input = st.text_input("질문을 해보세요", "")

if len(input) > 0:
    st.write(f"입력: {input}")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # 시스템 메시지와 사용자 입력을 포함한 메시지 리스트 생성
    messages = [
        {"role": "system", "content": "당신은 친절한 여행 전문가입니다."},
        {"role": "user", "content": input}
    ]
    
    # LangChain을 사용하여 챗봇 응답 생성
    response = llm.invoke(messages).content
    
    # 생성된 응답 표시
    st.write(response)
