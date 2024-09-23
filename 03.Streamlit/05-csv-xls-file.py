import streamlit as st
import pandas as pd
import chardet

# 파일 업로드 버튼 (업로드 기능)
file = st.file_uploader("파일 선택(csv or excel)", type=["csv", "xls", "xlsx"])

def detect_encoding(file):
    # 파일의 시작 부분을 읽어서 인코딩을 감지합니다.
    rawdata = file.read(10000)
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    file.seek(0)  # 파일 포인터를 처음으로 되돌립니다.
    return encoding

# 파일이 정상 업로드 된 경우
if file is not None:
    ext = file.name.split(".")[-1]
    
    if ext == "csv":
        encoding = detect_encoding(file)
        df = pd.read_csv(file, encoding=encoding)
        st.write(f"감지된 인코딩: {encoding}")
        st.table(df)
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(file)
        st.table(df)
