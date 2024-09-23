import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Alpha Vantage API 설정
API_KEY = "A3JTG8U1BPPI5NKD"
STOCK_SYMBOL = "AAPL"

def get_stock_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data

# Streamlit 앱 제목 설정
st.title(f'{STOCK_SYMBOL} 주식 차트')

# 데이터 가져오기
data = get_stock_data(STOCK_SYMBOL)

# 데이터프레임으로 변환
df = pd.DataFrame(data['Time Series (Daily)']).T
df = df.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
df = df.astype(float)
df.index = pd.to_datetime(df.index)
df = df.sort_index(ascending=True)

# 최근 30일 데이터 선택
df = df.last('30D')

# 차트 생성
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('캔들스틱', '거래량'), row_heights=[0.7, 0.3])

# 캔들스틱 차트 추가
fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='캔들스틱'),
                row=1, col=1)

# 거래량 차트 추가
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(0, 0, 255, 0.5)'), row=2, col=1)

# 레이아웃 설정
fig.update_layout(
    title=f'{STOCK_SYMBOL} 주식 차트 (최근 30일)',
    yaxis_title='주가',
    xaxis_rangeslider_visible=False,
    height=800,
    showlegend=False
)

fig.update_yaxes(title_text="거래량", row=2, col=1)

# Streamlit에 차트 표시
st.plotly_chart(fig, use_container_width=True)

# 데이터 테이블 표시 (옵션)
if st.checkbox('원본 데이터 보기'):
    st.write(df)