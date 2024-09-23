import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# Streamlit 앱 제목 설정
st.title('애플 주식 캔들 차트')

# CSV 파일을 읽어옵니다.
file_path = './AAPL.csv'  # 파일 경로를 여기에 입력하세요
data = pd.read_csv(file_path)

# 날짜 열을 datetime 형식으로 변환합니다.
data['Date'] = pd.to_datetime(data['Unnamed: 0'])
data.set_index('Date', inplace=True)

# 캔들스틱 차트에 필요한 데이터 준비
ohlc_data = data[['1. open', '2. high', '3. low', '4. close']].copy()
ohlc_data.reset_index(inplace=True)
ohlc_data['Date'] = mdates.date2num(ohlc_data['Date'])
ohlc_data = ohlc_data[['Date', '1. open', '2. high', '3. low', '4. close']]

# 캔들스틱 차트 그리기
fig, ax = plt.subplots(figsize=(14, 7))

for idx, row in ohlc_data.iterrows():
    if row['1. open'] < row['4. close']:
        color = 'g'
        lower = row['1. open']
        height = row['4. close'] - row['1. open']
    else:
        color = 'r'
        lower = row['4. close']
        height = row['1. open'] - row['4. close']
    
    # 캔들스틱 몸통 그리기
    ax.add_patch(plt.Rectangle((row['Date'] - 0.2, lower), 0.4, height, color=color, alpha=0.8))
    # 캔들스틱 고가-저가 선 그리기
    ax.plot([row['Date'], row['Date']], [row['3. low'], row['2. high']], color=color, linewidth=1.5)

# x축 포맷 설정
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
fig.autofmt_xdate()

ax.set_title('Apple Stock Price Chart')
ax.set_xlabel('date')
ax.set_ylabel('price (USD)')
plt.grid(True)

# Streamlit을 사용하여 그래프 표시
st.pyplot(fig)