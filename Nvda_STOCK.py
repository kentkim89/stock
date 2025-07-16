import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="NVIDIA 주가 모니터링",
    page_icon=" NVIDIA_logo.png", # 로고 이미지를 추가할 수 있습니다.
    layout="wide",
)

# --- 캐싱을 사용한 데이터 로딩 함수 ---
# yfinance의 요청 횟수를 줄이기 위해 캐싱을 사용합니다. TTL(Time To Live)을 설정하여 일정 시간마다 데이터를 갱신합니다.
@st.cache_data(ttl=60) # 60초마다 데이터 갱신
def get_stock_data(ticker):
    """지정된 티커의 주식 데이터를 가져옵니다."""
    stock = yf.Ticker(ticker)
    history = stock.history(period="1d", interval="1m") # 1일간의 1분봉 데이터
    info = stock.info
    news = stock.news
    return history, info, news

# --- UI ---
st.title("엔비디아(NVDA) 주식 실시간 모니터링")
st.write(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- 사이드바 설정 ---
st.sidebar.header("알림 설정")
high_alert_price = st.sidebar.number_input("고점 알림 가격 ($)", min_value=0.0, format="%.2f")
low_alert_price = st.sidebar.number_input("저점 알림 가격 ($)", min_value=0.0, format="%.2f")

# --- 데이터 로딩 및 표시 ---
ticker = "NVDA"
try:
    history_data, info, news = get_stock_data(ticker)

    if not history_data.empty:
        # --- 최신 가격 및 변동 정보 ---
        latest_price = history_data['Close'].iloc[-1]
        previous_close = info.get('previousClose', 0)
        price_change = latest_price - previous_close
        percent_change = (price_change / previous_close) * 100 if previous_close else 0

        # 3개의 컬럼으로 정보 표시
        col1, col2, col3 = st.columns(3)
        col1.metric("현재가", f"${latest_price:,.2f}", f"{price_change:,.2f} ({percent_change:.2f}%)")
        col2.metric("장중 고가", f"${history_data['High'].max():,.2f}")
        col3.metric("장중 저가", f"${history_data['Low'].min():,.2f}")

        # --- 알림 기능 ---
        if high_alert_price > 0 and latest_price >= high_alert_price:
            st.success(f"📈 **고점 도달 알림:** 현재가가 설정하신 ${high_alert_price:,.2f} 이상입니다.")
        if low_alert_price > 0 and latest_price <= low_alert_price:
            st.warning(f"📉 **저점 도달 알림:** 현재가가 설정하신 ${low_alert_price:,.2f} 이하입니다.")

        # --- 실시간 주가 차트 ---
        st.subheader("실시간 주가 차트 (1분봉)")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=history_data.index,
                                     open=history_data['Open'],
                                     high=history_data['High'],
                                     low=history_data['Low'],
                                     close=history_data['Close'],
                                     name='실시간 주가'))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 종합 평가 및 요약 ---
        st.subheader("종합 평가 및 요약")
        col1_info, col2_info = st.columns([2, 1])

        with col1_info:
            st.write("#### 🏢 기업 개요")
            st.write(info.get('longBusinessSummary', '정보 없음'))

        with col2_info:
            st.write("#### 📊 주요 재무 정보")
            st.write(f"**시가총액:** ${info.get('marketCap', 0):,}")
            st.write(f"**52주 최고가:** ${info.get('fiftyTwoWeekHigh', 0):,.2f}")
            st.write(f"**52주 최저가:** ${info.get('fiftyTwoWeekLow', 0):,.2f}")
            st.write(f"**주가수익비율(PER):** {info.get('trailingPE', 0):.2f}")
            st.write(f"**배당수익률:** {info.get('dividendYield', 0) * 100:.2f}%")

        # --- 관련 뉴스 ---
        st.subheader("📰 최신 뉴스")
        if news:
            for item in news[:5]:  # 최근 5개 뉴스
                st.write(f"[{item['title']}]({item['link']}) - *{item['publisher']}*")
        else:
            st.write("최신 뉴스를 가져올 수 없습니다.")

    else:
        st.error("데이터를 가져오는 데 실패했습니다. 잠시 후 다시 시도해주세요.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
    st.info("yfinance API의 요청 제한일 수 있습니다. 잠시 후 페이지를 새로고침 해주세요.")

# 페이지 하단에 자동 새로고침 버튼 추가 (선택 사항)
if st.button('수동으로 새로고침'):
    st.rerun()
