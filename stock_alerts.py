import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import time
import smtplib
from email.mime.text import MIMEText
import ssl
from multiprocessing import cpu_count  # Kept but not used for parallel to avoid rate limits

# Streamlit 테마 커스터마이징 (화려한 UI) - unchanged
st.set_page_config(page_title="주식 알림 대시보드", page_icon="📈", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #f0f8ff; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; }
    .stSlider .st-ae { color: #2196F3; }
    .stTabs [data-baseweb="tab"] { font-weight: bold; }
    .error { background-color: #ffcccc; padding: 10px; border-radius: 5px; }
    .success { background-color: #ccffcc; padding: 10px; border-radius: 5px; }
    .warning { background-color: #fff3cd; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# 변경: multiprocessing 제거, sequential loop with delay for rate limit avoidance
@st.cache_data(ttl=86400)  # 변경: 캐시 TTL을 1일로 증가하여 반복 호출 줄임
def get_undervalued_stocks(per_threshold, max_screen_stocks):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500 = tables[0]['Symbol'].tolist()[:max_screen_stocks]  # 변경: max_screen_stocks로 제한
    undervalued = []
    for ticker in sp500:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if 'forwardPE' in info and info['forwardPE'] < per_threshold:
                undervalued.append(ticker)
        except:
            pass
        time.sleep(0.5)  # 변경: 각 호출 사이 0.5초 지연 추가 (rate limit 방지)
    return undervalued

# 변경: yf.download으로 bulk fetch (하나의 호출로 모든 티커 데이터 가져옴, threads=False로 sequential)
@st.cache_data(ttl=300)
def get_stock_data(tickers, rsi_period, sma_short=50, sma_long=200):
    if not tickers:
        return pd.DataFrame()
    multi_data = yf.download(tickers, period="1y", group_by='ticker', auto_adjust=True, threads=False)  # 변경: bulk download, threads=False
    data = {}
    for ticker in tickers:
        if ticker in multi_data.columns.levels[0]:
            hist = multi_data[ticker].dropna()
            if len(hist) >= max(rsi_period + 1, sma_long):
                prev_close = hist['Close'].iloc[-2]
                current_close = hist['Close'].iloc[-1]
                change = (current_close - prev_close) / prev_close * 100
                prev_volume = hist['Volume'].iloc[-2]
                current_volume = hist['Volume'].iloc[-1]
                volume_change = (current_volume - prev_volume) / prev_volume * 100 if prev_volume != 0 else 0
                rsi = calculate_rsi(hist, rsi_period).iloc[-1]
                sma50 = calculate_sma(hist, sma_short).iloc[-1]
                sma200 = calculate_sma(hist, sma_long).iloc[-1]
                data[ticker] = {
                    'Current Price': current_close,
                    'Previous Close': prev_close,
                    'Change (%)': change,
                    'Current Volume': current_volume,
                    'Previous Volume': prev_volume,
                    'Volume Change (%)': volume_change,
                    'RSI': rsi,
                    'SMA50': sma50,
                    'SMA200': sma200
                }
    return pd.DataFrame.from_dict(data, orient='index')

def predict_price(hist):
    hist['Day'] = np.arange(len(hist))
    X = hist[['Day']]
    y = hist['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    next_day = np.array([[len(hist)]])
    predicted = model.predict(next_day)[0]
    return predicted, (predicted - hist['Close'].iloc[-1]) / hist['Close'].iloc[-1] * 100

def send_email(sender_email, sender_pw, receiver_email, subject, body):
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_pw)
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = receiver_email
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"이메일 전송 실패: {e}")
        return False

def backtest_strategy(hist, rsi_period, rsi_oversold, rsi_overbought, sma_short, sma_long):
    hist['RSI'] = calculate_rsi(hist, rsi_period)
    hist['SMA_short'] = calculate_sma(hist, sma_short)
    hist['SMA_long'] = calculate_sma(hist, sma_long)
    hist['Signal'] = 0
    hist.loc[(hist['RSI'] < rsi_oversold) & (hist['SMA_short'] > hist['SMA_long']), 'Signal'] = 1
    hist.loc[(hist['RSI'] > rsi_overbought) & (hist['SMA_short'] < hist['SMA_long']), 'Signal'] = -1
    hist['Return'] = hist['Close'].pct_change()
    hist['Strategy_Return'] = hist['Return'] * hist['Signal'].shift(1)
    cumulative_return = (1 + hist['Strategy_Return']).cumprod() - 1
    return cumulative_return.iloc[-1] * 100, hist

# 헤더 - unchanged
st.header("📈 주식 알림 대시보드", divider='rainbow')

# 사이드바 설정 - 추가: max_screen_stocks 슬라이더 for rate limit control
with st.sidebar:
    st.title("⚙️ 설정")
    portfolio = st.text_input("보유 주식 티커 (콤마로 구분) 📊", "").split(',')
    max_screen_stocks = st.slider("스크리닝 최대 주식 수 (rate limit 방지)", 50, 500, 100)  # 추가: 사용자 조정 가능
    per_threshold = st.slider("저평가 PER 임계값", 5.0, 30.0, 15.0, help="Forward PE 기준")
    volume_threshold = st.slider("거래량 증가 (%)", 10, 200, 50)
    rsi_period = st.slider("RSI 기간", 5, 30, 14)
    rsi_oversold = st.slider("RSI 과매도 (<)", 10, 50, 30)
    rsi_overbought = st.slider("RSI 과매수 (>)", 50, 90, 70)
    stop_loss_threshold = st.slider("스탑로스 (%)", -10, -1, -5)
    sender_email = st.text_input("발신자 이메일 📧")
    sender_pw = st.text_input("비밀번호 🔑", type="password")
    receiver_email = st.text_input("수신자 이메일")
    auto_refresh = st.toggle("실시간 모니터링 (1분) 🔄", value=True)

# 저평가 주식 로딩 - unchanged
with st.spinner("저평가 주식 스크리닝 중... ⏳"):
    undervalued_stocks = get_undervalued_stocks(per_threshold, max_screen_stocks)
st.success(f"스크리닝된 주식: {len(undervalued_stocks)}개 (상위 10: {', '.join(undervalued_stocks[:10])} ...)")

# 탭 구조 - unchanged, but df fetch uses updated function
tab1, tab2, tab3, tab4 = st.tabs(["🔔 알림", "💼 포트폴리오", "📊 백테스트", "📉 차트"])

with tab1:
    st.subheader("실시간 알림")
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)
    df = get_stock_data(undervalued_stocks + [p.strip() for p in portfolio if p.strip()], rsi_period)
    if not df.empty:
        st.dataframe(df.style.background_gradient(cmap='viridis'))

        declined_stocks = df[df['Change (%)'] < 0]
        if not declined_stocks.empty:
            st.markdown('<div class="error">⚠️ 가격 하락 알림!</div>', unsafe_allow_html=True)
            for ticker, row in declined_stocks.iterrows():
                st.write(f"📉 {ticker}: {row['Change (%)']:.2f}% 하락")

        volume_increased_stocks = df[df['Volume Change (%)'] > volume_threshold]
        if not volume_increased_stocks.empty:
            st.markdown('<div class="error">⚠️ 거래량 증가 알림!</div>', unsafe_allow_html=True)
            for ticker, row in volume_increased_stocks.iterrows():
                st.write(f"📈 {ticker}: {row['Volume Change (%)']:.2f}% 증가")

        buy_signals = df[(df['Change (%)'] < 0) & (df['Volume Change (%)'] > volume_threshold) & (df['RSI'] < rsi_oversold) & (df['SMA50'] > df['SMA200'])]
        if not buy_signals.empty:
            st.markdown('<div class="warning">💰 매수 기회 알림!</div>', unsafe_allow_html=True)
            for ticker, row in buy_signals.iterrows():
                hist = yf.download(ticker, period="1y")  # 개별 hist for predict (rate limit ok, few calls)
                predicted, pred_change = predict_price(hist)
                st.write(f"🟢 {ticker}: RSI {row['RSI']:.2f}, 예측 {pred_change:.2f}%")
                if sender_email and receiver_email and sender_pw:
                    send_email(sender_email, sender_pw, receiver_email, f"{ticker} 매수", f"예측: {pred_change:.2f}%")

        sell_signals = df[(df['Change (%)'] > 0) & (df['RSI'] > rsi_overbought) & (df['SMA50'] < df['SMA200']) | (df['Change (%)'] < stop_loss_threshold)]
        if not sell_signals.empty:
            st.markdown('<div class="warning">💸 매도 기회 알림!</div>', unsafe_allow_html=True)
            for ticker, row in sell_signals.iterrows():
                st.write(f"🔴 {ticker}: RSI {row['RSI']:.2f}")
                if sender_email and receiver_email and sender_pw:
                    send_email(sender_email, sender_pw, receiver_email, f"{ticker} 매도", "매도 타이밍!")

with tab2:
    st.subheader("포트폴리오 관리")
    if portfolio:
        for tick in portfolio:
            if tick.strip() in df.index:
                row = df.loc[tick.strip()]
                color = "green" if row['Change (%)'] > 0 else "red"
                st.metric(label=tick, value=f"${row['Current Price']:.2f}", delta=f"{row['Change (%)']:.2f}%", delta_color=color)

with tab3:
    st.subheader("백테스트 결과")
    selected_ticker = st.selectbox("주식 선택", undervalued_stocks + portfolio)
    if selected_ticker:
        hist = yf.download(selected_ticker, period="1y")
        return_pct, back_hist = backtest_strategy(hist, rsi_period, rsi_oversold, rsi_overbought, 50, 200)
        st.metric("수익률", f"{return_pct:.2f}%")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=back_hist['Close'], label='Price', ax=ax)
        sns.lineplot(data=back_hist['SMA_short'], label='SMA50', ax=ax)
        sns.lineplot(data=back_hist['SMA_long'], label='SMA200', ax=ax)
        st.pyplot(fig)

with tab4:
    st.subheader("차트 분석")
    if selected_ticker:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=hist['Close'], label='Price', color='blue', ax=ax1)
        sns.lineplot(data=calculate_sma(hist, 50), label='SMA50', color='green', ax=ax1)
        sns.lineplot(data=calculate_sma(hist, 200), label='SMA200', color='red', ax=ax1)
        ax2 = ax1.twinx()
        sns.lineplot(data=calculate_rsi(hist), label='RSI', color='purple', style=True, dashes=[(2,2)], ax=ax2)
        st.pyplot(fig)

# 푸터 - unchanged
st.markdown("---")
st.info("데이터: Yahoo Finance | 2025 개발 by Grok")

# 실시간 모니터링 - unchanged
if auto_refresh:
    time.sleep(60)
    st.rerun()
