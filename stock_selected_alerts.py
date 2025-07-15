import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import time
import smtplib
from email.mime.text import MIMEText
import ssl

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

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.dropna()  # NaN 제거
    return rsi

def calculate_sma(data, window):
    return data['Close'].rolling(window=window, min_periods=1).mean()

@st.cache_data(ttl=300)
def get_stock_data(tickers, rsi_period, sma_short=50, sma_long=200):
    if not tickers:
        return pd.DataFrame()
    multi_data = yf.download(tickers, period="1y", group_by='ticker', auto_adjust=True, threads=False)
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
                rsi = calculate_rsi(hist, rsi_period).iloc[-1] if not calculate_rsi(hist, rsi_period).empty else np.nan
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

# 헤더
st.header("📈 주식 알림 대시보드", divider='rainbow')

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 설정")
    portfolio = st.text_input("보유 주식 티커 (콤마로 구분) 📊", "").split(',')
    use_screening = st.toggle("S&P 500 저평가 스크리닝 사용", value=False)  # 기본 off
    if use_screening:
        max_screen_stocks = st.slider("스크리닝 최대 주식 수", 10, 200, 50)
        per_threshold = st.slider("저평가 PER 임계값", 5.0, 30.0, 15.0)
    volume_threshold = st.slider("거래량 증가 (%)", 10, 200, 50)
    rsi_period = st.slider("RSI 기간", 5, 30, 14)
    rsi_oversold = st.slider("RSI 과매도 (<)", 10, 50, 30)
    rsi_overbought = st.slider("RSI 과매수 (>)", 50, 90, 70)
    stop_loss_threshold = st.slider("스탑로스 (%)", -10, -1, -5)
    sender_email = st.text_input("발신자 이메일 📧")
    sender_pw = st.text_input("비밀번호 🔑", type="password")
    receiver_email = st.text_input("수신자 이메일")
    auto_refresh = st.toggle("실시간 모니터링 (1분) 🔄", value=True)

# S&P 500 티커 목록 로드
sp500_tickers = get_sp500_tickers(500)  # 전체 로드

# 포트폴리오 티커 S&P 500으로 필터
portfolio = [p.strip() for p in portfolio if p.strip() in sp500_tickers]

# 저평가 주식 스크리닝
undervalued_stocks = []
if use_screening:
    st.write("저평가 주식 스크리닝 중... ⏳")
    progress_bar = st.progress(0)
    undervalued_stocks = []
    company_names_eng = {}
    company_names_kor = {  # 한글 매핑
        'AAPL': '애플',
        # ... (이전 목록)
    }
    for i, ticker in enumerate(sp500_tickers[:max_screen_stocks]):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            english_name = info.get('longName', 'N/A')
            if 'forwardPE' in info and info['forwardPE'] < per_threshold:
                undervalued_stocks.append(ticker)
                company_names_eng[ticker] = english_name
        except:
            pass
        time.sleep(0.2)
        progress_bar.progress((i + 1) / max_screen_stocks)
    st.success(f"스크리닝된 주식: {len(undervalued_stocks)}개")

# df 로드
tickers_to_fetch = undervalued_stocks + portfolio
df = get_stock_data(tickers_to_fetch, rsi_period)

# 회사명 추가 (Korean Name N/A 시 English Name 사용)
if not df.empty:
    df['English Name'] = df.index.map(lambda t: yf.Ticker(t).info.get('longName', 'N/A'))
    df['Korean Name'] = df.index.map(company_names_kor.get)
    df['Display Name'] = df.apply(lambda row: row['Korean Name'] if pd.notna(row['Korean Name']) else row['English Name'], axis=1)

# 탭 구조 (백테스트 제거)
tab1, tab2, tab3 = st.tabs(["🔔 알림", "💼 포트폴리오", "📉 차트"])

with tab1:
    st.subheader("실시간 알림")
    if not df.empty:
        st.dataframe(df.style.background_gradient(cmap='viridis'))

        declined_stocks = df[df['Change (%)'] < 0]
        if not declined_stocks.empty:
            st.markdown('<div class="error">⚠️ 가격 하락 알림!</div>', unsafe_allow_html=True)
            for ticker, row in declined_stocks.iterrows():
                st.write(f"📉 {ticker} ({row['Display Name']}): {row['Change (%)']:.2f}% 하락")

        volume_increased_stocks = df[df['Volume Change (%)'] > volume_threshold]
        if not volume_increased_stocks.empty:
            st.markdown('<div class="error">⚠️ 거래량 증가 알림!</div>', unsafe_allow_html=True)
            for ticker, row in volume_increased_stocks.iterrows():
                st.write(f"📈 {ticker} ({row['Display Name']}): {row['Volume Change (%)']:.2f}% 증가")

        buy_signals = df[(df['Change (%)'] < 0) & (df['Volume Change (%)'] > 0) & (df['RSI'] < rsi_oversold) & (df['SMA50'] > df['SMA200'])]
        if not buy_signals.empty:
            st.markdown('<div class="warning">💰 매수 기회 알림!</div>', unsafe_allow_html=True)
            for ticker, row in buy_signals.iterrows():
                hist = yf.download(ticker, period="1y")
                predicted, pred_change = predict_price(hist)
                st.write(f"🟢 {ticker} ({row['Display Name']}): RSI {row['RSI']:.2f}, 예측 {pred_change:.2f}%")
                if sender_email and receiver_email and sender_pw:
                    send_email(sender_email, sender_pw, receiver_email, f"{ticker} 매수", f"예측: {pred_change:.2f}%")
        else:
            st.info("현재 매수 신호 없음.")

        sell_signals = df[(df['Change (%)'] > 0) & (df['RSI'] > rsi_overbought) & (df['SMA50'] < df['SMA200']) | (df['Change (%)'] < stop_loss_threshold)]
        if not sell_signals.empty:
            st.markdown('<div class="warning">💸 매도 기회 알림!</div>', unsafe_allow_html=True)
            for ticker, row in sell_signals.iterrows():
                st.write(f"🔴 {ticker} ({row['Display Name']}): RSI {row['RSI']:.2f}")
                if sender_email and receiver_email and sender_pw:
                    send_email(sender_email, sender_pw, receiver_email, f"{ticker} 매도", "매도 타이밍!")
        else:
            st.info("현재 매도 신호 없음.")

        st.subheader("알림 요약")
        summary = f"하락 주식: {len(declined_stocks)}개, 거래량 증가: {len(volume_increased_stocks)}개, 매수 신호: {len(buy_signals)}개, 매도 신호: {len(sell_signals)}개"
        st.write(summary)
    else:
        st.warning("데이터 없음. 티커 입력 확인.")

with tab2:
    st.subheader("포트폴리오 관리")
    if not portfolio:
        st.info("포트폴리오 티커 입력 (S&P 500만 지원).")
    elif not df.empty:
        portfolio_df = df.loc[[t for t in portfolio if t in df.index]]
        if not portfolio_df.empty:
            st.dataframe(portfolio_df.style.background_gradient(cmap='viridis'))
        else:
            st.warning("포트폴리오 데이터 없음.")
    st.subheader("포트폴리오 요약")
    if portfolio and not df.empty:
        changes = portfolio_df['Change (%)']
        avg_change = changes.mean() if not changes.empty else 0
        st.write(f"평균 변화율: {avg_change:.2f}%, 종목 수: {len(portfolio_df)}개")

with tab3:
    st.subheader("차트 분석")
    selected_ticker = st.selectbox("주식 선택", portfolio if portfolio else undervalued_stocks)
    if selected_ticker:
        hist = yf.download(selected_ticker, period="1y")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(hist.index, hist['Close'], label='Price', color='blue')
        ax1.plot(hist.index, calculate_sma(hist, 50), label='SMA50', color='green')
        ax1.plot(hist.index, calculate_sma(hist, 200), label='SMA200', color='red')
        ax2 = ax1.twinx()
        rsi = calculate_rsi(hist)
        ax2.plot(hist.index, rsi, label='RSI', color='purple', linestyle='--')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        st.pyplot(fig)

        st.subheader("차트 요약")
        current_rsi = rsi.iloc[-1] if not rsi.empty else np.nan
        current_rsi_str = f"{current_rsi:.2f}" if pd.notnull(current_rsi) else "N/A (데이터 부족)"
        st.write(f"{selected_ticker} 가격 추세: SMA 크로스오버와 RSI 확인. 현재 RSI: {current_rsi_str}")

st.markdown("---")
st.info("데이터: Yahoo Finance | 2025 개발 by Grok")

if auto_refresh:
    time.sleep(60)
    st.rerun()
