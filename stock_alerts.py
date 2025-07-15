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

# 변경: get_undervalued_stocks 캐싱 제거, 메인에서 루프 실행으로 진행률 표시
def get_sp500_tickers(max_screen_stocks):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]['Symbol'].tolist()[:max_screen_stocks]

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

# 사이드바 설정 - default max_screen_stocks를 50으로 낮춤
with st.sidebar:
    st.title("⚙️ 설정")
    portfolio = st.text_input("보유 주식 티커 (콤마로 구분) 📊", "").split(',')
    max_screen_stocks = st.slider("스크리닝 최대 주식 수 (rate limit 방지)", 10, 200, 50)
    per_threshold = st.slider("저평가 PER 임계값", 5.0, 30.0, 15.0, help="Forward PE 기준")
    volume_threshold = st.slider("거래량 증가 (%)", 10, 200, 50)
    rsi_period = st.slider("RSI 기간", 5, 30, 14)
    rsi_oversold = st.slider("RSI 과매도 (<)", 10, 50, 30)
    rsi_overbought = st.slider("RSI 과매수 (>)", 50, 90, 70)
    stop_loss_threshold = st.slider("스탑로스 (%)", -10, -1, -5)
    sender_email = st.text_input("발신자 이메일 📧")
    sender_pw = st.text_input("비밀번호 🔑", type="password")
    receiver_email = st.text_input("수신자 이�멜")
    auto_refresh = st.toggle("실시간 모니터링 (1분) 🔄", value=True)

# 저평가 주식 스크리닝 with 진행률 바
st.write("저평가 주식 스크리닝 중... ⏳")
progress_bar = st.progress(0)
sp500 = get_sp500_tickers(max_screen_stocks)
undervalued_stocks = []
company_names_eng = {}  # 추가: 영어 회사명
company_names_kor = {  # 추가: 한글 이름 (부분, 웹 검색에서 추출)
    'AAPL': '애플',
    'MSFT': '마이크로소프트',
    'AMZN': '아마존',
    'NVDA': '엔비디아',
    'GOOGL': '알파벳 (A종)',
    'TSLA': '테슬라',
    'GOOG': '알파벳 (C종)',
    'META': '메타 플랫폼즈',
    'BRK.B': '버크셔 해서웨이 (B종)',
    'AVGO': '브로드컴',
    'LLY': '일라이 릴리',
    'JPM': 'JP모건 체이스',
    'UNH': '유나이티드헬스 그룹',
    'V': '비자',
    'MA': '마스터카드',
    'XOM': '엑슨모빌',
    'HD': '홈 디포',
    'PG': '프록터 & 갬블',
    'MRK': '머크',
    'MCD': '맥도날드',
    'LIN': '린데',
    'GE': 'GE 에어로스페이스',
    'PEP': '펩시코',
    'MO': '알트리아',
    'CB': '처브',
    'PNC': 'PNC 파이낸셜 서비스 그룹',
    'USB': 'US 뱅코프',
    'CME': 'CME 그룹',
    # 더 많은 추가 가능, 전체 목록은 별도 CSV 추천
}
for i, ticker in enumerate(sp500):
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
    progress_bar.progress((i + 1) / len(sp500))
st.success(f"스크리닝된 주식: {len(undervalued_stocks)}개 (상위 10: {', '.join(undervalued_stocks[:10])} ...)")

# df 정의를 탭 밖으로 이동 (NameError 해결)
df = get_stock_data(undervalued_stocks + [p.strip() for p in portfolio if p.strip()], rsi_period)

# 회사명 추가
if not df.empty:
    df['English Name'] = df.index.map(company_names_eng.get)
    df['Korean Name'] = df.index.map(company_names_kor.get)  # N/A if not in dict

# 탭 구조
tab1, tab2, tab3, tab4 = st.tabs(["🔔 알림", "💼 포트폴리오", "📊 백테스트", "📉 차트"])

with tab1:
    st.subheader("실시간 알림")
    if not df.empty:
        st.dataframe(df.style.background_gradient(cmap='viridis'))  # 회사명 포함 표시

        declined_stocks = df[df['Change (%)'] < 0]
        if not declined_stocks.empty:
            st.markdown('<div class="error">⚠️ 가격 하락 알림!</div>', unsafe_allow_html=True)
            for ticker, row in declined_stocks.iterrows():
                st.write(f"📉 {ticker} ({row['Korean Name']}): {row['Change (%)']:.2f}% 하락")

        volume_increased_stocks = df[df['Volume Change (%)'] > volume_threshold]
        if not volume_increased_stocks.empty:
            st.markdown('<div class="error">⚠️ 거래량 증가 알림!</div>', unsafe_allow_html=True)
            for ticker, row in volume_increased_stocks.iterrows():
                st.write(f"📈 {ticker} ({row['Korean Name']}): {row['Volume Change (%)']:.2f}% 증가")

        buy_signals = df[(df['Change (%)'] < 0) & (df['Volume Change (%)'] > volume_threshold) & (df['RSI'] < rsi_oversold) & (df['SMA50'] > df['SMA200'])]
        if not buy_signals.empty:
            st.markdown('<div class="warning">💰 매수 기회 알림!</div>', unsafe_allow_html=True)
            for ticker, row in buy_signals.iterrows():
                hist = yf.download(ticker, period="1y")
                predicted, pred_change = predict_price(hist)
                st.write(f"🟢 {ticker} ({row['Korean Name']}): RSI {row['RSI']:.2f}, 예측 {pred_change:.2f}%")
                if sender_email and receiver_email and sender_pw:
                    send_email(sender_email, sender_pw, receiver_email, f"{ticker} 매수", f"예측: {pred_change:.2f}%")

        sell_signals = df[(df['Change (%)'] > 0) & (df['RSI'] > rsi_overbought) & (df['SMA50'] < df['SMA200']) | (df['Change (%)'] < stop_loss_threshold)]
        if not sell_signals.empty:
            st.markdown('<div class="warning">💸 매도 기회 알림!</div>', unsafe_allow_html=True)
            for ticker, row in sell_signals.iterrows():
                st.write(f"🔴 {ticker} ({row['Korean Name']}): RSI {row['RSI']:.2f}")
                if sender_email and receiver_email and sender_pw:
                    send_email(sender_email, sender_pw, receiver_email, f"{ticker} 매도", "매도 타이밍!")

with tab2:
    st.subheader("포트폴리오 관리")
    if portfolio:
        for tick in portfolio:
            if tick.strip() in df.index:
                row = df.loc[tick.strip()]
                color = "green" if row['Change (%)'] > 0 else "red"
                st.metric(label=f"{tick} ({row['Korean Name']})", value=f"${row['Current Price']:.2f}", delta=f"{row['Change (%)']:.2f}%", delta_color=color)

with tab3:
    st.subheader("백테스트 결과")
    selected_ticker = st.selectbox("주식 선택", undervalued_stocks + portfolio)
    if selected_ticker:
        hist = yf.download(selected_ticker, period="1y")
        return_pct, back_hist = backtest_strategy(hist, rsi_period, rsi_oversold, rsi_overbought, 50, 200)
        st.metric("수익률", f"{return_pct:.2f}%")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=back_hist.index.to_numpy(), y=back_hist['Close'].to_numpy(), label='Price', ax=ax)  # 변경: to_numpy() for 1D
        sns.lineplot(x=back_hist.index.to_numpy(), y=back_hist['SMA_short'].to_numpy(), label='SMA50', ax=ax)
        sns.lineplot(x=back_hist.index.to_numpy(), y=back_hist['SMA_long'].to_numpy(), label='SMA200', ax=ax)
        st.pyplot(fig)

with tab4:
    st.subheader("차트 분석")
    if selected_ticker:
        hist = yf.download(selected_ticker, period="1y")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=hist.index.to_numpy(), y=hist['Close'].to_numpy(), label='Price', color='blue', ax=ax1)  # 변경: to_numpy()
        sns.lineplot(x=hist.index.to_numpy(), y=calculate_sma(hist, 50).to_numpy(), label='SMA50', color='green', ax=ax1)
        sns.lineplot(x=hist.index.to_numpy(), y=calculate_sma(hist, 200).to_numpy(), label='SMA200', color='red', ax=ax1)
        ax2 = ax1.twinx()
        sns.lineplot(x=hist.index.to_numpy(), y=calculate_rsi(hist).to_numpy(), label='RSI', color='purple', linestyle='--', ax=ax2)
        st.pyplot(fig)

# 푸터 - unchanged
st.markdown("---")
st.info("데이터: Yahoo Finance | 2025 개발 by Grok")

# 실시간 모니터링 - unchanged
if auto_refresh:
    time.sleep(60)
    st.rerun()
