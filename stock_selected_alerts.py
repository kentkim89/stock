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

# Streamlit 페이지 설정
st.set_page_config(page_title="주식 알림 대시보드", page_icon="📈", layout="wide")

# CSS 스타일 적용
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

# S&P 500 티커 목록을 가져오는 함수 (오류 수정)
@st.cache_data(ttl=86400)  # 하루에 한 번만 실행
def get_sp500_tickers():
    """
    Wikipedia에서 S&P 500 티커 목록을 가져옵니다.
    """
    try:
        # Wikipedia 페이지에서 S&P 500 목록이 포함된 테이블을 읽어옵니다.
        payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        # 첫 번째 테이블에서 티커 심볼을 가져옵니다.
        sp500_tickers = payload[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        return sp500_tickers
    except Exception as e:
        st.error(f"S&P 500 티커 목록을 가져오는 데 실패했습니다: {e}")
        # 실패 시 기본적인 티커 목록을 반환합니다.
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def calculate_rsi(data, period=14):
    """RSI(상대강도지수)를 계산합니다."""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.dropna()
    return rsi

def calculate_sma(data, window):
    """SMA(단순이동평균)를 계산합니다."""
    return data['Close'].rolling(window=window, min_periods=1).mean()

@st.cache_data(ttl=300) # 5분 캐시
def get_stock_data(tickers, rsi_period, sma_short=50, sma_long=200):
    """선택된 티커들의 주식 데이터를 가져오고 주요 지표를 계산합니다."""
    if not tickers:
        return pd.DataFrame()
    
    # yfinance를 통해 여러 티커의 데이터를 한 번에 다운로드합니다.
    multi_data = yf.download(tickers, period="1y", group_by='ticker', auto_adjust=True, threads=True)
    
    data = {}
    for ticker in tickers:
        # 다운로드한 데이터가 단일 티커에 대한 DataFrame인지 또는 여러 티커에 대한 MultiIndex DataFrame인지 확인합니다.
        if len(tickers) == 1:
            hist = multi_data.dropna()
        elif ticker in multi_data.columns.levels[0]:
            hist = multi_data[ticker].dropna()
        else:
            continue

        if len(hist) >= max(rsi_period + 1, sma_long):
            prev_close = hist['Close'].iloc[-2]
            current_close = hist['Close'].iloc[-1]
            change = (current_close - prev_close) / prev_close * 100
            
            prev_volume = hist['Volume'].iloc[-2] if len(hist) > 1 else 0
            current_volume = hist['Volume'].iloc[-1]
            volume_change = (current_volume - prev_volume) / prev_volume * 100 if prev_volume > 0 else 0
            
            rsi_series = calculate_rsi(hist, rsi_period)
            rsi = rsi_series.iloc[-1] if not rsi_series.empty else np.nan
            
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
    """선형 회귀 모델을 사용하여 다음 날의 주가를 예측합니다."""
    hist = hist.copy()
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
    """이메일 발송 기능을 처리합니다."""
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
        st.error(f"이메일 전송에 실패했습니다: {e}")
        return False

# --- UI 및 메인 로직 ---

st.header("📈 주식 알림 대시보드", divider='rainbow')

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 설정")
    # S&P 500 티커 목록 로드
    sp500_tickers = get_sp500_tickers()
    
    portfolio_input = st.text_input("보유 주식 티커 (콤마로 구분, 예: AAPL, MSFT) 📊", "")
    portfolio = [p.strip().upper() for p in portfolio_input.split(',') if p.strip()]

    use_screening = st.toggle("S&P 500 저평가 스크리닝 사용", value=False)
    if use_screening:
        max_screen_stocks = st.slider("스크리닝 최대 주식 수", 10, 500, 50)
        per_threshold = st.slider("저평가 PER 임계값", 5.0, 30.0, 15.0)

    volume_threshold = st.slider("거래량 급증 알림 기준 (%)", 10, 300, 100)
    rsi_period = st.slider("RSI 기간", 5, 30, 14)
    rsi_oversold = st.slider("RSI 과매도 기준 (<)", 10, 50, 30)
    rsi_overbought = st.slider("RSI 과매수 기준 (>)", 50, 90, 70)
    stop_loss_threshold = st.slider("스탑로스 알림 기준 (%)", -10, -1, -5)
    
    st.subheader("이메일 알림 설정")
    sender_email = st.text_input("발신자 이메일 (Gmail) 📧")
    sender_pw = st.text_input("발신자 앱 비밀번호 🔑", type="password")
    receiver_email = st.text_input("수신자 이메일")
    
    auto_refresh = st.toggle("실시간 모니터링 (1분마다 새로고침) 🔄", value=True)

# 저평가 주식 스크리닝 로직
undervalued_stocks = []
if use_screening:
    st.write("S&P 500 저평가 주식 스크리닝 중... ⏳")
    progress_bar = st.progress(0)
    
    screened_tickers = sp500_tickers[:max_screen_stocks]
    for i, ticker in enumerate(screened_tickers):
        try:
            stock_info = yf.Ticker(ticker).info
            # 'forwardPE'가 존재하고 None이 아니며, 설정된 임계값보다 낮은 경우
            if stock_info.get('forwardPE') and stock_info['forwardPE'] < per_threshold:
                undervalued_stocks.append(ticker)
        except Exception:
            # 특정 티커에서 오류 발생 시 건너뜁니다.
            pass
        time.sleep(0.1) # API 요청 속도 조절
        progress_bar.progress((i + 1) / len(screened_tickers))
    st.success(f"스크리닝 완료! {len(undervalued_stocks)}개의 저평가 주식을 찾았습니다.")

# 데이터를 가져올 전체 티커 목록 (중복 제거)
tickers_to_fetch = sorted(list(set(undervalued_stocks + portfolio)))

# 데이터 프레임 로드
if tickers_to_fetch:
    df = get_stock_data(tickers_to_fetch, rsi_period)

    # 회사명 추가
    if not df.empty:
        company_names = {}
        for ticker in df.index:
            try:
                # 회사 이름을 가져올 때도 캐시를 활용하면 더 효율적입니다.
                info = yf.Ticker(ticker).info
                company_names[ticker] = info.get('longName', ticker)
            except Exception:
                company_names[ticker] = ticker
        df['Company Name'] = df.index.map(company_names)
else:
    df = pd.DataFrame()

# 탭 구조
tab1, tab2, tab3 = st.tabs(["🔔 실시간 알림", "💼 포트폴리오", "📉 차트 분석"])

with tab1:
    st.subheader("종합 현황")
    if not df.empty:
        st.dataframe(df[['Company Name', 'Current Price', 'Change (%)', 'Current Volume', 'Volume Change (%)', 'RSI']].style.format({
            'Current Price': '${:,.2f}',
            'Change (%)': '{:,.2f}%',
            'Current Volume': '{:,}',
            'Volume Change (%)': '{:,.2f}%',
            'RSI': '{:.2f}'
        }).background_gradient(cmap='viridis', subset=['Change (%)', 'Volume Change (%)', 'RSI']))

        # 알림 로직
        declined_stocks = df[df['Change (%)'] < 0]
        volume_increased_stocks = df[df['Volume Change (%)'] > volume_threshold]
        buy_signals = df[(df['RSI'] < rsi_oversold) & (df['SMA50'] > df['SMA200'])]
        sell_signals = df[(df['RSI'] > rsi_overbought) & (df['SMA50'] < df['SMA200'])]
        stop_loss_signals = df[df['Change (%)'] < stop_loss_threshold]

        st.subheader("실시간 알림")
        if not (declined_stocks.empty and volume_increased_stocks.empty and buy_signals.empty and sell_signals.empty and stop_loss_signals.empty):
            # 매수 신호
            if not buy_signals.empty:
                st.markdown('<div class="warning">💰 매수 기회 알림! (RSI 과매도 & 골든 크로스)</div>', unsafe_allow_html=True)
                for ticker, row in buy_signals.iterrows():
                    st.write(f"🟢 {row['Company Name']} ({ticker}): RSI {row['RSI']:.2f}")

            # 매도 신호
            if not sell_signals.empty:
                st.markdown('<div class="warning">💸 매도 기회 알림! (RSI 과매수 & 데드 크로스)</div>', unsafe_allow_html=True)
                for ticker, row in sell_signals.iterrows():
                    st.write(f"🔴 {row['Company Name']} ({ticker}): RSI {row['RSI']:.2f}")

            # 스탑로스 신호
            if not stop_loss_signals.empty:
                st.markdown('<div class="error">🚨 스탑로스 알림!</div>', unsafe_allow_html=True)
                for ticker, row in stop_loss_signals.iterrows():
                    st.write(f"📉 {row['Company Name']} ({ticker}): {row['Change (%)']:.2f}% 하락")
            
            # 거래량 급증 신호
            if not volume_increased_stocks.empty:
                st.markdown('<div class="error">⚠️ 거래량 급증 알림!</div>', unsafe_allow_html=True)
                for ticker, row in volume_increased_stocks.iterrows():
                    st.write(f"📈 {row['Company Name']} ({ticker}): 거래량 {row['Volume Change (%)']:.2f}% 증가")
        else:
            st.success("현재 특별한 알림이 없습니다.")

    else:
        st.warning("표시할 데이터가 없습니다. 사이드바에서 티커를 입력하거나 스크리닝 옵션을 활성화하세요.")

with tab2:
    st.subheader("포트폴리오 관리")
    if not portfolio:
        st.info("사이드바에서 포트폴리오에 포함할 주식 티커를 입력하세요.")
    elif not df.empty:
        portfolio_df = df.loc[df.index.isin(portfolio)]
        if not portfolio_df.empty:
            st.dataframe(portfolio_df[['Company Name', 'Current Price', 'Change (%)', 'RSI']].style.background_gradient(cmap='viridis', subset=['Change (%)']))
            
            st.subheader("포트폴리오 요약")
            avg_change = portfolio_df['Change (%)'].mean()
            st.metric(label="포트폴리오 평균 수익률", value=f"{avg_change:.2f}%")
        else:
            st.warning("입력한 포트폴리오 티커에 대한 데이터를 찾을 수 없습니다. 티커가 정확한지 확인하세요.")

with tab3:
    st.subheader("차트 분석")
    chart_tickers = tickers_to_fetch
    if not chart_tickers:
        st.info("분석할 주식을 선택하려면 사이드바에서 티커를 입력하거나 스크리닝을 활성화하세요.")
    else:
        selected_ticker = st.selectbox("분석할 주식 선택", options=chart_tickers, format_func=lambda x: f"{df.loc[x, 'Company Name']} ({x})" if x in df.index else x)
        if selected_ticker:
            hist = yf.download(selected_ticker, period="1y", auto_adjust=True)
            if not hist.empty:
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # 가격 및 이동평균선
                ax1.plot(hist.index, hist['Close'], label='종가', color='blue', alpha=0.8)
                ax1.plot(hist.index, calculate_sma(hist, 50), label='50일 이동평균', color='green', linestyle='--')
                ax1.plot(hist.index, calculate_sma(hist, 200), label='200일 이동평균', color='red', linestyle='--')
                ax1.set_ylabel('주가 ($)', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
                
                # RSI
                ax2 = ax1.twinx()
                rsi = calculate_rsi(hist, rsi_period)
                ax2.plot(hist.index[-len(rsi):], rsi, label='RSI', color='purple', alpha=0.7)
                ax2.axhline(rsi_overbought, color='orange', linestyle=':', label=f'과매수 ({rsi_overbought})')
                ax2.axhline(rsi_oversold, color='brown', linestyle=':', label=f'과매도 ({rsi_oversold})')
                ax2.set_ylabel('RSI', color='purple')
                ax2.tick_params(axis='y', labelcolor='purple')
                ax2.set_ylim(0, 100)
                
                # 범례 합치기
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper left')

                plt.title(f"{df.loc[selected_ticker, 'Company Name']} ({selected_ticker}) 주가 및 RSI 차트")
                st.pyplot(fig)
            else:
                st.error(f"{selected_ticker}의 차트 데이터를 불러올 수 없습니다.")


# --- 하단 정보 및 자동 새로고침 ---

st.markdown("---")
st.info("데이터 출처: Yahoo Finance | 2025 개발 by Kent Kim")

if auto_refresh:
    time.sleep(60)
    st.rerun()
