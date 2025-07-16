import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import google.generativeai as genai
from gnews import GNews # 구글 뉴스 라이브러리 임포트

# --- 1. 페이지 기본 설정 및 함수 정의 ---
st.set_page_config(page_title="AI 주가 분석 플랫폼", page_icon="🚀", layout="wide")

# --- 제미나이 및 세션 상태 초기화 ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (FileNotFoundError, KeyError):
    st.error("오류: Gemini API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인하고 Streamlit Cloud에 Secrets를 등록해주세요.")
    st.stop()

if 'ticker' not in st.session_state: st.session_state.ticker = 'NVDA'
if 'gemini_briefing' not in st.session_state: st.session_state.gemini_briefing = {}
if 'analyst_view' not in st.session_state: st.session_state.analyst_view = None

# --- 데이터 로딩 함수 (구글 뉴스 연동) ---
@st.cache_data(ttl=300)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if not info.get('marketCap'): return None, None, None
    financials = stock.quarterly_financials
    
    # yfinance 뉴스 대신 GNews 사용
    google_news = GNews(language='ko', country='KR')
    # 회사명으로 검색하여 정확도 높임
    company_name = info.get('shortName', ticker)
    news = google_news.get_news(f'{company_name} 주가')
    
    return info, financials, news

@st.cache_data(ttl=60)
def get_history(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval)

# --- AI 브리핑 생성 함수 (고품질 뉴스 데이터 사용) ---
@st.cache_data(ttl=600)
def generate_gemini_briefing(info, history, news, analysis_type):
    model = genai.GenerativeModel('gemini-1.5-flash')
    company_name = info.get('longName', '해당 기업')
    prompt = ""

    if analysis_type == '뉴스':
        if not news: return "분석할 최신 뉴스가 없습니다."
        # gnews 데이터 구조에 맞게 수정: article['title']
        news_headlines = "\n".join([f"- {article['title']}" for article in news[:8]])
        prompt = f"""당신은 금융 뉴스 전문 애널리스트입니다. 다음 구글 뉴스에서 수집된 최신 헤드라인들을 기반으로 '{company_name}'에 대한 시장의 전반적인 분위기와 핵심 이슈를 요약해주세요.
        뉴스 목록:\n{news_headlines}\n\n**분석:**"""
    
    elif analysis_type == '차트':
        ma50 = history['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = history['Close'].rolling(window=200).mean().iloc[-1]
        prompt = f"""당신은 기술적 분석(Technical Analyst) 전문가입니다. 다음 데이터를 바탕으로 '{company_name}'의 현재 주가 차트 상태를 기술적으로 분석해주세요.
        - 현재가: {info.get('currentPrice', 'N/A'):.2f}
        - 50일 이동평균선: {ma50:.2f}
        - 200일 이동평균선: {ma200:.2f}
        **분석 (상승/하락 신호, 지지/저항선 등):**"""
        
    elif analysis_type == '재무':
        prompt = f"""당신은 기업 재무 분석 전문가입니다. 다음 핵심 재무 지표를 바탕으로 '{company_name}'의 최근 재무 건전성과 수익성을 간단하게 평가해주세요.
        - 총이익률(Gross Margins): {info.get('grossMargins', 0)*100:.2f}%
        - 영업이익률(Operating Margins): {info.get('operatingMargins', 0)*100:.2f}%
        - 부채비율(Debt to Equity): {info.get('debtToEquity', 'N/A')}
        **분석:**"""

    if not prompt: return "분석 유형이 올바르지 않습니다."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"AI 분석 중 오류 발생: {e}"

# --- AI 생성 애널리스트 시각 함수 (고품질 뉴스 데이터 사용) ---
@st.cache_data(ttl=600)
def generate_synthesized_analyst_view(info, news):
    model = genai.GenerativeModel('gemini-1.5-flash')
    company_name = info.get('longName', '해당 기업')
    
    news_summary = "최신 뉴스가 없습니다."
    if news:
        # gnews 데이터 구조에 맞게 수정: article['title']
        news_summary = "\n".join([f"- {article['title']}" for article in news[:5]])

    prompt = f"""당신은 월스트리트의 유능한 금융 애널리스트입니다. 다음 데이터를 **종합적으로 해석**하여 '{company_name}'에 대한 애널리스트 리포트 형식의 의견을 제시해주세요. **실시간 검색이 아닌, 제공된 데이터 기반으로 추론하세요.**
    - **핵심 데이터:**
      - 애널리스트 평균 목표가: ${info.get('targetMeanPrice', 'N/A')} / 현재가: ${info.get('currentPrice', 'N/A')}
      - PER: {info.get('trailingPE', 'N/A'):.2f}, PBR: {info.get('priceToBook', 'N/A'):.2f}
      - 최근 구글 뉴스 헤드라인:\n{news_summary}
    - **작성 지침:** 위 데이터를 바탕으로 **① 목표가에 대한 종합 평가**와 **② 최근 뉴스 및 밸류에이션을 고려한 투자 전략**을 구체적으로 제시해주세요."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"AI 분석 중 오류 발생: {e}"

# --- 가치평가 스코어카드 렌더링 함수 ---
def render_valuation_scorecard(info):
    scores = {}
    pe, pb = info.get('trailingPE'), info.get('priceToBook')
    pe_score = (4 if 0 < pe <= 15 else 2 if pe <= 25 else 1) if pe else 0
    pb_score = (2 if 0 < pb <= 1.5 else 1) if pb else 0
    scores['가치'] = pe_score + pb_score
    peg, rev_growth = info.get('pegRatio'), info.get('revenueGrowth', 0)
    peg_score = (4 if 0 < peg <= 1 else 2 if peg <= 2 else 0) if peg else 0
    growth_score = (4 if rev_growth > 0.2 else 2 if rev_growth > 0.1 else 0)
    scores['성장성'] = peg_score + growth_score
    roe, profit_margin = info.get('returnOnEquity', 0), info.get('profitMargins', 0)
    roe_score = (4 if roe > 0.2 else 2 if roe > 0.15 else 0)
    profit_score = (4 if profit_margin > 0.2 else 2 if profit_margin > 0.1 else 0)
    scores['수익성'] = roe_score + profit_score
    target_price, current_price = info.get('targetMeanPrice'), info.get('currentPrice', 0)
    analyst_score = 0
    if target_price and current_price:
        upside = (target_price / current_price - 1)
        analyst_score = (4 if upside > 0.3 else 2 if upside > 0.1 else 1)
    scores['애널리스트'] = analyst_score
    
    with st.container(border=True):
        st.subheader("⚖️ AI 가치평가 스코어카드")
        cols = st.columns(4)
        max_scores = {'가치': 6, '성장성': 8, '수익성': 8, '애널리스트': 4}
        for i, (cat, score) in enumerate(scores.items()):
            with cols[i]:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
                    domain={'x': [0, 1], 'y': [0, 1]}, title={'text': cat, 'font': {'size': 16}},
                    gauge={'axis': {'range': [0, max_scores[cat]]}, 'bar': {'color': "#0d6efd"}}))
                fig.update_layout(height=150, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

# --- 2. 앱 UI 렌더링 ---
st.sidebar.header("종목 검색")
search_ticker = st.sidebar.text_input("종목 코드 입력 (예: AAPL, GOOG)", value=st.session_state.ticker, key="ticker_input").upper()
if st.sidebar.button("분석 실행", key="run_button"):
    st.session_state.ticker = search_ticker
    st.session_state.gemini_briefing = {}
    st.session_state.analyst_view = None
    st.cache_data.clear()
    st.rerun()

try:
    info, financials, news = get_stock_data(st.session_state.ticker)

    if info is None:
        st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다. 종목 코드를 확인해주세요.")
    else:
        company_name = info.get('longName', st.session_state.ticker)
        st.title(f"🚀 {company_name} AI 주가 분석 플랫폼")

        tab1, tab2, tab3, tab4 = st.tabs(["**🤖 AI 종합 브리핑**", "**📈 차트 & 기술적 분석**", "**📂 상세 재무 및 가치평가**", "**💡 애널리스트 & 뉴스**"])

        with tab1:
            st.subheader("✨ AI 실시간 브리핑")
            briefing_cols = st.columns(3)
            with briefing_cols[0]:
                if st.button("📰 최신 뉴스 분석"):
                    with st.spinner("AI가 구글 뉴스를 분석 중입니다..."):
                        st.session_state.gemini_briefing['news'] = generate_gemini_briefing(info, None, news, '뉴스')
            with briefing_cols[1]:
                history_1y = get_history(st.session_state.ticker, "1y", "1d")
                if st.button("📊 주가 차트 분석"):
                     with st.spinner("AI가 차트를 분석 중입니다..."):
                        st.session_state.gemini_briefing['chart'] = generate_gemini_briefing(info, history_1y, None, '차트')
            with briefing_cols[2]:
                if st.button("💰 핵심 재무 분석"):
                    with st.spinner("AI가 재무를 분석 중입니다..."):
                        st.session_state.gemini_briefing['financials'] = generate_gemini_briefing(info, None, None, '재무')

            if st.session_state.gemini_briefing:
                st.markdown("---")
                # 각 분석 결과를 별도의 컨테이너에 표시
                for key, value in st.session_state.gemini_briefing.items():
                    if value:
                        container_title = {'news': '📰 뉴스 분석 요약', 'chart': '📊 기술적 분석 요약', 'financials': '💰 재무 분석 요약'}.get(key)
                        with st.container(border=True):
                            st.markdown(f"##### {container_title}")
                            st.write(value)
            else:
                st.info("위 버튼을 클릭하여 각 영역에 대한 AI 브리핑을 받아보세요.")

        with tab2:
            st.subheader("📈 주가 및 거래량 차트")
            # (차트 UI 부분은 변경 없음)
            period_options = {"오늘": "1d", "1주": "5d", "1개월": "1mo", "1년": "1y", "5년": "5y"}
            selected_period = st.radio("차트 기간 선택", options=period_options.keys(), horizontal=True, key="chart_period")
            period_val, interval_val = (period_options[selected_period], "5m") if selected_period == "오늘" else (period_options[selected_period], "1d")
            history = get_history(st.session_state.ticker, period_val, interval_val)
            if not history.empty:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name='주가'), row=1, col=1)
                if period_val not in ["1d", "5d"]:
                    ma50 = history['Close'].rolling(window=50).mean()
                    ma200 = history['Close'].rolling(window=200).mean()
                    fig.add_trace(go.Scatter(x=history.index, y=ma50, mode='lines', name='50일 이동평균', line=dict(color='orange', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=history.index, y=ma200, mode='lines', name='200일 이동평균', line=dict(color='purple', width=1)), row=1, col=1)
                fig.add_trace(go.Bar(x=history.index, y=history['Volume'], name='거래량'), row=2, col=1)
                fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                fig.update_yaxes(title_text="주가", row=1, col=1); fig.update_yaxes(title_text="거래량", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("차트 데이터를 불러올 수 없습니다.")

        with tab3:
            render_valuation_scorecard(info)
            st.divider()
            st.subheader(f"💰 {company_name} 재무 상태")
            if financials is not None and not financials.empty:
                financials_t = financials.T.iloc[:4]
                financials_t.index = pd.to_datetime(financials_t.index).strftime('%Y-%m')
                fig_fin = go.Figure(data=[go.Bar(name='매출', x=financials_t.index, y=financials_t.get('Total Revenue')),
                                          go.Bar(name='순이익', x=financials_t.index, y=financials_t.get('Net Income'))])
                fig_fin.update_layout(barmode='group', title_text="분기별 매출 및 순이익 추이")
                st.plotly_chart(fig_fin, use_container_width=True)
            else: st.info("재무 데이터를 가져올 수 없습니다.")
            
        with tab4:
            st.subheader("💡 AI 생성 애널리스트 시각")
            st.info("""**[안내]** 이 분석은 제미나이 AI가 제공된 최신 데이터(구글 뉴스, 주가, 재무)를 종합하여 애널리스트의 시각으로 **재해석한 분석**입니다.""")
            if st.button("최신 데이터로 AI 애널리스트 리포트 생성"):
                with st.spinner("AI가 최신 데이터를 종합하여 분석 중입니다..."):
                    st.session_state.analyst_view = generate_synthesized_analyst_view(info, news)
            
            if st.session_state.analyst_view:
                st.markdown(st.session_state.analyst_view)
            
            st.divider()
            st.subheader("📰 원본 뉴스 목록 (From Google News)")
            if news:
                for article in news:
                    st.write(f"[{article['title']}]({article['url']}) - *{article['publisher']['title']}*")
            else:
                st.info("구글 뉴스에서 관련 뉴스를 찾을 수 없습니다.")

except Exception as e:
    st.error(f"앱 실행 중 예상치 못한 오류가 발생했습니다: {e}")

# 사이드바 알림 기능
st.sidebar.markdown("---")
st.sidebar.header("가격 알림 설정")
high_alert = st.sidebar.number_input("고점 알림 가격", min_value=0.0, format="%.2f", key="high_alert")
low_alert = st.sidebar.number_input("저점 알림 가격", min_value=0.0, format="%.2f", key="low_alert")

if 'info' in locals() and info is not None:
    current_price = info.get('currentPrice', 0)
    if high_alert > 0 and current_price >= high_alert: st.sidebar.success(f"📈 목표 고점(${high_alert:,.2f}) 도달!")
    if low_alert > 0 and current_price <= low_alert: st.sidebar.warning(f"📉 목표 저점(${low_alert:,.2f}) 도달!")
