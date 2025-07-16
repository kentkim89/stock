import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import google.generativeai as genai
from gnews import GNews

# --- 1. 페이지 기본 설정 및 함수 정의 ---
st.set_page_config(page_title="AI 주가 분석 플랫폼", page_icon="🚀", layout="wide")

# --- 제미나이 및 세션 상태 초기화 ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (FileNotFoundError, KeyError):
    st.error("오류: Gemini API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인하고 Streamlit Cloud에 Secrets를 등록해주세요.")
    st.stop()

if 'ticker' not in st.session_state: st.session_state.ticker = 'NVDA'
if 'ai_analysis' not in st.session_state: st.session_state.ai_analysis = {}

# --- 데이터 로딩 함수 ---
@st.cache_data(ttl=300)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if not info.get('marketCap'): return None, None, None
    financials = stock.quarterly_financials
    google_news = GNews(language='ko', country='KR')
    company_name = info.get('shortName', ticker)
    news = google_news.get_news(f'{company_name} 주가')
    return info, financials, news

@st.cache_data(ttl=60)
def get_history(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval)

# --- AI 분석 함수들 ---
@st.cache_data(ttl=600)
def generate_ai_analysis(info, data, analysis_type):
    model = genai.GenerativeModel('gemini-1.5-flash')
    company_name = info.get('longName', '해당 기업')
    prompt = ""

    if analysis_type == 'chart':
        history = data
        ma50 = history['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = history['Close'].rolling(window=200).mean().iloc[-1]
        volume_ratio = history['Volume'].iloc[-20:].mean() / history['Volume'].iloc[-60:].mean()
        prompt = f"""당신은 차트 기술적 분석(CMT) 전문가입니다. 다음 데이터를 바탕으로 '{company_name}'의 현재 주가 차트를 상세히 분석해주세요. 거시적인 관점과 차트 패턴을 모두 고려하여 전문적인 의견을 제시하세요.
        - 현재가: {info.get('currentPrice', 'N/A'):.2f}
        - 50일 이동평균선: {ma50:.2f}
        - 200일 이동평균선: {ma200:.2f}
        - 최근 거래량 동향: {volume_ratio:.2f} (1 이상이면 최근 거래량 증가)
        **분석:** (현재 추세(상승/하락/횡보), 주요 지지선 및 저항선, 이동평균선의 의미, 거래량 분석, 종합적인 기술적 의견)"""
    
    elif analysis_type == 'financial':
        prompt = f"""당신은 최고재무책임자(CFO)입니다. 다음 핵심 재무 데이터를 보고 '{company_name}'의 재무 건전성을 분석하고 종합 평가를 내려주세요.
        - **수익성:** 총이익률 {info.get('grossMargins', 0)*100:.2f}%, 영업이익률 {info.get('operatingMargins', 0)*100:.2f}%, ROE {info.get('returnOnEquity', 0)*100:.2f}%
        - **안정성:** 부채비율(Debt/Equity) {info.get('debtToEquity', 'N/A')}
        - **현금흐름:** 영업활동 현금흐름 ${info.get('operatingCashflow', 0):,}
        **AI 재무 진단 리포트:** (각 지표의 의미를 설명하고, 이를 바탕으로 이 회사의 재무적 강점과 약점을 구체적으로 평가한 후, 최종적으로 '매우 우수', '양호', '주의 필요' 등급을 매겨주세요.)"""
    
    elif analysis_type == 'famous_investor':
        news = data
        news_headlines = "\n".join([f"- {article['title']}" for article in news]) if news else "관련 뉴스 없음"
        prompt = f"""당신은 금융 시장 분석가입니다. '캐시 우드', '워런 버핏', '낸시 펠로시' 등 유명 투자자들의 동향을 파악하는 전문가입니다. 구글 뉴스에서 '{company_name}'와 이들 투자자들에 대해 검색된 다음 최신 뉴스 헤드라인을 바탕으로, 이들의 최근 스탠스나 시장의 인식을 요약해주세요.
        - **검색된 뉴스:**\n{news_headlines}
        **유명 투자자 동향 브리핑:** (뉴스 내용을 기반으로 사실 위주로 요약하고, 만약 관련 뉴스가 없다면 '최근 직접적인 언급이나 거래 뉴스는 발견되지 않았습니다'라고 명시해주세요.)"""

    if not prompt: return "분석 유형 오류"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"AI 분석 중 오류 발생: {e}"

# --- 가치평가 스코어카드 & 최종 의견 함수 ---
def get_final_verdict_and_scores(info):
    scores = {}
    details = {}
    
    pe, pb = info.get('trailingPE'), info.get('priceToBook')
    pe_score = (4 if 0 < pe <= 15 else 2 if pe <= 25 else 1) if pe else 0
    pb_score = (2 if 0 < pb <= 1.5 else 1) if pb else 0
    scores['가치'] = pe_score + pb_score
    details['PER'] = f"{pe:.2f}" if pe else "N/A"
    details['PBR'] = f"{pb:.2f}" if pb else "N/A"

    peg, rev_growth = info.get('pegRatio'), info.get('revenueGrowth', 0)
    peg_score = (4 if 0 < peg <= 1 else 2 if peg <= 2 else 0) if peg else 0
    growth_score = (4 if rev_growth > 0.2 else 2 if rev_growth > 0.1 else 0)
    scores['성장성'] = peg_score + growth_score
    details['PEG'] = f"{peg:.2f}" if peg else "N/A"
    details['매출성장률'] = f"{rev_growth*100:.2f}%"

    roe, profit_margin = info.get('returnOnEquity', 0), info.get('profitMargins', 0)
    roe_score = (4 if roe > 0.2 else 2 if roe > 0.15 else 0)
    profit_score = (4 if profit_margin > 0.2 else 2 if profit_margin > 0.1 else 0)
    scores['수익성'] = roe_score + profit_score
    details['ROE'] = f"{roe*100:.2f}%"
    details['순이익률'] = f"{profit_margin*100:.2f}%"

    target_price, current_price = info.get('targetMeanPrice'), info.get('currentPrice', 0)
    analyst_score = 0
    if target_price and current_price and current_price > 0:
        upside = (target_price / current_price - 1)
        analyst_score = (4 if upside > 0.3 else 2 if upside > 0.1 else 1)
    scores['애널리스트'] = analyst_score
    
    total_score = sum(scores.values())
    verdict_info = {"verdict": "관망", "color": "#ffc107", "text_color": "black"}
    if total_score >= 18: verdict_info = {"verdict": "강력 매수", "color": "#198754"}
    elif total_score >= 12: verdict_info = {"verdict": "매수 고려", "color": "#0d6efd"}
    elif total_score < 6: verdict_info = {"verdict": "투자 주의", "color": "#dc3545"}
    
    return verdict_info, scores, details

# --- UI 렌더링 함수 ---
def render_valuation_scorecard(scores, details):
    with st.container(border=True):
        st.subheader("⚖️ AI 가치평가 스코어카드")
        cols = st.columns(4)
        max_scores = {'가치': 6, '성장성': 8, '수익성': 8, '애널리스트': 4}
        for i, (cat, score) in enumerate(scores.items()):
            with cols[i]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=score,
                    domain={'x': [0, 1], 'y': [0, 1]}, title={'text': cat, 'font': {'size': 16}},
                    gauge={'axis': {'range': [0, max_scores[cat]]}, 'bar': {'color': "#0d6efd"}}))
                fig.update_layout(height=150, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
        st.info(f"**상세 지표:** {', '.join([f'{k}: {v}' for k, v in details.items()])}")


# --- 2. 앱 UI 렌더링 ---
st.sidebar.header("종목 검색")
search_ticker = st.sidebar.text_input("종목 코드 입력 (예: AAPL, GOOG)", value=st.session_state.ticker, key="ticker_input").upper()
if st.sidebar.button("분석 실행", key="run_button"):
    st.session_state.ticker = search_ticker
    st.session_state.ai_analysis = {}
    st.cache_data.clear()
    st.rerun()

try:
    info, financials, news = get_stock_data(st.session_state.ticker)

    if info is None:
        st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다.")
    else:
        company_name = info.get('longName', st.session_state.ticker)
        final_verdict, scores, details = get_final_verdict_and_scores(info)
        text_color = final_verdict.get("text_color", "white")

        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h1 style="margin: 0;">🚀 {company_name} AI 분석</h1>
                <div style="padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: {final_verdict['color']}; color: {text_color};">
                    <span style="font-weight: bold; font-size: 1.2rem;">AI 종합 의견: {final_verdict['verdict']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.caption(f"종목코드: {st.session_state.ticker} | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["**📊 종합 대시보드 및 차트 분석**", "**📂 재무 및 가치평가**", "**💡 애널리스트 & 주요 투자자**"])

        with tab1:
            st.subheader("📈 주가 및 거래량 차트")
            period_options = {"오늘": "1d", "1주": "5d", "1개월": "1mo", "1년": "1y", "5년": "5y"}
            selected_period = st.radio("차트 기간 선택", options=period_options.keys(), horizontal=True, key="chart_period")
            period_val, interval_val = (period_options[selected_period], "5m") if selected_period == "오늘" else (period_options[selected_period], "1d")
            history = get_history(st.session_state.ticker, period_val, interval_val)

            if not history.empty:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name='주가'), row=1, col=1)
                if period_val not in ["1d", "5d"]:
                    ma50 = history['Close'].rolling(window=50).mean(); ma200 = history['Close'].rolling(window=200).mean()
                    fig.add_trace(go.Scatter(x=history.index, y=ma50, mode='lines', name='50일 이동평균', line=dict(color='orange', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=history.index, y=ma200, mode='lines', name='200일 이동평균', line=dict(color='purple', width=1)), row=1, col=1)
                fig.add_trace(go.Bar(x=history.index, y=history['Volume'], name='거래량'), row=2, col=1)
                fig.update_layout(height=500, xaxis_rangeslider_visible=False); fig.update_yaxes(title_text="주가", row=1, col=1); fig.update_yaxes(title_text="거래량", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

                if st.button("🤖 AI 차트 분석 실행"):
                    with st.spinner("AI가 차트를 심층 분석 중입니다..."):
                        history_for_ai = get_history(st.session_state.ticker, "1y", "1d") # 1년치 데이터로 분석
                        st.session_state.ai_analysis['chart'] = generate_ai_analysis(info, history_for_ai, 'chart')
                if 'chart' in st.session_state.ai_analysis and st.session_state.ai_analysis['chart']:
                    with st.container(border=True): st.markdown(st.session_state.ai_analysis['chart'])
            else: st.warning("차트 데이터를 불러올 수 없습니다.")

        with tab2:
            # *** 여기가 수정된 부분입니다 ***
            render_valuation_scorecard(scores, details)
            st.divider()

            st.subheader(f"💰 {company_name} 재무 상태 요약")
            fin_cols = st.columns(2)
            with fin_cols[0]:
                fin_summary = {
                    "시가총액": f"${info.get('marketCap', 0):,}",
                    "주가수익비율 (PER)": f"{info.get('trailingPE', 'N/A'):.2f}",
                    "주가순자산비율 (PBR)": f"{info.get('priceToBook', 'N/A'):.2f}",
                    "주가매출비율 (PSR)": f"{info.get('priceToSalesTrailing12Months', 'N/A'):.2f}",
                    "자기자본이익률 (ROE)": f"{info.get('returnOnEquity', 0)*100:.2f}%",
                    "부채비율 (Debt/Equity)": info.get('debtToEquity', 'N/A')
                }
                st.table(pd.DataFrame(fin_summary.items(), columns=['항목', '수치']))
            with fin_cols[1]:
                if st.button("🤖 AI 재무 진단 실행"):
                    with st.spinner("AI가 재무 데이터를 분석하고 등급을 매기는 중입니다..."):
                        st.session_state.ai_analysis['financial'] = generate_ai_analysis(info, None, 'financial')
                if 'financial' in st.session_state.ai_analysis and st.session_state.ai_analysis['financial']:
                     with st.container(border=True, height=350): st.markdown(st.session_state.ai_analysis['financial'])
                else:
                    st.info("버튼을 눌러 AI 재무 진단을 받아보세요.")

        with tab3:
            st.subheader("💡 유명 투자자 동향 분석 (AI 기반)")
            st.info("""'캐시 우드', '워런 버핏', '낸시 펠로시' 키워드로 검색된 최신 뉴스를 AI가 분석하여 이들의 스탠스를 요약합니다.""")
            if st.button("🤖 최신 동향 분석 실행"):
                with st.spinner("AI가 구글 뉴스에서 관련 동향을 분석 중입니다..."):
                    google_news_famous = GNews(language='ko', country='KR')
                    query = f"{company_name} (워런 버핏 | 캐시 우드 | 낸시 펠로시)"
                    news_famous = google_news_famous.get_news(query)
                    st.session_state.ai_analysis['famous_investor'] = generate_ai_analysis(info, news_famous, 'famous_investor')
            if 'famous_investor' in st.session_state.ai_analysis and st.session_state.ai_analysis['famous_investor']:
                with st.container(border=True): st.markdown(st.session_state.ai_analysis['famous_investor'])
            
            st.divider()
            st.subheader("📰 관련 최신 뉴스 (From Google News)")
            if news:
                for article in news[:10]:
                    st.write(f"[{article['title']}]({article['url']}) - *{article['publisher']['title']}*")
            else: st.info("구글 뉴스에서 관련 뉴스를 찾을 수 없습니다.")

except Exception as e:
    st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
