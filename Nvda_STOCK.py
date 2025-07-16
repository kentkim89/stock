import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import google.generativeai as genai
from gnews import GNews
# --- 새로운 라이브러리 임포트 ---
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from st_aggrid_redux import AgGrid, GridOptionsBuilder

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

# --- Lottie 애니메이션 로드 함수 ---
@st.cache_data
def load_lottie_url(url: str):
    import requests
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

# --- AI 분석 생성 함수 (이전 버전과 동일) ---
@st.cache_data(ttl=600)
def generate_ai_analysis(info, data, analysis_type):
    model = genai.GenerativeModel('gemini-1.5-flash')
    company_name = info.get('longName', '해당 기업')
    today_date = datetime.now().strftime('%Y년 %m월 %d일')
    prompt = ""

    if analysis_type == 'chart':
        history = data
        ma50 = history['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = history['Close'].rolling(window=200).mean().iloc[-1]
        prompt = f"""당신은 차트 기술적 분석(CMT) 전문가입니다. **오늘은 {today_date}입니다.** 다음 데이터를 바탕으로 '{company_name}'의 주가 차트를 상세히 분석해주세요.
        - 현재가: {info.get('currentPrice', 'N/A'):.2f}, 50일 이동평균선: {ma50:.2f}, 200일 이동평균선: {ma200:.2f}
        **분석:** (현재 추세(상승/하락/횡보), 이동평균선의 관계, 주요 지지선 및 저항선, 종합적인 기술적 의견)"""
    
    elif analysis_type == 'financial':
        financials = data
        latest_date = financials.columns[0].strftime('%Y년 %m월')
        prompt = f"""당신은 최고재무책임자(CFO)입니다. 다음은 **{latest_date} 기준**의 최신 재무 데이터입니다. 이를 보고 '{company_name}'의 재무 건전성을 분석하고 종합 평가를 내려주세요.
        - **수익성:** 총이익률 {info.get('grossMargins', 0)*100:.2f}%, ROE {info.get('returnOnEquity', 0)*100:.2f}%
        - **안정성:** 부채비율(Debt/Equity) {info.get('debtToEquity', 'N/A')}
        **AI 재무 진단 리포트:** (각 지표의 의미를 설명하고, 재무적 강점과 약점을 구체적으로 평가한 후, 최종적으로 '매우 우수', '양호', '주의 필요' 등급을 매겨주세요.)"""

    if not prompt: return "분석 유형 오류"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"AI 분석 중 오류 발생: {e}"

# --- 가치평가 스코어카드 & 최종 의견 함수 ---
def get_final_verdict_and_scores(info):
    scores, details = {}, {}
    pe, pb = info.get('trailingPE'), info.get('priceToBook'); scores['가치'] = ((4 if 0 < pe <= 15 else 2 if pe <= 25 else 1) if pe else 0) + ((2 if 0 < pb <= 1.5 else 1) if pb else 0)
    details['PER'] = f"{pe:.2f}" if pe else "N/A"; details['PBR'] = f"{pb:.2f}" if pb else "N/A"
    peg, rev_growth = info.get('pegRatio'), info.get('revenueGrowth', 0); scores['성장성'] = ((4 if 0 < peg <= 1 else 2 if peg <= 2 else 0) if peg else 0) + ((4 if rev_growth > 0.2 else 2 if rev_growth > 0.1 else 0))
    details['PEG'] = f"{peg:.2f}" if peg else "N/A"; details['매출성장률'] = f"{rev_growth*100:.2f}%"
    roe, profit_margin = info.get('returnOnEquity', 0), info.get('profitMargins', 0); scores['수익성'] = ((4 if roe > 0.2 else 2 if roe > 0.15 else 0)) + ((4 if profit_margin > 0.2 else 2 if profit_margin > 0.1 else 0))
    details['ROE'] = f"{roe*100:.2f}%"; details['순이익률'] = f"{profit_margin*100:.2f}%"
    target_price, current_price = info.get('targetMeanPrice'), info.get('currentPrice', 0); scores['애널리스트'] = (4 if (target_price/current_price-1)>0.3 else 2 if (target_price/current_price-1)>0.1 else 1) if target_price and current_price and current_price > 0 else 0
    total_score = sum(scores.values())
    verdict_info = {"verdict": "관망", "color": "#ffc107", "text_color": "black"}
    if total_score >= 18: verdict_info = {"verdict": "강력 매수", "color": "#198754"}
    elif total_score >= 12: verdict_info = {"verdict": "매수 고려", "color": "#0d6efd"}
    elif total_score < 6: verdict_info = {"verdict": "투자 주의", "color": "#dc3545"}
    return verdict_info, scores, details

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
    if info is None: st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다.")
    else:
        company_name = info.get('longName', st.session_state.ticker)
        final_verdict, scores, details = get_final_verdict_and_scores(info)
        text_color = final_verdict.get("text_color", "white")

        # --- 상단 헤더 및 내비게이션 메뉴 ---
        lottie_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_dtrqvxcm.json")

        st.markdown(f"""<div style="display: flex; justify-content: space-between; align-items: center;"><h1 style="margin: 0;">🚀 {company_name} AI 분석</h1><div style="padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: {final_verdict['color']}; color: {text_color};"><span style="font-weight: bold; font-size: 1.2rem;">AI 종합 의견: {final_verdict['verdict']}</span></div></div>""", unsafe_allow_html=True)
        st.caption(f"종목코드: {st.session_state.ticker} | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        selected_page = option_menu(
            menu_title=None,
            options=["종합 대시보드", "재무 & 가치평가", "뉴스 & 시장 동향"],
            icons=["bi-house-door-fill", "bi-cash-coin", "bi-newspaper"],
            menu_icon="cast", default_index=0, orientation="horizontal",
        )
        st.markdown("---")

        # --- 페이지별 콘텐츠 ---
        if selected_page == "종합 대시보드":
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

                if st.toggle("🤖 AI 심층 차트 분석 보기", key="chart_toggle"):
                    placeholder = st.empty()
                    with placeholder.container():
                        st_lottie(lottie_animation, height=100)
                        st.write("AI가 차트를 심층 분석 중입니다...")
                        history_for_ai = get_history(st.session_state.ticker, "1y", "1d")
                        st.session_state.ai_analysis['chart'] = generate_ai_analysis(info, history_for_ai, 'chart')
                    placeholder.empty()
                    if 'chart' in st.session_state.ai_analysis:
                        st.markdown(st.session_state.ai_analysis['chart'])
            else: st.warning("차트 데이터를 불러올 수 없습니다.")

        if selected_page == "재무 & 가치평가":
            st.subheader("⚖️ AI 가치평가 스코어카드")
            cols = st.columns(4)
            max_scores = {'가치': 6, '성장성': 8, '수익성': 8, '애널리스트': 4}
            for i, (cat, score) in enumerate(scores.items()):
                with cols[i]:
                    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': cat, 'font': {'size': 16}}, gauge={'axis': {'range': [0, max_scores[cat]]}, 'bar': {'color': "#0d6efd"}}))
                    fig_gauge.update_layout(height=150, margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.divider()
            st.subheader(f"💰 {company_name} 재무 데이터")
            if financials is not None and not financials.empty:
                fin_summary_df = financials.T.iloc[:4] # 최근 4분기
                gb = GridOptionsBuilder.from_dataframe(fin_summary_df)
                gb.configure_default_column(cellStyle={'text-align': 'right'})
                AgGrid(fin_summary_df.reset_index(), gridOptions=gb.build(), theme='streamlit', fit_columns_on_grid_load=True)

                if st.toggle("🤖 AI 재무 진단 보기", key="financial_toggle"):
                    placeholder = st.empty()
                    with placeholder.container():
                        st_lottie(lottie_animation, height=100)
                        st.write("AI가 재무 데이터를 분석하고 등급을 매기는 중입니다...")
                        st.session_state.ai_analysis['financial'] = generate_ai_analysis(info, financials, 'financial')
                    placeholder.empty()
                    if 'financial' in st.session_state.ai_analysis:
                         st.markdown(st.session_state.ai_analysis['financial'])
            else: st.info("재무 데이터를 가져올 수 없습니다.")

        if selected_page == "뉴스 & 시장 동향":
            st.subheader("📰 AI 뉴스 요약 및 시장 분위기 분석")
            if st.toggle("🤖 AI 뉴스 분석 보기", key="news_toggle"):
                placeholder = st.empty()
                with placeholder.container():
                    st_lottie(lottie_animation, height=100)
                    st.write("AI가 구글 뉴스에서 최신 동향을 분석 중입니다...")
                    st.session_state.ai_analysis['news'] = generate_ai_analysis(info, news, 'news')
                placeholder.empty()
                if 'news' in st.session_state.ai_analysis: st.markdown(st.session_state.ai_analysis['news'])

            st.divider()
            st.subheader("💡 유명 투자자 동향 분석 (AI 기반)")
            if st.toggle("🤖 최신 동향 분석 보기", key="famous_toggle"):
                placeholder = st.empty()
                with placeholder.container():
                    st_lottie(lottie_animation, height=100)
                    st.write("AI가 관련 뉴스를 검색하고 분석 중입니다...")
                    google_news_famous = GNews(language='ko', country='KR')
                    query = f"{company_name} (워런 버핏 | 캐시 우드 | 낸시 펠로시)"
                    news_famous = google_news_famous.get_news(query)
                    st.session_state.ai_analysis['famous_investor'] = generate_ai_analysis(info, news_famous, 'famous_investor')
                placeholder.empty()
                if 'famous_investor' in st.session_state.ai_analysis: st.markdown(st.session_state.ai_analysis['famous_investor'])

            st.divider()
            st.subheader("📜 관련 최신 뉴스 원문 (From Google News)")
            if news:
                for article in news[:10]: st.write(f"[{article['title']}]({article['url']}) - *{article['publisher']['title']}*")
            else: st.info("구글 뉴스에서 관련 뉴스를 찾을 수 없습니다.")

except Exception as e:
    st.error(f"앱 실행 중 오류가 발생했습니다: {e}")```
