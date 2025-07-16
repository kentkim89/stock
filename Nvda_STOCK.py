import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime
import google.generativeai as genai

# --- 1. 페이지 기본 설정 및 함수 정의 ---
st.set_page_config(page_title="AI 주가 분석 대시보드", page_icon="🧠", layout="wide")

# --- 제미나이 및 세션 상태 초기화 ---
# Streamlit Secrets에서 API 키 가져오기
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (FileNotFoundError, KeyError):
    st.error("오류: Gemini API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인하고 Streamlit Cloud에 Secrets를 등록해주세요.")
    st.stop()

# 세션 상태 초기화
if 'ticker' not in st.session_state:
    st.session_state.ticker = 'NVDA'
if 'gemini_report' not in st.session_state:
    st.session_state.gemini_report = None


# --- 데이터 로딩 함수 ---
@st.cache_data(ttl=300)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if not info.get('marketCap'): return None, None
    financials = stock.quarterly_financials
    return info, financials

@st.cache_data(ttl=60)
def get_history(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval)

# --- 제미나이 분석 함수 (안정성 강화 버전) ---
@st.cache_data(ttl=600)
def get_gemini_analysis(info):
    """제미나이 API를 호출하여 기업 분석 리포트를 생성합니다."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    company_name = info.get('longName', '해당 기업')
    
    # --- 데이터 안전하게 전처리하는 과정 (핵심 수정 부분) ---
    def format_value(value, precision=2, is_percent=False):
        """숫자 데이터는 서식을 적용하고, 아니면 'N/A'를 반환하는 안전한 함수"""
        if isinstance(value, (int, float)):
            return f"{value * 100:.{precision}f}%" if is_percent else f"{value:.{precision}f}"
        return "N/A"

    per = format_value(info.get('trailingPE'))
    pbr = format_value(info.get('priceToBook'))
    peg = format_value(info.get('pegRatio'))
    roe = format_value(info.get('returnOnEquity'), is_percent=True)
    target_price = format_value(info.get('targetMeanPrice'))
    current_price = format_value(info.get('currentPrice', info.get('regularMarketPrice')))
    market_cap = f"${info.get('marketCap', 0):,}" if info.get('marketCap') else "N/A"
    
    # --- 안전하게 전처리된 변수를 사용한 프롬프트 ---
    prompt = f"""
    당신은 월스트리트의 경험 많은 시니어 금융 애널리스트입니다. 다음 데이터를 기반으로 '{company_name}'에 대한 전문적인 투자 분석 보고서를 **Markdown 형식의 한국어**로 작성해주세요.

    **핵심 기업 데이터:**
    - **기업명:** {company_name} ({info.get('symbol')})
    - **업종:** {info.get('sector', 'N/A')}
    - **시가총액:** {market_cap}
    - **PER:** {per}
    - **PBR:** {pbr}
    - **PEG:** {peg}
    - **ROE:** {roe}
    - **애널리스트 평균 목표가:** ${target_price}
    - **현재가:** ${current_price}

    **보고서 작성 지침:**
    아래 목차에 따라, 각 항목을 구체적이고 논리적으로 분석하여 투자자들이 명확한 판단을 내릴 수 있도록 도와주세요.

    ### 1. 투자 하이라이트 (Investment Highlights)
    - **핵심 성장 동력:** 이 기업의 미래 성장을 이끌 가장 중요한 요소는 무엇인가? (최소 2가지 이상 구체적으로 설명)
    - **강력한 해자(Moat):** 경쟁사들이 쉽게 따라올 수 없는 이 기업만의 독점적인 강점은 무엇인가?

    ### 2. 주요 리스크 요인 (Key Risk Factors)
    - **시장 및 경쟁 리스크:** 시장의 변화나 경쟁사의 위협으로 인해 발생할 수 있는 위험은 무엇인가?
    - **내재적 리스크:** 이 기업이 내부적으로 가지고 있는 약점이나 재무적 위험은 무엇인가?

    ### 3. 종합 결론 및 투자 전략 (Final Verdict & Strategy)
    - **최종 투자 의견:** 모든 데이터를 종합했을 때, 당신의 최종 투자 의견은 무엇인가? ('적극 매수', '분할 매수', '중립(보유)', '비중 축소' 중 선택)
    - **투자 전략:** 위 의견에 따라, 투자자들은 어떤 전략을 취하는 것이 바람직한가? (예: '장기적인 관점에서 분할 매수 접근이 유효합니다.')
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini 분석 중 오류가 발생했습니다: {e}"

# --- 2. 앱 UI 렌더링 ---
st.sidebar.header("종목 검색")
search_ticker = st.sidebar.text_input("종목 코드 입력 (예: AAPL, GOOG)", value=st.session_state.ticker, key="ticker_input").upper()
if st.sidebar.button("분석 실행", key="run_button"):
    st.session_state.ticker = search_ticker
    st.session_state.gemini_report = None # 새로운 종목 검색 시 이전 리포트 삭제
    st.cache_data.clear()
    st.rerun()

try:
    info, financials = get_stock_data(st.session_state.ticker)

    if info is None:
        st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다. 종목 코드를 확인해주세요.")
    else:
        company_name = info.get('longName', st.session_state.ticker)
        st.title(f"🧠 {company_name} AI 주가 분석")
        st.caption(f"종목코드: {st.session_state.ticker} | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        tab1, tab2, tab3 = st.tabs(["**📊 종합 대시보드**", "**🤖 제미나이 AI 심층 분석**", "**📂 재무 및 기업 정보**"])

        with tab1:
            st.subheader("📈 주가 추이 차트")
            period_options = {"오늘": "1d", "1주": "5d", "1개월": "1mo", "1년": "1y", "5년": "5y"}
            selected_period = st.radio("차트 기간 선택", options=period_options.keys(), horizontal=True, key="chart_period")
            period_val, interval_val = (period_options[selected_period], "5m") if selected_period == "오늘" else (period_options[selected_period], "1d")
            history = get_history(st.session_state.ticker, period_val, interval_val)
            
            if not history.empty:
                chart_type = 'Candlestick' if selected_period == "오늘" else 'Scatter'
                fig = go.Figure(data=[go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name='분봉')] if chart_type == 'Candlestick' 
                                      else [go.Scatter(x=history.index, y=history['Close'], mode='lines', name='종가')])
                is_intraday = selected_period == "오늘"
                fig.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20),
                                  xaxis_rangeslider_visible=not is_intraday, dragmode='pan' if not is_intraday else False,
                                  xaxis=dict(fixedrange=is_intraday), yaxis=dict(fixedrange=is_intraday))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': not is_intraday})
            else:
                st.warning("차트 데이터를 불러올 수 없습니다.")
        
        with tab2:
            st.subheader(f"🤖 제미나이(Gemini)가 분석한 {company_name} 리포트")
            
            if st.button("실시간 AI 리포트 생성하기", key="gemini_button"):
                with st.spinner('제미나이 AI가 최신 데이터를 분석하고 있습니다... 약 30초 정도 소요될 수 있습니다.'):
                    # API 호출 후 결과를 세션 상태에 저장
                    st.session_state.gemini_report = get_gemini_analysis(info)
            
            st.markdown("---")

            # 세션 상태에 저장된 리포트가 있으면 표시
            if st.session_state.gemini_report:
                st.markdown(st.session_state.gemini_report)
            else:
                st.info("위에 있는 '실시간 AI 리포트 생성하기' 버튼을 클릭하시면, 제미나이가 투자 포인트, 리스크, 종합 의견을 포함한 상세 리포트를 생성합니다.")

        with tab3:
            st.subheader(f"💰 {company_name} 재무 상태")
            if financials is not None and not financials.empty:
                financials_t = financials.T.iloc[:4]
                financials_t.index = pd.to_datetime(financials_t.index).strftime('%Y-%m')
                fig_fin = go.Figure(data=[go.Bar(name='매출(Revenue)', x=financials_t.index, y=financials_t.get('Total Revenue')),
                                          go.Bar(name='순이익(Net Income)', x=financials_t.index, y=financials_t.get('Net Income'))])
                fig_fin.update_layout(barmode='group', title_text="분기별 매출 및 순이익 추이")
                st.plotly_chart(fig_fin, use_container_width=True)
            else: 
                st.info("재무 데이터를 가져올 수 없습니다.")
            
            st.divider()
            st.subheader(f"📑 {company_name} 기업 개요")
            st.write(info.get('longBusinessSummary', '기업 개요 정보가 없습니다.'))

except Exception as e:
    st.error(f"앱 실행 중 예상치 못한 오류가 발생했습니다: {e}")

# 사이드바 알림
st.sidebar.markdown("---")
st.sidebar.header("가격 알림 설정")
high_alert = st.sidebar.number_input("고점 알림 가격", min_value=0.0, format="%.2f", key="high_alert")
low_alert = st.sidebar.number_input("저점 알림 가격", min_value=0.0, format="%.2f", key="low_alert")

if 'info' in locals() and info is not None:
    current_price = info.get('currentPrice', 0)
    if high_alert > 0 and current_price >= high_alert:
        st.sidebar.success(f"📈 목표 고점(${high_alert:,.2f}) 도달!")
    if low_alert > 0 and current_price <= low_alert:
        st.sidebar.warning(f"📉 목표 저점(${low_alert:,.2f}) 도달!")
