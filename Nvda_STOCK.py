import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

# --- 1. 페이지 기본 설정 및 함수 정의 ---
st.set_page_config(
    page_title="AI 주가 분석 대시보드",
    page_icon="🤖",
    layout="wide",
)

# --- 캐싱을 사용한 데이터 로딩 함수 ---
@st.cache_data(ttl=300) # 5분마다 데이터 갱신
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    # 유효한 종목인지 확인 (marketCap이 없으면 데이터 없는 것으로 간주)
    if not info.get('marketCap'):
        return None, None
    financials = stock.quarterly_financials
    return info, financials

@st.cache_data(ttl=60)
def get_history(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval)

# --- 가치평가 및 AI 의견 생성 함수 ---
def render_valuation_analysis(info):
    st.subheader("⚖️ AI 가치평가 및 투자 의견")
    
    scores = {}
    details = {}

    # 1. 상대가치 평가 (Relative Valuation)
    pe = info.get('trailingPE')
    ps = info.get('priceToSalesTrailing12Months')
    pb = info.get('priceToBook')
    
    pe_score = 0
    if pe:
        if 0 < pe <= 15: pe_score = 4
        elif pe <= 25: pe_score = 2
        else: pe_score = 1
    scores['상대가치'] = pe_score
    details['PER (주가수익비율)'] = f"{pe:.2f}" if pe else "N/A"
    details['PBR (주가순자산비율)'] = f"{pb:.2f}" if pb else "N/A"
    details['PSR (주가매출비율)'] = f"{ps:.2f}" if ps else "N/A"

    # 2. 성장성 평가 (Growth)
    peg = info.get('pegRatio')
    rev_growth = info.get('revenueGrowth', 0)
    growth_score = 0
    if peg and 0 < peg <= 1: growth_score = 4
    elif peg and peg <= 2: growth_score = 2
    if rev_growth > 0.2: growth_score += 4
    elif rev_growth > 0.1: growth_score += 2
    scores['성장성'] = growth_score
    details['PEG (주가수익성장비율)'] = f"{peg:.2f}" if peg else "N/A"
    details['매출성장률(YoY)'] = f"{rev_growth*100:.2f}%"

    # 3. 수익성 평가 (Profitability)
    roe = info.get('returnOnEquity', 0)
    profit_margin = info.get('profitMargins', 0)
    profit_score = 0
    if roe > 0.2: profit_score = 4
    elif roe > 0.15: profit_score = 2
    if profit_margin > 0.2: profit_score += 4
    elif profit_margin > 0.1: profit_score += 2
    scores['수익성'] = profit_score
    details['ROE (자기자본이익률)'] = f"{roe*100:.2f}%"
    details['순이익률'] = f"{profit_margin*100:.2f}%"
    
    # 4. 애널리스트 평가 (Analyst Target)
    target_price = info.get('targetMeanPrice')
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    analyst_score = 0
    if target_price and current_price:
        upside = (target_price / current_price - 1) * 100
        if upside > 30: analyst_score = 4
        elif upside > 10: analyst_score = 2
        else: analyst_score = 1
    scores['애널리스트'] = analyst_score
    details['목표가 상승여력'] = f"{upside:.2f}%" if target_price and current_price else "N/A"
    details['애널리스트 수'] = info.get('numberOfAnalystOpinions', 'N/A')

    # 종합 의견 생성
    total_score = sum(scores.values())
    opinion = {"verdict": "정보 분석 중", "color": "#6c757d"}
    if total_score >= 12: opinion = {"verdict": "강력 매수 고려", "color": "#198754"}
    elif total_score >= 8: opinion = {"verdict": "긍정적, 분할 매수", "color": "#0d6efd"}
    elif total_score >= 4: opinion = {"verdict": "관망 필요, 리스크 확인", "color": "#ffc107", "text_color": "black"}
    else: opinion = {"verdict": "투자 주의", "color": "#dc3545"}

    # UI 렌더링
    text_color = opinion.get("text_color", "white")
    st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: {opinion['color']}; color: {text_color}; text-align: center;">
            <div style="font-weight: bold; font-size: 1.2rem;">AI 종합 투자 의견</div>
            <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{opinion['verdict']}</div>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("AI 평가 상세 분석 보기", expanded=True):
        cols = st.columns(4)
        categories = list(scores.keys())
        for i in range(4):
            with cols[i]:
                cat = categories[i]
                score = scores[cat]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': cat, 'font': {'size': 16}},
                    gauge={'axis': {'range': [0, 4]}, 'bar': {'color': "#0d6efd"}}))
                fig.update_layout(height=150, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
        st.info(f"**상세 지표:**\n{', '.join([f'{k}: {v}' for k, v in details.items()])}")


# --- 2. 앱 UI 렌더링 ---
# 세션 상태 초기화
if 'ticker' not in st.session_state:
    st.session_state.ticker = 'NVDA'

st.sidebar.header("종목 검색")
search_ticker = st.sidebar.text_input("종목 코드 입력 (예: AAPL, GOOG)", value=st.session_state.ticker, key="ticker_input").upper()
if st.sidebar.button("분석 실행", key="run_button"):
    st.session_state.ticker = search_ticker
    st.cache_data.clear()
    st.rerun()

try:
    info, financials = get_stock_data(st.session_state.ticker)

    if info is None:
        st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다. 종목 코드를 확인해주세요.")
    else:
        company_name = info.get('shortName', st.session_state.ticker)
        st.title(f"🤖 {company_name} AI 주가 분석")
        st.caption(f"종목코드: {st.session_state.ticker} | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        tab1, tab2 = st.tabs(["**📊 종합 대시보드**", "**📜 재무 및 기업 정보**"])

        with tab1:
            render_valuation_analysis(info)
            st.divider()

            st.subheader("📈 주가 추이 차트")
            period_options = {"오늘": "1d", "1주": "5d", "1개월": "1mo", "1년": "1y", "5년": "5y"}
            selected_period = st.radio("차트 기간 선택", options=period_options.keys(), horizontal=True, key="chart_period")
            
            period_val = period_options[selected_period]
            interval_val = "5m" if period_val == "1d" else "1d"
            
            history = get_history(st.session_state.ticker, period_val, interval_val)
            
            if not history.empty:
                chart_type = 'Candlestick' if interval_val == "5m" else 'Scatter'
                fig = go.Figure()
                if chart_type == 'Candlestick':
                    fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name='분봉'))
                else:
                    fig.add_trace(go.Scatter(x=history.index, y=history['Close'], mode='lines', name='종가'))
                
                is_intraday = selected_period == "오늘"
                fig.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20),
                                  xaxis_rangeslider_visible=not is_intraday,
                                  dragmode='pan' if not is_intraday else False,
                                  xaxis=dict(fixedrange=is_intraday), yaxis=dict(fixedrange=is_intraday))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': not is_intraday})
            else:
                st.warning("차트 데이터를 불러올 수 없습니다.")


        with tab2:
            st.subheader(f"💰 {company_name} 재무 상태")
            if financials is not None and not financials.empty:
                financials_t = financials.T
                financials_t.index = pd.to_datetime(financials_t.index).strftime('%Y-%m')
                fig_fin = go.Figure()
                fig_fin.add_trace(go.Bar(x=financials_t.index, y=financials_t.get('Total Revenue'), name='매출(Revenue)'))
                fig_fin.add_trace(go.Bar(x=financials_t.index, y=financials_t.get('Net Income'), name='순이익(Net Income)'))
                fig_fin.update_layout(title_text="분기별 매출 및 순이익 추이", barmode='group')
                st.plotly_chart(fig_fin, use_container_width=True)
            else:
                st.info("재무 데이터를 가져올 수 없습니다.")
            
            st.divider()
            st.subheader(f"📑 {company_name} 기업 개요")
            st.write(info.get('longBusinessSummary', '기업 개요 정보가 없습니다.'))

except Exception as e:
    st.error(f"앱 실행 중 예상치 못한 오류가 발생했습니다: {e}")
    st.info("종목 코드를 확인하시거나, 잠시 후 페이지를 새로고침 해주세요.")

# 사이드바 알림 기능
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
