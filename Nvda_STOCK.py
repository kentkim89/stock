import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# --- 1. 페이지 기본 설정 및 함수 정의 ---

st.set_page_config(
    page_title="AI 주가 분석 대시보드",
    page_icon="💡",
    layout="wide",
)

# --- 캐싱을 사용한 데이터 로딩 함수 ---
@st.cache_data(ttl=300) # 5분마다 데이터 갱신
def get_stock_data(ticker):
    """입력된 티커에 대한 모든 주식 데이터를 가져옵니다."""
    stock = yf.Ticker(ticker)
    info = stock.info
    # 유효한 종목인지 확인 (marketCap이 없으면 보통 비활성 종목)
    if not info.get('marketCap'):
        return None, None, None
    recs = stock.recommendations
    financials = stock.quarterly_financials
    return info, recs, financials

@st.cache_data(ttl=60)
def get_history(ticker, period="1y"):
    """차트 기간에 맞는 일봉 데이터를 가져옵니다."""
    return yf.Ticker(ticker).history(period=period, interval="1d")

# --- AI 기반 투자 의견 생성 함수 ---
def generate_investment_opinion(info, history):
    """여러 지표를 종합하여 AI 기반 투자 의견을 생성합니다."""
    scores = {}
    
    # 1. 가치 평가 (Valuation) - 10점 만점
    peg = info.get('pegRatio')
    pe = info.get('trailingPE')
    valuation_score = 0
    if peg and 0 < peg < 1: valuation_score += 5
    elif peg and peg < 2: valuation_score += 2
    if pe and 0 < pe < 20: valuation_score += 5
    elif pe and pe < 40: valuation_score += 2
    scores['가치'] = valuation_score

    # 2. 성장성 (Growth) - 10점 만점
    rev_growth = info.get('revenueGrowth', 0)
    growth_score = 0
    if rev_growth > 0.3: growth_score += 5
    elif rev_growth > 0.1: growth_score += 3
    # 분기별 순이익 성장률 (단순 계산)
    if info.get('earningsQuarterlyGrowth', 0) > 0.3: growth_score += 5
    elif info.get('earningsQuarterlyGrowth', 0) > 0.1: growth_score += 3
    scores['성장성'] = growth_score

    # 3. 수익성 (Profitability) - 10점 만점
    profit_margin = info.get('profitMargins', 0)
    roe = info.get('returnOnEquity', 0)
    profit_score = 0
    if profit_margin > 0.2: profit_score += 5
    elif profit_margin > 0.1: profit_score += 3
    if roe > 0.2: profit_score += 5
    elif roe > 0.15: profit_score += 3
    scores['수익성'] = profit_score

    # 4. 기술적 모멘텀 (Momentum) - 10점 만점
    if not history.empty:
        ma50 = history['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = history['Close'].rolling(window=200).mean().iloc[-1]
        current_price = history['Close'].iloc[-1]
        momentum_score = 0
        if current_price > ma50: momentum_score += 5
        if current_price > ma200: momentum_score += 5
        scores['모멘텀'] = momentum_score
    else:
        scores['모멘텀'] = 0

    # 최종 점수 및 의견
    total_score = sum(scores.values())
    opinion = {
        "verdict": "정보 분석 중", "color": "#6c757d", "score": total_score, "details": scores
    }
    if total_score >= 30:
        opinion.update({"verdict": "투자 적극 고려", "color": "#198754"})
    elif total_score >= 20:
        opinion.update({"verdict": "긍정적, 신중한 접근", "color": "#0d6efd"})
    elif total_score >= 10:
        opinion.update({"verdict": "투자 고려, 리스크 확인", "color": "#ffc107", "text_color": "black"})
    else:
        opinion.update({"verdict": "투자 주의 필요", "color": "#dc3545"})
    
    return opinion


# --- 2. 앱 UI 렌더링 ---

# 세션 상태 초기화
if 'ticker' not in st.session_state:
    st.session_state.ticker = 'NVDA'

# 사이드바
st.sidebar.header("종목 검색")
search_ticker = st.sidebar.text_input("종목 코드 입력 (예: AAPL, GOOG)", value=st.session_state.ticker, key="ticker_input").upper()
if st.sidebar.button("분석 실행"):
    st.session_state.ticker = search_ticker
    st.cache_data.clear()
    st.rerun()

try:
    info, recs, financials = get_stock_data(st.session_state.ticker)

    if info is None:
        st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다. 종목 코드를 확인해주세요.")
    else:
        company_name = info.get('shortName', st.session_state.ticker)
        st.title(f"💡 {company_name} AI 주가 분석 대시보드")

        # --- 메인 탭 구성 ---
        tab1, tab2, tab3 = st.tabs(["**📊 종합 & 차트 분석**", "**⚖️ 재무 & 애널리스트**", "**📰 기업 개요 & 뉴스**"])

        with tab1:
            # AI 투자 의견 생성을 위해 1년치 데이터 미리 로드
            history_1y = get_history(st.session_state.ticker, "1y")
            ai_opinion = generate_investment_opinion(info, history_1y)
            
            # --- 최상단 핵심 지표 ---
            latest_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            previous_close = info.get('previousClose', 0)
            price_change = latest_price - previous_close
            percent_change = (price_change / previous_close) * 100 if previous_close else 0

            cols = st.columns([1.5, 1.5, 2.5])
            with cols[0]:
                st.metric(label=f"현재가 ({info.get('currency', 'USD')})", value=f"{latest_price:,.2f}", delta=f"{price_change:,.2f} ({percent_change:.2f}%)")
            with cols[1]:
                st.metric(label="52주 최고가 / 최저가", value=f"{info.get('fiftyTwoWeekHigh', 0):,.2f}", delta=f"{info.get('fiftyTwoWeekLow', 0):,.2f}")
            with cols[2]:
                text_color = ai_opinion.get("text_color", "white")
                st.markdown(f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: {ai_opinion['color']}; color: {text_color};">
                        <span style="font-weight: bold; font-size: 1.1rem;">AI 종합 투자 의견</span><br>
                        <span style="font-size: 1.5rem; font-weight: bold;">{ai_opinion['verdict']}</span>
                    </div>
                """, unsafe_allow_html=True)
            st.divider()

            # --- 기간 선택 가능한 차트 ---
            st.subheader(f"{company_name} 주가 추이 차트")
            period_options = {"3개월": "3mo", "6개월": "6mo", "1년": "1y", "5년": "5y"}
            selected_period_label = st.radio("차트 기간 선택", options=period_options.keys(), horizontal=True)
            period = period_options[selected_period_label]
            history_chart = get_history(st.session_state.ticker, period)
            
            if not history_chart.empty:
                fig_chart = go.Figure(data=[go.Scatter(x=history_chart.index, y=history_chart['Close'], mode='lines', name='종가')])
                fig_chart.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_chart, use_container_width=True)
            else:
                st.warning("차트 데이터를 불러올 수 없습니다.")

            # --- AI 투자 의견 상세 분석 ---
            with st.container(border=True):
                st.subheader("AI 투자 매력도 상세 분석")
                details = ai_opinion['details']
                total_score = ai_opinion['score']
                st.progress(total_score / 40, text=f"종합 점수: {total_score} / 40")

                detail_cols = st.columns(4)
                for i, (cat, score) in enumerate(details.items()):
                    with detail_cols[i]:
                        st.metric(label=cat, value=f"{score} / 10")


        with tab2:
            st.subheader(f"{company_name} 재무 분석")
            if financials is not None and not financials.empty:
                financials_t = financials.T
                financials_t.index = pd.to_datetime(financials_t.index).strftime('%Y-%m')
                fig_fin = go.Figure()
                fig_fin.add_trace(go.Bar(x=financials_t.index, y=financials_t.get('Total Revenue'), name='매출'))
                fig_fin.add_trace(go.Bar(x=financials_t.index, y=financials_t.get('Net Income'), name='순이익'))
                fig_fin.update_layout(title_text="분기별 매출 및 순이익 추이", barmode='group')
                st.plotly_chart(fig_fin, use_container_width=True)
            else:
                st.info("재무 데이터를 가져올 수 없습니다.")
            st.divider()

            st.subheader("애널리스트 투자의견 분포")
            # 오류 수정: recs와 컬럼 존재 여부를 모두 확인
            if recs is not None and not recs.empty and 'To Grade' in recs.columns:
                recs_summary = recs.tail(25)['To Grade'].value_counts()
                fig_recs = px.bar(recs_summary, x=recs_summary.index, y=recs_summary.values,
                                  labels={'x': '투자의견', 'y': '의견 수'},
                                  title="최근 25개 투자의견 동향", color=recs_summary.index)
                st.plotly_chart(fig_recs, use_container_width=True)
            else:
                st.info(f"{company_name}에 대한 애널리스트 투자의견 데이터가 부족합니다.")


        with tab3:
            st.subheader(f"{company_name} 기업 개요")
            st.write(info.get('longBusinessSummary', '기업 개요 정보가 없습니다.'))
            st.divider()

            st.subheader("관련 최신 뉴스")
            # 뉴스 소스 안정화
            news_list = info.get('news', [])
            if news_list:
                for item in news_list[:8]:
                    st.write(f"[{item.get('title', '제목 없음')}]({item.get('link', '#')}) - *{item.get('publisher', '출처 불명')}*")
            else:
                st.info("관련 뉴스가 없습니다.")

except Exception as e:
    st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
    st.info("종목 코드를 확인하시거나, 잠시 후 다시 시도해주세요.")

# --- 사이드바 알림 기능 ---
st.sidebar.markdown("---")
st.sidebar.header("가격 알림 설정")
high_alert = st.sidebar.number_input("고점 알림 가격", min_value=0.0, format="%.2f", key="high_alert")
low_alert = st.sidebar.number_input("저점 알림 가격", min_value=0.0, format="%.2f", key="low_alert")

if 'latest_price' in locals():
    if high_alert > 0 and latest_price >= high_alert:
        st.sidebar.success(f"📈 목표 고점(${high_alert:,.2f}) 도달!")
    if low_alert > 0 and latest_price <= low_alert:
        st.sidebar.warning(f"📉 목표 저점(${low_alert:,.2f}) 도달!")
