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
    if not info.get('regularMarketPrice'): # 데이터가 없으면 None 반환
        return None, None, None, None
    recs = stock.recommendations
    financials = stock.quarterly_financials
    news = stock.news
    return info, recs, financials, news

@st.cache_data(ttl=60)
def get_history(ticker, period="1d", interval="1m"):
    """차트 기간에 맞는 주가 데이터를 가져옵니다."""
    return yf.Ticker(ticker).history(period=period, interval=interval)


# --- 수익성 진단 함수 ---
def check_profitability(info):
    """기업의 수익성을 진단하고 결과를 반환합니다."""
    # (이전 코드와 동일)
    profit_margin = info.get('profitMargins', 0)
    operating_margin = info.get('operatingMargins', 0)
    net_income = info.get('netIncomeToCommon', 0)
    free_cashflow = info.get('freeCashflow', 0)

    score = 0
    reasons = []

    if profit_margin > 0.1: score += 1
    if operating_margin > 0.15: score += 1
    if net_income > 0: score += 1
    if free_cashflow > 0:
        score += 1
        reasons.append(f"✅ **잉여현금흐름:** ${free_cashflow/1_000_000_000:.2f}B (투자 후 남는 현금 흑자)")
    else:
        reasons.append(f"⚠️ **잉여현금흐름:** ${free_cashflow/1_000_000_000:.2f}B (투자 후 현금 부족)")

    reasons.insert(0, f"✅ **순이익률:** {profit_margin*100:.2f}%")
    reasons.insert(1, f"✅ **영업이익률:** {operating_margin*100:.2f}%")

    if score >= 3:
        return {"verdict": "수익성 우수", "color": "#5cb85c", "reasons": reasons}
    elif score >= 1:
        return {"verdict": "수익성 보통", "color": "#0275d8", "reasons": reasons}
    else:
        return {"verdict": "수익성 부진", "color": "#d9534f", "reasons": reasons}


# --- 가치 평가 함수 ---
def calculate_valuation(info, current_price, recs):
    """현재 주가에 대한 가치 평가 총평을 생성합니다."""
    # (이전 코드와 동일, 안정성만 강화)
    summary = ""
    target_price = info.get('targetMeanPrice')
    peg_ratio = info.get('pegRatio')

    # 오류 수정: recs 데이터프레임과 'To Grade' 컬럼 존재 여부 확인
    has_recs = recs is not None and not recs.empty and 'To Grade' in recs.columns
    recs_summary = recs.tail(10)['To Grade'].value_counts() if has_recs else pd.Series()

    if target_price:
        if current_price > target_price:
            summary += f"현재 주가는 애널리스트 평균 목표가(${target_price:,.2f})를 **상회**하고 있어 단기적인 상승 여력에 대한 부담이 있습니다. "
        else:
            upside = (target_price / current_price - 1) * 100
            summary += f"현재 주가는 애널리스트 평균 목표가(${target_price:,.2f}) 대비 **{upside:.2f}%의 상승 여력**이 있는 것으로 평가됩니다. "
    if peg_ratio:
        if peg_ratio > 2.0:
            summary += f"다만, 성장성 대비 주가 수준을 나타내는 PEG 비율이 {peg_ratio:.2f}로 다소 높아, **미래 성장 기대치가 주가에 많이 반영**된 상태입니다. "
        elif 0 < peg_ratio < 1.2:
            summary += f"성장성 대비 주가 수준을 나타내는 PEG 비율이 {peg_ratio:.2f}로 **매력적인 수준**으로 평가됩니다. "
    if has_recs and ('Buy' in recs_summary.index or 'Strong Buy' in recs_summary.index):
        summary += "최근 애널리스트들은 대체로 **긍정적인 투자의견**을 유지하고 있습니다."
    elif has_recs:
        summary += "최근 애널리스트들의 투자의견은 다소 엇갈리고 있습니다."

    return summary if summary else "가치 평가 정보를 종합하기 어렵습니다."


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

st.title(f"💡 {st.session_state.ticker} AI 주가 분석 대시보드")

try:
    info, recs, financials, news = get_stock_data(st.session_state.ticker)

    if info is None:
        st.error("유효하지 않은 종목 코드이거나 데이터를 가져올 수 없습니다. 코드를 확인 후 다시 시도해주세요.")
    else:
        # --- 메인 탭 구성 ---
        tab1, tab2, tab3 = st.tabs(["**📊 종합 & 차트 분석**", "**⚖️ 가치 평가 & 재무**", "**📰 관련 뉴스**"])

        with tab1:
            # --- 최상단 핵심 지표 ---
            latest_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            previous_close = info.get('previousClose', 0)
            price_change = latest_price - previous_close
            percent_change = (price_change / previous_close) * 100 if previous_close else 0
            profitability = check_profitability(info)

            cols = st.columns([1.5, 1.5, 2.5])
            with cols[0]:
                st.metric(label=f"현재가 ({info.get('currency', 'USD')})", value=f"{latest_price:,.2f}", delta=f"{price_change:,.2f} ({percent_change:.2f}%)")
            with cols[1]:
                st.metric(label="52주 최고가 / 최저가", value=f"{info.get('fiftyTwoWeekHigh', 0):,.2f}", delta=f"{info.get('fiftyTwoWeekLow', 0):,.2f}")
            with cols[2]:
                st.markdown(f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: {profitability['color']}; color: white;">
                        <span style="font-weight: bold; font-size: 1.1rem;">수익성 진단</span><br>
                        <span style="font-size: 1.5rem; font-weight: bold;">{profitability['verdict']}</span>
                    </div>
                """, unsafe_allow_html=True)
            st.divider()

            # --- 기간 선택 가능한 차트 ---
            st.subheader("주가 추이 차트")
            period_options = {"1개월": "1mo", "6개월": "6mo", "1년": "1y", "5년": "5y", "실시간": "1d"}
            selected_period_label = st.radio("차트 기간 선택", options=period_options.keys(), horizontal=True, index=len(period_options)-1)
            period = period_options[selected_period_label]
            interval = "1m" if period == "1d" else "1d"
            
            history = get_history(st.session_state.ticker, period, interval)

            if not history.empty:
                chart_type = 'Candlestick' if interval == '1m' else 'Line'
                fig = go.Figure()
                if chart_type == 'Candlestick':
                     fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name='분봉'))
                else:
                     fig.add_trace(go.Scatter(x=history.index, y=history['Close'], mode='lines', name='일봉 종가'))

                # 모바일 환경 최적화 설정
                is_mobile_friendly = selected_period_label == "실시간"
                fig.update_layout(
                    height=400, margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_rangeslider_visible=not is_mobile_friendly,
                    dragmode='pan' if not is_mobile_friendly else False,
                    xaxis=dict(fixedrange=is_mobile_friendly),
                    yaxis=dict(fixedrange=is_mobile_friendly)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': not is_mobile_friendly})
                if is_mobile_friendly:
                    st.info("💡 실시간 차트는 스마트폰 사용 편의를 위해 확대/이동 기능이 비활성화됩니다.")
            else:
                 st.warning("선택된 기간의 차트 데이터를 불러올 수 없습니다.")

        with tab2:
            st.subheader("적정주가 종합 평가")
            valuation_summary = calculate_valuation(info, latest_price, recs)
            st.write(valuation_summary)
            st.divider()

            st.subheader("애널리스트 투자의견 분포")
            # 오류 수정: recs 데이터프레임과 'To Grade' 컬럼 존재 여부 확인
            if recs is not None and not recs.empty and 'To Grade' in recs.columns:
                recs_summary = recs.tail(25)['To Grade'].value_counts()
                fig_recs = px.bar(recs_summary, x=recs_summary.index, y=recs_summary.values,
                                  labels={'x': '투자의견', 'y': '의견 수'},
                                  title="최근 25개 투자의견 동향", color=recs_summary.index)
                st.plotly_chart(fig_recs, use_container_width=True)
            else:
                st.info("애널리스트 투자의견 데이터가 부족합니다.")
            st.divider()

            st.subheader("핵심 재무 지표 추이")
            if financials is not None and not financials.empty:
                financials_t = financials.T
                financials_t.index = pd.to_datetime(financials_t.index).strftime('%Y-%m')
                fig_fin = go.Figure()
                fig_fin.add_trace(go.Bar(x=financials_t.index, y=financials_t.get('Total Revenue'), name='매출'))
                fig_fin.add_trace(go.Bar(x=financials_t.index, y=financials_t.get('Net Income'), name='순이익'))
                fig_fin.update_layout(title_text="분기별 매출 및 순이익", barmode='group')
                st.plotly_chart(fig_fin, use_container_width=True)
            else:
                st.info("재무 데이터를 가져올 수 없습니다.")


        with tab3:
            st.subheader("관련 최신 뉴스")
            if news:
                for item in news[:8]:
                    st.write(f"[{item.get('title', '제목 없음')}]({item.get('link', '#')}) - *{item.get('publisher', '출처 불명')}*")
            else:
                st.info("관련 뉴스가 없습니다.")

except Exception as e:
    st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
    st.info("종목 코드를 확인하시거나, 잠시 후 다시 시도해주세요. 일부 데이터가 누락되었을 수 있습니다.")


# 사이드바 알림 기능
st.sidebar.markdown("---")
st.sidebar.header("가격 알림 설정")
high_alert = st.sidebar.number_input("고점 알림 가격", min_value=0.0, format="%.2f", key="high_alert")
low_alert = st.sidebar.number_input("저점 알림 가격", min_value=0.0, format="%.2f", key="low_alert")

if 'latest_price' in locals():
    if high_alert > 0 and latest_price >= high_alert:
        st.sidebar.success(f"📈 목표 고점(${high_alert:,.2f}) 도달!")
    if low_alert > 0 and latest_price <= low_alert:
        st.sidebar.warning(f"📉 목표 저점(${low_alert:,.2f}) 도달!")
