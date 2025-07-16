import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime

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
    # .info가 비어있으면 유효하지 않은 티커로 간주
    if not info or info.get('regularMarketPrice') is None:
        return None
    history = stock.history(period="1d", interval="1m")
    recs = stock.recommendations
    financials = stock.quarterly_financials
    return info, history, recs, financials

# --- 수익성 진단 함수 ---
def check_profitability(info):
    """기업의 수익성을 진단하고 결과를 반환합니다."""
    profit_margin = info.get('profitMargins', 0)
    operating_margin = info.get('operatingMargins', 0)
    net_income = info.get('netIncomeToCommon', 0)
    free_cashflow = info.get('freeCashflow', 0)

    score = 0
    reasons = []

    if profit_margin > 0.1: # 순이익률 10% 이상
        score += 1
    if operating_margin > 0.15: # 영업이익률 15% 이상
        score += 1
    if net_income > 0:
        score += 1
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
    summary = ""
    target_price = info.get('targetMeanPrice')
    peg_ratio = info.get('pegRatio')
    recs_summary = recs.tail(10)['To Grade'].value_counts() if recs is not None else pd.Series()

    # 1. 애널리스트 목표가 기반 평가
    if target_price:
        if current_price > target_price:
            summary += f"현재 주가는 애널리스트 평균 목표가(${target_price:,.2f})를 **상회**하고 있어 단기적인 상승 여력에 대한 부담이 있습니다. "
        else:
            upside = (target_price / current_price - 1) * 100
            summary += f"현재 주가는 애널리스트 평균 목표가(${target_price:,.2f}) 대비 **{upside:.2f}%의 상승 여력**이 있는 것으로 평가됩니다. "

    # 2. 성장성(PEG) 기반 평가
    if peg_ratio:
        if peg_ratio > 2.0:
            summary += f"다만, 성장성 대비 주가 수준을 나타내는 PEG 비율이 {peg_ratio:.2f}로 다소 높아, **미래 성장 기대치가 주가에 많이 반영**된 상태입니다. "
        elif 0 < peg_ratio < 1.2:
            summary += f"성장성 대비 주가 수준을 나타내는 PEG 비율이 {peg_ratio:.2f}로 **매력적인 수준**으로 평가됩니다. "

    # 3. 최근 추천 동향
    if not recs_summary.empty and ('Buy' in recs_summary.index or 'Strong Buy' in recs_summary.index):
        summary += "최근 애널리스트들은 대체로 **긍정적인 투자의견**을 유지하고 있습니다."
    else:
        summary += "최근 애널리스트들의 투자의견은 다소 엇갈리고 있습니다."

    return summary if summary else "가치 평가 정보를 종합하기 어렵습니다."


# --- 2. 앱 UI 렌더링 ---

# 세션 상태 초기화
if 'ticker' not in st.session_state:
    st.session_state.ticker = 'NVDA'

# 사이드바 구성
st.sidebar.header("종목 검색")
search_ticker = st.sidebar.text_input("종목 코드 입력 (예: AAPL, GOOG)", value=st.session_state.ticker).upper()
if st.sidebar.button("분석 실행"):
    st.session_state.ticker = search_ticker
    st.cache_data.clear() # 티커 변경 시 캐시 초기화
    st.rerun()

st.title(f"💡 {st.session_state.ticker} AI 주가 분석 대시보드")

try:
    # 데이터 로딩
    data_load_state = st.text("데이터를 불러오는 중입니다...")
    data = get_stock_data(st.session_state.ticker)
    data_load_state.empty()

    if data is None:
        st.error("유효하지 않은 종목 코드이거나 데이터를 가져올 수 없습니다. 코드를 확인 후 다시 시도해주세요.")
    else:
        info, history, recs, financials = data

        # --- 메인 탭 구성 ---
        tab1, tab2, tab3, tab4 = st.tabs(["**📊 종합 모니터링**", "**⚖️ 가치 평가 및 전망**", "** financially 재무 분석**", "**📰 최신 뉴스**"])

        with tab1:
            # --- 최상단 핵심 지표 ---
            if not history.empty:
                latest_price = history['Close'].iloc[-1]
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

                # --- 실시간 차트 (모바일 최적화) ---
                st.subheader("실시간 주가 차트 (1분봉)")
                fig = go.Figure(data=[go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'])])
                # 모바일 환경에서 확대/이동 방지
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    dragmode=False,  # 드래그 비활성화
                    xaxis=dict(fixedrange=True),  # X축 고정
                    yaxis=dict(fixedrange=True)   # Y축 고정
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.info("💡 위 차트는 스마트폰 사용 편의를 위해 확대/축소 및 이동 기능이 비활성화되어 있습니다.")

            else:
                st.info("장 마감으로 실시간 주가 정보를 표시할 수 없습니다.")

        with tab2:
            st.subheader("적정주가 종합 평가")
            if not history.empty:
                valuation_summary = calculate_valuation(info, latest_price, recs)
                st.write(valuation_summary)
            else:
                st.warning("장 마감으로 실시간 주가 기준 평가를 제공할 수 없습니다.")

            st.divider()

            st.subheader("애널리스트 투자의견 분포")
            if recs is not None and not recs.empty:
                recs_summary = recs.tail(25)['To Grade'].value_counts()
                fig_recs = px.bar(recs_summary, x=recs_summary.index, y=recs_summary.values,
                                  labels={'x': '투자의견', 'y': '의견 수'},
                                  title="최근 25개 투자의견 동향", color=recs_summary.index)
                st.plotly_chart(fig_recs, use_container_width=True)
            else:
                st.info("애널리스트 투자의견 데이터가 부족합니다.")

            with st.expander("기업 개요 보기"):
                 st.write(info.get('longBusinessSummary', '기업 개요 정보가 없습니다.'))

        with tab3:
            st.subheader("핵심 재무 지표 추이")
            if financials is not None and not financials.empty:
                # 분기별 매출 및 순이익 차트
                financials_t = financials.T
                financials_t.index = pd.to_datetime(financials_t.index).strftime('%Y-%m')
                fig_fin = go.Figure()
                fig_fin.add_trace(go.Bar(x=financials_t.index, y=financials_t['Total Revenue'], name='매출 (Revenue)'))
                fig_fin.add_trace(go.Bar(x=financials_t.index, y=financials_t['Net Income'], name='순이익 (Net Income)'))
                fig_fin.update_layout(title_text="분기별 매출 및 순이익", barmode='group')
                st.plotly_chart(fig_fin, use_container_width=True)

                st.write("#### 수익성 진단 상세")
                for reason in profitability['reasons']:
                    st.markdown(f"- {reason}")
            else:
                st.info("재무 데이터를 가져올 수 없습니다.")

        with tab4:
            st.subheader("관련 최신 뉴스")
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
high_alert = st.sidebar.number_input("고점 알림 가격", min_value=0.0, format="%.2f")
low_alert = st.sidebar.number_input("저점 알림 가격", min_value=0.0, format="%.2f")

if 'latest_price' in locals():
    if high_alert > 0 and latest_price >= high_alert:
        st.sidebar.success(f"📈 목표 고점(${high_alert:,.2f}) 도달!")
    if low_alert > 0 and latest_price <= low_alert:
        st.sidebar.warning(f"📉 목표 저점(${low_alert:,.2f}) 도달!")
