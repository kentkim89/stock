import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime

# --- 1. 모든 함수를 코드 상단에 먼저 정의 ---

@st.cache_data(ttl=60) # 1분마다 야후 파이낸스 데이터 갱신
def get_stock_data(ticker):
    """주식 기본 정보, 동종업체 정보, 1일치 분봉 데이터를 가져옵니다."""
    stock = yf.Ticker(ticker)
    info = stock.info
    history = stock.history(period="1d", interval="1m")
    peers = {'AMD': yf.Ticker('AMD').info} # 비교군으로 AMD 정보 추가
    return info, history, peers, stock.news

def calculate_valuation(info, peers, current_price):
    """현재 주가를 기반으로 가치 평가를 동적으로 계산합니다."""
    valuation = {'verdict': "판단 보류", 'color': "gray", 'reasons': []}
    points = 0

    # 1. 애널리스트 목표가 비교
    target_price = info.get('targetMeanPrice')
    if target_price:
        if current_price > target_price * 1.1: # 목표가보다 10% 이상 높으면
            points -= 2
        elif current_price > target_price:
            points -= 1
        else:
            points += 1
        valuation['reasons'].append(f"🎯 **애널리스트 목표가:** ${target_price:,.2f} (현재가 대비: {((current_price/target_price-1)*100):.1f}%)")

    # 2. PEG 비율
    peg_ratio = info.get('pegRatio', 0)
    if peg_ratio > 2.0:
        points -= 1
    elif 0 < peg_ratio < 1.2:
        points += 1
    valuation['reasons'].append(f"📈 **PEG 비율:** {peg_ratio:.2f} (성장성 대비 주가 수준, 1 미만일수록 좋음)")

    # 3. 동종업체 PER 비교
    current_pe = info.get('trailingPE', 0)
    amd_pe = peers['AMD'].get('trailingPE', 0)
    if current_pe > 0 and amd_pe > 0:
        if current_pe > amd_pe * 1.5: # AMD보다 PER이 50% 이상 높으면
            points -= 1
        valuation['reasons'].append(f"📊 **주가수익비율(PER):** {current_pe:.2f} (경쟁사 AMD: {amd_pe:.2f})")

    # 최종 판단
    if points <= -2:
        valuation.update({'verdict': "고평가 가능성", 'color': "#d9534f"}) # 빨간색
    elif points == -1:
        valuation.update({'verdict': "적정 ~ 고평가 구간", 'color': "#f0ad4e"}) # 주황색
    elif points >= 1:
        valuation.update({'verdict': "적정 ~ 저평가 구간", 'color': "#5cb85c"}) # 초록색
    else:
        valuation.update({'verdict': "적정 주가 수준", 'color': "#0275d8"}) # 파란색

    return valuation

def get_ai_outlook_analysis():
    """엔비디아의 AI 관련 전망을 분석하여 텍스트로 반환합니다."""
    analysis = {
        "summary": """
        **AI 시대의 '곡괭이'를 파는 기업**으로 비유되며, AI 산업의 성장에 가장 직접적인 수혜를 받는 기업입니다.
        GPU의 압도적인 성능과 CUDA라는 강력한 소프트웨어 생태계를 기반으로 한 경제적 해자는 단기간에 무너지기 어렵습니다.
        """,
        "strengths": "✅ **독점적 시장 지배력:** AI 학습 및 추론용 GPU 시장의 80% 이상을 점유한 강력한 리더입니다. \n✅ **CUDA 생태계:** 수백만 개발자를 보유한 CUDA 플랫폼은 경쟁사가 넘볼 수 없는 강력한 기술적 해자입니다.",
        "risks": "⚠️ **높은 밸류에이션:** 미래의 성장 기대치가 현재 주가에 상당 부분 반영되어 있어, 시장 성장 둔화 시 변동성이 클 수 있습니다. \n⚠️ **지정학적 리스크:** 미-중 기술 분쟁 심화 시, 중국 관련 매출에 타격이 발생할 수 있습니다."
    }
    return analysis

# --- 2. 앱 UI 렌더링 시작 ---

# 페이지 기본 설정
st.set_page_config(
    page_title="NVIDIA AI 주가 분석 대시보드",
    page_icon="🤖",
    layout="wide",
)

# CSS 스타일 적용 (UI 렌더링 시작 부분으로 이동)
st.markdown("""
    <style>
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .st-emotion-cache-r421ms {
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .st-emotion-cache-1rpb2s1 {
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 NVIDIA AI 주가 분석 대시보드")

try:
    # 데이터 로딩
    info, history, peers, news = get_stock_data("NVDA")
    ai_outlook = get_ai_outlook_analysis()

    if history.empty:
        st.error("현재 주가 데이터를 가져올 수 없습니다. 장 마감 또는 API 일시적 오류일 수 있습니다.")
    else:
        # 최상단 핵심 지표
        latest_price = history['Close'].iloc[-1]
        previous_close = info.get('previousClose', 0)
        price_change = latest_price - previous_close
        percent_change = (price_change / previous_close) * 100 if previous_close else 0
        valuation = calculate_valuation(info, peers, latest_price)

        cols = st.columns([1.5, 1.5, 2.5])
        # ... (이하 나머지 UI 코드는 이전과 동일) ...
        with cols[0]:
            st.metric(
                label="현재가 (USD)",
                value=f"${latest_price:,.2f}",
                delta=f"{price_change:,.2f} ({percent_change:.2f}%)"
            )
        with cols[1]:
            st.metric(
                label="장중 최고 / 최저",
                value=f"${history['High'].max():.2f}",
                delta=f"${history['Low'].min():.2f}"
            )
        with cols[2]:
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {valuation['color']}; color: white;">
                <span style="font-weight: bold; font-size: 1.1rem;">실시간 주가 평가</span><br>
                <span style="font-size: 1.5rem; font-weight: bold;">{valuation['verdict']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        tab1, tab2, tab3 = st.tabs(["**📈 차트 및 가치 평가**", "**🧠 AI 전망 및 기업 정보**", "**📰 최신 뉴스**"])
        with tab1:
            st.subheader("실시간 주가 차트 (1분봉)")
            fig = go.Figure(data=[go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'])])
            fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            with st.container(border=True):
                st.subheader("실시간 가치 평가 상세 근거")
                st.write("현재 주가를 기준으로 애널리스트 목표가, 성장성(PEG), 동종업체(AMD)와의 PER을 종합하여 판단합니다.")
                for reason in valuation['reasons']:
                    st.markdown(f"- {reason}")
        with tab2:
            st.subheader("AI 산업 전망 및 총평")
            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**👍 강점 (Strengths)**")
                    st.markdown(ai_outlook['strengths'])
                with c2:
                    st.write("**👎 리스크 (Risks)**")
                    st.markdown(ai_outlook['risks'])
                st.info(f"**총평:** {ai_outlook['summary']}")

            with st.expander("🏢 **엔비디아 기업 개요 및 주요 재무 정보 보기**"):
                st.write(info.get('longBusinessSummary', '기업 개요 정보 없음'))
                st.markdown(f"""
                - **시가총액:** ${info.get('marketCap', 0):,}
                - **52주 변동폭:** ${info.get('fiftyTwoWeekLow', 0):,.2f} ~ ${info.get('fiftyTwoWeekHigh', 0):,.2f}
                - **배당수익률:** {info.get('dividendYield', 0) * 100:.2f}%
                """)
        with tab3:
            st.subheader("관련 최신 뉴스")
            for item in news[:7]:
                st.write(f"[{item.get('title', '제목 없음')}]({item.get('link', '#')}) - *{item.get('publisher', '출처 불명')}*")

except Exception as e:
    st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
    st.warning("데이터를 불러오지 못했습니다. API 요청 제한 또는 네트워크 문제를 확인해주세요.")

# 사이드바
st.sidebar.header("⚙️ 설정")
if st.sidebar.button('🔄 데이터 새로고침'):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🔔 가격 알림")
high_alert = st.sidebar.number_input("고점 알림 가격 ($)", min_value=0.0, format="%.2f")
low_alert = st.sidebar.number_input("저점 알림 가격 ($)", min_value=0.0, format="%.2f")

if 'latest_price' in locals():
    if high_alert > 0 and latest_price >= high_alert:
        st.sidebar.success(f"📈 목표 고점(${high_alert:,.2f}) 도달!")
    if low_alert > 0 and latest_price <= low_alert:
        st.sidebar.warning(f"📉 목표 저점(${low_alert:,.2f}) 도달!")
