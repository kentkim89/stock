import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime

# --- 1. 페이지 기본 설정 및 함수 정의 ---
st.set_page_config(page_title="AI 주가 분석 대시보드", page_icon="🤖", layout="wide")

# --- 데이터 로딩 함수 (캐싱) ---
@st.cache_data(ttl=300)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if not info.get('marketCap'): return None, None, None
    recs = stock.recommendations
    financials = stock.quarterly_financials
    return info, recs, financials

@st.cache_data(ttl=60)
def get_history(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval)

# --- 상세 지표 해설 렌더링 함수 ---
def render_metric_explanations():
    with st.expander("상세 지표 해설 보기"):
        st.markdown("""
        - **PER (주가수익비율, Price Earnings Ratio):** 현재 주가를 주당순이익(EPS)으로 나눈 값입니다. 기업이 벌어들이는 이익에 비해 주가가 높은지 낮은지를 판단하는 대표적인 지표입니다. **일반적으로 낮을수록 저평가**된 것으로 간주합니다.
        - **PBR (주가순자산비율, Price Book-value Ratio):** 현재 주가를 주당순자산(BPS)으로 나눈 값입니다. 기업의 자산 가치에 비해 주가가 어느 수준인지를 나타냅니다. **보통 1에 가까울수록 적정, 1 미만이면 저평가**된 것으로 봅니다.
        - **PSR (주가매출비율, Price Sales Ratio):** 현재 주가를 주당매출액으로 나눈 값입니다. 아직 이익을 내지 못하는 성장주를 평가할 때 유용하며, **낮을수록 매출 대비 저평가** 상태임을 의미합니다.
        - **PEG (주가수익성장비율, Price Earnings to Growth Ratio):** PER을 주당순이익(EPS) 증가율로 나눈 값입니다. 기업의 성장성을 고려하여 주가의 고평가 여부를 판단하며, **보통 1 미만일 때 성장성에 비해 저평가**된 것으로 봅니다.
        - **ROE (자기자본이익률, Return On Equity):** 기업이 자기자본을 이용하여 얼마나 효율적으로 이익을 냈는지를 나타내는 지표입니다. **ROE가 높을수록 수익성이 좋은 기업**으로 평가됩니다.
        """)

# --- AI 가치평가 스코어카드 렌더링 함수 ---
def render_valuation_scorecard(info):
    st.subheader("⚖️ AI 투자 매력도 분석")
    scores, details = {}, {}

    # (이전 버전의 평가 로직은 그대로 사용)
    pe, pb, ps = info.get('trailingPE'), info.get('priceToBook'), info.get('priceToSalesTrailing12Months')
    pe_score = (4 if 0 < pe <= 15 else 2 if pe <= 25 else 1) if pe else 0
    pb_score = (2 if 0 < pb <= 1.5 else 1) if pb else 0
    scores['가치'] = pe_score + pb_score
    details['PER (주가수익비율)'] = f"{pe:.2f}" if pe else "N/A"
    details['PBR (주가순자산비율)'] = f"{pb:.2f}" if pb else "N/A"
    details['PSR (주가매출비율)'] = f"{ps:.2f}" if ps else "N/A"

    peg, rev_growth = info.get('pegRatio'), info.get('revenueGrowth', 0)
    peg_score = (4 if 0 < peg <= 1 else 2 if peg <= 2 else 0) if peg else 0
    growth_score = (4 if rev_growth > 0.2 else 2 if rev_growth > 0.1 else 0)
    scores['성장성'] = peg_score + growth_score
    details['PEG (주가수익성장비율)'] = f"{peg:.2f}" if peg else "N/A"
    details['매출성장률(YoY)'] = f"{rev_growth*100:.2f}%"

    roe, profit_margin = info.get('returnOnEquity', 0), info.get('profitMargins', 0)
    roe_score = (4 if roe > 0.2 else 2 if roe > 0.15 else 0)
    profit_score = (4 if profit_margin > 0.2 else 2 if profit_margin > 0.1 else 0)
    scores['수익성'] = roe_score + profit_score
    details['ROE (자기자본이익률)'] = f"{roe*100:.2f}%"
    details['순이익률'] = f"{profit_margin*100:.2f}%"
    
    target_price, current_price = info.get('targetMeanPrice'), info.get('currentPrice', 0)
    analyst_score = 0
    if target_price and current_price:
        upside = (target_price / current_price - 1)
        analyst_score = (4 if upside > 0.3 else 2 if upside > 0.1 else 1)
    scores['애널리스트'] = analyst_score
    details['목표가 상승여력'] = f"{upside*100:.2f}%" if target_price and current_price else "N/A"
    
    total_score = sum(scores.values())
    opinion = {"verdict": "관망 필요", "color": "#ffc107", "text_color": "black"}
    if total_score >= 18: opinion = {"verdict": "강력 매수 고려", "color": "#198754"}
    elif total_score >= 12: opinion = {"verdict": "긍정적, 분할 매수", "color": "#0d6efd"}
    elif total_score < 6: opinion = {"verdict": "투자 주의", "color": "#dc3545"}

    text_color = opinion.get("text_color", "white")
    st.markdown(f"""<div style="padding: 1rem; border-radius: 0.5rem; background-color: {opinion['color']}; color: {text_color}; text-align: center;">
            <div style="font-weight: bold; font-size: 1.2rem;">AI 종합 투자 의견</div>
            <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{opinion['verdict']}</div></div>""", unsafe_allow_html=True)
    
    with st.expander("AI 평가 상세 분석 및 지표 해설 보기", expanded=True):
        st.info(f"**상세 지표:** {', '.join([f'{k}: {v}' for k, v in details.items()])}")
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
        # 상세 지표 해설 함수 호출
        render_metric_explanations()

# --- 2. 앱 UI 렌더링 ---
if 'ticker' not in st.session_state: st.session_state.ticker = 'NVDA'

st.sidebar.header("종목 검색")
search_ticker = st.sidebar.text_input("종목 코드 입력 (예: AAPL, GOOG)", value=st.session_state.ticker, key="ticker_input").upper()
if st.sidebar.button("분석 실행", key="run_button"):
    st.session_state.ticker = search_ticker
    st.cache_data.clear()
    st.rerun()

try:
    info, recs, financials = get_stock_data(st.session_state.ticker)

    if info is None:
        st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다. 종목 코드를 확인해주세요.")
    else:
        company_name = info.get('longName', info.get('shortName', st.session_state.ticker))
        st.title(f"🤖 {company_name} AI 주가 분석")
        st.caption(f"종목코드: {st.session_state.ticker} | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        tab1, tab2 = st.tabs(["**📊 종합 대시보드**", "**📂 재무 및 애널리스트 상세**"])

        with tab1:
            render_valuation_scorecard(info)
            st.divider()

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
            else: st.warning("차트 데이터를 불러올 수 없습니다.")

        with tab2:
            st.subheader(f"🎯 {company_name} 애널리스트 컨센서스")
            target_price, current_price = info.get('targetMeanPrice'), info.get('currentPrice', 0)
            if target_price and current_price:
                upside = (target_price/current_price-1)*100
                st.metric(label=f"애널리스트 평균 목표가 (상승여력: {upside:.2f}%)", value=f"{target_price:,.2f} {info.get('financialCurrency')}", delta=f"현재가: {current_price:,.2f}")
            
            # 안정화된 애널리스트 추천 막대그래프 (한글 번역 포함)
            if recs is not None and not recs.empty and 'To Grade' in recs.columns:
                # 번역 맵
                opinion_map = {'Strong Buy': '강력 매수', 'Buy': '매수', 'Hold': '보유', 'Sell': '매도', 'Strong Sell': '강력 매도', 
                               'Overweight': '비중 확대', 'Equal-Weight': '중립', 'Underweight': '비중 축소'}
                recs['번역'] = recs['To Grade'].map(opinion_map).fillna(recs['To Grade'])
                recs_summary = recs['번역'].value_counts()
                
                fig_recs = px.bar(recs_summary, x=recs_summary.index, y=recs_summary.values,
                                  labels={'x': '투자의견', 'y': '의견 수'}, title="최근 투자의견 동향", color=recs_summary.index)
                st.plotly_chart(fig_recs, use_container_width=True)
            else:
                st.info(f"{company_name}에 대한 애널리스트 투자의견 데이터가 부족하여 표시할 수 없습니다.")
            
            st.divider()

            st.subheader(f"💰 {company_name} 재무 상태")
            if financials is not None and not financials.empty:
                financials_t = financials.T.iloc[:4] # 최근 4분기 데이터만 선택
                financials_t.index = pd.to_datetime(financials_t.index).strftime('%Y-%m')
                fig_fin = go.Figure(data=[go.Bar(name='매출(Revenue)', x=financials_t.index, y=financials_t.get('Total Revenue')),
                                          go.Bar(name='순이익(Net Income)', x=financials_t.index, y=financials_t.get('Net Income'))])
                fig_fin.update_layout(barmode='group', title_text="분기별 매출 및 순이익 추이")
                st.plotly_chart(fig_fin, use_container_width=True)
            else: st.info("재무 데이터를 가져올 수 없습니다.")
            
            st.divider()
            st.subheader(f"📑 {company_name} 기업 개요")
            st.write(info.get('longBusinessSummary', '기업 개요 정보가 없습니다.'))

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
