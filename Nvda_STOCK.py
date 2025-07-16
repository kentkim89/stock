import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import google.generativeai as genai
from gnews import GNews

# --- 1. PAGE CONFIG & SETUP ---
st.set_page_config(page_title="AI Stock Analysis Platform", page_icon="🚀", layout="wide")

# --- GEMINI & SESSION STATE ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (FileNotFoundError, KeyError):
    st.error("ERROR: Gemini API Key not set. Please check your .streamlit/secrets.toml file and add it to your Streamlit Cloud Secrets.")
    st.stop()

if 'ticker' not in st.session_state: st.session_state.ticker = 'NVDA'
if 'ai_analysis' not in st.session_state: st.session_state.ai_analysis = {}

# --- DATA LOADING FUNCTIONS ---
@st.cache_data(ttl=86400)
def get_latest_tickers():
    """Downloads and refines the latest list of stocks and ETFs from NASDAQ servers."""
    try:
        nasdaq_df = pd.read_csv("ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt", sep='|')
        other_df = pd.read_csv("ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt", sep='|')
        nasdaq_tickers = nasdaq_df[['Symbol', 'Security Name']]
        other_tickers = other_df[['ACT Symbol', 'Security Name']]
        other_tickers = other_tickers.rename(columns={'ACT Symbol': 'Symbol'})
        all_tickers = pd.concat([nasdaq_tickers, other_tickers]).dropna()
        all_tickers = all_tickers[~all_tickers['Symbol'].str.contains(r'[\$\.]', regex=True)]
        all_tickers = all_tickers.rename(columns={'Security Name': 'Name'})
        all_tickers['display'] = all_tickers['Symbol'] + " - " + all_tickers['Name']
        return all_tickers.sort_values(by='Symbol').reset_index(drop=True)
    except Exception:
        # Failsafe default list
        return pd.DataFrame({
            'Symbol': ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ'],
            'Name': ['NVIDIA Corporation', 'Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 'SPDR S&P 500 ETF Trust', 'Invesco QQQ Trust'],
            'display': ['NVDA - NVIDIA Corporation', 'AAPL - Apple Inc.', 'MSFT - Microsoft Corporation', 'GOOGL - Alphabet Inc.', 'SPY - SPDR S&P 500 ETF Trust', 'QQQ - Invesco QQQ Trust']
        })

@st.cache_data(ttl=300)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if not info.get('marketCap') and not info.get('totalAssets'): return None, None, None
    financials = stock.quarterly_financials if info.get('quoteType') == 'EQUITY' else None
    google_news = GNews(language='ko', country='KR')
    company_name = info.get('shortName', ticker)
    news = google_news.get_news(f'{company_name} 주가')
    return info, financials, news

@st.cache_data(ttl=60)
def get_history(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval)

# --- AI & VALUATION FUNCTIONS ---
@st.cache_data(ttl=600)
def generate_ai_analysis(info, data, analysis_type):
    model = genai.GenerativeModel('gemini-1.5-flash')
    company_name = info.get('longName', '해당 종목')
    today_date = datetime.now().strftime('%Y년 %m월 %d일')
    prompt = ""

    if analysis_type == 'chart':
        history = data
        ma50 = history['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = history['Close'].rolling(window=200).mean().iloc[-1]
        prompt = f"""당신은 차트 기술적 분석(CMT) 전문가입니다. **오늘은 {today_date}입니다.** 다음 데이터를 바탕으로 '{company_name}'의 주가 차트를 상세히 분석해주세요.
        - 현재가: {info.get('currentPrice', 'N/A'):.2f}, 50일 이동평균선: {ma50:.2f}, 200일 이동평균선: {ma200:.2f}
        **분석:** (현재 추세, 이동평균선의 관계, 주요 지지/저항선, 종합적인 기술적 의견)"""
    elif analysis_type == 'financial':
        financials = data
        latest_date = financials.columns[0].strftime('%Y년 %m월')
        prompt = f"""당신은 최고재무책임자(CFO)입니다. 다음은 **{latest_date} 기준**의 최신 재무 데이터입니다. 이를 보고 '{company_name}'의 재무 건전성을 분석하고 종합 평가를 내려주세요.
        - **수익성:** 총이익률 {info.get('grossMargins', 0)*100:.2f}%, ROE {info.get('returnOnEquity', 0)*100:.2f}%
        - **안정성:** 부채비율(Debt/Equity) {info.get('debtToEquity', 'N/A')}
        **AI 재무 진단 리포트:** (각 지표의 의미를 설명하고, 재무적 강점과 약점을 평가한 후, 최종 등급을 매겨주세요.)"""
    elif analysis_type == 'news':
        news = data
        news_headlines = "\n".join([f"- {article['title']}" for article in news[:7]]) if news else "관련 뉴스 없음"
        prompt = f"""당신은 금융 시장 분석가입니다. **오늘은 {today_date}입니다.** 다음은 구글 뉴스에서 수집된 '{company_name}' 관련 최신 뉴스 헤드라인입니다. 이를 바탕으로 현재 시장의 분위기와 핵심 이슈를 요약해주세요.
        - **최신 뉴스:**\n{news_headlines}
        **뉴스 요약 및 시장 분위기 분석:** (긍정적, 부정적, 중립적 요소를 구분하여 분석하고, 현재 투자자들이 가장 주목하는 이슈가 무엇인지 설명해주세요.)"""

    if not prompt: return "분석 유형 오류"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 분석 중 오류 발생: {e}"

def get_valuation_scores_and_verdict(info):
    scores, details = {}, {}
    pe, pb = info.get('trailingPE'), info.get('priceToBook')
    scores['가치'] = ((4 if 0 < pe <= 15 else 2 if pe <= 25 else 1) if pe else 0) + ((2 if 0 < pb <= 1.5 else 1) if pb else 0)
    details['PER'] = f"{pe:.2f}" if pe else "N/A"
    details['PBR'] = f"{pb:.2f}" if pb else "N/A"
    peg, rev_growth = info.get('pegRatio'), info.get('revenueGrowth', 0)
    scores['성장성'] = ((4 if 0 < peg <= 1 else 2 if peg <= 2 else 0) if peg else 0) + ((4 if rev_growth > 0.2 else 2 if rev_growth > 0.1 else 0))
    details['PEG'] = f"{peg:.2f}" if peg else "N/A"
    details['매출성장률'] = f"{rev_growth*100:.2f}%"
    roe, profit_margin = info.get('returnOnEquity', 0), info.get('profitMargins', 0)
    scores['수익성'] = ((4 if roe > 0.2 else 2 if roe > 0.15 else 0)) + ((4 if profit_margin > 0.2 else 2 if profit_margin > 0.1 else 0))
    details['ROE'] = f"{roe*100:.2f}%"
    details['순이익률'] = f"{profit_margin*100:.2f}%"
    total_score = sum(scores.values())
    verdict_info = {"verdict": "관망", "color": "#ffc107", "text_color": "black"}
    if total_score >= 18: verdict_info = {"verdict": "강력 매수", "color": "#198754"}
    elif total_score >= 12: verdict_info = {"verdict": "매수 고려", "color": "#0d6efd"}
    elif total_score < 6: verdict_info = {"verdict": "투자 주의", "color": "#dc3545"}
    return verdict_info, scores, details

# --- 2. MAIN APP UI ---
st.sidebar.header("종목 검색")
ticker_data_df = get_latest_tickers()

if ticker_data_df is not None:
    # Failsafe selectbox implementation
    options_list = ticker_data_df['display'].tolist()
    symbols_list = ticker_data_df['Symbol'].tolist()
    default_index = 0
    try:
        default_index = symbols_list.index(st.session_state.ticker)
    except ValueError:
        default_index = 0 # If current ticker not in list, default to first item

    selected_display = st.sidebar.selectbox(
        "종목 선택 (이름 또는 코드로 검색)",
        options=options_list,
        index=default_index,
        key="ticker_select"
    )

    selected_ticker = symbols_list[options_list.index(selected_display)]
    if selected_ticker != st.session_state.ticker:
        st.session_state.ticker = selected_ticker
        st.session_state.ai_analysis = {} # Reset AI analysis on new ticker
        st.cache_data.clear()
        st.rerun()
else:
    st.sidebar.error("최신 종목 목록을 불러오는 데 실패했습니다.")

# --- Main App Logic ---
try:
    info, financials, news = get_stock_data(st.session_state.ticker)
    if info is None:
        st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다.")
    else:
        company_name = info.get('longName', st.session_state.ticker)
        quote_type = info.get('quoteType')
        final_verdict, scores, details = get_valuation_scores_and_verdict(info)
        text_color = final_verdict.get("text_color", "white")

        # --- HEADER ---
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h1 style="margin: 0;">🚀 {company_name} AI 분석</h1>
                <div style="padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: {final_verdict['color']}; color: {text_color};">
                    <span style="font-weight: bold; font-size: 1.2rem;">AI 종합 의견: {final_verdict['verdict']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.caption(f"종목코드: {st.session_state.ticker} ({quote_type}) | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("---")

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["**📊 대시보드 및 차트**", "**📂 재무 및 가치평가**", "**💡 뉴스 및 시장 동향**"])

        with tab1:
            # --- Key Metrics ---
            with st.container(border=True):
                st.subheader("📌 핵심 지표 요약")
                current_price = info.get('currentPrice', 0)
                prev_close = info.get('previousClose', 0)
                price_change = current_price - prev_close if current_price and prev_close else 0
                percent_change = (price_change / prev_close) * 100 if prev_close else 0
                cols = st.columns(4)
                cols[0].metric("현재가", f"${current_price:,.2f}", f"{price_change:,.2f} ({percent_change:.2f}%)")
                cols[1].metric("시가총액", f"${info.get('marketCap', 0):,}")
                cols[2].metric("52주 최고가", f"${info.get('fiftyTwoWeekHigh', 0):,.2f}")
                cols[3].metric("52주 최저가", f"${info.get('fiftyTwoWeekLow', 0):,.2f}")

            # --- Chart ---
            st.subheader("📈 주가 및 거래량 차트")
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
                fig.update_yaxes(title_text="주가", row=1, col=1)
                fig.update_yaxes(title_text="거래량", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

                # --- AI Chart Analysis ---
                with st.expander("🤖 AI 심층 차트 분석 보기"):
                    analysis_key = 'chart'
                    if analysis_key not in st.session_state.ai_analysis:
                        with st.spinner("AI가 차트를 심층 분석 중입니다..."):
                             history_for_ai = get_history(st.session_state.ticker, "1y", "1d")
                             st.session_state.ai_analysis[analysis_key] = generate_ai_analysis(info, history_for_ai, analysis_key)
                    st.markdown(st.session_state.ai_analysis[analysis_key])
            else:
                st.warning("차트 데이터를 불러올 수 없습니다.")

        with tab2:
            # --- Valuation Scorecard ---
            with st.container(border=True):
                st.subheader("⚖️ AI 가치평가 스코어카드")
                cols = st.columns(4)
                max_scores = {'가치': 6, '성장성': 8, '수익성': 8}
                for i, (cat, score) in enumerate(scores.items()):
                    with cols[i]:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': cat, 'font': {'size': 16}},
                            gauge={'axis': {'range': [0, max_scores[cat]]}, 'bar': {'color': "#0d6efd"}}
                        ))
                        fig_gauge.update_layout(height=150, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                with st.expander("상세 평가지표 보기"):
                    st.table(pd.DataFrame(details.items(), columns=['지표', '수치']))

            # --- Financials ---
            with st.container(border=True):
                st.subheader(f"💰 {company_name} 재무 상태 요약")
                if financials is not None and not financials.empty:
                    st.dataframe(financials.T.iloc[:4]) # Display recent 4 quarters
                    # --- AI Financial Analysis ---
                    with st.expander("🤖 AI 재무 진단 보기"):
                        analysis_key = 'financial'
                        if analysis_key not in st.session_state.ai_analysis:
                             with st.spinner("AI가 재무 데이터를 분석하고 등급을 매기는 중입니다..."):
                                 st.session_state.ai_analysis[analysis_key] = generate_ai_analysis(info, financials, analysis_key)
                        st.markdown(st.session_state.ai_analysis[analysis_key])
                else:
                    st.info("재무 데이터를 가져올 수 없습니다.")

        with tab3:
            # --- AI News Analysis ---
            with st.container(border=True):
                st.subheader("📰 AI 뉴스 요약 및 시장 분위기 분석")
                analysis_key = 'news'
                if analysis_key not in st.session_state.ai_analysis:
                    with st.spinner("AI가 구글 뉴스에서 최신 동향을 분석 중입니다..."):
                        st.session_state.ai_analysis[analysis_key] = generate_ai_analysis(info, news, analysis_key)
                st.markdown(st.session_state.ai_analysis[analysis_key])

            # --- News List ---
            with st.container(border=True):
                st.subheader("📜 관련 최신 뉴스 원문 (From Google News)")
                if news:
                    for article in news[:10]:
                        st.write(f"[{article['title']}]({article['url']}) - *{article['publisher']['title']}*")
                else:
                    st.info("구글 뉴스에서 관련 뉴스를 찾을 수 없습니다.")

except Exception as e:
    st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
