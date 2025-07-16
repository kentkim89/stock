import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
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
@st.cache_data(ttl=86400)
def get_latest_tickers():
    """NASDAQ 서버에서 최신 주식 및 ETF 목록을 직접 다운로드합니다."""
    try:
        nasdaq_df = pd.read_csv("ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt", sep='|')
        other_df = pd.read_csv("ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt", sep='|')
        
        nasdaq_tickers = nasdaq_df[['Symbol', 'Security Name']]
        other_tickers = other_df[['ACT Symbol', 'Security Name']]
        other_tickers.rename(columns={'ACT Symbol': 'Symbol'}, inplace=True)
        
        all_tickers = pd.concat([nasdaq_tickers, other_tickers]).dropna()
        all_tickers = all_tickers[~all_tickers['Symbol'].str.contains('\$')]
        all_tickers = all_tickers[~all_tickers['Symbol'].str.contains('\.')]
        
        all_tickers.rename(columns={'Security Name': 'Name'}, inplace=True)
        all_tickers['display'] = all_tickers['Symbol'] + " - " + all_tickers['Name']
        return all_tickers.sort_values(by='Symbol').reset_index(drop=True)
    except Exception as e:
        st.error(f"최신 종목 목록을 불러오는 데 실패했습니다: {e}")
        return None

@st.cache_data(ttl=300)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if not info.get('marketCap') and not info.get('totalAssets'): return None, None
    financials = stock.quarterly_financials if info.get('quoteType') == 'EQUITY' else None
    return info, financials

@st.cache_data(ttl=900)
def get_news_data(query):
    google_news = GNews(language='ko', country='KR')
    news = google_news.get_news(query)
    return news

@st.cache_data(ttl=60)
def get_history(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval)

# --- (이하 모든 AI 분석 및 UI 렌더링 함수는 이전과 동일하게 유지) ---
@st.cache_data(ttl=600)
def generate_ai_analysis(info, data, analysis_type):
    model = genai.GenerativeModel('gemini-1.5-flash'); company_name = info.get('longName', '해당 종목')
    today_date = datetime.now().strftime('%Y년 %m월 %d일'); prompt = ""
    if analysis_type == 'verdict_stock':
        scores, details = data
        prompt = f"""당신은 최고 투자 책임자(CIO)입니다. **오늘은 {today_date}입니다.** '{company_name}'에 대한 아래의 모든 분석 결과를 종합하여, 최종 투자 의견과 그 이유를 명확하게 서술해주세요.
        - **AI 가치평가 스코어카드:** 가치: {scores['가치']}/6, 성장성: {scores['성장성']}/8, 수익성: {scores['수익성']}/8
        - **주요 지표:** {', '.join([f'{k}: {v}' for k, v in details.items()])}
        **최종 투자 의견 및 전략:** (서론-본론-결론 형식으로, 최종 투자 등급('강력 매수', '매수 고려', '관망', '투자 주의' 중 하나)을 결정하고, 그 이유와 투자 전략을 논리적으로 설명해주세요.)"""
    elif analysis_type == 'verdict_etf':
        holdings_summary = "\n".join([f"- {h['holdingName']} ({h['holdingPercent']*100:.2f}%)" for h in info.get('holdings', [])[:5]])
        prompt = f"""당신은 ETF 전문 애널리스트입니다. **오늘은 {today_date}입니다.** 다음 데이터를 바탕으로 '{company_name}' ETF를 종합적으로 분석하고 투자 의견을 제시해주세요.
        - **ETF 개요:** {info.get('longBusinessSummary')}
        - **운용보수(Expense Ratio):** {info.get('annualReportExpenseRatio', 'N/A')}
        - **상위 보유 종목:**\n{holdings_summary}
        **ETF 종합 분석 리포트:** (ETF의 투자 전략, 보유 종목의 매력도, 운용보수의 적절성을 종합적으로 평가하고, 이 ETF가 어떤 유형의 투자자에게 적합한지에 대한 최종 의견을 제시해주세요.)"""
    if not prompt: return "분석 유형 오류"
    try:
        response = model.generate_content(prompt); return response.text
    except Exception as e: return f"AI 분석 중 오류 발생: {e}"

def get_valuation_scores(info):
    scores, details = {}, {}; pe, pb = info.get('trailingPE'), info.get('priceToBook')
    scores['가치'] = ((4 if 0 < pe <= 15 else 2 if pe <= 25 else 1) if pe else 0) + ((2 if 0 < pb <= 1.5 else 1) if pb else 0)
    details['PER'] = f"{pe:.2f}" if pe else "N/A"; details['PBR'] = f"{pb:.2f}" if pb else "N/A"
    peg, rev_growth = info.get('pegRatio'), info.get('revenueGrowth', 0); scores['성장성'] = ((4 if 0 < peg <= 1 else 2 if peg <= 2 else 0) if peg else 0) + ((4 if rev_growth > 0.2 else 2 if rev_growth > 0.1 else 0))
    details['PEG'] = f"{peg:.2f}" if peg else "N/A"; details['매출성장률'] = f"{rev_growth*100:.2f}%"
    roe, profit_margin = info.get('returnOnEquity', 0), info.get('profitMargins', 0); scores['수익성'] = ((4 if roe > 0.2 else 2 if roe > 0.15 else 0)) + ((4 if profit_margin > 0.2 else 2 if profit_margin > 0.1 else 0))
    details['ROE'] = f"{roe*100:.2f}%"; details['순이익률'] = f"{profit_margin*100:.2f}%"
    return scores, details

# --- 2. 앱 UI 렌더링 ---
st.sidebar.header("종목 검색")
ticker_data = get_latest_tickers()
if ticker_data is not None:
    # --- 여기가 수정된 부분입니다: 더 안전하고 확실한 인덱스 검색 로직 ---
    options_list = ticker_data['display'].tolist()
    default_index = 0
    
    # 현재 세션의 티커에 해당하는 전체 표시 이름(display name)을 찾습니다.
    current_display_series = ticker_data[ticker_data['Symbol'] == st.session_state.ticker]['display']

    if not current_display_series.empty:
        default_display_value = current_display_series.iloc[0]
        try:
            # 파이썬 리스트의 내장 .index() 메소드를 사용하여 위치를 찾습니다.
            # 이 방법은 항상 순수한 정수(int)를 반환하여 오류가 없습니다.
            default_index = options_list.index(default_display_value)
        except ValueError:
            # 만약 리스트에 값이 없는 매우 드문 경우, 0으로 초기화합니다.
            default_index = 0

    selected_display = st.sidebar.selectbox(
        "종목 선택 (이름 또는 코드로 검색)", 
        options=options_list, 
        index=default_index, # 안전하게 찾은 정수 인덱스 사용
        key="ticker_select"
    )
    # --- 여기까지 수정 ---
    
    selected_ticker = ticker_data[ticker_data['display'] == selected_display]['Symbol'].iloc[0]
    if selected_ticker != st.session_state.ticker:
        st.session_state.ticker = selected_ticker
        st.session_state.ai_analysis = {}
        st.cache_data.clear()
        st.rerun()
else:
    st.sidebar.error("최신 종목 목록을 불러오는 데 실패했습니다.")

try:
    info, financials = get_stock_data(st.session_state.ticker)
    if info is None: st.error(f"'{st.session_state.ticker}'에 대한 데이터를 찾을 수 없습니다.")
    else:
        company_name = info.get('longName', st.session_state.ticker)
        quote_type = info.get('quoteType')

        st.markdown(f"<h1 style='margin-bottom:0;'>🚀 {company_name} AI 분석</h1>", unsafe_allow_html=True)
        st.caption(f"종목코드: {st.session_state.ticker} ({quote_type}) | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("---")

        with st.container(border=True):
            st.subheader("🤖 AI 종합 투자 의견")
            analysis_key, analysis_type = ('verdict', 'verdict_etf' if quote_type == 'ETF' else 'verdict_stock')
            if analysis_key not in st.session_state.ai_analysis or st.button("AI 의견 새로고침", key="verdict_refresh"):
                with st.spinner("AI가 모든 데이터를 종합하여 최종 투자 의견을 생성 중입니다..."):
                    data_for_ai = info if quote_type == 'ETF' else get_valuation_scores(info)
                    st.session_state.ai_analysis[analysis_key] = generate_ai_analysis(info, data_for_ai, analysis_type)
            st.markdown(st.session_state.ai_analysis.get(analysis_key, "AI 의견을 생성하려면 버튼을 클릭하세요."))
        
        tab1, tab2 = st.tabs(["**📊 대시보드 및 차트**", "**💡 뉴스 및 시장 동향**"])
        with tab1:
            if quote_type == 'ETF':
                st.subheader("📌 ETF 핵심 정보")
                cols = st.columns(3); cols[0].metric(label="순자산가치 (NAV)", value=f"${info.get('navPrice', 0):,.2f}")
                cols[1].metric(label="운용보수", value=f"{info.get('annualReportExpenseRatio', 0)*100:.3f}%")
                cols[2].metric(label="총자산 (AUM)", value=f"${info.get('totalAssets', 0):,}")
                st.subheader("📋 상위 10개 보유 종목")
                holdings = info.get('holdings', [])
                if holdings:
                    holdings_df = pd.DataFrame(holdings); holdings_df['holdingPercent'] *= 100
                    fig_pie = px.pie(holdings_df.head(10), values='holdingPercent', names='holdingName', title='Top 10 Holdings', hole=.3)
                    st.plotly_chart(fig_pie, use_container_width=True)
            elif quote_type == 'EQUITY':
                st.subheader("⚖️ AI 가치평가 스코어카드")
                scores, details = get_valuation_scores(info)
                cols = st.columns(4); max_scores = {'가치': 6, '성장성': 8, '수익성': 8}
                for i, (cat, score) in enumerate(scores.items()):
                    with cols[i]:
                        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': cat, 'font': {'size': 16}}, gauge={'axis': {'range': [0, max_scores[cat]]}, 'bar': {'color': "#0d6efd"}}))
                        fig_gauge.update_layout(height=150, margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(fig_gauge, use_container_width=True)
                with st.expander("상세 평가지표 보기"): st.table(pd.DataFrame(details.items(), columns=['지표', '수치']))

            st.subheader("📈 주가 차트")
            period_options = {"1개월": "1mo", "1년": "1y", "5년": "5y"}
            selected_period = st.radio("차트 기간 선택", options=period_options.keys(), horizontal=True, key="chart_period")
            history = get_history(st.session_state.ticker, selected_period, "1d")
            if not history.empty:
                fig_main_chart = go.Figure(data=[go.Scatter(x=history.index, y=history['Close'], mode='lines', name='종가')])
                st.plotly_chart(fig_main_chart, use_container_width=True)

        with tab2:
            st.subheader("📰 관련 최신 뉴스 (From Google News)")
            news = get_news_data(f'{company_name} 주가')
            if news:
                for article in news[:10]: st.write(f"[{article['title']}]({article['url']}) - *{article['publisher']['title']}*")
            else: st.info("구글 뉴스에서 관련 뉴스를 찾을 수 없습니다.")

except Exception as e:
    st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
