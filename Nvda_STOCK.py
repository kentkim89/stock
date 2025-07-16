import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime
import google.generativeai as genai # 제미나이 라이브러리 임포트

# --- 1. 페이지 기본 설정 및 함수 정의 ---
st.set_page_config(page_title="AI 주가 분석 대시보드", page_icon="🧠", layout="wide")

# 제미나이 API 키 설정
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (FileNotFoundError, KeyError):
    st.error("오류: Gemini API 키가 secrets.toml 파일에 설정되지 않았습니다.")
    st.stop()

# --- 데이터 로딩 함수 (이전과 동일) ---
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

# --- 제미나이 분석 함수 (신규 추가) ---
@st.cache_data(ttl=600) # API 호출 비용과 시간을 줄이기 위해 10분 캐싱
def get_gemini_analysis(info, recs):
    """제미나이 API를 호출하여 기업 분석 리포트를 생성합니다."""
    model = genai.GenerativeModel('gemini-pro')
    company_name = info.get('longName', '이 기업')
    
    # 제미나이에게 전달할 프롬프트 (가장 중요한 부분!)
    prompt = f"""
    당신은 월스트리트의 유능한 금융 애널리스트입니다. 다음 데이터를 기반으로 '{company_name}'에 대한 전문적인 투자 분석 보고서를 한국어로 작성해주세요.

    **기업 데이터:**
    - 기업명: {company_name} ({info.get('symbol')})
    - 업종: {info.get('industry')}
    - 시가총액: ${info.get('marketCap', 0):,}
    - PER (주가수익비율): {info.get('trailingPE', 'N/A'):.2f}
    - PBR (주가순자산비율): {info.get('priceToBook', 'N/A'):.2f}
    - PEG (주가수익성장비율): {info.get('pegRatio', 'N/A'):.2f}
    - ROE (자기자본이익률): {info.get('returnOnEquity', 0)*100:.2f}%
    - 애널리스트 평균 목표가: ${info.get('targetMeanPrice', 'N/A')}
    - 현재가: ${info.get('currentPrice', 'N/A')}

    **분석 요청:**
    아래 형식에 맞춰, 각 항목을 구체적이고 논리적으로 분석해주세요.

    **1. 투자 포인트 (Investment Thesis):**
    - 이 기업의 핵심적인 강점과 성장 동력은 무엇인가? (최소 2가지 이상)

    **2. 리스크 요인 (Risk Factors):**
    - 이 기업에 투자할 때 반드시 고려해야 할 잠재적 위험은 무엇인가? (최소 2가지 이상)

    **3. 최종 결론 (Final Verdict):**
    - 모든 데이터를 종합했을 때, 현재 시점에서 이 기업에 대한 당신의 최종 투자 의견은 무엇인가? (예: '매수', '보유', '매도'와 함께 그 이유를 설명)
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini 분석 중 오류가 발생했습니다: {e}"


# --- (이하 기존 UI 렌더링 함수들은 그대로 사용) ---
# ... (render_valuation_scorecard, render_metric_explanations 등) ...

# --- 2. 앱 UI 렌더링 ---
# ... (사이드바 및 기본 UI 코드) ...

# 이전 코드와 동일하게 진행되다가, 탭 구성만 변경합니다.
try:
    info, recs, financials = get_stock_data(st.session_state.ticker)

    if info is None:
        st.error("...")
    else:
        company_name = info.get('longName', st.session_state.ticker)
        st.title(f"🧠 {company_name} AI 주가 분석")
        
        # 탭 구성 변경: 제미나이 분석 탭 추가
        tab1, tab2, tab3 = st.tabs(["**📊 종합 대시보드**", "**🧠 제미나이 AI 심층 분석**", "**📂 재무 및 애널리스트 상세**"])

        with tab1:
            # 기존 '종합 대시보드' 탭의 내용
            # ... render_valuation_scorecard(info) ...
            # ... st.divider() ...
            # ... 주가 추이 차트 코드 ...
            st.write("기존 종합 대시보드 내용이 여기에 표시됩니다.")


        with tab2:
            st.subheader(f"🤖 제미나이(Gemini)가 분석한 {company_name} 리포트")
            
            if st.button("실시간 AI 리포트 생성하기"):
                with st.spinner('제미나이 AI가 데이터를 분석하고 있습니다... 잠시만 기다려주세요.'):
                    gemini_report = get_gemini_analysis(info, recs)
                    st.markdown(gemini_report)
            else:
                st.info("버튼을 클릭하면 제미나이 AI가 최신 데이터로 상세 분석 리포트를 생성합니다.")
        
        with tab3:
            # 기존 '재무 및 애널리스트 상세' 탭의 내용
            st.write("기존 재무 및 애널리스트 상세 정보가 여기에 표시됩니다.")

except Exception as e:
    st.error(f"앱 실행 중 오류가 발생했습니다: {e}")

# ... (사이드바 알림 기능 등 나머지 코드는 동일) ...
