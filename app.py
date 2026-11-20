import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="FinRAG | Multi-Agent Financial AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

from finrag.orchestrator.india_orchestrator import IndiaMultiAgentOrchestrator

@st.cache_resource
def get_orchestrator():
    return IndiaMultiAgentOrchestrator()

orchestrator = get_orchestrator()

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏛️ FinRAG: Multi-Agent Equity & Geopolitics AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Live Financial News + Market Fundamentals + Annual Report RAG for NSE & BSE Equities</div>', unsafe_allow_html=True)

st.sidebar.header("🎯 Research Presets")
preset_options = [
    "Select a pre-configured query...",
    "🛡️ Compare BDL vs HAL on global affairs and defense orders",
    "🏦 Compare ICICI Bank with HDFC Bank and SBI on NIMs and asset quality",
    "⚡ How do recent events affect Reliance and Bharti Airtel?",
    "🏗️ Analyze Larsen & Toubro (LT) order book & infrastructure execution",
    "📈 Evaluate Adani Enterprises exposure across energy and airports",
    "🚀 Whats better sector right now - space or EV? Best company from each",
    "💊 Compare pharma sector vs IT sector for investment",
    "🔋 Best renewable energy stocks in India right now",
]

selected_preset = st.sidebar.selectbox("Choose a scenario:", preset_options)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Model Pipeline")
st.sidebar.info("""
- **LLM:** Google Gemini (Flash)
- **Agents:** Stock Agent, Sector Research Agent, News Agent, Quant Agent
- **Data:** Live Google News / Yahoo Finance / NSE-BSE Fundamentals
- **Filings:** HAL, BDL, L&T, Reliance Power Annual Reports
- **Cost:** $0.00 (Free Tier API)
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Covered Sectors")
st.sidebar.markdown("""
Defence · Banking · IT · Pharma · Auto/EV · Energy · Power
Infrastructure · Metals · Oil & Gas · FMCG · Telecom · Insurance
""")

default_query = ""
if selected_preset and not selected_preset.startswith("Select"):
    default_query = selected_preset.split(" ", 1)[1]

user_query = st.text_area(
    "Enter your investment research query or compare Indian stocks:",
    value=default_query or "Compare BDL vs HAL on global affairs, defense orders, and recent geopolitical outlook",
    height=80
)

run_button = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)

if run_button and user_query:
    with st.spinner("🔄 Multi-Agent Pipeline Active: Fetching live news, querying fundamentals, running sector research..."):
        result = orchestrator.run_comparative_analysis(user_query)

    st.success("✅ Research Briefing Generated!")

    fundamentals = result.get("fundamentals", [])
    if fundamentals:
        st.subheader("📊 Live Market Fundamentals (NSE/BSE)")
        # show max 4 per row
        for row_start in range(0, len(fundamentals), 4):
            batch = fundamentals[row_start:row_start + 4]
            cols = st.columns(len(batch))
            for i, item in enumerate(batch):
                with cols[i]:
                    st.markdown(f"### {item['ticker']}")
                    st.caption(item['name'])
                    st.metric("Live Price", f"₹{item['price']:,.2f}" if isinstance(item['price'], (int, float)) else str(item['price']))
                    st.metric("P/E Ratio", f"{item['pe']:.2f}" if isinstance(item['pe'], (int, float)) else str(item['pe']))
                    st.metric("Market Cap", str(item['mcap']))

    st.subheader("📑 Executive Research Dossier")
    st.markdown(result.get("answer", ""))

    st.markdown("---")
    st.caption("⚡ Powered by FinRAG Multi-Agent System | Open Source Indian Equity Intelligence")
