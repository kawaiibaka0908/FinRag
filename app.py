"""
Vortex-India: Multi-Agent Financial & Geopolitical Intelligence Platform
Streamlit Web Application for NSE & BSE Equities.
"""
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Vortex-India | Multi-Agent Financial AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

from finrag.orchestrator.india_orchestrator import IndiaMultiAgentOrchestrator

@st.cache_resource
def get_orchestrator():
    return IndiaMultiAgentOrchestrator()

orchestrator = get_orchestrator()

# Custom Styling
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

# Header
st.markdown('<div class="main-title">🏛️ Vortex-India: Multi-Agent Equity & Geopolitics AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dual-Horizon Intelligence for NSE & BSE: Combining Live Financial News, Market Fundamentals, and Annual Report Disclosures</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎯 Research Presets")
preset_options = [
    "Select a pre-configured query...",
    "🛡️ Compare BDL vs HAL on global affairs and defense orders",
    "🏦 Compare ICICI Bank with HDFC Bank and SBI on NIMs and asset quality",
    "⚡ How do recent events affect Reliance and Bharti Airtel?",
    "🏗️ Analyze Larsen & Toubro (LT) order book & infrastructure execution",
    "📈 Evaluate Adani Enterprises exposure across energy and airports"
]

selected_preset = st.sidebar.selectbox("Choose a scenario:", preset_options)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Model Pipeline Architecture")
st.sidebar.info("""
- **LLM Engine:** Google Gemini (Flash / Pro)
- **Agent Roles:** Defence, Banking, Conglomerates, News, Quant
- **Data Feeds:** Live Reuters / Business Standard / Yahoo Finance
- **Filings Ingested:** HAL, BDL, L&T, Reliance Power
- **Cost:** $0.00 (Free Tier API)
""")

# Main Input Section
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
    with st.spinner("🔄 Multi-Agent Pipeline Active: Fetching live breaking news, querying NSE/BSE fundamentals, and synthesizing research with Gemini..."):
        result = orchestrator.run_comparative_analysis(user_query)

    st.success("✅ Research Briefing Generated Successfully!")

    # 1. Valuation & Fundamentals Metrics Display
    fundamentals = result.get("fundamentals", [])
    if fundamentals:
        st.subheader("📊 Live Market Fundamentals (NSE/BSE)")
        cols = st.columns(len(fundamentals))
        for i, item in enumerate(fundamentals):
            with cols[i]:
                st.markdown(f"### {item['ticker']}")
                st.caption(item['name'])
                st.metric("Live Price", f"₹{item['price']:,.2f}" if isinstance(item['price'], (int, float)) else str(item['price']))
                st.metric("P/E Ratio", f"{item['pe']:.2f}" if isinstance(item['pe'], (int, float)) else str(item['pe']))
                st.metric("Market Cap", str(item['mcap']))

    # 2. Comprehensive AI Research Report
    st.subheader("📑 Executive Research Dossier")
    st.markdown(result.get("answer", ""))

    st.markdown("---")
    st.caption("⚡ Powered by Vortex-India Multi-Agent System | Open Source Indian Equity Intelligence")
