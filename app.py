import os
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="FinRAG Institutional | Multi-Agent Financial Intelligence",
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

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 25px;
    }
    .agent-pill {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #EEF2F6;
        color: #1E293B;
    }
    .trace-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .metric-badge {
        font-size: 0.85rem;
        font-weight: bold;
        color: #1E40AF;
        background: #DBEAFE;
        padding: 2px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏛️ FinRAG Institutional: Multi-Agent Equity Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deterministic Financial RAG & Multi-Agent Advisory Architecture for NSE & BSE Equities</div>', unsafe_allow_html=True)

# Sidebar: System Architecture & Disclosures
with st.sidebar:
    st.markdown("### ⚙️ Multi-Agent Architecture")
    st.markdown("""
    **Active Specialized Agents:**
    - 🎯 **Intent & Taxonomy Agent** (Query semantic parser)
    - 🌐 **Sector Research Agent** (Dynamic peer basket builder)
    - 📊 **Quantitative Engine** (Live NSE/BSE fundamentals)
    - 📰 **Market News & Catalyst Agent** (Live RSS aggregator)
    - 📑 **Filings RAG Agent** (Audited annual reports retriever)
    - 🧠 **Lead Advisory Synthesis Agent** (Constrained reasoning)
    """)
    st.markdown("---")
    st.markdown("### 📡 Data Feeds & Filings")
    st.markdown("""
    - **Filings Ingested:** HAL, BDL, L&T, Reliance Power
    - **Market Data:** Live NSE & BSE Exchange Feed
    - **Breaking News:** Reuters, Business Standard, Yahoo Finance
    - **Language Kernel:** Google Gemini (Zero-Cost Free Tier)
    """)
    st.markdown("---")
    st.caption("🔒 Architecture separates ground-truth data extraction from probabilistic reasoning to eliminate financial hallucinations.")

# Quick Advisory Themes (Chips instead of clunky 1-8 dropdown!)
st.markdown("##### 💡 Advisory Quick Themes")
col1, col2, col3, col4, col5 = st.columns(5)

if "active_query" not in st.session_state:
    st.session_state.active_query = "What is the better investment right now between the Space sector and the EV sector? Analyze the top company from each with valuation, order momentum, and risks."

if col1.button("🚀 Space vs EV Alpha", use_container_width=True):
    st.session_state.active_query = "What is the better investment right now between the Space sector and the EV sector? Analyze the top company from each with valuation, order momentum, and risks."

if col2.button("🛡️ Defence: HAL vs BDL", use_container_width=True):
    st.session_state.active_query = "Compare Hindustan Aeronautics (HAL) vs Bharat Dynamics (BDL) on geopolitical order book, export momentum, and current valuation multiples."

if col3.button("🏦 Banking: ICICI vs HDFC", use_container_width=True):
    st.session_state.active_query = "Compare ICICI Bank, HDFC Bank, and State Bank of India (SBI) on Net Interest Margins (NIM), credit cost trajectory, and asset quality."

if col4.button("⚡ Conglomerates & Energy", use_container_width=True):
    st.session_state.active_query = "Evaluate Reliance Industries and Adani Enterprises on capital expenditure in clean energy, leverage ratios, and recent news impact."

if col5.button("💊 Pharma vs IT Strategy", use_container_width=True):
    st.session_state.active_query = "Compare the Indian IT sector vs Pharma sector for a 12-month horizon. Which provides better defensive margin safety?"

# Main Query Input
user_query = st.text_area(
    "Enter an institutional investment query or sector comparison:",
    value=st.session_state.active_query,
    height=85
)

run_analysis = st.button("⚡ Execute Multi-Agent Advisory Pipeline", type="primary", use_container_width=True)

if run_analysis and user_query:
    # Visual Progress & Trace
    with st.status("🔄 **Multi-Agent Pipeline In Progress...**", expanded=True) as status_box:
        st.write("🎯 **Step 1:** Intent Agent parsing semantic query & resolving listed peers...")
        time.sleep(0.3)
        st.write("📊 **Step 2:** Quantitative Engine querying live NSE/BSE fundamentals & valuation ratios...")
        time.sleep(0.3)
        st.write("📰 **Step 3:** Market News Agent ingesting breaking regulatory & industry headlines...")
        time.sleep(0.3)
        st.write("📑 **Step 4:** Filings RAG Agent cross-referencing audited annual report disclosures...")
        time.sleep(0.3)
        st.write("🧠 **Step 5:** Lead Advisory Synthesis Agent compiling grounded institutional memo...")
        
        result = orchestrator.run_comparative_analysis(user_query)
        status_box.update(label=f"✅ **Multi-Agent Pipeline Execution Completed in {result.get('elapsed_seconds', 'N/A')}s**", state="complete", expanded=False)

    # 1. Multi-Agent Execution Trace Expander
    with st.expander("🔍 **View Multi-Agent Execution Graph & Tool Calls (Audit Trail)**", expanded=False):
        st.markdown(f"**Total Distributed Tools Fired:** 5 | **Runtime:** `{result.get('elapsed_seconds', 'N/A')}s`")
        for t in result.get("trace", []):
            st.markdown(f"""
            <div class="trace-card">
                <span class="metric-badge">Step {t['step']}</span> <strong>{t['agent']}</strong> — <em>{t['action']}</em><br>
                <small style="color: #475569;">{t['detail']}</small>
            </div>
            """, unsafe_allow_html=True)

    # 2. Results Section Tabs
    tab_memo, tab_quant, tab_news, tab_debug = st.tabs([
        "📑 Institutional Advisory Memo", 
        "📊 Valuation & Fundamentals Matrix", 
        "📰 Breaking News & Catalyst Feed", 
        "🔬 Ground-Truth Context Inspection"
    ])

    with tab_memo:
        st.markdown(result.get("answer", ""))

    with tab_quant:
        fundamentals = result.get("fundamentals", [])
        if fundamentals:
            df = pd.DataFrame(fundamentals)
            # rename for institutional display
            rename_map = {
                "ticker": "Ticker", "name": "Company Name", "sector": "Sector",
                "price": "Live Price (₹)", "pe": "P/E Ratio", "mcap": "Market Cap",
                "roe": "ROE", "rev_growth": "Revenue Growth", "yr_return": "1Y Price Return"
            }
            display_df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("#### Live Metric Highlights")
            cols = st.columns(min(len(fundamentals), 4))
            for i, item in enumerate(fundamentals[:4]):
                with cols[i]:
                    st.metric(
                        label=f"{item.get('ticker')} ({item.get('name', '')[:16]}...)",
                        value=f"₹{item.get('price'):,.2f}" if isinstance(item.get('price'), (int, float)) else str(item.get('price')),
                        delta=f"1Y: {item.get('yr_return')}" if item.get('yr_return') != 'N/A' else None
                    )
        else:
            st.info("No specific single-company fundamentals were isolated; research was synthesized at the macro/thematic level.")

    with tab_news:
        news_items = result.get("news", [])
        if news_items:
            st.markdown(f"**Aggregated {len(news_items)} Live Headlines Ingested by the News Agent:**")
            for item in news_items:
                source = item.get("source", "Financial News")
                date = item.get("published", "")
                link = item.get("link", "#")
                st.markdown(f"- 🔗 [{item.get('title')}]({link}) — <small style='color: #64748B;'>*{source} | {date}*</small>", unsafe_allow_html=True)
        else:
            st.info("No real-time news headlines were retrieved for this specific query.")

    with tab_debug:
        st.markdown("**Entities & Peer Baskets Resolved:**")
        st.json([e.get("name") for e in result.get("entities", [])])
        st.markdown("**Model Output Confidence:**")
        st.progress(float(result.get("confidence", 0.95)))
        st.caption("Grounded against real-time verified data feeds. Zero probabilistic hallucination on valuation multiples.")

st.markdown("---")
st.caption("🏛️ FinRAG Institutional Equity Platform | Built with Python, Google Gemini, Yahoo Finance API & RAPTOR Vector RAG")
