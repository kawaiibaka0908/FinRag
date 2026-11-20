import os
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="FinRAG Institutional | LangGraph Multi-Agent Architecture",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

from finrag.orchestrator.langgraph_orchestrator import FinRAGLangGraphOrchestrator
from finrag.vectorstore.annual_report_store import load_annual_report_store

@st.cache_resource
def get_orchestrator():
    vector_store, store_message = load_annual_report_store()
    return FinRAGLangGraphOrchestrator(vector_store=vector_store), store_message

orchestrator, annual_report_store_message = get_orchestrator()
if annual_report_store_message:
    st.warning(annual_report_store_message)

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
st.markdown('<div class="main-header">🏛️ FinRAG: LangGraph Multi-Agent Financial Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Stateful Directed Multi-Agent Graph for NSE & BSE Equities (LangGraph + Deterministic Market Feeds + RAG)</div>', unsafe_allow_html=True)

# Sidebar: System Architecture & LangGraph Graph
with st.sidebar:
    st.markdown("### 🕸️ LangGraph State Machine")
    st.markdown("""
    **Compiled Graph Nodes:**
    - `START` ──► `intent_agent`
    - `intent_agent` ──► (Conditional Branch)
      - ──► `sector_agent`
      - ──► `quant_engine`
    - `quant_engine` ──► `news_agent`
    - `news_agent` ──► `filings_rag`
    - `filings_rag` ──► `context_builder`
    - `context_builder` ──► `synthesis_agent` ──► `END`
    """)
    st.markdown("---")
    st.markdown("### 📡 Live Feed Status")
    st.markdown("""
    - **LangGraph Version:** Active
    - **Filings Ingested:** HAL, BDL, L&T, Reliance Power
    - **Market Data:** Live NSE & BSE Exchange API
    - **Language Model:** Google Gemini (Zero-Cost Free Tier)
    """)
    st.markdown("---")
    st.caption("🔒 Architecture separates ground-truth data extraction from probabilistic reasoning to eliminate financial hallucinations.")

# LangGraph Architecture Diagram Expander
with st.expander("📊 **View Active LangGraph StateGraph Architecture (Mermaid Graph)**", expanded=False):
    st.markdown("```mermaid\n" + orchestrator.get_mermaid_graph() + "\n```")

# Quick Advisory Themes
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

run_analysis = st.button("⚡ Execute LangGraph Workflow", type="primary", use_container_width=True)

if run_analysis and user_query:
    with st.status("🔄 **LangGraph Workflow Executing Across Nodes...**", expanded=True) as status_box:
        st.write("🎯 **Node 1 (intent_agent):** Semantic classification & conditional edge evaluation...")
        time.sleep(0.3)
        st.write("🌐 **Node 2 (sector_agent):** Thematic peer basket enumeration...")
        time.sleep(0.3)
        st.write("📊 **Node 3 (quant_engine):** Querying real-time exchange ratios via Yahoo Finance API...")
        time.sleep(0.3)
        st.write("📰 **Node 4 (news_agent):** Ingesting breaking regulatory & market RSS feeds...")
        time.sleep(0.3)
        st.write("📑 **Node 5 (filings_rag):** Vector search over corporate filings index...")
        time.sleep(0.3)
        st.write("🧠 **Node 6 (synthesis_agent):** Grounded institutional memo generation with Gemini...")

        result = orchestrator.run(user_query)
        status_box.update(label=f"✅ **LangGraph Completed Successfully in {result.get('elapsed_seconds', 'N/A')}s**", state="complete", expanded=False)

    # 1. Multi-Agent Execution Trace Expander
    with st.expander("🔍 **View LangGraph Execution Audit Trail (Node-by-Node State Output)**", expanded=False):
        st.markdown(f"**Total Graph Nodes Fired:** {len(result.get('trace', []))} | **Runtime:** `{result.get('elapsed_seconds', 'N/A')}s`")
        for t in result.get("trace", []):
            st.markdown(f"""
            <div class="trace-card">
                <span class="metric-badge">Node {t['step']}</span> <strong>{t['agent']}</strong> — <em>{t['action']}</em><br>
                <small style="color: #475569;">{t['detail']}</small>
            </div>
            """, unsafe_allow_html=True)

    # 2. Results Section Tabs
    tab_memo, tab_quant, tab_news, tab_debug = st.tabs([
        "📑 Institutional Advisory Memo", 
        "📊 Valuation & Fundamentals Matrix", 
        "📰 Breaking News & Catalyst Feed", 
        "🔬 LangGraph State Inspection"
    ])

    with tab_memo:
        st.markdown(result.get("answer", ""))

    with tab_quant:
        fundamentals = result.get("fundamentals", [])
        if fundamentals:
            df = pd.DataFrame(fundamentals)
            rename_map = {
                "ticker": "Ticker", "name": "Company Name", "sector": "Sector",
                "price": "Live Price (₹)", "pe": "P/E Ratio", "mcap": "Market Cap",
                "roe": "ROE", "rev_growth": "Revenue Growth", "yr_return": "1Y Price Return",
                "debt_equity": "Debt/Equity", "margin": "Profit Margin"
            }
            display_df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("#### Live Valuation Metrics")
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
            st.markdown(f"**Aggregated {len(news_items)} Live Headlines Ingested by the LangGraph News Node:**")
            for item in news_items:
                source = item.get("source", "")
                date = item.get("published", "")
                link = item.get("link", "#")
                st.markdown(f"- 🔗 [{item.get('title')}]({link}) — <small style='color: #64748B;'>*{source} | {date}*</small>", unsafe_allow_html=True)
        else:
            st.info("No real-time news headlines were retrieved for this specific query.")

    with tab_debug:
        st.markdown("**LangGraph State - Resolved Entities:**")
        st.json([e.get("name") for e in result.get("entities", [])])
        st.markdown("**Model Output Confidence:**")
        st.progress(float(result.get("confidence", 0.0)))
        st.caption("Confidence reflects the synthesis and evidence available for this query; it is not a guarantee of accuracy.")
        if result.get("sources_used"):
            st.markdown("**Sources used for this query:**")
            st.json(result["sources_used"])

st.markdown("---")
st.caption("🏛️ FinRAG Institutional Equity Platform | Powered by LangGraph StateGraph, Google Gemini, Yahoo Finance API & RAPTOR Vector Store")
