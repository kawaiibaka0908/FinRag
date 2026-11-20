# 🏛️ Vortex-India: Multi-Agent Financial & Geopolitical Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-orange.svg)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, multi-agent equity and geopolitical intelligence platform specialized for Indian equities (**NSE & BSE**). 

Unlike naive RAG systems that query static documents in isolation, **Vortex-India** pioneers a **Dual-Horizon Intelligence Architecture**: it dynamically synthesizes long-term corporate filings (Annual Reports via RAPTOR hierarchical RAG) with real-time Indian market news (Reuters, Business Standard) and live quantitative exchange metrics (P/E, Market Cap, ROE, Debt/Equity via Yahoo Finance).

---

## 🌟 Key Architecture & Capabilities

```
                                  [ User Query ]
                                        │
                                        ▼
             ┌────────────────────────────────────────────────────────┐
             │       🧠 India Multi-Agent Orchestrator (Gemini)       │
             └──────────────────────────┬─────────────────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
  [🛡️ Defence Agent]             [🏦 Banking Agent]            [⚡ Energy Agent]
   (HAL, BDL, Missiles)          (ICICI, HDFC, SBI)            (Reliance, Adani, L&T)
         │                              │                              │
         └──────────────────────────────┼──────────────────────────────┘
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        ▼                               ▼                               ▼
 [📚 RAPTOR RAG Engine]      [📰 Live News Fetcher]        [📊 Quant Scorer]
  Hierarchical Index of       Google News India RSS,        Live NSE/BSE Ratios,
  Annual Reports (ChromaDB)   Reuters, Business Standard    P/E, ROE, Market Cap
        │                               │                               │
        └───────────────────────────────┬───────────────────────────────┘
                                        ▼
             ┌────────────────────────────────────────────────────────┐
             │    📑 Executive Investment & Risk Dossier Synthesizer  │
             └────────────────────────────────────────────────────────┘
```

### 1. Specialized Sector Agents
- **Defence & Geopolitics Agent (`HAL.NS`, `BDL.NS`):** Tracks MoD capital expenditure allocations, indigenous export order books (Tejas LCA, Akash missile systems), and global geopolitical conflicts.
- **Banking & Macro Agent (`ICICIBANK.NS`, `HDFCBANK.NS`, `SBIN.NS`):** Compares Net Interest Margins (NIM), gross/net NPAs, Return on Equity (ROE), and RBI interest rate cycle sensitivities.
- **Conglomerates & Energy Agent (`RELIANCE.NS`, `ADANIENT.NS`, `BHARTIARTL.NS`):** Cross-analyzes green energy capex, corporate debt leverage, and breaking regulatory announcements.
- **Infrastructure & Heavy Engineering (`LT.NS`):** Evaluates multi-year order backlogs and domestic infrastructure execution.

### 2. Dual-Horizon Data Synthesis
- **Long-Horizon Disclosures:** Ingests 400+ page Integrated Annual Reports (e.g., FY24-25) into semantic vector collections using ChromaDB and sentence-transformers.
- **Short-Horizon Breaking News:** Automatically pulls, filters, and ranks live financial news articles from leading Indian financial publications.
- **Live Fundamentals:** Queries Yahoo Finance for real-time market valuations, current share prices, and debt-to-equity ratios.

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.10+
- A free Google Gemini API Key (`GEMINI_API_KEY`)

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/vortex-india.git
cd vortex-india/FinRAGFinal
pip install -r requirements.txt
```

### 2. Environment Configuration
Set your Gemini API Key:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_gemini_api_key_here"

# Linux / MacOS
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### 3. Launch Interactive Terminal
Run the terminal-based interactive multi-agent research CLI:
```bash
python run_india_finrag.py
```

### 4. Launch Web Dashboard (Streamlit)
Run the browser dashboard with live valuation cards and interactive comparative queries:
```bash
streamlit run app.py
```

---

## 📊 Sample Output: BDL vs HAL Defense Comparison

```text
=== LIVE VALUATION SUMMARY ===
Ticker   | Company Name                   | Price (₹) | P/E Ratio | Market Cap
--------------------------------------------------------------------------------
HAL.NS   | Hindustan Aeronautics Limited  | ₹4,180.50 | 33.46x    | ₹320,811 Cr
BDL.NS   | Bharat Dynamics Limited        | ₹1,085.20 | 77.49x    | ₹37,865 Cr

=== EXECUTIVE DOSSIER HIGHLIGHTS ===
• Order Book Scale: HAL benefits from long-cycle mega contracts (Tejas Mk1A, ALH Dhruv)
  backed by proposed 20% defence capex hike.
• Valuation Arbitrage: HAL trades at 33.4x P/E with 23.6% ROE and 0.02 D/E ratio,
  presenting superior risk-adjusted return vs BDL's stretched 77.5x P/E.
• Geopolitical Catalyst: Recent ₹1,307 Cr Indian Navy missile contract positions BDL
  as high-beta tactical trade, whereas HAL serves as strategic compounder.
```

---

## 🛠️ Tech Stack
- **Frameworks & Orchestration:** Python, LangChain, Custom Multi-Agent State Machine
- **LLM Engine:** Google Gemini (Gemini 2.0 / 3.8 Flash via modern `google-genai` SDK)
- **Vector Storage:** ChromaDB with Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Financial APIs:** yfinance, feedparser (Google News India RSS)
- **Web UI:** Streamlit with custom CSS layout

---

## 📜 License
MIT License. Created for advanced financial research and AI engineering portfolios.
