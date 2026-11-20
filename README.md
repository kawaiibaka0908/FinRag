# FinRag — Indian Stock Market Multi-Agent RAG

A multi-agent financial research assistant for Indian equities (NSE & BSE).

Most RAG setups just take a bunch of PDF documents, chop them into chunks, and do vector search. That works okay for static manuals, but for the stock market, looking at an annual report from 8 months ago without knowing what happened in the news yesterday gives you incomplete answers.

FinRag fixes this by combining three things whenever you ask a question:
1. **Corporate Filings (Annual Reports)** via vector search over parsed PDFs.
2. **Live Breaking News** pulled in real time from Indian financial news feeds (Reuters, Business Standard, Google News India).
3. **Real-time Fundamentals** directly from the market (live share prices, P/E ratio, Market Cap, ROE, Debt-to-Equity).

Then, a specialized LLM agent synthesizes everything into a balanced investment brief.

---

## What Can It Do?

- **Compare Indian stocks side-by-side**: 
  - Compare Defence giants: **HAL vs BDL** (order book, Tejas deliveries, MoD contracts, valuations).
  - Compare Private & Public Banks: **ICICI Bank vs HDFC Bank vs SBI** (NIMs, NPA health, credit growth).
  - Compare Conglomerates & Telecom: **Reliance vs Bharti Airtel**, or check **Adani Enterprises** across its energy/infrastructure segments.
- **Sector-aware context**: It knows whether a company belongs to Defence, Banking, Infrastructure, or Conglomerates, and highlights metrics relevant to that sector.
- **Zero API Cost**: Powered by Google Gemini's free tier (`gemini-3.8-flash` / `gemini-3.1-flash-lite`) and local embedding models.

---

## Project Structure

```text
FinRAGFinal/
├── app.py                     # Streamlit web dashboard
├── run_india_finrag.py        # Terminal CLI with quick presets
├── requirements.txt           # Python dependencies
├── data/                      # Annual reports (PDFs) and company mappings
└── src/
    └── finrag/
        ├── models/
        │   └── gemini_models.py       # Gemini QA and summarizer wrapper
        ├── scoring/
        │   ├── news_fetcher.py        # Live RSS & news aggregator
        │   └── financial_data_fetcher.py # yfinance data collector
        └── orchestrator/
            └── india_orchestrator.py  # Entity detection and multi-source synthesis
```

---

## Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/kawaiibaka0908/FinRag.git
cd FinRag
pip install -r requirements.txt
```

### 2. Set Up Your API Key

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/):

```bash
# On Windows (PowerShell):
$env:GEMINI_API_KEY="your_api_key_here"

# On Linux / Mac:
export GEMINI_API_KEY="your_api_key_here"
```

### 3. Run It

**Option A: Web App (Streamlit)**
```bash
streamlit run app.py
```

**Option B: Terminal CLI**
```bash
python run_india_finrag.py
```

---

## Sample Query & Output

**Question:**
> *"Compare BDL vs HAL on defense orders and recent geopolitical outlook"*

**What the system does:**
1. Recognizes **HAL.NS** (Hindustan Aeronautics) and **BDL.NS** (Bharat Dynamics).
2. Grabs latest headlines (e.g. HAL Tejas deliveries, BDL ₹1,307 Cr Navy missile order).
3. Fetches live market stats (HAL: ~₹4,180, P/E ~33x | BDL: ~₹1,085, P/E ~77x).
4. Pulls strategy & order backlog disclosures from their annual reports.
5. Produces a side-by-side breakdown covering valuation, order books, and key risks.

---

## Tech Used

- **Python 3.10+**
- **LLM:** Google Gemini (`google-genai` SDK)
- **Vector DB:** ChromaDB
- **Data & News:** `yfinance`, `feedparser`
- **UI:** Streamlit & Rich CLI
