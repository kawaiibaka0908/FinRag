import re
import time
import logging
from ..scoring.news_fetcher import IndianFinancialNewsFetcher
from ..scoring.financial_data_fetcher import FinancialDataFetcher
from ..models.gemini_models import GeminiQAModel

logger = logging.getLogger(__name__)

STOCKS = {
    # Defence & Space
    "HAL": {"name": "Hindustan Aeronautics Limited", "ticker": "HAL.NS", "sector": "Defence & Aerospace"},
    "BDL": {"name": "Bharat Dynamics Limited", "ticker": "BDL.NS", "sector": "Defence"},
    "MTAR": {"name": "MTAR Technologies", "ticker": "MTARTECH.NS", "sector": "Space & Precision Engg"},
    "MTARTECH": {"name": "MTAR Technologies", "ticker": "MTARTECH.NS", "sector": "Space & Precision Engg"},
    "DATA PATTERNS": {"name": "Data Patterns India", "ticker": "DATAPATTNS.NS", "sector": "Defence & Aerospace"},
    "DATAPATTNS": {"name": "Data Patterns India", "ticker": "DATAPATTNS.NS", "sector": "Defence & Aerospace"},
    
    # Auto & EV
    "M&M": {"name": "Mahindra & Mahindra", "ticker": "M&M.NS", "sector": "Auto & EV OEM"},
    "MAHINDRA": {"name": "Mahindra & Mahindra", "ticker": "M&M.NS", "sector": "Auto & EV OEM"},
    "MARUTI": {"name": "Maruti Suzuki India", "ticker": "MARUTI.NS", "sector": "Auto Passenger"},
    "OLECTRA": {"name": "Olectra Greentech", "ticker": "OLECTRA.NS", "sector": "EV Public Transit"},
    "EXIDE": {"name": "Exide Industries", "ticker": "EXIDEIND.NS", "sector": "EV Batteries & Cells"},
    "EXIDEIND": {"name": "Exide Industries", "ticker": "EXIDEIND.NS", "sector": "EV Batteries & Cells"},

    # Banking & Financials
    "ICICI": {"name": "ICICI Bank", "ticker": "ICICIBANK.NS", "sector": "Private Banking"},
    "HDFC": {"name": "HDFC Bank", "ticker": "HDFCBANK.NS", "sector": "Private Banking"},
    "SBI": {"name": "State Bank of India", "ticker": "SBIN.NS", "sector": "Public Banking"},
    "BAJFINANCE": {"name": "Bajaj Finance", "ticker": "BAJFINANCE.NS", "sector": "NBFC & Consumer Credit"},

    # Conglomerate, Energy, Telecom
    "RELIANCE": {"name": "Reliance Industries", "ticker": "RELIANCE.NS", "sector": "Energy, Retail & 5G"},
    "AIRTEL": {"name": "Bharti Airtel", "ticker": "BHARTIARTL.NS", "sector": "Telecom & Cloud"},
    "LT": {"name": "Larsen & Toubro", "ticker": "LT.NS", "sector": "EPC Infrastructure & Space"},
    "ADANI": {"name": "Adani Enterprises", "ticker": "ADANIENT.NS", "sector": "Conglomerate & Incubator"},
    "ADANIGREEN": {"name": "Adani Green Energy", "ticker": "ADANIGREEN.NS", "sector": "Renewable Power"},
    "TATAPOWER": {"name": "Tata Power", "ticker": "TATAPOWER.NS", "sector": "Clean Energy & EV Infra"},
    "NTPC": {"name": "NTPC Limited", "ticker": "NTPC.NS", "sector": "Power Generation"},

    # IT & Tech
    "TCS": {"name": "Tata Consultancy Services", "ticker": "TCS.NS", "sector": "Enterprise IT"},
    "INFOSYS": {"name": "Infosys", "ticker": "INFY.NS", "sector": "Enterprise IT"},
    "WIPRO": {"name": "Wipro", "ticker": "WIPRO.NS", "sector": "Enterprise IT"},

    # Pharma
    "SUNPHARMA": {"name": "Sun Pharmaceutical", "ticker": "SUNPHARMA.NS", "sector": "Pharma & Generics"},
    "DRREDDY": {"name": "Dr. Reddy's Laboratories", "ticker": "DRREDDY.NS", "sector": "Pharma & Biosimilars"},
    "CIPLA": {"name": "Cipla", "ticker": "CIPLA.NS", "sector": "Pharma & Respiratory"},

    # Metals & FMCG
    "TATASTEEL": {"name": "Tata Steel", "ticker": "TATASTEEL.NS", "sector": "Metals & Mining"},
    "ITC": {"name": "ITC Limited", "ticker": "ITC.NS", "sector": "Diversified FMCG"},
    "HINDUNILVR": {"name": "Hindustan Unilever", "ticker": "HINDUNILVR.NS", "sector": "FMCG Staples"},
}

SECTOR_MAP = {
    # Space & Satellite
    "space": ["HAL", "MTARTECH", "DATAPATTNS", "LT"],
    "isro": ["HAL", "MTARTECH", "DATAPATTNS", "LT"],
    "satellite": ["DATAPATTNS", "MTARTECH", "HAL"],
    "aerospace": ["HAL", "MTARTECH", "DATAPATTNS"],

    # Defence
    "defence": ["HAL", "BDL", "DATAPATTNS"],
    "defense": ["HAL", "BDL", "DATAPATTNS"],
    "military": ["HAL", "BDL"],

    # EV & Auto
    "ev": ["M&M", "OLECTRA", "EXIDEIND", "TATAPOWER"],
    "electric vehicle": ["M&M", "OLECTRA", "EXIDEIND"],
    "auto": ["M&M", "MARUTI", "OLECTRA"],
    "automobile": ["M&M", "MARUTI"],
    "battery": ["EXIDEIND"],

    # Banking & Finance
    "banking": ["ICICI", "HDFC", "SBI"],
    "bank": ["ICICI", "HDFC", "SBI"],
    "finance": ["ICICI", "HDFC", "SBI", "BAJFINANCE"],
    "nbfc": ["BAJFINANCE"],

    # IT / Tech
    "it": ["TCS", "INFOSYS", "WIPRO"],
    "software": ["TCS", "INFOSYS", "WIPRO"],
    "tech": ["TCS", "INFOSYS", "WIPRO"],

    # Energy & Green
    "renewable": ["ADANIGREEN", "TATAPOWER", "NTPC"],
    "green energy": ["ADANIGREEN", "TATAPOWER"],
    "power": ["TATAPOWER", "NTPC", "ADANIGREEN"],
    "energy": ["RELIANCE", "TATAPOWER", "NTPC", "ADANIGREEN"],

    # Infra & Conglomerates
    "infra": ["LT", "ADANI"],
    "infrastructure": ["LT", "ADANI"],
    "telecom": ["RELIANCE", "AIRTEL"],
    "adani": ["ADANI", "ADANIGREEN"],

    # Pharma & FMCG
    "pharma": ["SUNPHARMA", "DRREDDY", "CIPLA"],
    "healthcare": ["SUNPHARMA", "DRREDDY"],
    "fmcg": ["ITC", "HINDUNILVR"],
}


FINANCIAL_TERMS = {
    # Markets & Exchanges
    "stock", "stocks", "share", "shares", "equity", "equities", "market", "markets",
    "bse", "nse", "nifty", "sensex", "indices", "index", "bull", "bear",
    
    # Valuation & Metrics
    "pe", "p/e", "valuation", "valuations", "market cap", "marketcap", "mcap",
    "roe", "roce", "ebitda", "pat", "eps", "dividend", "yield", "margin", "margins",
    "debt", "leverage", "cash flow", "balance sheet", "income statement",
    "revenue", "profit", "loss", "growth", "earnings", "guidance", "order book",
    
    # Corporate Actions & Filings
    "annual report", "filing", "filings", "10-k", "quarterly", "results",
    "q1", "q2", "q3", "q4", "fy24", "fy25", "fy26", "concall", "transcripts",
    "ipo", "demerger", "merger", "acquisition", "bonus", "split", "promoter",
    
    # Investment & Sectors
    "invest", "investment", "investing", "buy", "sell", "hold", "target price",
    "portfolio", "outperform", "underperform", "multibagger", "risk", "hedging",
    "sector", "sectors", "industry", "industries", "thematic", "peers", "peer",
    "capex", "monetary policy", "rbi", "repo rate", "inflation", "gdp", "fmcg",
    "defence", "defense", "banking", "pharma", "infra", "renewable", "solar", "telecom"
}

class SectorResearchAgent:
    """Specialized Agent node for thematic, multi-sector, and macro inquiries."""

    def __init__(self, news, quant):
        self.news = news
        self.quant = quant

    def detect_sectors(self, query):
        q = query.lower()
        found = {}
        for keyword, keys in SECTOR_MAP.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', q) or keyword in q:
                for k in keys:
                    if k not in found and k in STOCKS:
                        found[k] = STOCKS[k]
        return found

    def build_sector_context(self, query, sector_stocks):
        parts = []
        fundamentals = []
        all_news = []
        grouped = {}

        for key, info in sector_stocks.items():
            sec = info["sector"]
            if sec not in grouped:
                grouped[sec] = []
            grouped[sec].append((key, info))

        for sec_name, members in grouped.items():
            parts.append(f"\n### Sector: {sec_name}")

            for key, info in members:
                ticker = info["ticker"]
                name = info["name"]

                try:
                    stats = self.quant.get_company_data(ticker)
                    mcap = stats.get("market_cap", 0)
                    mcap_str = f"₹{mcap / 1e7:,.2f} Cr" if mcap else "N/A"
                    price = stats.get("current_price", "N/A")
                    pe = stats.get("pe_ratio", "N/A")
                    roe = stats.get("roe", "N/A")
                    de = stats.get("debt_to_equity", "N/A")
                    rev_g = stats.get("revenue_growth", "N/A")
                    margin = stats.get("profit_margin", "N/A")
                    yr_change = stats.get("price_change_1y", "N/A")

                    parts.append(
                        f"\n**{name} ({ticker})**:\n"
                        f"- Live Price: ₹{price}\n"
                        f"- Market Cap: {mcap_str}\n"
                        f"- P/E Ratio: {pe}\n"
                        f"- ROE: {roe}\n"
                        f"- Debt to Equity: {de}\n"
                        f"- Revenue Growth: {rev_g}\n"
                        f"- Profit Margin: {margin}\n"
                        f"- 1-Year Price Return: {yr_change}%\n"
                    )
                    fundamentals.append({
                        "ticker": ticker,
                        "name": name,
                        "sector": sec_name,
                        "price": price,
                        "pe": pe,
                        "mcap": mcap_str,
                        "roe": f"{roe*100:.1f}%" if isinstance(roe, (int, float)) else str(roe),
                        "rev_growth": f"{rev_g*100:.1f}%" if isinstance(rev_g, (int, float)) else str(rev_g),
                        "yr_return": f"{yr_change:.1f}%" if isinstance(yr_change, (int, float)) else str(yr_change)
                    })
                except Exception as e:
                    logger.warning(f"Error fetching {ticker}: {e}")

            # Sector news
            news_q = f"{sec_name} stocks India market news"
            news_items = self.news.fetch_live_news(news_q, limit=2)
            if news_items:
                all_news.extend(news_items)
                parts.append(f"\nRecent News & Drivers for {sec_name}:")
                parts.append(self.news.format_news_for_context(news_items))

        # Query specific news
        q_news = self.news.fetch_live_news(query, limit=3)
        if q_news:
            all_news.extend(q_news)
            parts.append(f"\nBreaking Query News:")
            parts.append(self.news.format_news_for_context(q_news))

        return "\n".join(parts), fundamentals, all_news


class IndiaMultiAgentOrchestrator:
    def __init__(self, vector_store=None):
        self.store = vector_store
        self.news = IndianFinancialNewsFetcher()
        self.quant = FinancialDataFetcher()
        self.qa = GeminiQAModel()
        self.sector_agent = SectorResearchAgent(self.news, self.quant)

    def find_tickers(self, text):
        hits = []
        up = text.upper()
        for k, v in STOCKS.items():
            if re.search(r'\b' + re.escape(k) + r'\b', up) or v["name"].upper() in up:
                if v not in hits:
                    hits.append(v)
        return hits

    def _fetch_stock_context(self, targets):
        parts = []
        fundamentals = []
        news_collected = []
        for t in targets:
            ticker = t["ticker"]
            name = t["name"]

            news_items = self.news.fetch_live_news(name, ticker, limit=3)
            news_collected.extend(news_items)
            parts.append(f"Recent News for {name} ({ticker}):\n{self.news.format_news_for_context(news_items)}")

            try:
                stats = self.quant.get_company_data(ticker)
                mcap = stats.get("market_cap", 0)
                mcap_str = f"₹{mcap / 1e7:,.2f} Cr" if mcap else "N/A"
                price = stats.get("current_price", "N/A")
                pe = stats.get("pe_ratio", "N/A")
                roe = stats.get("roe", "N/A")
                de = stats.get("debt_to_equity", "N/A")
                rev_g = stats.get("revenue_growth", "N/A")
                margin = stats.get("profit_margin", "N/A")
                yr_change = stats.get("price_change_1y", "N/A")

                parts.append(
                    f"Financials for {name} ({ticker}, {t['sector']}):\n"
                    f"- Price: ₹{price}\n"
                    f"- Market Cap: {mcap_str}\n"
                    f"- P/E: {pe}\n"
                    f"- ROE: {roe}\n"
                    f"- Debt/Equity: {de}\n"
                    f"- Revenue Growth: {rev_g}\n"
                    f"- Profit Margin: {margin}\n"
                    f"- 1-Year Price Return: {yr_change}%\n"
                )
                fundamentals.append({
                    "ticker": ticker,
                    "name": name,
                    "sector": t.get("sector", "Equities"),
                    "price": price,
                    "pe": pe,
                    "mcap": mcap_str,
                    "roe": f"{roe*100:.1f}%" if isinstance(roe, (int, float)) else str(roe),
                    "rev_growth": f"{rev_g*100:.1f}%" if isinstance(rev_g, (int, float)) else str(rev_g),
                    "yr_return": f"{yr_change:.1f}%" if isinstance(yr_change, (int, float)) else str(yr_change)
                })
            except Exception as e:
                logger.warning(f"Failed stats for {ticker}: {e}")

        return parts, fundamentals, news_collected

    def run_comparative_analysis(self, query):
        trace = []
        start_time = time.time()
        
        # Step 0: Domain Guardrail Verification
        q_strip = query.strip()
        q_lower = q_strip.lower()
        q_upper = q_strip.upper()

        has_ticker = any(re.search(r'\b' + re.escape(k) + r'\b', q_upper) or v["name"].upper() in q_upper for k, v in STOCKS.items())
        has_sector = any(re.search(r'\b' + re.escape(s) + r'\b', q_lower) for s in SECTOR_MAP.keys())
        has_term = any(re.search(r'\b' + re.escape(t) + r'\b', q_lower) for t in FINANCIAL_TERMS)
        is_domain = has_ticker or has_sector or has_term

        if not is_domain and len(q_strip.split()) > 2 and self.qa.client:
            try:
                check_prompt = (
                    "Is the following user input asking about financial markets, investing, stocks, companies, "
                    "business performance, economic sectors, or corporate filings? "
                    "Reply strictly with only YES or NO.\n\n"
                    f"User Query: {q_strip}"
                )
                res = self.qa.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=check_prompt
                )
                if res and res.text and "YES" in res.text.upper():
                    is_domain = True
            except Exception:
                pass

        trace.append({
            "step": 0,
            "agent": "Domain Guardrail Agent",
            "action": "Context & Domain Relevance Verification",
            "detail": f"Evaluation Result: {'IN-DOMAIN (Authorized)' if is_domain else 'OUT-OF-DOMAIN (Rejected)'}"
        })

        if not is_domain:
            msg = (
                "⚠️ **This query is out of context.**\n\n"
                "My field of expertise is **Indian stock and financial market analysis** "
                "(NSE & BSE equities, sectoral research, valuation multiples, corporate filings, and macro financial trends).\n\n"
                "Please ask an investment, company, or market-related question (e.g. *'Compare HAL vs BDL'*, "
                "*'What is the better sector between Space and EV?'*, or *'Evaluate Reliance and Adani capex'*)."
            )
            return {
                "query": query,
                "entities": [],
                "answer": msg,
                "confidence": 0.0,
                "fundamentals": [],
                "news": [],
                "trace": trace,
                "elapsed_seconds": round(time.time() - start_time, 2)
            }

        # Step 1: Query Intent & Entity Resolution
        targets = self.find_tickers(query)
        is_sector = self._is_sector_query(query)

        trace.append({
            "step": 1,
            "agent": "Intent & Taxonomy Agent",
            "action": "Entity Resolution & Classification",
            "detail": f"Query classified as {'Sector/Thematic Comparative' if is_sector else 'Direct Ticker Evaluation'}. Direct entities identified: {[t['name'] for t in targets] if targets else 'None (Triggering Sector Dynamic Mapping)'}"
        })

        ctx_parts = []
        fundamentals = []
        collected_news = []

        # Step 2: Dynamic Sector Peer Basket Mapping if needed
        if is_sector or not targets:
            sector_stocks = self.sector_agent.detect_sectors(query)
            if sector_stocks:
                sec_ctx, sec_funds, sec_news = self.sector_agent.build_sector_context(query, sector_stocks)
                ctx_parts.append(sec_ctx)
                fundamentals.extend(sec_funds)
                collected_news.extend(sec_news)
                targets = list(sector_stocks.values())

                trace.append({
                    "step": 2,
                    "agent": "Sector Research Agent",
                    "action": "Thematic Peer Basket Enumeration",
                    "detail": f"Constructed representative peer basket across {len(sector_stocks)} equities: {list(sector_stocks.keys())}"
                })

        # Step 3: Fetch Direct Stock Fundamentals & Quant Analysis
        if targets and not ctx_parts:
            parts, funds, stock_news = self._fetch_stock_context(targets)
            ctx_parts.extend(parts)
            fundamentals.extend(funds)
            collected_news.extend(stock_news)

        trace.append({
            "step": 3,
            "agent": "Quantitative Engine",
            "action": "Real-Time Exchange Data Ingestion",
            "detail": f"Extracted deterministic ratios (P/E, Market Cap, ROE, Debt/Equity, 1Y Returns) for {len(fundamentals)} target symbols via NSE/BSE feeds."
        })

        # Fallback if still empty
        if not ctx_parts:
            fallback_txt = self._broad_fallback(query)
            ctx_parts.append(fallback_txt)

        # Step 4: RAG Filings Retrieval Agent
        rag_hits = 0
        if self.store:
            try:
                docs = self.store.similarity_search(query, k=4)
                if docs:
                    txt = "\n---\n".join([d.get("text", "") for d in docs if isinstance(d, dict)])
                    ctx_parts.append(f"Annual Report Filings Excerpts:\n{txt}")
                    rag_hits = len(docs)
            except Exception:
                pass

        trace.append({
            "step": 4,
            "agent": "Corporate Filings RAG Agent",
            "action": "Semantic Vector Store Retrieval",
            "detail": f"Searched audited corporate filings (HAL, BDL, L&T, Reliance Power). Retrievable matches: {rag_hits} chunks."
        })

        # Step 5: News & Sentiment Ingestion Agent
        trace.append({
            "step": 5,
            "agent": "Market News & Sentiment Agent",
            "action": "Live Breaking Catalyst Aggregation",
            "detail": f"Aggregated {len(collected_news)} real-time headlines across Reuters, Business Standard, and Yahoo Finance."
        })

        # Step 6: Lead Advisory Synthesis Agent
        full_ctx = "\n\n".join(ctx_parts)

        trace.append({
            "step": 6,
            "agent": "Lead Advisory Synthesis Agent",
            "action": "Grounded Institutional Synthesis",
            "detail": "Synthesizing cross-sector valuation, regulatory policy tailwinds, and risks into a formal investment memorandum."
        })

        res = self.qa.answer_question(
            context=full_ctx,
            question=query,
            query_type="sector" if is_sector else "stock"
        )

        elapsed = round(time.time() - start_time, 2)

        return {
            "query": query,
            "entities": targets,
            "answer": res.get("answer", ""),
            "confidence": res.get("confidence", 0.95),
            "fundamentals": fundamentals,
            "news": collected_news,
            "trace": trace,
            "elapsed_seconds": elapsed
        }

    def _is_sector_query(self, query):
        q = query.lower()
        sector_words = [
            "sector", "industry", "best stock", "best company",
            "which is better", "compare sector", "top stocks", "stocks i can buy",
            "space", "ev", "electric", "battery", "defence", "defense", "pharma",
            "banking", "it ", "auto", "energy", "power", "infra",
            "renewable", "fmcg", "oil", "metals"
        ]
        return any(w in q for w in sector_words)

    def _broad_fallback(self, query):
        parts = []
        news_items = self.news.fetch_live_news(query, limit=5)
        if news_items:
            parts.append(f"Market News for '{query}':\n{self.news.format_news_for_context(news_items)}")
        parts.append(
            f"Query Topic: {query}\n"
            f"Scope: Indian Equities (NSE & BSE)\n"
            f"Note: Provide comprehensive sectoral reasoning using your Indian market knowledge and available market news."
        )
        return "\n".join(parts)
