import re
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
    "M&M": {"name": "Mahindra & Mahindra", "ticker": "M&M.NS", "sector": "Auto & EV"},
    "MAHINDRA": {"name": "Mahindra & Mahindra", "ticker": "M&M.NS", "sector": "Auto & EV"},
    "MARUTI": {"name": "Maruti Suzuki India", "ticker": "MARUTI.NS", "sector": "Auto"},
    "OLECTRA": {"name": "Olectra Greentech", "ticker": "OLECTRA.NS", "sector": "EV & Electric Buses"},
    "EXIDE": {"name": "Exide Industries (EV Battery)", "ticker": "EXIDEIND.NS", "sector": "EV Batteries"},
    "EXIDEIND": {"name": "Exide Industries (EV Battery)", "ticker": "EXIDEIND.NS", "sector": "EV Batteries"},

    # Banking & Financials
    "ICICI": {"name": "ICICI Bank", "ticker": "ICICIBANK.NS", "sector": "Banking"},
    "HDFC": {"name": "HDFC Bank", "ticker": "HDFCBANK.NS", "sector": "Banking"},
    "SBI": {"name": "State Bank of India", "ticker": "SBIN.NS", "sector": "Banking"},
    "BAJFINANCE": {"name": "Bajaj Finance", "ticker": "BAJFINANCE.NS", "sector": "NBFC"},

    # Conglomerate, Energy, Telecom
    "RELIANCE": {"name": "Reliance Industries", "ticker": "RELIANCE.NS", "sector": "Energy & Telecom"},
    "AIRTEL": {"name": "Bharti Airtel", "ticker": "BHARTIARTL.NS", "sector": "Telecom"},
    "LT": {"name": "Larsen & Toubro", "ticker": "LT.NS", "sector": "Infrastructure & Space"},
    "ADANI": {"name": "Adani Enterprises", "ticker": "ADANIENT.NS", "sector": "Conglomerate"},
    "ADANIGREEN": {"name": "Adani Green Energy", "ticker": "ADANIGREEN.NS", "sector": "Renewable Energy"},
    "TATAPOWER": {"name": "Tata Power", "ticker": "TATAPOWER.NS", "sector": "Power & EV Charging"},
    "NTPC": {"name": "NTPC Limited", "ticker": "NTPC.NS", "sector": "Power & Energy"},

    # IT & Tech
    "TCS": {"name": "Tata Consultancy Services", "ticker": "TCS.NS", "sector": "IT"},
    "INFOSYS": {"name": "Infosys", "ticker": "INFY.NS", "sector": "IT"},
    "WIPRO": {"name": "Wipro", "ticker": "WIPRO.NS", "sector": "IT"},

    # Pharma
    "SUNPHARMA": {"name": "Sun Pharmaceutical", "ticker": "SUNPHARMA.NS", "sector": "Pharma"},
    "DRREDDY": {"name": "Dr. Reddy's Laboratories", "ticker": "DRREDDY.NS", "sector": "Pharma"},
    "CIPLA": {"name": "Cipla", "ticker": "CIPLA.NS", "sector": "Pharma"},

    # Metals & FMCG
    "TATASTEEL": {"name": "Tata Steel", "ticker": "TATASTEEL.NS", "sector": "Metals"},
    "ITC": {"name": "ITC Limited", "ticker": "ITC.NS", "sector": "FMCG"},
    "HINDUNILVR": {"name": "Hindustan Unilever", "ticker": "HINDUNILVR.NS", "sector": "FMCG"},
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
                        "price": price,
                        "pe": pe,
                        "mcap": mcap_str
                    })
                except Exception as e:
                    logger.warning(f"Error fetching {ticker}: {e}")

            # Sector news
            news_q = f"{sec_name} stocks India market news"
            news_items = self.news.fetch_live_news(news_q, limit=2)
            if news_items:
                parts.append(f"\nRecent News & Drivers for {sec_name}:")
                parts.append(self.news.format_news_for_context(news_items))

        # Query specific news
        q_news = self.news.fetch_live_news(query, limit=3)
        if q_news:
            parts.append(f"\nBreaking Query News:")
            parts.append(self.news.format_news_for_context(q_news))

        return "\n".join(parts), fundamentals


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
        for t in targets:
            ticker = t["ticker"]
            name = t["name"]

            news_items = self.news.fetch_live_news(name, ticker, limit=3)
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

                parts.append(
                    f"Financials for {name} ({ticker}, {t['sector']}):\n"
                    f"- Price: ₹{price}\n"
                    f"- Market Cap: {mcap_str}\n"
                    f"- P/E: {pe}\n"
                    f"- ROE: {roe}\n"
                    f"- Debt/Equity: {de}\n"
                    f"- Revenue Growth: {rev_g}\n"
                    f"- Profit Margin: {margin}\n"
                )
                fundamentals.append({
                    "ticker": ticker,
                    "name": name,
                    "price": price,
                    "pe": pe,
                    "mcap": mcap_str
                })
            except Exception as e:
                logger.warning(f"Failed stats for {ticker}: {e}")

        return parts, fundamentals

    def run_comparative_analysis(self, query):
        targets = self.find_tickers(query)
        ctx_parts = []
        fundamentals = []

        is_sector = self._is_sector_query(query)

        # If query asks about sectors/themes or doesn't match direct tickers
        if is_sector or not targets:
            sector_stocks = self.sector_agent.detect_sectors(query)
            if sector_stocks:
                sec_ctx, sec_funds = self.sector_agent.build_sector_context(query, sector_stocks)
                ctx_parts.append(sec_ctx)
                fundamentals.extend(sec_funds)
                targets = list(sector_stocks.values())

        # If specific stocks were targeted and not covered yet
        if targets and not ctx_parts:
            parts, funds = self._fetch_stock_context(targets)
            ctx_parts.extend(parts)
            fundamentals.extend(funds)

        # Fallback if still empty
        if not ctx_parts:
            ctx_parts.append(self._broad_fallback(query))

        # Vector store filings chunks if available
        if self.store:
            try:
                docs = self.store.similarity_search(query, k=4)
                if docs:
                    txt = "\n---\n".join([d.get("text", "") for d in docs if isinstance(d, dict)])
                    ctx_parts.append(f"Annual Report Filings Excerpts:\n{txt}")
            except Exception:
                pass

        full_ctx = "\n\n".join(ctx_parts)

        res = self.qa.answer_question(
            context=full_ctx,
            question=query,
            query_type="sector" if is_sector else "stock"
        )

        return {
            "query": query,
            "entities": targets,
            "answer": res.get("answer", ""),
            "confidence": res.get("confidence", 0.95),
            "fundamentals": fundamentals
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
