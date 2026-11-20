import re
import logging
from ..scoring.news_fetcher import IndianFinancialNewsFetcher
from ..scoring.financial_data_fetcher import FinancialDataFetcher
from ..models.gemini_models import GeminiQAModel

logger = logging.getLogger(__name__)

STOCKS = {
    "HAL": {"name": "Hindustan Aeronautics Limited", "ticker": "HAL.NS", "sector": "Defence"},
    "BDL": {"name": "Bharat Dynamics Limited", "ticker": "BDL.NS", "sector": "Defence"},
    "ICICI": {"name": "ICICI Bank", "ticker": "ICICIBANK.NS", "sector": "Banking"},
    "HDFC": {"name": "HDFC Bank", "ticker": "HDFCBANK.NS", "sector": "Banking"},
    "SBI": {"name": "State Bank of India", "ticker": "SBIN.NS", "sector": "Banking"},
    "RELIANCE": {"name": "Reliance Industries", "ticker": "RELIANCE.NS", "sector": "Energy & Telecom"},
    "AIRTEL": {"name": "Bharti Airtel", "ticker": "BHARTIARTL.NS", "sector": "Telecom"},
    "LT": {"name": "Larsen & Toubro", "ticker": "LT.NS", "sector": "Infrastructure"},
    "ADANI": {"name": "Adani Enterprises", "ticker": "ADANIENT.NS", "sector": "Conglomerate"},
}

class IndiaMultiAgentOrchestrator:
    def __init__(self, vector_store=None):
        self.store = vector_store
        self.news = IndianFinancialNewsFetcher()
        self.quant = FinancialDataFetcher()
        self.qa = GeminiQAModel()

    def find_tickers(self, text: str):
        hits = []
        up = text.upper()
        for k, v in STOCKS.items():
            if re.search(r'\b' + re.escape(k) + r'\b', up) or v["name"].upper() in up:
                hits.append(v)
        return hits

    def run_comparative_analysis(self, query: str):
        targets = self.find_tickers(query)
        context_parts = []
        fundamentals = []

        # If specific stocks were mentioned
        if targets:
            for t in targets:
                ticker = t["ticker"]
                name = t["name"]

                # news
                news_items = self.news.fetch_live_news(name, ticker, limit=4)
                context_parts.append(f"Recent News for {name} ({ticker}):\n{self.news.format_news_for_context(news_items)}")

                # financials
                try:
                    stats = self.quant.get_company_data(ticker)
                    mcap = stats.get("market_cap", 0)
                    mcap_str = f"₹{mcap / 1e7:,.2f} Cr" if mcap else "N/A"
                    price = stats.get("current_price", "N/A")
                    pe = stats.get("pe_ratio", "N/A")
                    roe = stats.get("roe", "N/A")
                    de = stats.get("debt_to_equity", "N/A")

                    context_parts.append(
                        f"Financials for {ticker} ({t['sector']}):\n"
                        f"- Price: ₹{price}\n"
                        f"- Market Cap: {mcap_str}\n"
                        f"- P/E: {pe}\n"
                        f"- ROE: {roe}\n"
                        f"- Debt/Equity: {de}\n"
                    )
                    fundamentals.append({
                        "ticker": ticker,
                        "name": name,
                        "price": price,
                        "pe": pe,
                        "mcap": mcap_str
                    })
                except Exception as err:
                    logger.warning(f"Failed to fetch stats for {ticker}: {err}")
        else:
            # Thematic / Sector / Broad query (e.g. "space stocks", "best EV stocks", etc.)
            theme_news = self.news.fetch_live_news(query, limit=5)
            context_parts.append(f"Live Market News for '{query}':\n{self.news.format_news_for_context(theme_news)}")
            context_parts.append(
                f"Topic: {query}\n"
                f"Scope: Indian Equities (NSE & BSE), ISRO supply chain, defence, energy, and thematic sectors."
            )

        # document chunks if available
        if self.store:
            try:
                docs = self.store.similarity_search(query, k=4)
                if docs:
                    txt = "\n---\n".join([d.get("text", "") for d in docs if isinstance(d, dict)])
                    context_parts.append(f"Annual Report Excerpts:\n{txt}")
            except Exception:
                pass

        full_context = "\n\n".join(context_parts)
        res = self.qa.answer_question(context=full_context, question=query)

        return {
            "query": query,
            "entities": targets,
            "answer": res.get("answer", ""),
            "confidence": res.get("confidence", 0.9),
            "fundamentals": fundamentals
        }
