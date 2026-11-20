"""
India Multi-Agent Financial & Geopolitical Intelligence Orchestrator.
Specializes in NSE & BSE equities: Defence (HAL, BDL), Banking (ICICI, HDFC, SBI),
Conglomerates (Reliance, Adani), and Infrastructure (L&T).
"""
from typing import Dict, Any, List, Optional
import os
import re
import logging
from datetime import datetime

from ..scoring.news_fetcher import IndianFinancialNewsFetcher
from ..scoring.financial_data_fetcher import FinancialDataFetcher
from ..models.gemini_models import GeminiQAModel

logger = logging.getLogger(__name__)


class IndiaMultiAgentOrchestrator:
    """Multi-Agent Orchestrator for Indian Equity & Geopolitical Analysis."""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.news_fetcher = IndianFinancialNewsFetcher()
        self.data_fetcher = FinancialDataFetcher()
        self.qa_model = GeminiQAModel()

        # Sector taxonomy mapping
        self.known_tickers = {
            "HAL": {"name": "Hindustan Aeronautics Limited", "ticker": "HAL.NS", "sector": "Defence & Aerospace"},
            "BDL": {"name": "Bharat Dynamics Limited", "ticker": "BDL.NS", "sector": "Defence & Missiles"},
            "ICICI": {"name": "ICICI Bank", "ticker": "ICICIBANK.NS", "sector": "Banking & Financial Services"},
            "HDFC": {"name": "HDFC Bank", "ticker": "HDFCBANK.NS", "sector": "Banking & Financial Services"},
            "SBI": {"name": "State Bank of India", "ticker": "SBIN.NS", "sector": "Banking & Financial Services"},
            "RELIANCE": {"name": "Reliance Industries", "ticker": "RELIANCE.NS", "sector": "Conglomerate & Energy"},
            "AIRTEL": {"name": "Bharti Airtel", "ticker": "BHARTIARTL.NS", "sector": "Telecom & Digital"},
            "LT": {"name": "Larsen & Toubro", "ticker": "LT.NS", "sector": "Infrastructure & Engineering"},
            "ADANI": {"name": "Adani Enterprises", "ticker": "ADANIENT.NS", "sector": "Energy & Infrastructure"},
        }

    def detect_entities(self, query: str) -> List[Dict[str, str]]:
        """Identify which companies/tickers are mentioned in the query."""
        detected = []
        q_upper = query.upper()
        for key, info in self.known_tickers.items():
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, q_upper) or info["name"].upper() in q_upper:
                detected.append(info)
        return detected

    def run_comparative_analysis(self, query: str) -> Dict[str, Any]:
        """
        Execute multi-agent workflow:
        1. Entity & Sector Detection
        2. Live Financial News Collection (Agent News)
        3. Fundamental Metrics & Valuation (Agent Quant)
        4. Annual Report Document Retrieval (Agent RAG)
        5. Deep Synthesis via Gemini Multi-Agent Evaluator
        """
        entities = self.detect_entities(query)
        logger.info("Detected %d entities for query: %s", len(entities), [e['ticker'] for e in entities])

        context_blocks = []
        quant_summary = []

        # 1. Collect Live News & Fundamentals for each detected entity
        for entity in entities:
            ticker = entity["ticker"]
            name = entity["name"]

            # Agent News: Fetch breaking Indian financial headlines
            news_items = self.news_fetcher.fetch_live_news(name, ticker, max_articles=4)
            formatted_news = self.news_fetcher.format_news_for_context(news_items)
            context_blocks.append(f"### Live News Context: {name} ({ticker})\n{formatted_news}")

            # Agent Quant: Fetch fundamentals from Yahoo Finance
            try:
                company_data = self.data_fetcher.get_company_data(ticker)
                curr_price = company_data.get("current_price", "N/A")
                pe = company_data.get("pe_ratio", "N/A")
                market_cap = company_data.get("market_cap", 0)
                mcap_in_cr = f"₹{market_cap / 1e7:,.2f} Cr" if isinstance(market_cap, (int, float)) and market_cap > 0 else "N/A"
                debt_to_equity = company_data.get("debt_to_equity", "N/A")
                roe = company_data.get("roe", "N/A")

                quant_block = (
                    f"### Live Valuation ({ticker} - {entity['sector']}):\n"
                    f"- Current Price: ₹{curr_price}\n"
                    f"- Market Cap: {mcap_in_cr}\n"
                    f"- P/E Ratio: {pe}\n"
                    f"- Debt-to-Equity: {debt_to_equity}\n"
                    f"- Return on Equity (ROE): {roe}\n"
                )
                context_blocks.append(quant_block)
                quant_summary.append({
                    "ticker": ticker,
                    "name": name,
                    "price": curr_price,
                    "pe": pe,
                    "mcap": mcap_in_cr
                })
            except Exception as e:
                logger.warning("Could not fetch quant data for %s: %s", ticker, e)

        # 2. Agent RAG: Retrieve matching Annual Report chunks if vector_store available
        if self.vector_store:
            try:
                rag_results = self.vector_store.similarity_search(query, k=5)
                rag_texts = [r.get("text", "") for r in rag_results if isinstance(r, dict)]
                if rag_texts:
                    context_blocks.append("### Corporate Annual Report Disclosures (RAG):\n" + "\n---\n".join(rag_texts[:4]))
            except Exception as e:
                logger.warning("Vector store query failed: %s", e)

        full_context = "\n\n".join(context_blocks)

        # 3. Agent Synthesizer: Call Gemini with structured sector instructions
        qa_result = self.qa_model.answer_question(context=full_context, question=query)

        return {
            "query": query,
            "entities": entities,
            "answer": qa_result.get("answer", ""),
            "confidence": qa_result.get("confidence", 0.9),
            "fundamentals": quant_summary,
            "context_used_length": len(full_context),
        }
