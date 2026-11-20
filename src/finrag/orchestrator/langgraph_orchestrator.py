"""
FinRAG LangGraph Multi-Agent Orchestration Graph.
Implements a stateful, cyclic/acyclic multi-agent workflow using official LangGraph.
Now features a leading Domain Guardrail Node that validates financial context.
"""

import re
import time
import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END

from .india_orchestrator import (
    STOCKS,
    SECTOR_MAP,
    SectorResearchAgent,
    IndianFinancialNewsFetcher,
    FinancialDataFetcher,
    GeminiQAModel
)

logger = logging.getLogger(__name__)

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


class AgentState(TypedDict):
    """LangGraph shared agent memory state."""
    query: str
    is_in_domain: bool
    is_sector_query: bool
    entities: List[Dict[str, Any]]
    fundamentals: List[Dict[str, Any]]
    news: List[Dict[str, Any]]
    rag_excerpts: List[Dict[str, Any]]
    context_text: str
    answer: str
    confidence: float
    sources_used: List[Dict[str, Any]]
    unsupported_tickers: List[str]
    validated_ticker_data: Dict[str, Dict[str, Any]]
    trace: List[Dict[str, Any]]
    elapsed_seconds: float


class FinRAGLangGraphOrchestrator:
    """Stateful LangGraph runner for Indian equity intelligence."""

    def __init__(self, vector_store=None):
        self.store = vector_store
        self.news = IndianFinancialNewsFetcher()
        self.quant = FinancialDataFetcher()
        self.qa = GeminiQAModel()
        self.sector_agent = SectorResearchAgent(self.news, self.quant)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 1. Register Agent Nodes (guardrail_node is FIRST)
        workflow.add_node("guardrail_node", self._node_guardrail)
        workflow.add_node("out_of_domain_node", self._node_out_of_domain)
        workflow.add_node("unsupported_ticker_node", self._node_unsupported_ticker)
        workflow.add_node("intent_agent", self._node_intent_and_entities)
        workflow.add_node("sector_agent", self._node_sector_research)
        workflow.add_node("quant_engine", self._node_quant_engine)
        workflow.add_node("news_agent", self._node_news_and_catalysts)
        workflow.add_node("filings_rag", self._node_filings_rag)
        workflow.add_node("context_builder", self._node_context_builder)
        workflow.add_node("synthesis_agent", self._node_advisory_synthesis)

        # 2. Wire Edges & Conditional Routing
        # Entry point -> Guardrail
        workflow.add_edge(START, "guardrail_node")

        # Guardrail conditional branch
        workflow.add_conditional_edges(
            "guardrail_node",
            self._route_guardrail,
            {
                "out_of_domain_node": "out_of_domain_node",
                "unsupported_ticker_node": "unsupported_ticker_node",
                "intent_agent": "intent_agent"
            }
        )

        # If rejected -> Terminate immediately
        workflow.add_edge("out_of_domain_node", END)
        workflow.add_edge("unsupported_ticker_node", END)

        # If approved -> Intent conditional branch
        workflow.add_conditional_edges(
            "intent_agent",
            self._route_intent,
            {
                "unsupported_ticker_node": "unsupported_ticker_node",
                "sector_agent": "sector_agent",
                "quant_engine": "quant_engine"
            }
        )

        workflow.add_edge("sector_agent", "quant_engine")
        workflow.add_edge("quant_engine", "news_agent")
        workflow.add_edge("news_agent", "filings_rag")
        workflow.add_edge("filings_rag", "context_builder")
        workflow.add_edge("context_builder", "synthesis_agent")
        workflow.add_edge("synthesis_agent", END)

        return workflow.compile()

    # --- Routing Condition 1: Guardrail ---
    def _route_guardrail(self, state: AgentState) -> str:
        if state.get("unsupported_tickers"):
            return "unsupported_ticker_node"
        if not state.get("is_in_domain", True):
            return "out_of_domain_node"
        return "intent_agent"

    # --- Routing Condition 2: Intent ---
    def _route_intent(self, state: AgentState) -> str:
        if state.get("unsupported_tickers"):
            return "unsupported_ticker_node"
        if state.get("is_sector_query") or not state.get("entities"):
            return "sector_agent"
        return "quant_engine"

    # --- Node 0: Leading Domain Guardrail Node ---
    def _node_guardrail(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"].strip()
        q_lower = query.lower()
        q_upper = query.upper()

        trace = list(state.get("trace", []))

        # Check 1: Ticker match
        has_ticker = any(re.search(r'\b' + re.escape(k) + r'\b', q_upper) or v["name"].upper() in q_upper for k, v in STOCKS.items())
        explicit_tickers = re.findall(r"\b[A-Z0-9&-]+\.(?:NS|BO)\b", q_upper)
        has_ticker = has_ticker or bool(explicit_tickers)
        
        # Check 2: Sector match
        has_sector = any(re.search(r'\b' + re.escape(s) + r'\b', q_lower) for s in SECTOR_MAP.keys())

        # Check 3: Financial terminology match
        has_term = any(re.search(r'\b' + re.escape(t) + r'\b', q_lower) for t in FINANCIAL_TERMS)

        is_domain = has_ticker or has_sector or has_term

        # Check 4: If still uncertain and query is longer than 2 words, ask LLM guardrail
        if not is_domain and len(query.split()) > 2 and self.qa.client:
            try:
                check_prompt = (
                    "Is the following user input asking about financial markets, investing, stocks, companies, "
                    "business performance, economic sectors, or corporate filings? "
                    "Reply strictly with only YES or NO.\n\n"
                    f"User Query: {query}"
                )
                res = self.qa.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=check_prompt
                )
                if res and res.text and "YES" in res.text.upper():
                    is_domain = True
            except Exception as e:
                logger.warning(f"Guardrail LLM check failed: {e}")

        trace.append({
            "step": 1,
            "agent": "Domain Guardrail Node",
            "action": "Context & Domain Relevance Verification",
            "detail": f"Evaluation Result: {'IN-DOMAIN (Authorized)' if is_domain else 'OUT-OF-DOMAIN (Rejected)'}. Query verified against financial and market criteria."
        })

        return {
            "is_in_domain": is_domain,
            "unsupported_tickers": [],
            "validated_ticker_data": {},
            "trace": trace
        }

    def _node_unsupported_ticker(self, state: AgentState) -> Dict[str, Any]:
        tickers = state.get("unsupported_tickers", [])
        answer = (
            "I could not validate the following NSE/BSE ticker(s) through the configured market data source: "
            + ", ".join(tickers)
            + ". Please verify the exchange symbol. No company analysis or annual-report retrieval was run."
        )
        trace = list(state.get("trace", []))
        trace.append({
            "step": len(trace) + 1,
            "agent": "Ticker Mapping Validation",
            "action": "Unsupported Ticker Rejected",
            "detail": f"Yahoo Finance validation failed for: {', '.join(tickers)}",
        })
        return {"answer": answer, "confidence": 0.0, "sources_used": [], "trace": trace}

    # --- Node Rejection: Out of Domain Node ---
    def _node_out_of_domain(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        trace.append({
            "step": 2,
            "agent": "Domain Guardrail Node",
            "action": "Execution Interrupted",
            "detail": "Pipeline halted early. Query dismissed as out of scope to avoid hallucination."
        })

        msg = (
            "⚠️ **This query is out of context.**\n\n"
            "My field of expertise is **Indian stock and financial market analysis** "
            "(NSE & BSE equities, sectoral research, valuation multiples, corporate filings, and macro financial trends).\n\n"
            "Please ask an investment, company, or market-related question. For example:\n"
            "- *'What's the better investment right now between the Space sector and the EV sector?'*\n"
            "- *'Compare HAL vs BDL on defense order book and current P/E multiples.'*\n"
            "- *'Analyze ICICI Bank vs HDFC Bank on Net Interest Margins and asset quality.'*\n"
            "- *'How do recent events affect Reliance Industries and Adani Enterprises?'*"
        )

        return {
            "answer": msg,
            "confidence": 0.0,
            "fundamentals": [],
            "news": [],
            "trace": trace
        }

    # --- Node 1: Intent & Entity Classification ---
    def _node_intent_and_entities(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        up = query.upper()
        hits = []
        validated_ticker_data = dict(state.get("validated_ticker_data", {}))
        unsupported_tickers = []
        explicit_tickers = list(dict.fromkeys(re.findall(r"\b[A-Z0-9&-]+\.(?:NS|BO)\b", up)))

        for k, v in STOCKS.items():
            if re.search(r'\b' + re.escape(k) + r'\b', up) or v["name"].upper() in up:
                if v not in hits:
                    hits.append(v)

        mapped_by_ticker = {item["ticker"].upper(): item for item in STOCKS.values()}
        for ticker in explicit_tickers:
            mapped = mapped_by_ticker.get(ticker)
            if mapped and mapped not in hits:
                hits.append(mapped)
                continue
            if mapped:
                continue

            # The existing Yahoo Finance integration also validates explicitly
            # supplied Indian symbols that are not in the local alias mapping.
            try:
                stats = self.quant.get_company_data(ticker)
                company_name = stats.get("company_name")
                has_market_identity = any((
                    stats.get("current_price"), stats.get("market_cap"),
                    (stats.get("data_completeness") or 0) > 0,
                    stats.get("sector") not in (None, "", "Unknown"),
                    stats.get("industry") not in (None, "", "Unknown"),
                ))
                if stats.get("error") or not company_name or company_name.upper() == ticker or not has_market_identity:
                    unsupported_tickers.append(ticker)
                    continue
                validated_ticker_data[ticker] = stats
                entity = {
                    "name": company_name,
                    "ticker": ticker,
                    "sector": stats.get("sector") if stats.get("sector") not in (None, "", "Unknown") else "Equities",
                }
                if entity not in hits:
                    hits.append(entity)
            except Exception as exc:
                logger.warning("Yahoo Finance could not validate explicit ticker %s: %s", ticker, exc)
                unsupported_tickers.append(ticker)

        q_lower = query.lower()
        sector_words = [
            "sector", "industry", "best stock", "best company",
            "which is better", "compare sector", "top stocks", "stocks i can buy",
            "space", "ev", "electric", "battery", "defence", "defense", "pharma",
            "banking", "it ", "auto", "energy", "power", "infra",
            "renewable", "fmcg", "oil", "metals"
        ]
        is_sec = not hits and any(w in q_lower for w in sector_words)

        trace = list(state.get("trace", []))
        trace.append({
            "step": len(trace) + 1,
            "agent": "LangGraph Intent Node",
            "action": "Classification & Ticker Parsing",
            "detail": f"Type: {'Sector/Thematic' if is_sec else 'Direct Ticker'}. Direct hits: {[t['name'] for t in hits] or 'None'}"
        })

        return {
            "entities": hits,
            "is_sector_query": is_sec,
            "unsupported_tickers": unsupported_tickers,
            "validated_ticker_data": validated_ticker_data,
            "trace": trace
        }

    # --- Node 2: Sector Research Agent ---
    def _node_sector_research(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        entities = list(state.get("entities", []))

        # Only expand a peer basket for a sector query. A company name such as
        # "Reliance Power" must not accidentally fan out to the power sector.
        sector_stocks = self.sector_agent.detect_sectors(query) if state.get("is_sector_query") else {}

        for k, v in sector_stocks.items():
            if v not in entities:
                entities.append(v)

        trace = list(state.get("trace", []))
        trace.append({
            "step": len(trace) + 1,
            "agent": "LangGraph Sector Node",
            "action": "Dynamic Peer Enumeration",
            "detail": f"Constructed multi-stock basket ({len(entities)} tickers): {[e['ticker'] for e in entities]}"
        })

        return {
            "entities": entities,
            "trace": trace
        }

    # --- Node 3: Quantitative Engine ---
    def _node_quant_engine(self, state: AgentState) -> Dict[str, Any]:
        entities = state.get("entities", [])
        fundamentals = []

        for t in entities:
            ticker = t["ticker"]
            name = t["name"]
            sector = t.get("sector", "Equities")

            try:
                stats = state.get("validated_ticker_data", {}).get(ticker)
                if stats is None:
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

                fundamentals.append({
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "price": price,
                    "pe": pe,
                    "mcap": mcap_str,
                    "roe": f"{roe*100:.1f}%" if isinstance(roe, (int, float)) else str(roe),
                    "rev_growth": f"{rev_g*100:.1f}%" if isinstance(rev_g, (int, float)) else str(rev_g),
                    "yr_return": f"{yr_change:.1f}%" if isinstance(yr_change, (int, float)) else str(yr_change),
                    "debt_equity": de,
                    "margin": f"{margin*100:.1f}%" if isinstance(margin, (int, float)) else str(margin)
                })
            except Exception as e:
                logger.warning(f"Error fetching stats for {ticker}: {e}")

        trace = list(state.get("trace", []))
        trace.append({
            "step": len(trace) + 1,
            "agent": "LangGraph Quant Engine",
            "action": "Live Exchange Ratios Ingestion",
            "detail": f"Fetched verified fundamental ratios for {len(fundamentals)} equities via NSE/BSE feeds."
        })

        return {
            "fundamentals": fundamentals,
            "trace": trace
        }

    # --- Node 4: News & Catalysts Agent ---
    def _node_news_and_catalysts(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        entities = state.get("entities", [])
        collected = []

        q_news = self.news.fetch_live_news(query, limit=3)
        collected.extend(q_news)

        for e in entities[:3]:
            c_news = self.news.fetch_live_news(e["name"], e["ticker"], limit=2)
            collected.extend(c_news)

        trace = list(state.get("trace", []))
        trace.append({
            "step": len(trace) + 1,
            "agent": "LangGraph News Agent",
            "action": "Breaking Catalyst Feeds Aggregation",
            "detail": f"Aggregated {len(collected)} live financial headlines returned by the configured news fetcher."
        })

        return {
            "news": collected,
            "trace": trace
        }

    # --- Node 5: Corporate Filings RAG Agent ---
    def _node_filings_rag(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        rag_excerpts = []
        retrieval_status = []
        entities = state.get("entities", [])

        if not entities:
            retrieval_status.append("Skipped: no company was selected for company-level research.")
        elif not self.store:
            retrieval_status.extend(
                f"{entity['name']} ({entity['ticker']}): skipped because ChromaDB is unavailable."
                for entity in entities
            )
        else:
            for entity in entities:
                ticker = entity.get("ticker")
                name = entity.get("name", ticker or "Unknown company")
                if not ticker:
                    retrieval_status.append(f"{name}: skipped because no reliable ticker is available.")
                    continue
                try:
                    docs = self.store.similarity_search(query, k=3, where={"ticker": ticker})
                    company_docs = []
                    for doc in docs:
                        metadata = doc.get("metadata") or {}
                        # Defend against a broken/legacy backend that ignores `where`.
                        if metadata.get("ticker") != ticker or not doc.get("text"):
                            continue
                        company_docs.append({
                            "text": doc["text"],
                            "company_name": metadata.get("company_name", name),
                            "ticker": ticker,
                            "financial_year": metadata.get("financial_year", "Not recorded"),
                            "source_document": metadata.get("source_document", metadata.get("source", "Not recorded")),
                            "document_type": metadata.get("document_type", "annual_report"),
                        })
                    rag_excerpts.extend(company_docs)
                    retrieval_status.append(
                        f"{name} ({ticker}): matched {len(company_docs)} company-filtered report chunks."
                        if company_docs else f"{name} ({ticker}): no indexed company report; RAG skipped."
                    )
                except Exception as exc:
                    logger.exception("Annual report retrieval failed for %s", ticker)
                    retrieval_status.append(f"{name} ({ticker}): retrieval failed: {exc}")

        trace = list(state.get("trace", []))
        trace.append({
            "step": len(trace) + 1,
            "agent": "LangGraph Filings RAG",
            "action": "Vector Similarity Retrieval",
            "detail": "Company Knowledge Layer: " + " ".join(retrieval_status)
        })

        return {
            "rag_excerpts": rag_excerpts,
            "trace": trace
        }

    # --- Node 6: Context Builder & Grounding ---
    def _node_context_builder(self, state: AgentState) -> Dict[str, Any]:
        parts = []

        funds = state.get("fundamentals", [])
        if funds:
            parts.append("### Current Market Fundamentals (Yahoo Finance via yfinance, retrieved for this query):")
            for f in funds:
                parts.append(
                    f"- {f['name']} ({f['ticker']}, {f['sector']}): "
                    f"Price ₹{f['price']}, P/E {f['pe']}, MCap {f['mcap']}, "
                    f"ROE {f['roe']}, RevGrowth {f['rev_growth']}, 1Y Return {f['yr_return']}%, D/E {f['debt_equity']}"
                )

        news_items = state.get("news", [])
        if news_items:
            parts.append("\n### Breaking Market News & Catalysts:")
            for i, n in enumerate(news_items[:6], 1):
                src = f" ({n.get('source', '')})" if n.get('source') else ""
                pub = f" [{n.get('published', '')}]" if n.get('published') else ""
                parts.append(f"{i}. {n.get('title')}{src}{pub}")

        excerpts = state.get("rag_excerpts", [])
        if excerpts:
            parts.append("\n### Historical Company Disclosures (annual reports; not current market data):")
            for excerpt in excerpts:
                parts.append(
                    f"[Company: {excerpt['company_name']} | Ticker: {excerpt['ticker']} | "
                    f"Financial year: {excerpt['financial_year']} | Source: {excerpt['source_document']} | "
                    f"Document type: {excerpt['document_type']}]\n{excerpt['text']}"
                )

        full_context = "\n\n".join(parts)

        trace = list(state.get("trace", []))
        trace.append({
            "step": len(trace) + 1,
            "agent": "LangGraph Normalization Node",
            "action": "Deterministic Grounding Construction",
            "detail": f"Compiled {len(full_context)} characters of verified multi-source financial context."
        })

        return {
            "context_text": full_context,
            "trace": trace
        }

    def _synthesis_is_in_scope(self, answer: str, entities: List[Dict[str, Any]], is_sector_query: bool) -> bool:
        """Reject a synthesis that centers on known, unrelated Indian companies/sectors."""
        selected_tickers = {entity.get("ticker") for entity in entities}
        answer_lower = answer.lower()
        selected_sector_text = " ".join(entity.get("sector", "") for entity in entities).lower()

        known_by_ticker = {value["ticker"]: value for value in STOCKS.values()}
        for ticker, info in known_by_ticker.items():
            if ticker in selected_tickers:
                continue
            name = info.get("name", "").lower()
            if name and len(name) > 3 and name in answer_lower:
                candidate_sector = info.get("sector", "").lower()
                shared_terms = {
                    token for token in re.findall(r"[a-z]+", candidate_sector)
                    if token not in {"and", "the", "limited", "company", "equities", "enterprise"}
                } & {
                    token for token in re.findall(r"[a-z]+", selected_sector_text)
                    if token not in {"and", "the", "limited", "company", "equities", "enterprise"}
                }
                if candidate_sector and selected_sector_text and not shared_terms:
                    return False

        if is_sector_query:
            for keyword in ("it", "software", "technology"):
                selected_for_sector = any(
                    entity.get("ticker") in SECTOR_MAP.get(keyword, []) for entity in entities
                )
                if not selected_for_sector and re.search(
                    rf"\b{re.escape(keyword)}\s+(?:sector|stocks|companies)\b", answer_lower
                ):
                    return False
        return True

    @staticmethod
    def _sources_from_state(state: AgentState) -> List[Dict[str, Any]]:
        """Describe only evidence actually fetched and placed in this query context."""
        sources = []
        fundamentals = state.get("fundamentals", [])
        if fundamentals:
            sources.append({
                "type": "market_data",
                "provider": "Yahoo Finance via yfinance",
                "tickers": list(dict.fromkeys(item.get("ticker") for item in fundamentals if item.get("ticker"))),
            })
        for item in state.get("news", []):
            title = item.get("title")
            if title:
                source = {"type": "news", "title": title}
                for field in ("source", "published", "link"):
                    if item.get(field):
                        source[field] = item[field]
                sources.append(source)
        for item in state.get("rag_excerpts", []):
            sources.append({
                "type": "annual_report",
                "company_name": item.get("company_name"),
                "ticker": item.get("ticker"),
                "financial_year": item.get("financial_year"),
                "source_document": item.get("source_document"),
            })
        return sources

    @staticmethod
    def _has_unsupported_source_attribution(answer: str, state: AgentState) -> bool:
        """Reject explicit source attributions that are absent from this query's evidence."""
        attributions = re.findall(
            r"\b(?:according to|as per|as reported by|reported by|data from|source:)\s+([^.;\n]+)",
            answer,
            flags=re.IGNORECASE,
        )
        if not attributions:
            return False

        available = []
        if state.get("fundamentals"):
            available.extend(("yahoo finance", "yfinance"))
        for item in state.get("news", []):
            if item.get("source"):
                available.append(item["source"])
        for item in state.get("rag_excerpts", []):
            available.extend(
                value for value in (
                    item.get("source_document"), item.get("company_name"),
                    item.get("ticker"), item.get("financial_year"),
                ) if value and value != "Not recorded"
            )

        normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        allowed = [normalize(source) for source in available]
        return any(
            not any(source and source in normalize(claim) for source in allowed)
            for claim in attributions
        )

    @staticmethod
    def _grounded_fallback_answer(state: AgentState, reason: str) -> str:
        """Provide query-scoped source data when Gemini fails or returns off-scope text."""
        parts = [
            "## Grounded Research Summary",
            f"Gemini synthesis was omitted: {reason}",
            "The following facts come from the selected entities and sources for this query.",
        ]
        fundamentals = state.get("fundamentals", [])
        if fundamentals:
            parts.append("### Market data (Yahoo Finance via yfinance)")
            for item in fundamentals:
                parts.append(
                    f"- {item.get('name')} ({item.get('ticker')}): price {item.get('price')}, "
                    f"P/E {item.get('pe')}, ROE {item.get('roe')}, revenue growth {item.get('rev_growth')}, "
                    f"debt/equity {item.get('debt_equity')}, 1Y return {item.get('yr_return')}."
                )
        news_items = state.get("news", [])
        if news_items:
            parts.append("### Recent news")
            for item in news_items[:6]:
                attribution = [value for value in (item.get("source"), item.get("published")) if value]
                suffix = f" ({', '.join(attribution)})" if attribution else ""
                parts.append(f"- {item.get('title')}{suffix}")
        excerpts = state.get("rag_excerpts", [])
        if excerpts:
            parts.append("### Company report excerpts")
            parts.extend(
                f"- {item['company_name']} ({item['ticker']}), FY {item['financial_year']}, "
                f"{item['source_document']}: {item['text'][:500]}"
                for item in excerpts
            )
        return "\n\n".join(parts)

    # --- Node 7: Advisory Synthesis Agent ---
    def _node_advisory_synthesis(self, state: AgentState) -> Dict[str, Any]:
        context = state.get("context_text", "")
        query = state["query"]
        is_sec = state.get("is_sector_query", False)

        res = self.qa.answer_question(
            context=context,
            question=query,
            query_type="sector" if is_sec else "stock"
        )
        answer = res.get("answer", "")
        fallback_reason = None
        if not answer or res.get("confidence", 0.0) <= 0:
            fallback_reason = "the synthesis service did not return a usable answer."
        elif not self._synthesis_is_in_scope(answer, state.get("entities", []), is_sec):
            fallback_reason = "the generated answer referred to a company or sector outside this query."
        elif self._has_unsupported_source_attribution(answer, state):
            fallback_reason = "the generated answer attributed information to a source absent from this query's evidence."
        if fallback_reason:
            answer = self._grounded_fallback_answer(state, fallback_reason)

        trace = list(state.get("trace", []))
        trace.append({
            "step": len(trace) + 1,
            "agent": "LangGraph Lead Advisory Synthesis",
            "action": "Grounded Institutional Memorandum Generation",
            "detail": (
                "Returned a query-scoped grounded summary after rejecting unusable synthesis."
                if fallback_reason else "Synthesized the query-scoped market and company evidence."
            )
        })

        return {
            "answer": answer,
            "confidence": min(
                float(res.get("confidence", 0.0)),
                0.95 if state.get("fundamentals") and state.get("news") and state.get("rag_excerpts")
                else 0.75 if state.get("fundamentals") and state.get("news")
                else 0.55 if (state.get("fundamentals") or state.get("news") or state.get("rag_excerpts"))
                else 0.0,
            ) if not fallback_reason else 0.0,
            "sources_used": self._sources_from_state(state),
            "trace": trace
        }

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the LangGraph workflow end-to-end."""
        start_t = time.time()
        initial_state: AgentState = {
            "query": query,
            "is_in_domain": True,
            "is_sector_query": False,
            "entities": [],
            "fundamentals": [],
            "news": [],
            "rag_excerpts": [],
            "context_text": "",
            "answer": "",
            "confidence": 0.0,
            "sources_used": [],
            "unsupported_tickers": [],
            "validated_ticker_data": {},
            "trace": [],
            "elapsed_seconds": 0.0
        }

        final_state = self.graph.invoke(initial_state)
        final_state["elapsed_seconds"] = round(time.time() - start_t, 2)
        return final_state

    def get_mermaid_graph(self) -> str:
        """Return Mermaid graph visualization of the LangGraph state machine with the guardrail node."""
        return """flowchart TD
    __start__([START: User Query]) --> guardrail_node[Domain Guardrail Node]
    
    guardrail_node -->|Out of Domain| out_of_domain_node[Out-of-Domain Filter Node]
    guardrail_node -->|Authorized Market Query| intent_agent[Intent & Taxonomy Agent]
    
    out_of_domain_node --> __end_reject__([END: Rejection Notice])
    
    intent_agent -->|Sector / Thematic| sector_agent[Sector Research Agent]
    intent_agent -->|Direct Ticker| quant_engine[Quantitative Engine]
    
    sector_agent --> quant_engine
    quant_engine --> news_agent[Market News & Catalysts Agent]
    news_agent --> filings_rag[Filings Vector RAG Agent]
    filings_rag --> context_builder[Context Normalization Node]
    context_builder --> synthesis_agent[Lead Advisory Synthesis Agent]
    synthesis_agent --> __end__([END: Institutional Advisory Memo])
    
    style __start__ fill:#0F172A,stroke:#38BDF8,stroke-width:2px,color:#fff
    style __end__ fill:#0F172A,stroke:#22C55E,stroke-width:2px,color:#fff
    style __end_reject__ fill:#450A0A,stroke:#EF4444,stroke-width:2px,color:#fff
    style guardrail_node fill:#1E1B4B,stroke:#818CF8,stroke-width:2px,color:#fff
    style out_of_domain_node fill:#7F1D1D,stroke:#F87171,color:#fff
    style intent_agent fill:#1E293B,stroke:#94A3B8,color:#fff
    style sector_agent fill:#1E3A8A,stroke:#60A5FA,color:#fff
    style quant_engine fill:#064E3B,stroke:#34D399,color:#fff
    style news_agent fill:#701A75,stroke:#F472B6,color:#fff
    style filings_rag fill:#78350F,stroke:#FBBF24,color:#fff
    style context_builder fill:#312E81,stroke:#818CF8,color:#fff
    style synthesis_agent fill:#831843,stroke:#F43F5E,color:#fff
"""
