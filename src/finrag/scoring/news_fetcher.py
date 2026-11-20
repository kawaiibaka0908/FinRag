"""
Live Indian Financial News Fetcher & Analyzer for NSE & BSE Stocks.
Integrates live financial news from Google News India and Yahoo Finance.
"""
from typing import List, Dict, Any, Optional
import urllib.parse
import logging
from datetime import datetime
import feedparser
import yfinance as yf

logger = logging.getLogger(__name__)


class IndianFinancialNewsFetcher:
    """Fetches real-time financial news for Indian companies (NSE/BSE)."""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def fetch_live_news(self, company_name: str, ticker: Optional[str] = None, max_articles: int = 6) -> List[Dict[str, Any]]:
        """
        Fetch real-time news headlines, summaries, and publications.
        Combines Google News India RSS and yfinance news.
        """
        articles = []
        seen_titles = set()

        # 1. Fetch from Google News India (Focused on corporate/earnings/geopolitics)
        query = f"{company_name} share stock news India"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:max_articles]:
                title = entry.get("title", "").strip()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    articles.append({
                        "title": title,
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": entry.get("source", {}).get("title", "Financial Press"),
                        "summary": entry.get("summary", ""),
                    })
        except Exception as e:
            logger.warning("Error fetching Google News for %s: %s", company_name, e)

        # 2. Fetch from yfinance (if ticker provided)
        if ticker:
            try:
                stock = yf.Ticker(ticker)
                yf_news = getattr(stock, "news", []) or []
                for item in yf_news[:4]:
                    title = item.get("title", "").strip()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        pub_time = item.get("providerPublishTime")
                        pub_date = datetime.fromtimestamp(pub_time).strftime("%a, %d %b %Y %H:%M:%S GMT") if pub_time else ""
                        articles.append({
                            "title": title,
                            "link": item.get("link", ""),
                            "published": pub_date,
                            "source": item.get("publisher", "Yahoo Finance"),
                            "summary": "",
                        })
            except Exception as e:
                logger.warning("Error fetching yfinance news for %s: %s", ticker, e)

        return articles[:max_articles]

    def format_news_for_context(self, articles: List[Dict[str, Any]]) -> str:
        """Format retrieved news articles into structured text context for the LLM."""
        if not articles:
            return "No recent news headlines retrieved."

        lines = ["--- LIVE INDIAN MARKET NEWS HEADLINES ---"]
        for i, art in enumerate(articles, 1):
            date_str = f" [{art['published']}]" if art.get("published") else ""
            source_str = f" ({art.get('source', '')})" if art.get("source") else ""
            lines.append(f"{i}. {art['title']}{source_str}{date_str}")
        lines.append("------------------------------------------")
        return "\n".join(lines)
