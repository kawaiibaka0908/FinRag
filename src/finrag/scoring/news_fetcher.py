import urllib.parse
from datetime import datetime
import feedparser
import yfinance as yf

class IndianFinancialNewsFetcher:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def fetch_live_news(self, name: str, ticker: str = None, limit: int = 5):
        articles = []
        seen = set()

        # Google News India RSS
        q = urllib.parse.quote(f"{name} share stock news India")
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            feed = feedparser.parse(rss_url)
            for item in feed.entries[:limit]:
                title = item.get("title", "").strip()
                if title and title not in seen:
                    seen.add(title)
                    articles.append({
                        "title": title,
                        "link": item.get("link", ""),
                        "published": item.get("published", ""),
                        "source": item.get("source", {}).get("title", "News"),
                    })
        except Exception:
            pass

        # yfinance news
        if ticker:
            try:
                stock = yf.Ticker(ticker)
                for item in (stock.news or [])[:3]:
                    title = item.get("title", "").strip()
                    if title and title not in seen:
                        seen.add(title)
                        pub = item.get("providerPublishTime")
                        date_str = datetime.fromtimestamp(pub).strftime("%d %b %Y") if pub else ""
                        articles.append({
                            "title": title,
                            "link": item.get("link", ""),
                            "published": date_str,
                            "source": item.get("publisher", "Yahoo Finance"),
                        })
            except Exception:
                pass

        return articles[:limit]

    def format_news_for_context(self, articles):
        if not articles:
            return "No recent news found."

        out = ["Recent News & Market Updates:"]
        for i, a in enumerate(articles, 1):
            src = f" ({a['source']})" if a.get("source") else ""
            date = f" [{a['published']}]" if a.get("published") else ""
            out.append(f"{i}. {a['title']}{src}{date}")
        return "\n".join(out)
