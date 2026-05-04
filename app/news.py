import asyncio
import aiohttp
import feedparser
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class NewsRetriever:
    """Retrieves real-time news from RSS feeds and GDELT project."""
    
    def __init__(self):
        self.rss_feeds = [
            {
                "name": "BBC News",
                "url": "http://feeds.bbci.co.uk/news/rss.xml",
                "category": "general"
            },
            {
                "name": "Reuters",
                "url": "https://www.reuters.com/rssFeed/worldNews",
                "category": "world"
            },
            {
                "name": "Al Jazeera",
                "url": "https://www.aljazeera.com/xml/rss/all.xml",
                "category": "world"
            }
        ]
        
        self.gdelt_base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.cache = {}
        self.cache_duration = timedelta(minutes=10)
    
    async def fetch_rss_feed(self, session: aiohttp.ClientSession, feed: Dict[str, str]) -> List[Dict[str, Any]]:
        """Fetch articles from a single RSS feed."""
        try:
            async with session.get(feed["url"], timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    content = await response.text()
                    parsed = feedparser.parse(content)
                    
                    articles = []
                    for entry in parsed.entries[:10]:  # Limit to 10 articles per feed
                        article = {
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", ""),
                            "link": entry.get("link", ""),
                            "source": feed["name"],
                            "category": feed["category"],
                            "published": entry.get("published", ""),
                            "content": entry.get("summary", "")
                        }
                        articles.append(article)
                    
                    logger.info(f"Fetched {len(articles)} articles from {feed['name']}")
                    return articles
                else:
                    logger.warning(f"Failed to fetch {feed['name']}: HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching {feed['name']}: {e}")
            return []
    
    async def fetch_gdelt_articles(self, query: str, max_articles: int = 20) -> List[Dict[str, Any]]:
        """Fetch articles from GDELT project based on query."""
        try:
            # GDELT API parameters
            params = {
                "query": query,
                "mode": "artlist",
                "maxrecords": max_articles,
                "format": "json",
                "sort": "DateDesc"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.gdelt_base_url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        data = await response.text()
                        # Parse GDELT response (tab-separated format)
                        articles = []
                        lines = data.strip().split('\n')
                        
                        for line in lines[1:]:  # Skip header
                            parts = line.split('\t')
                            if len(parts) >= 10:
                                article = {
                                    "title": parts[2] if len(parts) > 2 else "",
                                    "summary": parts[3] if len(parts) > 3 else "",
                                    "link": parts[1] if len(parts) > 1 else "",
                                    "source": parts[4] if len(parts) > 4 else "GDELT",
                                    "category": "world",
                                    "published": parts[0] if len(parts) > 0 else "",
                                    "content": parts[3] if len(parts) > 3 else ""
                                }
                                articles.append(article)
                        
                        logger.info(f"Fetched {len(articles)} articles from GDELT")
                        return articles
                    else:
                        logger.warning(f"GDELT API failed: HTTP {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching GDELT articles: {e}")
            return []
    
    def rank_relevance(self, articles: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank articles by relevance to the query."""
        query_words = set(query.lower().split())
        
        for article in articles:
            title_words = set(article["title"].lower().split())
            summary_words = set(article["summary"].lower().split())
            content_words = set(article["content"].lower().split())
            
            # Calculate relevance score
            title_match = len(query_words.intersection(title_words))
            summary_match = len(query_words.intersection(summary_words))
            content_match = len(query_words.intersection(content_words))
            
            # Weight title matches higher
            article["relevance_score"] = (title_match * 3 + summary_match * 2 + content_match)
        
        # Sort by relevance score
        ranked_articles = sorted(articles, key=lambda x: x["relevance_score"], reverse=True)
        return ranked_articles
    
    async def get_news(self, query: str = "", max_articles: int = 5) -> List[Dict[str, Any]]:
        """Get relevant news articles based on query."""
        cache_key = f"news_{query}_{max_articles}"
        
        # Check cache
        if cache_key in self.cache:
            cached_time, cached_articles = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                logger.info(f"Using cached news for query: {query}")
                return cached_articles
        
        all_articles = []
        
        # Fetch RSS feeds
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_rss_feed(session, feed) for feed in self.rss_feeds]
            rss_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in rss_results:
                if isinstance(result, list):
                    all_articles.extend(result)
        
        # Fetch GDELT articles if query is provided
        if query:
            gdelt_articles = await self.fetch_gdelt_articles(query)
            all_articles.extend(gdelt_articles)
        
        # Rank by relevance
        if query:
            all_articles = self.rank_relevance(all_articles, query)
        
        # Get top articles
        top_articles = all_articles[:max_articles]
        
        # Cache results
        self.cache[cache_key] = (datetime.now(), top_articles)
        
        logger.info(f"Returning {len(top_articles)} relevant articles")
        return top_articles
    
    def format_news_context(self, articles: List[Dict[str, Any]]) -> str:
        """Format articles into context string for LLM."""
        if not articles:
            return "No recent news articles found."
        
        context = "Recent News Context:\n\n"
        for i, article in enumerate(articles, 1):
            context += f"{i}. {article['title']}\n"
            context += f"   Source: {article['source']}\n"
            context += f"   Summary: {article['summary']}\n"
            context += f"   Link: {article['link']}\n\n"
        
        return context
