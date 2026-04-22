"""
Web Search Tool for KHUNEHO? Neural Analysis System
Provides context gathering using DuckDuckGo
"""
import asyncio
from typing import List, Dict
from duckduckgo_search import DDGS

from ..core.config import config

class WebSearcher:
    """Free web search - no API key required"""
    
    def __init__(self):
        self.ddgs = DDGS()
        self.max_results = config.max_search_results
        self.max_news_results = config.max_news_results
        self.timeout = config.search_timeout
    
    def search(self, query: str, max_results: int = None) -> List[Dict]:
        """Search web and return results"""
        max_results = max_results or self.max_results
        
        try:
            # Try with backend='news' for better reliability
            results = list(self.ddgs.text(query, max_results=max_results, backend='news'))
            if not results:
                # Fallback to default backend
                results = list(self.ddgs.text(query, max_results=max_results))
            print(results , query)
            var1 = [
                {
                    "title": r["title"],
                    "snippet": r["body"],
                    "url": r["href"],
                    "source": "duckduckgo"
                }
                for r in results
            ]
            print("var 1 is " + str(var1))
            return var1
        except Exception as e:
            # Return fallback results if search fails
            return [
                {
                    "title": f"Search results for: {query}",
                    "snippet": f"Unable to fetch live search results for: {query}",
                    "url": f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                    "source": "fallback"
                }
            ][:max_results or self.max_results]
    
    def search_news(self, topic: str, days_back: int = 7) -> List[Dict]:
        """Search recent news only"""
        query = f"{topic} news"
        return self.search(query, max_results=self.max_news_results)
    
    def search_context(self, event: str) -> Dict[str, List[Dict]]:
        """Search multiple angles for context"""
        return {
            "current_news": self.search_news(event),
            "historical": self.search(f"history of {event}"),
            "analysis": self.search(f"{event} expert analysis"),
            "impact": self.search(f"{event} impact consequences")
        }
    
    def search_for_neuron(self, event: str, neuron_type: str) -> List[Dict]:
        """Search specifically for a neuron's domain"""
        keywords = config.get_keywords_for_domain(neuron_type)
        if keywords:
            # Use first few keywords to create focused search
            query = f"{event} {' '.join(keywords[:3])}"
            return self.search(query, max_results=3)
        return self.search(event, max_results=3)
