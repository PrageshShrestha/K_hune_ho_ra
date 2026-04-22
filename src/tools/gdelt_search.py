"""
GDELT Web Search Tool for KHUNEHO? Neural Analysis System
Provides real-time news data from GDELT Global Database
"""
import asyncio
from typing import List, Dict
from datetime import datetime, timedelta
from gdeltdoc import GdeltDoc, Filters
import pandas as pd

from ..core.config import config

class GDELTSearcher:
    """GDELT-based news search - real-time global news analysis"""
    
    def __init__(self):
        self.gd = GdeltDoc()
        self.max_results = config.max_search_results
        self.max_news_results = config.max_news_results
        self.timeout = config.search_timeout
        
    def search(self, query: str, max_results: int = None, days_back: int = 70) -> List[Dict]:
        """Search GDELT for recent news articles"""
        max_results = max_results or self.max_results
        
        try:
            # Set up date range for recent news
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            print("starting the search")
            # Create filters for GDELT search
            f = Filters(
                keyword=query,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            # Search for articles
            articles = self.gd.article_search(f)
            
            # Convert to our format
            print(articles)
            results = []

            for _, article in articles.head(max_results).iterrows():
                results.append({
                    "title": article.get('title', 'No title'),
                    "snippet": f"From {article.get('domain', 'Unknown source')} - {article.get('seendate', 'Unknown date')}",
                    "url": article.get('url', ''),
                    "source": "gdelt",
                    "domain": article.get('domain', 'Unknown'),
                    "date": article.get('seendate', 'Unknown'),
                    "language": article.get('language', 'Unknown'),
                    "country": article.get('sourcecountry', 'Unknown')
                })
            print(results)
            return results
            
        except Exception as e:
            # Return fallback results if GDELT fails
            return [
                {
                    "title": f"GDELT search for: {query}",
                    "snippet": f"Unable to fetch GDELT results for: {query}",
                    "url": f"https://www.gdeltproject.org/",
                    "source": "gdelt_fallback"
                }
            ][:max_results or self.max_results]
    
    def search_news(self, topic: str, days_back: int = 7) -> List[Dict]:
        """Search recent news specifically"""
        return self.search(topic, max_results=self.max_news_results, days_back=days_back)
    
    def search_context(self, event: str) -> Dict[str, List[Dict]]:
        """Search multiple angles for context using GDELT"""
        return {
            "current_news": self.search_news(event, days_back=3),  # Recent news (3 days)
            "historical": self.search(f"history of {event}", max_results=3, days_back=30),  # Historical context
            "analysis": self.search(f"{event} expert analysis", max_results=3, days_back=7),  # Expert analysis
            "impact": self.search(f"{event} impact consequences", max_results=3, days_back=7)  # Impact analysis
        }
    
    def search_for_neuron(self, event: str, neuron_type: str) -> List[Dict]:
        """Search specifically for a neuron's domain using GDELT"""
        keywords = config.get_keywords_for_domain(neuron_type)
        if keywords:
            # Use first few keywords to create focused search
            query = f"{event} {' '.join(keywords[:3])}"
            return self.search(query, max_results=3, days_back=7)
        return self.search(event, max_results=3, days_back=7)
    
    def get_tone_analysis(self, query: str, days_back: int = 7) -> Dict:
        """Get tone analysis for a topic from GDELT"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            f = Filters(
                keyword=query,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            # Get tone timeline
            timeline = self.gd.timeline_search("timelinetone", f)
            
            if not timeline.empty:
                latest_tone = timeline.iloc[-1]['timelinetone']
                return {
                    "average_tone": latest_tone,
                    "tone_trend": "increasing" if len(timeline) > 1 and timeline.iloc[-1]['timelinetone'] > timeline.iloc[-2]['timelinetone'] else "decreasing",
                    "data_points": len(timeline)
                }
            
        except Exception as e:
            pass
        
        return {
            "average_tone": 0.0,
            "tone_trend": "unknown",
            "data_points": 0
        }
    
    def get_volume_analysis(self, query: str, days_back: int = 7) -> Dict:
        """Get volume analysis for a topic from GDELT"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            f = Filters(
                keyword=query,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            # Get volume timeline
            timeline = self.gd.timeline_search("timelinevol", f)
            
            if not timeline.empty:
                latest_volume = timeline.iloc[-1]['timelinevol']
                return {
                    "average_volume": latest_volume,
                    "volume_trend": "increasing" if len(timeline) > 1 and timeline.iloc[-1]['timelinevol'] > timeline.iloc[-2]['timelinevol'] else "decreasing",
                    "data_points": len(timeline)
                }
            
        except Exception as e:
            pass
        
        return {
            "average_volume": 0.0,
            "volume_trend": "unknown", 
            "data_points": 0
        }
