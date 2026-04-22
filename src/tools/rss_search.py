"""
RSS Feed Search Tool for KHUNEHO? Neural Analysis System
Provides news data from RSS feeds using feedparser
"""
import asyncio
import feedparser
from typing import List, Dict
from datetime import datetime, timedelta
import re
from urllib.parse import quote

from ..core.config import config

class RSSSearcher:
    """RSS feed-based news search - real-time news from RSS feeds"""
    
    def __init__(self):
        self.max_results = config.max_search_results
        self.max_news_results = config.max_news_results
        self.timeout = config.search_timeout
        
        # Major news RSS feeds with more specialized sources
        self.rss_feeds = {
            'general': [
                'http://feeds.bbci.co.uk/news/rss.xml',
                'https://rss.cnn.com/rss/edition.rss',
                'https://feeds.reuters.com/reuters/topNews',
                'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
                'https://feeds.npr.org/1001/rss.xml',
                'https://feeds.feedburner.com/axios/rss',
                'https://feeds.theguardian.com/theguardian/world/rss'
            ],
            'financial': [
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://feeds.reuters.com/news/wealth',
                'https://www.ft.com/rss/home',
                'https://feeds.marketwatch.com/marketwatch/topstories',
                'https://feeds.feedburner.com/wallstreetjournal/WSJrealtimeRSS',
                'https://feeds.feedburner.com/seekingalpha/SectorInvesting'
            ],
            'technology': [
                'https://techcrunch.com/feed/',
                'https://feeds.arstechnica.com/arstechnica/index',
                'https://www.wired.com/feed/rss',
                'https://feeds.macrumors.com/',
                'https://feeds.theverge.com/theverge/main',
                'https://feeds.feedburner.com/venturebeat/SZYFH',
                'https://www.zdnet.com/news/rss.xml'
            ],
            'geopolitical': [
                'https://feeds.reuters.com/worldNews',
                'https://feeds.bbci.co.uk/news/world/rss.xml',
                'https://feeds.aljazeera.com/xml/rss/all.xml',
                'https://feeds.npr.org/1004/rss.xml',
                'https://feeds.feedburner.com/foreignpolicy/articles',
                'https://feeds.feedburner.com/theeconomist/all',
                'https://www.cfr.org/rss/feeds/feed'
            ],
            'environmental': [
                'https://feeds.bbci.co.uk/news/science/environment/rss.xml',
                'https://feeds.reuters.com/environment',
                'https://www.nature.com/nature.rss',
                'https://feeds.feedburner.com/NationalGeographicNews',
                'https://feeds.feedburner.com/sciencedaily/news',
                'https://www.euronews.com/rss/science'
            ],
            'political': [
                'https://feeds.reuters.com/politicsNews',
                'https://feeds.feedburner.com/politico/politics',
                'https://www.c-span.org/rss/',
                'https://feeds.feedburner.com/thehill/home',
                'https://feeds.washingtonpost.com/rss/politics'
            ]
        }
        
    def search(self, query: str, max_results: int = None, category: str = 'general') -> List[Dict]:
        """Search RSS feeds for news articles matching query"""
        max_results = max_results or self.max_results
        results = []
        
        # Smart category selection based on query content
        query_lower = query.lower()
        if any(word in query_lower for word in ['prime minister', 'president', 'election', 'government', 'politics', 'minister']):
            category = 'political'
        elif any(word in query_lower for word in ['federal reserve', 'bank', 'market', 'economy', 'financial', 'economic']):
            category = 'financial'
        elif any(word in query_lower for word in ['technology', 'tech', 'software', 'ai', 'digital']):
            category = 'technology'
        elif any(word in query_lower for word in ['climate', 'environment', 'energy', 'pollution']):
            category = 'environmental'
        elif any(word in query_lower for word in ['international', 'geopolitical', 'foreign', 'diplomatic']):
            category = 'geopolitical'
        
        # Get feeds for category
        feeds = self.rss_feeds.get(category, self.rss_feeds['general'])
        
        try:
            for feed_url in feeds[:3]:  # Limit to 3 feeds for performance
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:max_results // 2]:  # Get more entries to work with
                        # Extract content first
                        content = entry.get('summary', entry.get('description', ''))
                        if len(content) > 200:
                            content = content[:200] + '...'
                        
                        # Check if query matches title or summary (more lenient)
                        title_lower = entry.get('title', '').lower()
                        summary_lower = content.lower()
                        query_lower = query.lower()
                        
                        # More sophisticated matching
                        query_words = [word for word in query_lower.split() if len(word) > 2]
                        
                        # Check for meaningful matches (at least 2 words or specific keywords)
                        matched_words = []
                        for word in query_words:
                            if word in title_lower or word in summary_lower:
                                matched_words.append(word)
                        
                        # Require at least 2 matching words OR 1 specific keyword match
                        specific_keywords = ['prime minister', 'president', 'federal reserve', 'central bank', 'election', 'government', 'policy']
                        has_specific_match = any(keyword in title_lower or keyword in summary_lower for keyword in specific_keywords)
                        
                        matches = len(matched_words) >= 2 or (len(matched_words) >= 1 and has_specific_match)
                        
                        # Only include if actually matches
                        include_entry = matches
                        
                        if include_entry:
                            results.append({
                                "title": entry.get('title', 'No title'),
                                "snippet": content,
                                "url": entry.get('link', ''),
                                "source": "rss",
                                "domain": self._extract_domain(feed_url),
                                "date": self._format_date(entry.get('published')),
                                "language": "en",
                                "country": "US"
                            })
                            
                            if len(results) >= max_results:
                                break
                                
                except Exception as e:
                    continue  # Skip failed feeds
            
            # If still no results, return context-aware headlines
            if not results:
                results = self._get_latest_headlines(max_results, query)
            
            return results[:max_results]
            
        except Exception as e:
            # Return fallback results if RSS fails
            return [
                {
                    "title": f"RSS search for: {query}",
                    "snippet": f"Unable to fetch RSS results for: {query}",
                    "url": f"https://news.google.com/search?q={quote(query)}",
                    "source": "rss_fallback"
                }
            ][:max_results or self.max_results]
    
    def _get_latest_headlines(self, max_results: int, query: str = '') -> List[Dict]:
        """Get latest headlines with context awareness"""
        results = []
        
        # Create context-aware fallback if no RSS matches found
        if query:
            return [{
                "title": f"Analysis of: {query}",
                "snippet": f"Context-based analysis for {query}. Current RSS feeds may not have specific coverage for this topic. Analysis will proceed based on neural network processing of the event.",
                "url": f"https://news.google.com/search?q={query.replace(' ', '%20')}",
                "source": "context_fallback",
                "domain": "analysis",
                "date": datetime.now().strftime('%Y-%m-%d'),
                "language": "en",
                "country": "US"
            }]
        
        # Generic fallback only if no query provided
        feeds = self.rss_feeds['general'][:2]  # Use 2 main feeds
        
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_results // 2]:
                    content = entry.get('summary', entry.get('description', ''))
                    if len(content) > 200:
                        content = content[:200] + '...'
                    
                    results.append({
                        "title": entry.get('title', 'No title'),
                        "snippet": content,
                        "url": entry.get('link', ''),
                        "source": "rss",
                        "domain": self._extract_domain(feed_url),
                        "date": self._format_date(entry.get('published')),
                        "language": "en",
                        "country": "US"
                    })
                    
                    if len(results) >= max_results:
                        break
                        
            except Exception as e:
                continue
        
        return results
    
    def _broad_search(self, query: str, max_results: int) -> List[Dict]:
        """Broader search across all feeds"""
        results = []
        all_feeds = []
        
        # Collect all feeds
        for category_feeds in self.rss_feeds.values():
            all_feeds.extend(category_feeds)
        
        for feed_url in all_feeds[:5]:  # Check 5 feeds
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    title_lower = entry.get('title', '').lower()
                    summary_lower = entry.get('summary', '').lower()
                    query_lower = query.lower()
                    
                    # Check for partial matches
                    query_words = query_lower.split()
                    matches = any(word in title_lower or word in summary_lower for word in query_words)
                    
                    if matches:
                        content = entry.get('summary', entry.get('description', ''))
                        if len(content) > 200:
                            content = content[:200] + '...'
                        
                        results.append({
                            "title": entry.get('title', 'No title'),
                            "snippet": content,
                            "url": entry.get('link', ''),
                            "source": "rss",
                            "domain": self._extract_domain(feed_url),
                            "date": self._format_date(entry.get('published')),
                            "language": "en",
                            "country": "US"
                        })
                        
                        if len(results) >= max_results:
                            break
                            
            except Exception as e:
                continue
        
        return results
    
    def search_news(self, topic: str, days_back: int = 7) -> List[Dict]:
        """Search recent news specifically"""
        return self.search(topic, max_results=self.max_news_results, category='general')
    
    def search_context(self, event: str) -> Dict[str, List[Dict]]:
        """Search multiple angles for context using RSS feeds"""
        return {
            "current_news": self.search_news(event),
            "historical": self.search(f"history {event}", max_results=3, category='general'),
            "analysis": self.search(f"analysis {event}", max_results=3, category='general'),
            "impact": self.search(f"impact {event}", max_results=3, category='general')
        }
    
    def search_for_neuron(self, event: str, neuron_type: str) -> List[Dict]:
        """Search specifically for a neuron's domain using RSS feeds"""
        # Map neuron types to RSS categories
        category_map = {
            'financial': 'financial',
            'technological': 'technology', 
            'geopolitical': 'geopolitical',
            'environmental': 'environmental',
            'health': 'general',
            'military': 'geopolitical',
            'legal': 'general',
            'economic': 'financial'
        }
        
        category = category_map.get(neuron_type, 'general')
        
        # Try category-specific search first
        results = self.search(event, max_results=3, category=category)
        
        # If no results, try general search with keywords
        if not results:
            keywords = config.get_keywords_for_domain(neuron_type)
            if keywords:
                query = f"{event} {keywords[0]}"
                results = self.search(query, max_results=3, category='general')
        
        return results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '')
        except:
            return 'Unknown'
    
    def _format_date(self, date_str: str) -> str:
        """Format date string"""
        try:
            if date_str:
                return date_str.split(' ')[0]  # Just return date part
            return datetime.now().strftime('%Y-%m-%d')
        except:
            return 'Unknown'
    
    def get_feed_stats(self) -> Dict[str, int]:
        """Get statistics about available feeds"""
        stats = {}
        for category, feeds in self.rss_feeds.items():
            stats[category] = len(feeds)
        return stats
