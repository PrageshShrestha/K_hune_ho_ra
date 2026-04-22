"""
Dynamic AI-Driven Analyzer for KHUNEHO? Neural Analysis System
Generates dynamic categories, parameters, and comprehensive reports
"""
import torch
import psutil
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from pathlib import Path
import requests

class DynamicAnalyzer:
    """AI-driven dynamic analysis with resource tracking and comprehensive reporting"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_time = time.time()
        self.category_timings = {}
        self.resource_usage = {}
        self.internet_usage = {}
        
        # Dynamic category generation
        self.base_categories = {
            'market_dynamics': 'Market movements, trading patterns, investor behavior',
            'economic_factors': 'Economic indicators, monetary policy, fiscal measures',
            'political_landscape': 'Government actions, policy changes, geopolitical events',
            'social_sentiment': 'Public opinion, social trends, cultural impact',
            'technological_impact': 'Innovation, digital transformation, tech disruption',
            'environmental_considerations': 'Climate impact, sustainability, ecological effects',
            'regulatory_framework': 'Legal requirements, compliance, regulatory changes',
            'competitive_landscape': 'Industry competition, market positioning, strategic moves'
        }
        
    def generate_dynamic_categories(self, event: str, agent_reports: List[Dict], web_context: Dict) -> Dict[str, Any]:
        """Generate AI-driven dynamic categories based on event context"""
        start_time = time.time()
        
        # Analyze event and agent outputs to determine relevant categories
        event_lower = event.lower()
        agent_insights = self._extract_agent_insights(agent_reports)
        web_themes = self._extract_web_themes(web_context)
        
        # Dynamic category selection and generation
        dynamic_categories = {}
        
        # Base categories with dynamic parameters
        for category, description in self.base_categories.items():
            if self._is_category_relevant(category, event_lower, agent_insights, web_themes):
                dynamic_categories[category] = {
                    'description': description,
                    'relevance_score': self._calculate_relevance_score(category, event_lower, agent_insights, web_themes),
                    'dynamic_parameters': self._generate_dynamic_parameters(category, event, agent_insights),
                    'ai_reasoning': self._generate_category_reasoning(category, event, agent_insights, web_themes),
                    'key_indicators': self._generate_key_indicators(category, event, agent_insights),
                    'time_horizons': self._generate_time_horizons(category, event)
                }
        
        # Generate event-specific custom categories
        custom_categories = self._generate_custom_categories(event, agent_insights, web_themes)
        dynamic_categories.update(custom_categories)
        
        # Track resource usage
        end_time = time.time()
        self.category_timings['category_generation'] = end_time - start_time
        self._track_resource_usage('category_generation')
        
        return dynamic_categories
    
    def _extract_agent_insights(self, agent_reports: List[Dict]) -> Dict[str, List[str]]:
        """Extract key insights from neural agent reports"""
        insights = {}
        
        for report in agent_reports:
            if hasattr(report, 'neuron_id') and hasattr(report, 'reasoning'):
                neuron_id = report.neuron_id
                reasoning = getattr(report, 'reasoning', '')
                prediction = report.class_labels[report.predicted_class] if report.class_labels and hasattr(report, 'predicted_class') else 'unknown'
                confidence = getattr(report, 'confidence', 0)
                
                insights[neuron_id] = [
                    f"Prediction: {prediction} (confidence: {confidence:.3f})",
                    f"Reasoning: {reasoning}",
                    f"Key themes: {self._extract_themes_from_reasoning(reasoning)}"
                ]
        
        return insights
    
    def _extract_web_themes(self, web_context: Dict) -> Dict[str, List[str]]:
        """Extract themes from web context"""
        themes = {}
        
        for category, articles in web_context.items():
            if articles:
                category_themes = []
                for article in articles[:5]:  # Limit to top 5 articles
                    title = article.get('title', '').lower()
                    snippet = article.get('snippet', '').lower()
                    
                    # Extract key themes
                    title_words = [word for word in title.split() if len(word) > 3]
                    snippet_words = [word for word in snippet.split() if len(word) > 4]
                    
                    category_themes.extend(title_words[:3])  # Top 3 words from title
                    category_themes.extend(snippet_words[:2])  # Top 2 words from snippet
                
                themes[category] = list(set(category_themes))  # Remove duplicates
        
        return themes
    
    def _is_category_relevant(self, category: str, event: str, agent_insights: Dict, web_themes: Dict) -> bool:
        """Determine if a category is relevant to the current event"""
        category_keywords = {
            'market_dynamics': ['market', 'trading', 'stock', 'investment', 'financial'],
            'economic_factors': ['economy', 'economic', 'federal', 'reserve', 'inflation', 'gdp'],
            'political_landscape': ['government', 'policy', 'political', 'election', 'minister', 'president'],
            'social_sentiment': ['social', 'public', 'sentiment', 'opinion', 'culture'],
            'technological_impact': ['technology', 'tech', 'digital', 'innovation', 'software'],
            'environmental_considerations': ['environment', 'climate', 'green', 'sustainability', 'energy'],
            'regulatory_framework': ['regulation', 'legal', 'compliance', 'law', 'policy'],
            'competitive_landscape': ['competition', 'competitive', 'market', 'industry', 'business']
        }
        
        # Check event keywords
        event_keywords = category_keywords.get(category, [])
        event_relevance = any(keyword in event for keyword in event_keywords)
        
        # Check agent insights
        agent_relevance = False
        for neuron_id, insights in agent_insights.items():
            insight_text = ' '.join(insights).lower()
            if any(keyword in insight_text for keyword in event_keywords):
                agent_relevance = True
                break
        
        # Check web themes
        web_relevance = False
        for category_themes in web_themes.values():
            theme_text = ' '.join(category_themes).lower()
            if any(keyword in theme_text for keyword in event_keywords):
                web_relevance = True
                break
        
        return event_relevance or agent_relevance or web_relevance
    
    def _calculate_relevance_score(self, category: str, event: str, agent_insights: Dict, web_themes: Dict) -> float:
        """Calculate relevance score for a category"""
        score = 0.0
        
        # Event relevance (40% weight)
        if self._is_category_relevant(category, event, agent_insights, web_themes):
            score += 0.4
        
        # Agent insights relevance (35% weight)
        relevant_agents = 0
        total_agents = len(agent_insights)
        for neuron_id, insights in agent_insights.items():
            insight_text = ' '.join(insights).lower()
            if category.replace('_', ' ') in insight_text or any(word in insight_text for word in category.split('_')):
                relevant_agents += 1
        
        if total_agents > 0:
            score += (relevant_agents / total_agents) * 0.35
        
        # Web themes relevance (25% weight)
        relevant_themes = 0
        total_themes = len(web_themes)
        for category_name, themes in web_themes.items():
            theme_text = ' '.join(themes).lower()
            if category.replace('_', ' ') in theme_text:
                relevant_themes += 1
        
        if total_themes > 0:
            score += (relevant_themes / total_themes) * 0.25
        
        return min(score, 1.0)
    
    def _generate_dynamic_parameters(self, category: str, event: str, agent_insights: Dict) -> List[Dict[str, str]]:
        """Generate practical predictions and impacts instead of technical parameters"""
        
        # Generate practical predictions based on category and event
        event_lower = event.lower()
        
        if category == 'market_dynamics':
            return [
                {'prediction': 'Stock market will see increased volatility', 'impact': 'Expect daily swings of 2-3% in major indices', 'timeline': 'Next 2-4 weeks'},
                {'prediction': 'Trading volumes will spike significantly', 'impact': 'Daily trading volume could increase by 30-50%', 'timeline': 'Immediate to 1 week'},
                {'prediction': 'Investor sentiment will shift dramatically', 'impact': 'Risk assets may sell off while safe-haven assets rise', 'timeline': '1-2 weeks'},
                {'prediction': 'Currency markets will react strongly', 'impact': 'Exchange rates could move 5-10% against major currencies', 'timeline': 'Next 1-3 weeks'}
            ]
        
        elif category == 'political_landscape':
            return [
                {'prediction': 'Government stability will be tested', 'impact': 'Policy implementation may slow down by 40-60%', 'timeline': 'Next 3-6 months'},
                {'prediction': 'International relations will shift', 'impact': 'Trade agreements may be renegotiated, affecting imports/exports', 'timeline': '6-12 months'},
                {'prediction': 'Public approval ratings will fluctuate', 'impact': 'Support levels could swing 15-25 points up or down', 'timeline': 'Next 1-3 months'},
                {'prediction': 'Legislative agenda will change priorities', 'impact': 'Current bills may be delayed or cancelled', 'timeline': 'Immediate to 6 months'}
            ]
        
        elif category == 'social_sentiment':
            return [
                {'prediction': 'Public opinion will polarize further', 'impact': 'Social media discussions will increase 200-300% with strong opinions', 'timeline': 'Next 2-4 weeks'},
                {'prediction': 'Media coverage will intensify', 'impact': 'News stories about this event will dominate headlines for weeks', 'timeline': 'Next 1-2 months'},
                {'prediction': 'Community tensions may rise', 'impact': 'Local protests or gatherings could increase by 50-100%', 'timeline': 'Next 1-3 months'},
                {'prediction': 'Cultural discussions will emerge', 'impact': 'New social movements or hashtags may trend globally', 'timeline': 'Next 2-6 weeks'}
            ]
        
        elif category == 'economic_factors':
            return [
                {'prediction': 'Economic growth forecasts will be revised', 'impact': 'GDP growth predictions may drop 0.5-1.5 percentage points', 'timeline': 'Next 1-3 months'},
                {'prediction': 'Inflation expectations will change', 'impact': 'Consumer prices may rise/fall 2-4% faster than expected', 'timeline': 'Next 3-6 months'},
                {'prediction': 'Job market will feel the effects', 'impact': 'Unemployment could change by 0.5-1.5 percentage points', 'timeline': 'Next 3-9 months'},
                {'prediction': 'Consumer spending will adjust', 'impact': 'Retail sales may increase/decrease 5-15% depending on confidence', 'timeline': 'Next 1-6 months'}
            ]
        
        elif category == 'leadership_transition':
            return [
                {'prediction': 'Government operations will temporarily slow', 'impact': 'Decision-making may take 2-3 times longer than usual', 'timeline': 'Next 1-3 months'},
                {'prediction': 'Policy continuity will be challenged', 'impact': '30-50% of current policies may be reviewed or changed', 'timeline': 'Next 6-12 months'},
                {'prediction': 'Diplomatic relationships will reset', 'impact': 'Foreign policy priorities may shift, affecting international partnerships', 'timeline': 'Next 3-9 months'},
                {'prediction': 'Domestic support will be tested', 'impact': 'Approval ratings could swing 20-30 points in either direction', 'timeline': 'Next 2-6 months'}
            ]
        
        elif category == 'technological_impact':
            return [
                {'prediction': 'Tech sector will see opportunity or risk', 'impact': 'Tech stocks may move 10-20% based on policy implications', 'timeline': 'Next 1-3 months'},
                {'prediction': 'Innovation focus will shift', 'impact': 'R&D spending may redirect to new priorities', 'timeline': 'Next 6-12 months'},
                {'prediction': 'Digital adoption will accelerate', 'impact': 'Technology usage could increase 15-25% in affected sectors', 'timeline': 'Next 3-6 months'},
                {'prediction': 'Cybersecurity concerns will rise', 'impact': 'Security spending may increase 20-40% across organizations', 'timeline': 'Next 1-6 months'}
            ]
        
        # Default predictions for other categories
        else:
            return [
                {'prediction': 'Market conditions will change significantly', 'impact': 'Expect notable shifts in related metrics and behaviors', 'timeline': 'Next 1-3 months'},
                {'prediction': 'Stakeholder reactions will be substantial', 'impact': 'Key players will adjust strategies and operations', 'timeline': 'Next 2-6 weeks'},
                {'prediction': 'Long-term implications will emerge', 'impact': 'Structural changes may affect the sector for years', 'timeline': '6 months to 2 years'},
                {'prediction': 'Monitoring requirements will increase', 'impact': 'More frequent analysis and tracking will be necessary', 'timeline': 'Ongoing'}
            ]
    
    def _generate_category_reasoning(self, category: str, event: str, agent_insights: Dict, web_themes: Dict) -> str:
        """Generate AI reasoning for category relevance"""
        reasoning_parts = []
        
        # Event-based reasoning
        event_lower = event.lower()
        if self._is_category_relevant(category, event_lower, agent_insights, web_themes):
            reasoning_parts.append(f"Event '{event}' directly impacts {category.replace('_', ' ')} through key terms and context")
        
        # Agent-based reasoning
        relevant_agents = []
        for neuron_id, insights in agent_insights.items():
            insight_text = ' '.join(insights).lower()
            if category.replace('_', ' ') in insight_text:
                relevant_agents.append(neuron_id)
        
        if relevant_agents:
            reasoning_parts.append(f"Neural agents {', '.join(relevant_agents)} provide strong signals for {category.replace('_', ' ')}")
        
        # Web context reasoning
        relevant_web_sources = []
        for source_category, themes in web_themes.items():
            theme_text = ' '.join(themes).lower()
            if category.replace('_', ' ') in theme_text:
                relevant_web_sources.append(source_category)
        
        if relevant_web_sources:
            reasoning_parts.append(f"Web sources from {', '.join(relevant_web_sources)} confirm {category.replace('_', ' ')} relevance")
        
        # Synthesize reasoning
        if reasoning_parts:
            return " | ".join(reasoning_parts)
        else:
            return f"Standard {category.replace('_', ' ')} analysis applicable to current event context"
    
    def _generate_key_indicators(self, category: str, event: str, agent_insights: Dict) -> List[str]:
        """Generate key indicators for category monitoring"""
        indicator_templates = {
            'market_dynamics': [
                'Market volatility index',
                'Trading volume patterns',
                'Price trend analysis',
                'Market sentiment indicators'
            ],
            'economic_factors': [
                'GDP growth projections',
                'Inflation rate changes',
                'Employment statistics',
                'Interest rate movements'
            ],
            'political_landscape': [
                'Policy approval ratings',
                'Legislative progress',
                'Public opinion polls',
                'Government stability metrics'
            ],
            'social_sentiment': [
                'Social media sentiment',
                'Public opinion surveys',
                'Media coverage tone',
                'Cultural impact assessments'
            ],
            'technological_impact': [
                'Innovation index',
                'Technology adoption rates',
                'Digital transformation metrics',
                'R&D investment trends'
            ],
            'environmental_considerations': [
                'Carbon emission levels',
                'Sustainability indices',
                'Environmental impact assessments',
                'Green technology adoption'
            ],
            'regulatory_framework': [
                'Compliance cost metrics',
                'Regulatory change frequency',
                'Legal risk assessments',
                'Policy implementation tracking'
            ],
            'competitive_landscape': [
                'Market share changes',
                'Competitive positioning',
                'Industry consolidation trends',
                'Barrier to entry analysis'
            ]
        }
        
        indicators = indicator_templates.get(category, ['General monitoring indicators'])
        
        # Customize based on event
        event_lower = event.lower()
        if 'prime minister' in event_lower or 'president' in event_lower:
            if category == 'political_landscape':
                indicators.append('Leadership approval ratings')
                indicators.append('Policy implementation timeline')
        
        return indicators
    
    def _generate_time_horizons(self, category: str, event: str) -> Dict[str, str]:
        """Generate time horizons for category analysis"""
        base_horizons = {
            'immediate': '0-7 days',
            'short_term': '1-4 weeks',
            'medium_term': '1-3 months',
            'long_term': '3-12 months',
            'extended': '1-3 years'
        }
        
        # Customize based on category
        category_customization = {
            'market_dynamics': {
                'immediate': 'Market trading sessions',
                'short_term': 'Weekly market cycles',
                'medium_term': 'Quarterly earnings impact',
                'long_term': 'Annual market trends'
            },
            'political_landscape': {
                'immediate': 'Policy announcement impact',
                'short_term': 'Legislative process',
                'medium_term': 'Policy implementation',
                'long_term': 'Election cycle effects'
            },
            'technological_impact': {
                'immediate': 'Technology adoption signals',
                'short_term': 'Early implementation',
                'medium_term': 'Market penetration',
                'long_term': 'Industry transformation'
            }
        }
        
        return category_customization.get(category, base_horizons)
    
    def _generate_custom_categories(self, event: str, agent_insights: Dict, web_themes: Dict) -> Dict[str, Any]:
        """Generate event-specific custom categories"""
        custom_categories = {}
        event_lower = event.lower()
        
        # Generate custom categories based on event content
        if 'prime minister' in event_lower or 'president' in event_lower:
            custom_categories['leadership_transition'] = {
                'description': 'Leadership change dynamics and governance impact',
                'relevance_score': 0.9,
                'dynamic_parameters': [
                    {'parameter': 'transition_stability', 'description': 'Smoothness of leadership change'},
                    {'parameter': 'policy_continuity', 'description': 'Existing policy maintenance'},
                    {'parameter': 'international_relations', 'description': 'Diplomatic relationship impact'},
                    {'parameter': 'domestic_support', 'description': 'Internal backing level'}
                ],
                'ai_reasoning': f"Event involves leadership change requiring specialized analysis of governance dynamics",
                'key_indicators': ['Leadership approval', 'Policy continuity index', 'International response'],
                'time_horizons': {
                    'immediate': 'Transition period',
                    'short_term': 'First 100 days',
                    'medium_term': 'Policy establishment',
                    'long_term': 'Leadership impact'
                }
            }
        
        if 'federal reserve' in event_lower or 'interest rate' in event_lower:
            custom_categories['monetary_policy'] = {
                'description': 'Central bank policy and monetary decisions',
                'relevance_score': 0.95,
                'dynamic_parameters': [
                    {'parameter': 'rate_decision_impact', 'description': 'Interest rate decision effect'},
                    {'parameter': 'inflation_targeting', 'description': 'Price stability goals'},
                    {'parameter': 'market_liquidity', 'description': 'Financial system liquidity'},
                    {'parameter': 'economic_outlook', 'description': 'Growth projections'}
                ],
                'ai_reasoning': f"Event involves monetary policy requiring specialized financial analysis",
                'key_indicators': ['Interest rate changes', 'Inflation metrics', 'Market response'],
                'time_horizons': {
                    'immediate': 'Market reaction',
                    'short_term': 'Policy transmission',
                    'medium_term': 'Economic effect',
                    'long_term': 'Structural impact'
                }
            }
        
        return custom_categories
    
    def _extract_themes_from_reasoning(self, reasoning: str) -> str:
        """Extract key themes from agent reasoning"""
        # Simple theme extraction - can be enhanced with NLP
        words = reasoning.lower().split()
        themes = [word for word in words if len(word) > 4 and word.isalpha()]
        return ', '.join(themes[:5])  # Top 5 themes
    
    def _track_resource_usage(self, operation: str):
        """Track computer resource usage for operations"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        else:
            gpu_memory = 0
            gpu_utilization = 0
        
        self.resource_usage[operation] = {
            'cpu_percent': cpu_percent,
            'memory_used_gb': memory.used / 1024**3,
            'memory_percent': memory.percent,
            'gpu_memory_gb': gpu_memory,
            'gpu_utilization': gpu_utilization,
            'timestamp': datetime.now().isoformat()
        }
    
    def track_internet_usage(self, operation: str, urls: List[str], data_size: int = 0):
        """Track internet resource usage"""
        self.internet_usage[operation] = {
            'urls_accessed': urls,
            'estimated_data_size_mb': data_size / (1024 * 1024),
            'request_count': len(urls),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive analysis report with all tracking data"""
        total_time = time.time() - self.start_time
        
        return {
            'analysis_metadata': {
                'total_analysis_time': total_time,
                'category_timings': self.category_timings,
                'resource_usage': self.resource_usage,
                'internet_usage': self.internet_usage,
                'device_info': {
                    'device': self.device,
                    'cpu_count': psutil.cpu_count(),
                    'total_memory_gb': psutil.virtual_memory().total / 1024**3,
                    'gpu_available': torch.cuda.is_available()
                }
            }
        }
