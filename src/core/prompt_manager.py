"""
AI Prompt Manager for KHUNEHO? Neural Analysis System
Analyzes Step 1 web context and generates dynamic prompts for Step 2 agents
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any
from datetime import datetime

from .config import config

class PromptManager:
    """AI-powered prompt generation based on web context analysis"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.orchestrator = None  # Will be set by orchestrator
        self.tokenizer = None
        self.model = None
        
        # Agent domain descriptions for better prompt generation
        self.agent_domains = {
            'sentiment': 'public sentiment, emotional response, social mood',
            'financial': 'economic impact, market reactions, financial consequences',
            'geopolitical': 'international relations, diplomatic implications, global politics',
            'legal': 'legal implications, regulatory impact, compliance issues',
            'technological': 'technology impact, innovation effects, digital transformation',
            'social': 'societal impact, community response, cultural implications',
            'environmental': 'environmental impact, ecological consequences, climate effects',
            'health': 'health implications, medical impact, public health consequences',
            'military': 'military implications, defense impact, security consequences',
            'economic': 'economic trends, market impact, financial systems',
            'cultural': 'cultural impact, societal values, identity implications',
            'ethical': 'ethical considerations, moral implications, value judgments',
            'strategic': 'strategic implications, long-term planning, competitive advantage',
            'historical': 'historical context, precedent setting, historical significance',
            'predictive': 'future predictions, trend analysis, forecasting implications'
        }
        
    def _load_model(self):
        """Use existing loaded models from orchestrator"""
        if self.orchestrator and hasattr(self.orchestrator, 'preloaded_models'):
            # Use the sentiment model for prompt generation
            if 'sentiment' in self.orchestrator.preloaded_models:
                neuron = self.orchestrator.preloaded_models['sentiment']
                if neuron and hasattr(neuron, 'tokenizer') and hasattr(neuron, 'model'):
                    self.tokenizer = neuron.tokenizer
                    self.model = neuron.model
                    print(f"Using loaded sentiment model for prompt generation")
                    return
            
            # Fallback to any available model
            for model_id, neuron in self.orchestrator.preloaded_models.items():
                if neuron and hasattr(neuron, 'tokenizer') and hasattr(neuron, 'model'):
                    self.tokenizer = neuron.tokenizer
                    self.model = neuron.model
                    print(f"Using loaded {model_id} model for prompt generation")
                    return
        
        print("Warning: No suitable loaded model found for prompt generation")
        self.model = None
    
    def analyze_context_and_generate_prompts(self, event: str, web_context: Dict[str, List[Dict]]) -> Dict[str, str]:
        """Analyze web context and generate dynamic prompts for each agent"""
        self._load_model()
        
        if self.model is None:
            # Fallback to rule-based prompts if model fails
            return self._generate_fallback_prompts(event, web_context)
        
        # Analyze the web context
        context_summary = self._summarize_context(web_context)
        
        # Generate prompts for each agent
        dynamic_prompts = {}
        
        for agent_id, domain_desc in self.agent_domains.items():
            try:
                prompt = self._generate_agent_prompt(event, context_summary, agent_id, domain_desc)
                dynamic_prompts[agent_id] = prompt
            except Exception as e:
                # Fallback if generation fails
                dynamic_prompts[agent_id] = self._generate_fallback_prompt(event, agent_id, domain_desc)
        
        return dynamic_prompts
    
    def _summarize_context(self, web_context: Dict[str, List[Dict]]) -> str:
        """Summarize the web context for analysis"""
        summary_parts = []
        
        for category, articles in web_context.items():
            if articles:
                titles = [article.get('title', '') for article in articles[:3]]
                summary_parts.append(f"{category.upper()}: {' | '.join(titles)}")
        
        return " | ".join(summary_parts)
    
    def _generate_agent_prompt(self, event: str, context_summary: str, agent_id: str, domain_desc: str) -> str:
        """Generate a dynamic prompt for a specific agent using rule-based approach"""
        
        # Since we're using classification models, use rule-based prompt generation
        try:
            # Extract key themes from event and context
            event_lower = event.lower()
            context_lower = context_summary.lower()
            
            # Generate context-aware prompt based on agent domain
            if agent_id == 'sentiment':
                prompt = f"Analyze the emotional tone and public sentiment surrounding: {event}. Consider the social mood and emotional response patterns evident in current events."
            elif agent_id == 'financial':
                prompt = f"Evaluate the financial implications and economic impact of: {event}. Focus on market reactions, investment considerations, and economic consequences."
            elif agent_id == 'geopolitical':
                prompt = f"Assess the geopolitical significance and international relations impact of: {event}. Consider diplomatic implications and global political consequences."
            elif agent_id == 'legal':
                prompt = f"Examine the legal implications and regulatory considerations of: {event}. Focus on compliance requirements and potential legal challenges."
            elif agent_id == 'technological':
                prompt = f"Analyze the technological impact and innovation implications of: {event}. Consider digital transformation and technology sector effects."
            elif agent_id == 'social':
                prompt = f"Evaluate the societal impact and community response to: {event}. Focus on social dynamics and cultural implications."
            elif agent_id == 'environmental':
                prompt = f"Assess the environmental impact and ecological consequences of: {event}. Consider climate effects and sustainability implications."
            elif agent_id == 'health':
                prompt = f"Analyze the health implications and medical impact of: {event}. Focus on public health consequences and healthcare system effects."
            elif agent_id == 'military':
                prompt = f"Evaluate the military implications and defense impact of: {event}. Consider security consequences and strategic defense posture."
            elif agent_id == 'economic':
                prompt = f"Assess the economic trends and market impact of: {event}. Focus on economic indicators and financial system effects."
            elif agent_id == 'cultural':
                prompt = f"Analyze the cultural impact and identity implications of: {event}. Consider societal values and cultural shifts."
            elif agent_id == 'ethical':
                prompt = f"Evaluate the ethical considerations and moral implications of: {event}. Focus on value judgments and ethical frameworks."
            elif agent_id == 'strategic':
                prompt = f"Assess the strategic implications and long-term planning impact of: {event}. Consider competitive advantage and strategic positioning."
            elif agent_id == 'historical':
                prompt = f"Analyze the historical context and precedent setting of: {event}. Consider historical significance and lessons from past events."
            elif agent_id == 'predictive':
                prompt = f"Forecast future trends and predictions based on: {event}. Consider short-term and long-term implications and potential outcomes."
            else:
                prompt = f"Analyze the implications and consequences of: {event}. Focus on relevant factors and potential impacts."
            
            # Add context-specific enhancement
            if "government" in event_lower or "policy" in event_lower:
                prompt += " Pay special attention to policy and governance aspects."
            elif "market" in event_lower or "economic" in event_lower:
                prompt += " Focus on market dynamics and economic indicators."
            elif "technology" in event_lower or "innovation" in event_lower:
                prompt += " Emphasize technological disruption and innovation trends."
            
            return prompt
            
        except Exception as e:
            # Fallback if generation fails
            return self._generate_fallback_prompt(event, agent_id, domain_desc)
    
    def _generate_fallback_prompts(self, event: str, web_context: Dict[str, List[Dict]]) -> Dict[str, str]:
        """Generate rule-based prompts as fallback"""
        fallback_prompts = {}
        
        for agent_id, domain_desc in self.agent_domains.items():
            fallback_prompts[agent_id] = self._generate_fallback_prompt(event, agent_id, domain_desc)
        
        return fallback_prompts
    
    def _generate_fallback_prompt(self, event: str, agent_id: str, domain_desc: str) -> str:
        """Generate a rule-based prompt for an agent"""
        
        # Extract keywords from event
        event_words = event.lower().split()
        
        # Domain-specific prompt templates
        prompt_templates = {
            'sentiment': f"Analyze the public sentiment and emotional response to: {event}",
            'financial': f"Evaluate the financial and economic impact of: {event}",
            'geopolitical': f"Assess the geopolitical and international implications of: {event}",
            'legal': f"Examine the legal and regulatory consequences of: {event}",
            'technological': f"Analyze the technological impact and innovation effects of: {event}",
            'social': f"Evaluate the societal and community impact of: {event}",
            'environmental': f"Assess the environmental and ecological implications of: {event}",
            'health': f"Analyze the health and medical consequences of: {event}",
            'military': f"Evaluate the military and security implications of: {event}",
            'economic': f"Assess the economic trends and market impact of: {event}",
            'cultural': f"Analyze the cultural and identity implications of: {event}",
            'ethical': f"Evaluate the ethical and moral considerations of: {event}",
            'strategic': f"Assess the strategic and long-term planning implications of: {event}",
            'historical': f"Analyze the historical context and precedent setting of: {event}",
            'predictive': f"Forecast future trends and predictions based on: {event}"
        }
        
        return prompt_templates.get(agent_id, f"Analyze the implications of: {event}")
    
    def get_prompt_analysis_report(self, event: str, web_context: Dict[str, List[Dict]], prompts: Dict[str, str]) -> Dict[str, Any]:
        """Generate a report about the prompt analysis"""
        return {
            "event": event,
            "context_summary": self._summarize_context(web_context),
            "total_articles": sum(len(articles) for articles in web_context.values()),
            "prompts_generated": len(prompts),
            "prompt_analysis": {
                agent_id: {
                    "prompt": prompt,
                    "length": len(prompt),
                    "domain": self.agent_domains.get(agent_id, "Unknown")
                }
                for agent_id, prompt in prompts.items()
            },
            "generation_method": "ai_model" if self.model is not None else "rule_based",
            "timestamp": datetime.now().isoformat()
        }
