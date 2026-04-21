"""
Configuration loader for KHUNEHO? Neural Analysis System
Loads all settings from .env file
"""
import os
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

class Config:
    """Central configuration manager"""
    
    def __init__(self):
        # Load .env file from project root
        env_path = Path(__file__).parent.parent.parent / '.env'
        load_dotenv(env_path)
        
        # System settings
        self.system_name = os.getenv('SYSTEM_NAME', 'KHUNEHO?')
        self.system_version = os.getenv('SYSTEM_VERSION', '1.0.0')
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Model configurations
        self.models = self._load_model_configs()
        
        # Web search settings
        self.search_engine = os.getenv('SEARCH_ENGINE', 'duckduckgo')
        self.max_search_results = int(os.getenv('MAX_SEARCH_RESULTS', '5'))
        self.max_news_results = int(os.getenv('MAX_NEWS_RESULTS', '8'))
        self.search_timeout = int(os.getenv('SEARCH_TIMEOUT', '30'))
        
        # Weight calculation settings
        self.weight_keyword_relevance = float(os.getenv('WEIGHT_KEYWORD_RELEVANCE', '0.5'))
        self.weight_confidence = float(os.getenv('WEIGHT_CONFIDENCE', '0.3'))
        self.weight_source_quality = float(os.getenv('WEIGHT_SOURCE_QUALITY', '0.2'))
        self.min_neuron_weight = float(os.getenv('MIN_NEURON_WEIGHT', '0.05'))
        
        # VRAM management
        self.max_vram_usage_gb = float(os.getenv('MAX_VRAM_USAGE_GB', '4.0'))
        self.unload_after_use = os.getenv('UNLOAD_AFTER_USE', 'true').lower() == 'true'
        self.memory_logging = os.getenv('MEMORY_LOGGING', 'true').lower() == 'true'
        
        # Interface settings
        self.interface_style = os.getenv('INTERFACE_STYLE', 'clean')
        self.show_progress = os.getenv('SHOW_PROGRESS', 'true').lower() == 'true'
        self.show_system_info = os.getenv('SHOW_SYSTEM_INFO', 'true').lower() == 'true'
        self.max_reasoning_length = int(os.getenv('MAX_REASONING_LENGTH', '500'))
        
        # Action mappings
        self.action_mappings = self._load_action_mappings()
        
        # Domain keywords
        self.domain_keywords = self._load_domain_keywords()
    
    def _load_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load model configurations from environment"""
        models = {}
        neuron_types = [
            'sentiment', 'financial', 'geopolitical', 'legal', 'technological',
            'social', 'environmental', 'health', 'military', 'economic',
            'cultural', 'ethical', 'strategic', 'historical', 'predictive'
        ]
        
        for neuron_type in neuron_types:
            model_key = f'MODEL_{neuron_type.upper()}'
            labels_key = f'LABELS_{neuron_type.upper()}'
            
            models[neuron_type] = {
                'model_path': os.getenv(model_key, ''),
                'class_labels': [label.strip() for label in os.getenv(labels_key, '').split(',')] if os.getenv(labels_key) else []
            }
        
        return models
    
    def _load_action_mappings(self) -> Dict[str, str]:
        """Load action mappings from environment"""
        actions = {}
        for key, value in os.environ.items():
            if key.startswith('ACTION_'):
                actions[key[7:]] = value  # Remove 'ACTION_' prefix
        return actions
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain keywords from environment"""
        keywords = {}
        for key, value in os.environ.items():
            if key.startswith('KEYWORDS_'):
                neuron_type = key[9:].lower()  # Remove 'KEYWORDS_' prefix
                keywords[neuron_type] = [kw.strip() for kw in value.split(',')]
        return keywords
    
    def get_model_config(self, neuron_type: str) -> Dict[str, Any]:
        """Get configuration for a specific neuron type"""
        return self.models.get(neuron_type, {})
    
    def get_action_for_prediction(self, neuron: str, prediction: str) -> str:
        """Get action mapping for neuron prediction"""
        key = f"{neuron.upper()}_{prediction.upper().replace(' ', '_')}"
        return self.action_mappings.get(key, "Monitor situation - gather more data before acting")
    
    def get_keywords_for_domain(self, domain: str) -> List[str]:
        """Get keywords for a specific domain"""
        return self.domain_keywords.get(domain, [])

# Global config instance
config = Config()
