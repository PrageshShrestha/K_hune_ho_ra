"""
Base Neuron Class for KHUNEHO? Neural Analysis System
Provides foundation for all specialized neurons
"""
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..core.config import config

@dataclass
class NeuronReport:
    """Detailed report from each neuron"""
    neuron_id: str
    timestamp: datetime
    confidence: float
    logits: List[float]
    predicted_class: int
    class_labels: List[str]
    web_sources: List[Dict] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    reasoning: str = ""
    raw_data_snippets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "neuron": self.neuron_id,
            "confidence": self.confidence,
            "prediction": self.class_labels[self.predicted_class] if self.class_labels else str(self.predicted_class),
            "logits": [round(x, 3) for x in self.logits],
            "sources_used": len(self.web_sources),
            "search_queries": self.search_queries,
            "reasoning": self.reasoning[:config.max_reasoning_length]
        }

class BaseNeuron:
    """Base class for all specialized neurons"""
    
    def __init__(self, neuron_id: str, model_path: str = None, class_labels: List[str] = None):
        self.neuron_id = neuron_id
        
        # Load from config if not provided
        model_config = config.get_model_config(neuron_id)
        self.model_path = model_path or model_config.get('model_path', '')
        self.class_labels = class_labels or model_config.get('class_labels', [])
        
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        if not self.model_path:
            raise ValueError(f"No model path configured for neuron {neuron_id}")
    
    def _load_model(self):
        """Lazy loading - called by VRAM manager"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
    
    def _unload_model(self):
        """Free memory"""
        self.tokenizer = None
        self.model = None
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def forward(self, text: str, context: str = "") -> NeuronReport:
        """
        Process input and return report
        To be overridden by subclasses
        """
        full_text = f"{context}\n{text}" if context else text
        
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        outputs = self.model(**inputs)
        logits = outputs.logits[0].cpu().tolist()
        probs = torch.softmax(torch.tensor(logits), dim=-1)
        confidence = max(probs).item()
        predicted_class = torch.argmax(torch.tensor(logits)).item()
        
        return NeuronReport(
            neuron_id=self.neuron_id,
            timestamp=datetime.now(),
            confidence=confidence,
            logits=logits,
            predicted_class=predicted_class,
            class_labels=self.class_labels,
            reasoning=self._generate_reasoning(logits, predicted_class)
        )
    
    def _generate_reasoning(self, logits: List[float], predicted_class: int) -> str:
        """Generate human-readable reasoning from logits"""
        probs = torch.softmax(torch.tensor(logits), dim=-1).tolist()
        return f"Class distribution: {dict(zip(self.class_labels, [round(p, 3) for p in probs]))}"
    
    def get_action_for_prediction(self, prediction: str) -> str:
        """Get action mapping for prediction"""
        return config.get_action_for_prediction(self.neuron_id, prediction)
