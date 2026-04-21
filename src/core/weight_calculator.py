"""
Dynamic Weight Calculator for KHUNEHO? Neural Analysis System
Calculates per-event neuron weights based on relevance and confidence
"""
import torch
import numpy as np
from typing import Dict, List, Any

from ..core.config import config

class DynamicWeightCalculator:
    """Calculate per-event neuron weights"""
    
    def __init__(self):
        self.keyword_weight = config.weight_keyword_relevance
        self.confidence_weight = config.weight_confidence
        self.source_weight = config.weight_source_quality
        self.min_weight = config.min_neuron_weight
        self.domain_keywords = config.domain_keywords
    
    def compute_weights(self, event_text: str, neuron_reports: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute dynamic weights for all neurons
        Returns dict of neuron_id -> weight (0-1)
        """
        weights = {}
        event_lower = event_text.lower()
        
        for neuron_id, report in neuron_reports.items():
            # Factor 1: Keyword relevance
            keywords = self.domain_keywords.get(neuron_id, [])
            relevance = sum(1 for kw in keywords if kw in event_lower) / max(len(keywords), 1)
            relevance = min(relevance * 2, 1.0)  # Scale up
            
            # Factor 2: Neuron confidence
            confidence = report.confidence if hasattr(report, 'confidence') else 0.5
            
            # Factor 3: Source quality
            source_quality = 0.5
            if hasattr(report, 'web_sources') and report.web_sources:
                source_quality = min(0.8, len(report.web_sources) / 10)
            
            # Combine weights
            weight = (relevance * self.keyword_weight) + (confidence * self.confidence_weight) + (source_quality * self.source_weight)
            weights[neuron_id] = weight
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        # Filter out very low weights
        weights = {k: v for k, v in weights.items() if v >= self.min_weight}
        
        # Re-normalize if we filtered some out
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def get_top_neurons(self, weights: Dict[str, float], top_n: int = 3) -> List[tuple]:
        """Get top N neurons by weight"""
        return sorted(weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def analyze_weight_distribution(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Analyze the distribution of weights"""
        if not weights:
            return {"analysis": "No weights calculated"}
        
        weight_values = list(weights.values())
        return {
            "total_neurons": len(weights),
            "max_weight": max(weight_values),
            "min_weight": min(weight_values),
            "mean_weight": np.mean(weight_values),
            "std_weight": np.std(weight_values),
            "top_heavy": max(weight_values) > 3 * np.mean(weight_values)
        }
