"""
Main Orchestrator for KHUNEHO? Neural Analysis System
Coordinates sequential model loading and analysis
"""
import asyncio
import torch
from typing import Dict, List, Any
from datetime import datetime

from .vram_manager import VRAMManager
from .weight_calculator import DynamicWeightCalculator
from .config import config
from ..neurons import NEURON_REGISTRY
from ..neurons.base_neuron import NeuronReport
from ..tools.web_search import WebSearcher

class KhunehoOrchestrator:
    """
    Main orchestrator - loads one model at a time, executes, unloads
    """
    
    def __init__(self):
        self.vram = VRAMManager()
        self.weight_calculator = DynamicWeightCalculator()
        self.searcher = WebSearcher()
        self.conversation_history = []
        self.system_name = config.system_name
        self.show_progress = config.show_progress
        
        # Pre-load all models at startup
        if self.show_progress:
            print("Pre-loading all neural models...")
        self._preload_models()
        if self.show_progress:
            print("All models loaded and ready!\n")
    
    def _preload_models(self):
        """Pre-load all models to avoid loading during analysis"""
        self.preloaded_models = {}
        
        for neuron_id, NeuronClass in NEURON_REGISTRY.items():
            if self.show_progress:
                print(f"  Loading {neuron_id}...", end=" ", flush=True)
            
            try:
                neuron = NeuronClass()
                neuron._load_model()
                self.preloaded_models[neuron_id] = neuron
                
                if self.show_progress:
                    print("✓")
                    
            except Exception as e:
                if self.show_progress:
                    print(f"✗ ({str(e)[:30]})")
                # Create a dummy neuron for failed loads
                self.preloaded_models[neuron_id] = None
    
    async def analyze(self, event_text: str) -> Dict[str, Any]:
        """
        Main analysis pipeline - sequential model loading
        """
        if self.show_progress:
            print(f"\n[{self.system_name}] Processing: {event_text[:80]}...")
        
        # Step 1: Gather web context
        if self.show_progress:
            print("[1/3] Gathering web intelligence...")
        web_context = self.searcher.search_context(event_text)
        
        # Step 2: Run each neuron using pre-loaded models
        if self.show_progress:
            print("[2/3] Running neural analysis...")
        neuron_reports = {}
        
        for neuron_id in NEURON_REGISTRY.keys():
            if self.show_progress:
                print(f"    - {neuron_id}...", end=" ", flush=True)
            
            try:
                # Use pre-loaded neuron
                neuron = self.preloaded_models.get(neuron_id)
                
                if neuron is None:
                    # Handle failed pre-load
                    dummy_report = NeuronReport(
                        neuron_id=neuron_id,
                        timestamp=datetime.now(),
                        confidence=0.0,
                        logits=[0.0],
                        predicted_class=0,
                        class_labels=[],
                        reasoning="Model failed to load during initialization"
                    )
                    neuron_reports[neuron_id] = dummy_report
                    if self.show_progress:
                        print("not loaded")
                    continue
                
                # Run analysis with pre-loaded model
                report = neuron.forward(event_text, context=str(web_context)[:1000])
                
                # Add neuron-specific web sources
                neuron_sources = self.searcher.search_for_neuron(event_text, neuron_id)
                report.web_sources = neuron_sources[:3]
                report.search_queries = [f"{event_text} {neuron_id}"]
                neuron_reports[neuron_id] = report
                
                if self.show_progress:
                    print(f"done (conf: {report.confidence:.2f})")
                    
            except Exception as e:
                if self.show_progress:
                    print(f"failed ({str(e)[:30]})")
                # Create a dummy report for failed neurons
                dummy_report = NeuronReport(
                    neuron_id=neuron_id,
                    timestamp=datetime.now(),
                    confidence=0.0,
                    logits=[0.0],
                    predicted_class=0,
                    class_labels=[],
                    reasoning=f"Analysis failed: {str(e)}"
                )
                neuron_reports[neuron_id] = dummy_report
        
        # Step 3: Compute weights and synthesize
        if self.show_progress:
            print("[3/3] Synthesizing final verdict...")
        weights = self.weight_calculator.compute_weights(event_text, neuron_reports)
        
        final_verdict = self._synthesize(event_text, neuron_reports, weights, web_context)
        
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_text,
            "verdict": final_verdict
        })
        
        return final_verdict
    
    def _synthesize(self, event: str, reports: Dict, weights: Dict, context: Dict) -> Dict:
        """
        Generate final course of action from weighted reports
        """
        # Find top 3 influential neurons
        top_neurons = self.weight_calculator.get_top_neurons(weights, 3)
        
        # Extract predictions from top neurons
        top_predictions = []
        for neuron_id, weight in top_neurons:
            report = reports[neuron_id]
            prediction = report.class_labels[report.predicted_class] if report.class_labels else "unknown"
            top_predictions.append({
                "neuron": neuron_id,
                "weight": round(weight, 3),
                "prediction": prediction,
                "confidence": round(report.confidence, 3),
                "reasoning": report.reasoning[:200]
            })
        
        # Generate course of action based on weighted combination
        course_of_action = self._generate_course_of_action(event, reports, weights)
        
        return {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "top_influencers": top_predictions,
            "all_weights": {k: round(v, 3) for k, v in weights.items()},
            "course_of_action": course_of_action,
            "web_sources_summary": {
                "total_sources": sum(len(r.web_sources) for r in reports.values()),
                "top_source": context.get("current_news", [{}])[0].get("title", "none")
            },
            "system_info": self.vram.get_memory_stats()
        }
    
    def _generate_course_of_action(self, event: str, reports: Dict, weights: Dict) -> List[str]:
        """Generate actionable recommendations"""
        actions = []
        
        # Rule-based action generation from top weighted neurons
        sorted_neurons = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for neuron_id, weight in sorted_neurons[:5]:
            if weight < config.min_neuron_weight:
                continue
                
            report = reports[neuron_id]
            pred = report.class_labels[report.predicted_class] if report.class_labels else ""
            
            # Get action from config
            action = config.get_action_for_prediction(neuron_id, pred)
            if action and action != "Monitor situation - gather more data before acting":
                actions.append(f"[{neuron_id.upper()}] {action}")
        
        if not actions:
            actions = ["Monitor situation - gather more data before acting"]
        
        return actions
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "system_name": self.system_name,
            "version": config.system_version,
            "memory_stats": self.vram.get_memory_stats(),
            "total_analyses": len(self.conversation_history),
            "available_neurons": list(NEURON_REGISTRY.keys())
        }
