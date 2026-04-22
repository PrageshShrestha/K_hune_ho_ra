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
from .prompt_manager import PromptManager
from .research_synthesizer import ResearchSynthesizer
from .dynamic_analyzer import DynamicAnalyzer
from .report_generator import ReportGenerator
from ..neurons import NEURON_REGISTRY
from ..neurons.base_neuron import NeuronReport
from ..tools.rss_search import RSSSearcher

class KhunehoOrchestrator:
    """
    Main orchestrator - loads one model at a time, executes, unloads
    """
    
    def __init__(self):
        self.vram = VRAMManager()
        self.weight_calculator = DynamicWeightCalculator()
        self.searcher = RSSSearcher()
        self.prompt_manager = PromptManager()
        self.prompt_manager.orchestrator = self  # Set reference for using loaded models
        self.research_synthesizer = ResearchSynthesizer()
        self.research_synthesizer.orchestrator = self  # Set reference for using loaded models
        self.dynamic_analyzer = DynamicAnalyzer(self)  # AI-driven dynamic analysis
        self.report_generator = ReportGenerator()  # Comprehensive report generation
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
        print(web_context)
        
        # Step 1.5: Generate dynamic prompts for each agent
        if self.show_progress:
            print("[1.5/3] Generating AI-powered prompts...")
        dynamic_prompts = self.prompt_manager.analyze_context_and_generate_prompts(event_text, web_context)
        
        # Step 2: Run each neuron using pre-loaded models with dynamic prompts
        if self.show_progress:
            print("[2/3] Running neural analysis with dynamic prompts...")
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
                
                # Get dynamic prompt for this agent
                dynamic_prompt = dynamic_prompts.get(neuron_id, f"Analyze: {event_text}")
                
                # Run analysis with pre-loaded model and dynamic prompt
                enhanced_context = f"Dynamic Analysis Prompt: {dynamic_prompt}\n\nContext: {str(web_context)[:1000]}"
                report = neuron.forward(event_text, context=enhanced_context)
                
                # Add neuron-specific web sources
                neuron_sources = self.searcher.search_for_neuron(event_text, neuron_id)
                report.web_sources = neuron_sources[:3]
                report.search_queries = [f"{event_text} {neuron_id}"]
                report.dynamic_prompt = dynamic_prompt  # Store the prompt used
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
            print("[3/4] Computing neural weights...")
        weights = self.weight_calculator.compute_weights(event_text, neuron_reports)
        
        # Step 4: Generate AI-driven dynamic categories and parameters
        if self.show_progress:
            print("[4/5] Generating AI-driven dynamic categories...")
        dynamic_categories = self.dynamic_analyzer.generate_dynamic_categories(
            event_text, 
            list(neuron_reports.values()), 
            web_context
        )
        
        # Step 5: Generate LLM-level research synthesis
        if self.show_progress:
            print("[5/5] Generating professional research analysis...")
        research_analysis = self.research_synthesizer.generate_comprehensive_analysis(
            event_text, 
            list(neuron_reports.values()), 
            web_context
        )
        
        # Generate comprehensive report
        if self.show_progress:
            print("Generating comprehensive markdown reports...")
        resource_data = self.dynamic_analyzer.get_comprehensive_report()
        report_path = self.report_generator.generate_comprehensive_report(
            event_text, 
            dynamic_categories, 
            list(neuron_reports.values()), 
            web_context, 
            resource_data, 
            research_analysis
        )
        
        if self.show_progress:
            print(f"Comprehensive report generated: {report_path}")
        
        final_verdict = self._synthesize_dynamic(
            event_text, neuron_reports, weights, web_context, research_analysis, dynamic_categories, resource_data, report_path
        )
        print("here3")
        # Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_text,
            "verdict": final_verdict
        })
        
        return final_verdict
    
    def _synthesize_dynamic(self, event: str, reports: Dict, weights: Dict, context: Dict, research_analysis: Dict, dynamic_categories: Dict, resource_data: Dict, report_path: str) -> Dict:
        """
        Generate final course of action from weighted reports
        """
        import traceback
        # Find top 3 influential neurons
        try:
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
            
            # Include detailed reasoning from all agents
            all_agent_reports = []
            for neuron_id, report in reports.items():
                prediction = report.class_labels[report.predicted_class] if report.class_labels else "unknown"
                all_agent_reports.append({
                    "neuron": neuron_id,
                    "prediction": prediction,
                    "confidence": round(report.confidence, 3),
                    "reasoning": report.reasoning,
                    "weight": round(weights.get(neuron_id, 0.0), 3),
                    "web_sources": len(report.web_sources),
                    "dynamic_prompt": getattr(report, 'dynamic_prompt', 'Standard prompt'),
                    "prompt_length": len(getattr(report, 'dynamic_prompt', 'Standard prompt'))
                })

            return {
                "event": event,
                "timestamp": datetime.now().isoformat(),
                "top_influencers": top_predictions,
                "all_weights": {k: round(v, 3) for k, v in weights.items()},
                "all_agent_reports": all_agent_reports,
                "course_of_action": course_of_action,
                "research_analysis": research_analysis,
                "dynamic_categories": dynamic_categories,
                "resource_usage": resource_data,
                "report_path": report_path,
                "web_sources_summary": {
                    "total_sources": sum(len(r.web_sources) for r in reports.values()),
                    "top_source": context.get("current_news", [{}])[0].get("title", "none")
                },
                "system_info": self.vram.get_memory_stats()
            }
        except Exception as e:
            traceback.print_exc()
    
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
