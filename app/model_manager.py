import gc
import torch
import logging
from typing import Dict, Any, Optional
from ctransformers import AutoModelForCausalLM
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages sequential loading and unloading of LLM models to respect GPU memory constraints."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.current_model = None
        self.current_model_name = None
        self.models = {
            "qwen2": {
                "path": self.models_dir / "qwen2-7b-instruct-q4_k_m.gguf",
                "neurons": ["sentiment", "social", "cultural", "ethical", "historical"]
            },
            "mistral": {
                "path": self.models_dir / "mistral-7b-instruct-v0.3-q4_k_m.gguf", 
                "neurons": ["geopolitical", "strategic", "military", "technological"]
            },
            "deepseek": {
                "path": self.models_dir / "deepseek-llm-7b-chat-q4_k_m.gguf",
                "neurons": ["predictive", "future_reasoning"]
            }
        }
        
        # Fallback neurons for Qwen2 if needed
        self.fallback_neurons = ["economic", "health", "environmental", "legal", "financial"]
        
    def load_model(self, model_name: str) -> bool:
        """Load a specific model into GPU memory."""
        if self.current_model_name == model_name:
            logger.info(f"Model {model_name} already loaded")
            return True
            
        # Unload current model if any
        self.unload_model()
        
        model_config = self.models.get(model_name)
        if not model_config:
            logger.error(f"Unknown model: {model_name}")
            return False
            
        if not model_config["path"].exists():
            logger.error(f"Model file not found: {model_config['path']}")
            return False
            
        try:
            logger.info(f"Loading model: {model_name}")
            self.current_model = AutoModelForCausalLM.from_pretrained(
                str(model_config["path"]),
                model_type="llama",
                gpu_layers=33,  # Use GPU layers for 16GB VRAM
                temperature=0.7,
                max_new_tokens=1024,
                context_length=4096
            )
            self.current_model_name = model_name
            logger.info(f"Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def unload_model(self):
        """Unload current model from GPU memory."""
        if self.current_model:
            logger.info(f"Unloading model: {self.current_model_name}")
            del self.current_model
            self.current_model = None
            self.current_model_name = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            logger.info("Model unloaded and GPU cache cleared")
    
    def generate_response(self, prompt: str, model_name: Optional[str] = None) -> str:
        """Generate response using specified or current model."""
        if model_name and model_name != self.current_model_name:
            if not self.load_model(model_name):
                return "Error: Failed to load model"
        
        if not self.current_model:
            return "Error: No model loaded"
            
        try:
            response = self.current_model(prompt)
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: Generation failed - {e}"
    
    def get_model_for_neuron(self, neuron: str) -> str:
        """Get the appropriate model for a specific neuron."""
        for model_name, config in self.models.items():
            if neuron in config["neurons"]:
                return model_name
        
        # Fallback to qwen2 for remaining neurons
        return "qwen2"
    
    def process_neuron(self, neuron: str, prompt: str) -> str:
        """Process a specific neuron using the appropriate model."""
        model_name = self.get_model_for_neuron(neuron)
        return self.generate_response(prompt, model_name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()
