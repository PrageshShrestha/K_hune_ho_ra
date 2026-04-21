"""
VRAM Manager for KHUNEHO? Neural Analysis System
Sequential model loading to minimize memory usage
"""
import torch
import gc
import logging
from typing import Dict, Any, Callable, Optional
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from .config import config

logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    timestamp: datetime
    model_name: str
    vram_allocated_gb: float
    vram_reserved_gb: float
    cpu_ram_gb: float

class VRAMManager:
    """
    Sequential model loader - never loads more than one model at a time
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_history: list[MemorySnapshot] = []
        self.current_model: Optional[str] = None
        self.max_vram_gb = config.max_vram_usage_gb
        
        if self.device == "cuda":
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Total VRAM: {total_vram:.1f} GB")
            logger.info(f"Max allowed VRAM: {self.max_vram_gb:.1f} GB")
        else:
            logger.warning("CUDA not available, using CPU (slower)")
    
    @contextmanager
    def load(self, model_name: str, loader: Callable):
        """
        Context manager that loads model, yields it, then unloads
        Usage:
            with vram.load("model_name", lambda: load_model()) as model:
                result = model(input)
        """
        self._unload_current()
        
        logger.info(f"Loading {model_name}...")
        model = loader()
        self.current_model = model_name
        
        if config.memory_logging:
            self._log_memory(model_name)
        
        # Check VRAM usage
        if self.device == "cuda":
            current_vram = torch.cuda.memory_allocated() / 1e9
            if current_vram > self.max_vram_gb:
                logger.warning(f"VRAM usage {current_vram:.2f}GB exceeds limit {self.max_vram_gb:.2f}GB")
        
        try:
            yield model
        finally:
            if config.unload_after_use:
                self._unload_current()
    
    def _unload_current(self):
        if self.current_model is not None:
            logger.info(f"Unloading {self.current_model}...")
            self.current_model = None
        
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _log_memory(self, model_name: str):
        if self.device == "cuda":
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                model_name=model_name,
                vram_allocated_gb=torch.cuda.memory_allocated() / 1e9,
                vram_reserved_gb=torch.cuda.memory_reserved() / 1e9,
                cpu_ram_gb=0  # Would need psutil for CPU RAM
            )
            self.load_history.append(snapshot)
            logger.info(f"VRAM after load: {snapshot.vram_allocated_gb:.2f} GB")
    
    def get_peak_vram(self) -> float:
        if not self.load_history:
            return 0.0
        return max(s.vram_allocated_gb for s in self.load_history)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        stats = {
            "device": self.device,
            "current_model": self.current_model,
            "models_loaded": len(self.load_history)
        }
        
        if self.device == "cuda":
            stats.update({
                "vram_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "vram_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "vram_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9,
                "peak_vram_gb": self.get_peak_vram()
            })
        
        return stats
