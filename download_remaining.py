#!/usr/bin/env python3
"""
Download the remaining 2 models with better alternatives
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

def download_remaining_models():
    """Download remaining models"""
    base_dir = Path("models")
    
    # Military: Use a smaller, freely available model
    print("Downloading MILITARY model...")
    try:
        military_dir = base_dir / "military"
        military_dir.mkdir(exist_ok=True)
        
        # Use distilgpt2 as alternative (much smaller, freely available)
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        model = AutoModelForCausalLM.from_pretrained('distilgpt2')
        
        tokenizer.save_pretrained(military_dir)
        model.save_pretrained(military_dir)
        
        with open(military_dir / "model_info.txt", 'w') as f:
            f.write("Model Name: distilgpt2\n")
            f.write("Domain: military\n")
            f.write("Alternative to: meta-llama/Llama-3.1-8B-Instruct\n")
        
        print("✓ Military model downloaded successfully")
        
    except Exception as e:
        print(f"✗ Military model failed: {e}")
    
    # Ethical: Retry with deberta-v3-base
    print("Downloading ETHICAL model...")
    try:
        ethical_dir = base_dir / "ethical"
        ethical_dir.mkdir(exist_ok=True)
        
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
        model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base')
        
        tokenizer.save_pretrained(ethical_dir)
        model.save_pretrained(ethical_dir)
        
        with open(ethical_dir / "model_info.txt", 'w') as f:
            f.write("Model Name: microsoft/deberta-v3-base\n")
            f.write("Domain: ethical\n")
        
        print("✓ Ethical model downloaded successfully")
        
    except Exception as e:
        print(f"✗ Ethical model failed: {e}")
        # Try alternative
        print("Trying alternative ethical model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
            
            tokenizer.save_pretrained(ethical_dir)
            model.save_pretrained(ethical_dir)
            
            with open(ethical_dir / "model_info.txt", 'w') as f:
                f.write("Model Name: bert-base-uncased\n")
                f.write("Domain: ethical\n")
                f.write("Alternative to: microsoft/deberta-v3-base\n")
            
            print("✓ Ethical alternative model downloaded successfully")
            
        except Exception as e2:
            print(f"✗ Ethical alternative also failed: {e2}")

if __name__ == "__main__":
    download_remaining_models()
    print("Download process completed!")
