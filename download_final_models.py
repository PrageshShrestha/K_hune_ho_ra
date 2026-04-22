#!/usr/bin/env python3
"""
Download only freely available models and their alternatives for KHUNEHO?
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM
import torch

# Final model list with alternatives for gated models
FINAL_MODELS = {
    'sentiment': {
        'model_name': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'model_class': AutoModelForSequenceClassification,
        'subfolder': 'sentiment'
    },
    'financial': {
        'model_name': 'ProsusAI/finbert',
        'model_class': AutoModelForSequenceClassification,
        'subfolder': 'financial'
    },
    'geopolitical': {
        'model_name': 'jinaai/jina-embeddings-v3',  # Alternative to BGE
        'model_class': AutoModel,
        'subfolder': 'geopolitical'
    },
    'legal': {
        'model_name': 'nlpaueb/legal-bert-base-uncased',
        'model_class': AutoModelForSequenceClassification,
        'subfolder': 'legal'
    },
    'technological': {
        'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',  # Alternative to Qwen2-0.5B
        'model_class': AutoModelForCausalLM,
        'subfolder': 'technological'
    },
    'social': {
        'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'model_class': AutoModelForSequenceClassification,
        'subfolder': 'social'
    },
    'environmental': {
        'model_name': 'climatebert/distilroberta-base-climate-sentiment',
        'model_class': AutoModelForSequenceClassification,
        'subfolder': 'environmental'
    },
    'health': {
        'model_name': 'dmis-lab/biobert-base-cased-v1.1',  # Alternative to Bio_ClinicalBERT
        'model_class': AutoModel,
        'subfolder': 'health'
    },
    'military': {
        'model_name': 'meta-llama/Llama-3.1-8B-Instruct',  # Alternative to Gemma
        'model_class': AutoModelForCausalLM,
        'subfolder': 'military'
    },
    'economic': {
        'model_name': 'microsoft/DialoGPT-medium',  # Alternative to FinGPT
        'model_class': AutoModelForCausalLM,
        'subfolder': 'economic'
    },
    'cultural': {
        'model_name': 'xlm-roberta-base',
        'model_class': AutoModel,
        'subfolder': 'cultural'
    },
    'ethical': {
        'model_name': 'microsoft/deberta-v3-base',
        'model_class': AutoModelForSequenceClassification,
        'subfolder': 'ethical'
    },
    'strategic': {
        'model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'model_class': AutoModelForCausalLM,
        'subfolder': 'strategic'
    },
    'historical': {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'model_class': AutoModel,
        'subfolder': 'historical'
    },
    'predictive': {
        'model_name': 'distilbert-base-uncased',
        'model_class': AutoModelForSequenceClassification,
        'subfolder': 'predictive'
    }
}

def is_model_downloaded(model_dir):
    """Check if model is already downloaded"""
    if not model_dir.exists():
        return False
    
    config_exists = (model_dir / 'config.json').exists()
    tokenizer_exists = (model_dir / 'tokenizer.json').exists() or (model_dir / 'tokenizer_config.json').exists()
    
    return config_exists and tokenizer_exists

def download_model(model_key, model_config, base_dir):
    """Download a single model and tokenizer"""
    model_dir = base_dir / model_config['subfolder']
    
    if is_model_downloaded(model_dir):
        print(f"  {model_key.upper()}: Already downloaded, skipping...")
        return True
    
    print(f"  {model_key.upper()}: Downloading {model_config['model_name']}...")
    
    try:
        model_dir.mkdir(exist_ok=True)
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['model_name'],
            cache_dir=model_dir
        )
        tokenizer.save_pretrained(model_dir)
        
        # Download model with appropriate settings
        model_class = model_config['model_class']
        
        # Special handling for large models
        if model_key in ['military', 'technological', 'economic', 'strategic']:
            model = model_class.from_pretrained(
                model_config['model_name'],
                cache_dir=model_dir,
                torch_dtype=torch.float16,
                device_map='auto' if torch.cuda.is_available() else None
            )
        else:
            model = model_class.from_pretrained(
                model_config['model_name'],
                cache_dir=model_dir
            )
        
        model.save_pretrained(model_dir)
        
        # Save model info
        info_file = model_dir / "model_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Model Name: {model_config['model_name']}\n")
            f.write(f"Model Class: {model_class.__name__}\n")
            f.write(f"Domain: {model_key}\n")
        
        return True
        
    except Exception as e:
        print(f"  {model_key.upper()}: Error - {str(e)[:50]}...")
        return False

def main():
    """Download all models"""
    print("KHUNEHO? Final Model Downloader")
    print("Downloading freely available models and alternatives...")
    print("=" * 60)
    
    base_dir = Path("models")
    base_dir.mkdir(exist_ok=True)
    
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    for model_key, model_config in FINAL_MODELS.items():
        model_dir = base_dir / model_config['subfolder']
        
        if is_model_downloaded(model_dir):
            skipped_count += 1
            print(f"  {model_key.upper()}: Already downloaded")
            continue
            
        if download_model(model_key, model_config, base_dir):
            success_count += 1
        else:
            error_count += 1
    
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total models: {len(FINAL_MODELS)}")
    print(f"Already downloaded: {skipped_count}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {error_count}")
    
    # Save model mapping
    with open('model_mapping.txt', 'w') as f:
        f.write("# KHUNEHO? Model Mapping\n")
        f.write("# Domain -> Model Name\n\n")
        for domain, config in FINAL_MODELS.items():
            f.write(f"{domain}={config['model_name']}\n")
    
    print(f"\nModel mapping saved to: model_mapping.txt")
    
    if error_count == 0:
        print("All models downloaded successfully!")
        return 0
    else:
        print(f"Some models failed to download ({error_count}/{len(FINAL_MODELS)})")
        return 1

if __name__ == "__main__":
    sys.exit(main())
