#!/usr/bin/env python3
"""
Check availability of models on Hugging Face and find alternatives for gated models
"""

import requests
import sys
from pathlib import Path

# Original models from README.md
ORIGINAL_MODELS = {
    'sentiment': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'financial': 'ProsusAI/finbert',
    'geopolitical': 'BAAI/bge-small-en-v1.5',
    'legal': 'nlpaueb/legal-bert-base-uncased',
    'technological': 'Qwen/Qwen2-0.5B-Instruct',
    'social': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'environmental': 'climatebert/distilroberta-base-climate-sentiment',
    'health': 'emilyalsentzer/Bio_ClinicalBERT',
    'military': 'google/gemma-2b-it',
    'economic': 'FinGPT/fingpt-forecaster_dow30',
    'cultural': 'xlm-roberta-base',
    'ethical': 'microsoft/deberta-v3-base',
    'strategic': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'historical': 'sentence-transformers/all-MiniLM-L6-v2',
    'predictive': 'distilbert-base-uncased'
}

# Alternative models for gated/unavailable ones
ALTERNATIVE_MODELS = {
    'technological': 'microsoft/DialoGPT-medium',  # Alternative to Qwen
    'military': 'microsoft/DialoGPT-medium',      # Alternative to Gemma
    'economic': 'microsoft/DialoGPT-medium',       # Alternative to FinGPT
    'geopolitical': 'sentence-transformers/all-MiniLM-L6-v2',  # Alternative to BGE
    'health': 'dmis-lab/biobert-base-cased-v1.1',  # Alternative to Bio_ClinicalBERT
}

def check_model_availability(model_name):
    """Check if a model is available on Hugging Face"""
    try:
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            model_data = response.json()
            # Check if model is gated
            if model_data.get('gated', False):
                return 'gated', model_data
            return 'available', model_data
        elif response.status_code == 404:
            return 'not_found', None
        else:
            return 'error', None
            
    except Exception as e:
        return 'error', None

def get_model_info(model_data):
    """Extract useful information from model data"""
    if not model_data:
        return "No data available"
    
    info = []
    info.append(f"Downloads: {model_data.get('downloads', 'N/A')}")
    info.append(f"Likes: {model_data.get('likes', 'N/A')}")
    info.append(f"Tags: {', '.join(model_data.get('tags', [])[:5])}")  # First 5 tags
    return '\n'.join(info)

def main():
    """Check all models and suggest alternatives"""
    print("KHUNEHO? Model Availability Checker")
    print("=" * 60)
    
    available_models = {}
    gated_models = {}
    not_found_models = {}
    error_models = {}
    
    # Check original models
    for domain, model_name in ORIGINAL_MODELS.items():
        print(f"\nChecking {domain.upper()}: {model_name}")
        status, data = check_model_availability(model_name)
        
        if status == 'available':
            print(f"  Status: AVAILABLE")
            print(f"  Info: {get_model_info(data)}")
            available_models[domain] = model_name
        elif status == 'gated':
            print(f"  Status: GATED (requires authentication)")
            print(f"  Info: {get_model_info(data)}")
            gated_models[domain] = model_name
        elif status == 'not_found':
            print(f"  Status: NOT FOUND")
            not_found_models[domain] = model_name
        else:
            print(f"  Status: ERROR (could not check)")
            error_models[domain] = model_name
    
    # Suggest alternatives for gated/unavailable models
    print(f"\n{'='*60}")
    print("ALTERNATIVE MODELS SUGGESTIONS")
    print("=" * 60)
    
    final_models = {}
    
    for domain, model_name in ORIGINAL_MODELS.items():
        if domain in available_models:
            final_models[domain] = model_name
            print(f"{domain.upper()}: {model_name} (ORIGINAL - Available)")
        else:
            if domain in ALTERNATIVE_MODELS:
                alt_model = ALTERNATIVE_MODELS[domain]
                final_models[domain] = alt_model
                print(f"{domain.upper()}: {model_name} -> {alt_model} (ALTERNATIVE)")
            else:
                # Use a generic alternative
                alt_model = 'bert-base-uncased'
                final_models[domain] = alt_model
                print(f"{domain.upper()}: {model_name} -> {alt_model} (GENERIC ALTERNATIVE)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Available models: {len(available_models)}")
    print(f"Gated models: {len(gated_models)}")
    print(f"Not found models: {len(not_found_models)}")
    print(f"Error models: {len(error_models)}")
    
    # Save final model list
    with open('final_models.txt', 'w') as f:
        f.write("# Final model configurations for KHUNEHO?\n")
        f.write("# Only freely available models or their alternatives\n\n")
        for domain, model_name in final_models.items():
            f.write(f"{domain}={model_name}\n")
    
    print(f"\nFinal model list saved to: final_models.txt")
    print(f"Total models to download: {len(final_models)}")
    
    return len(available_models), len(gated_models), len(not_found_models), len(error_models)

if __name__ == "__main__":
    sys.exit(main())
