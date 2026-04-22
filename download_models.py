#!/usr/bin/env python3
"""
Script to download all required models for KHUNEHO? Neural Analysis System
Downloads models locally to avoid runtime download issues
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers import T5ForConditionalGeneration, GPT2LMHeadModel
from transformers import DebertaV2ForSequenceClassification
import torch

# Model configurations from README.md
MODELS = {
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
        'model_name': 'BAAI/bge-small-en-v1.5',
        'model_class': AutoModel,
        'subfolder': 'geopolitical'
    },
    'legal': {
        'model_name': 'nlpaueb/legal-bert-base-uncased',
        'model_class': AutoModelForSequenceClassification,
        'subfolder': 'legal'
    },
    'technological': {
        'model_name': 'Qwen/Qwen2-0.5B-Instruct',
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
        'model_name': 'emilyalsentzer/Bio_ClinicalBERT',
        'model_class': AutoModel,
        'subfolder': 'health'
    },
    'military': {
        'model_name': 'google/gemma-2b-it',
        'model_class': AutoModelForCausalLM,
        'subfolder': 'military'
    },
    'economic': {
        'model_name': 'FinGPT/fingpt-forecaster_dow30',
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

def download_model(model_key, model_config, base_dir):
    """Download a single model and tokenizer"""
    print(f"\n{'='*50}")
    print(f"Downloading {model_key.upper()} model...")
    print(f"Model: {model_config['model_name']}")
    print(f"{'='*50}")
    
    try:
        # Create subdirectory
        model_dir = base_dir / model_config['subfolder']
        model_dir.mkdir(exist_ok=True)
        
        # Download tokenizer
        print(f"Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['model_name'],
            cache_dir=model_dir
        )
        tokenizer.save_pretrained(model_dir)
        print(f"✓ Tokenizer saved to {model_dir}")
        
        # Download model
        print(f"Downloading model...")
        model_class = model_config['model_class']
        
        # Handle special cases
        if model_key == 'technological':
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_config['model_name'],
                cache_dir=model_dir,
                torch_dtype=torch.float16  # Use float16 for smaller models
            )
        elif model_key == 'military':
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_config['model_name'],
                cache_dir=model_dir,
                torch_dtype=torch.float16
            )
        elif model_key == 'economic':
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_config['model_name'],
                cache_dir=model_dir,
                torch_dtype=torch.float16
            )
        elif model_key == 'strategic':
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_config['model_name'],
                cache_dir=model_dir,
                torch_dtype=torch.float16
            )
        else:
            model = model_class.from_pretrained(
                model_config['model_name'],
                cache_dir=model_dir
            )
        
        model.save_pretrained(model_dir)
        print(f"✓ Model saved to {model_dir}")
        
        # Save model info
        info_file = model_dir / "model_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Model Name: {model_config['model_name']}\n")
            f.write(f"Model Class: {model_class.__name__}\n")
            f.write(f"Download Date: {torch.datetime.now() if hasattr(torch, 'datetime') else 'Unknown'}\n")
        
        print(f"✓ Model info saved to {info_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {model_key}: {str(e)}")
        return False

def main():
    """Main download function"""
    print("KHUNEHO? Model Downloader")
    print("Downloading all required models locally...")
    
    # Create models directory
    base_dir = Path("models")
    base_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_count = len(MODELS)
    
    for model_key, model_config in MODELS.items():
        if download_model(model_key, model_config, base_dir):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Download Summary:")
    print(f"Successfully downloaded: {success_count}/{total_count} models")
    print(f"Models directory: {base_dir.absolute()}")
    print(f"{'='*50}")
    
    if success_count == total_count:
        print("✓ All models downloaded successfully!")
        return 0
    else:
        print("✗ Some models failed to download. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
