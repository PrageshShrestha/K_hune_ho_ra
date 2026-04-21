# KHUNEHO? Neural News Analysis System

A sophisticated AI-powered news analysis system using 15 specialized neural networks to provide comprehensive event analysis and actionable recommendations.

## Features

- **15 Specialized Neural Networks**: Each optimized for specific domains (financial, geopolitical, legal, etc.)
- **Sequential VRAM Loading**: Never exceeds ~4GB VRAM usage
- **Dynamic Weight Calculation**: Automatically determines which neurons are most relevant
- **Web Search Integration**: Gathers real-time context using DuckDuckGo
- **Environment-based Configuration**: All settings configurable via .env file
- **Clean Terminal Interface**: No emojis, professional output

## Quick Start

1. **Clone and Setup**:
   ```bash
   cd k_hune_ho_ra
   chmod +x run.sh
   ./run.sh
   ```

2. **Manual Setup**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python3 main.py
   ```

## Configuration

All settings are configured via `.env` file. Key sections:

### Model Configuration
- All 15 neural network models and their class labels
- Easy to swap models without code changes
- Optimized for 16GB VRAM systems

### System Settings
- VRAM limits and memory management
- Logging and debug options
- Interface preferences

### Action Mappings
- Configurable recommendations for each neuron prediction
- Domain-specific keywords for weight calculation

## Neural Network Domains

| Neuron | Purpose | Model |
|--------|---------|-------|
| Sentiment | Public mood analysis | nlptown/bert-base-multilingual-uncased-sentiment |
| Financial | Market impact | ProsusAI/finbert |
| Geopolitical | International relations | BAAI/bge-small-en-v1.5 |
| Legal | Compliance assessment | nlpaueb/legal-bert-base-uncased |
| Technological | Tech disruption | Qwen/Qwen2-0.5B-Instruct |
| Social | Social impact | cardiffnlp/twitter-roberta-base-sentiment-latest |
| Environmental | Climate impact | climatebert/distilroberta-base-climate-sentiment |
| Health | Public health | emilyalsentzer/Bio_ClinicalBERT |
| Military | Defense implications | google/gemma-2b-it |
| Economic | Economic effects | FinGPT/fingpt-forecaster_dow30 |
| Cultural | Cultural impact | xlm-roberta-base |
| Ethical | Ethical considerations | microsoft/deberta-v3-base |
| Strategic | Strategic positioning | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Historical | Historical context | sentence-transformers/all-MiniLM-L6-v2 |
| Predictive | Future outlook | distilbert-base-uncased |

## Usage Examples

```
> Federal Reserve raises interest rates by 0.25%

[KHUNEHO?] Processing: Federal Reserve raises interest rates by 0.25%...
[1/3] Gathering web intelligence...
[2/3] Activating neural cluster...
    - sentiment... done (conf: 0.89)
    - financial... done (conf: 0.94)
    - geopolitical... done (conf: 0.67)
    ...
[3/3] Synthesizing final verdict...

------ ANALYSIS REPORT ------

Event: Federal Reserve raises interest rates by 0.25%
Time: 2024-01-15T10:30:00

Top Influencing Neurons:
  - financial: weight=0.342, prediction=positive, confidence=0.940
  - economic: weight=0.287, prediction=growth, confidence=0.856
  - sentiment: weight=0.198, prediction=positive, confidence=0.890

Recommended Course of Action:
  * [FINANCIAL] Consider increasing exposure
  * [ECONOMIC] Prepare implementation timeline

System Information:
  Device: cuda
  VRAM Usage: 3.2 GB
  Web Sources: 12
```

## Architecture

```
k_hune_ho_ra/
|-- .env                    # Configuration file
|-- main.py                 # Entry point
|-- run.sh                  # Setup and run script
|-- requirements.txt        # Dependencies
|-- src/
|   |-- core/
|   |   |-- config.py       # Configuration loader
|   |   |-- vram_manager.py # Memory management
|   |   |-- weight_calculator.py # Dynamic weighting
|   |   |-- orchestrator.py # Main coordinator
|   |-- neurons/
|   |   |-- base_neuron.py  # Base neuron class
|   |   |-- __init__.py     # 15 specialized neurons
|   |-- tools/
|   |   |-- web_search.py   # DuckDuckGo integration
|   |-- conversation/
|   |   |-- session.py      # Terminal interface
```

## Memory Management

The system uses sequential model loading to minimize VRAM usage:

1. Load one model at a time
2. Process input
3. Unload model
4. Load next model

**Peak VRAM**: ~4GB (configurable)
**Total VRAM if all loaded**: ~22GB

## Customization

### Adding New Neurons
1. Add model configuration to `.env`
2. Create neuron class in `src/neurons/__init__.py`
3. Add to `NEURON_REGISTRY`
4. Configure keywords and actions

### Modifying Actions
Edit the `ACTION_*` variables in `.env`:
```
ACTION_FINANCIAL_POSITIVE=Consider increasing exposure
ACTION_LEGAL_VIOLATION=Conduct compliance audit immediately
```

### Changing Models
Update model paths in `.env`:
```
MODEL_SENTIMENT=your-custom-model-name
LABELS_SENTIMENT=negative,neutral,positive
```

## Commands

- `help` or `?` - Show help
- `status` or `info` - Show system status
- `exit` or `quit` - Exit program

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ VRAM (for optimal performance)
- Internet connection (for web search)

## License

MIT License - feel free to modify and distribute.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request
