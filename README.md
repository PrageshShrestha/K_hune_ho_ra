# Multi-Neuron AI Chat System

A comprehensive AI system that provides real-time news reasoning across 15 specialized domains, similar to ChatGPT + Perplexity hybrid.

## 🎯 Features

- **15 Specialized Neurons**: Analyzes queries across sentiment, financial, geopolitical, legal, technological, social, environmental, health, military, economic, cultural, ethical, strategic, historical, and predictive domains
- **Real-Time News Integration**: Fetches news from BBC, Reuters, Al Jazeera, and GDELT Project
- **GPU-Optimized**: Sequential model loading for 16GB VRAM constraint
- **No API Keys**: Uses only free, open-source models and news sources
- **Modern Web Interface**: Clean, responsive chat UI with real-time status updates

## ⚙️ System Architecture

### Model Strategy
- **Qwen2-7B-Instruct**: Sentiment, Social, Cultural, Ethical, Historical analysis
- **Mistral-7B-Instruct-v0.3**: Geopolitical, Strategic, Military, Technological analysis  
- **DeepSeek-LLM-7B-Chat**: Predictive and future reasoning
- **Sequential Loading**: Only one model in GPU memory at any time

### News Sources
- BBC News RSS Feed
- Reuters RSS Feed
- Al Jazeera RSS Feed
- GDELT Project (global news database)

## 🚀 Quick Start

### Prerequisites
- Linux system
- Python 3.8+
- 16GB VRAM GPU (NVIDIA recommended)
- Internet connection for model downloads

### Installation

1. **Run the automated setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Start the system:**
   ```bash
   python app/main.py
   ```

4. **Access the web interface:**
   Open your browser and go to `http://localhost:8000`

## 📁 Project Structure

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI backend
│   ├── model_manager.py     # Sequential GPU model loading
│   ├── neurons.py           # 15-neuron reasoning system
│   ├── news.py              # RSS + GDELT news retrieval
│   └── aggregator.py        # Result aggregation engine
├── models/                   # Downloaded GGUF models
├── static/
│   └── index.html          # Web chat interface
├── setup.sh                # Automated installation script
├── requirements.txt        # Python dependencies
└── README.md
```

## 🧠 Neuron Domains

| Domain | Description | Model |
|--------|-------------|-------|
| Sentiment | Emotional tone analysis | Qwen2 |
| Financial | Market impact evaluation | Qwen2 |
| Geopolitical | International relations | Mistral |
| Legal | Regulatory implications | Qwen2 |
| Technological | Tech innovation impact | Mistral |
| Social | Community impact analysis | Qwen2 |
| Environmental | Sustainability assessment | Qwen2 |
| Health | Medical implications | Qwen2 |
| Military | Defense & security | Mistral |
| Economic | Market dynamics | Qwen2 |
| Cultural | Societal values | Qwen2 |
| Ethical | Moral considerations | Qwen2 |
| Strategic | Long-term planning | Mistral |
| Historical | Context & precedent | Qwen2 |
| Predictive | Future forecasting | DeepSeek |

## 🔧 API Endpoints

### Chat
```http
POST /chat
Content-Type: application/json

{
  "query": "What are the implications of AI regulation?",
  "max_news_articles": 5
}
```

### Health Check
```http
GET /health
```

### Model Management
```http
GET /models
POST /models/{model_name}/load
POST /models/unload
```

## 🎨 Web Interface Features

- **Real-time Status**: System health and model status indicators
- **Streaming Responses**: Live analysis progress display
- **Domain Breakdown**: Organized results by analysis domain
- **Confidence Scoring**: Analysis reliability indicators
- **Source Attribution**: News article links and citations
- **Responsive Design**: Works on desktop and mobile devices

## ⚡ Performance Optimizations

- **Sequential Model Loading**: Prevents GPU memory overflow
- **News Caching**: 10-minute cache for news articles
- **Async Processing**: Non-blocking I/O operations
- **Context Limiting**: Optimized prompt lengths
- **GPU Memory Management**: Automatic cache clearing and garbage collection

## 🔍 Example Queries

- "What are the economic implications of climate change policies?"
- "How might AI regulation affect technological innovation?"
- "Analyze the geopolitical impact of recent trade agreements"
- "What are the ethical considerations of genetic engineering?"
- "Predict the future of renewable energy adoption"

## 🛠️ Development

### Running in Development Mode
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Neurons
1. Add neuron definition to `app/neurons.py`
2. Update model assignments in `ModelManager`
3. Add emoji mapping in `AggregationEngine`
4. Update frontend neuron list

### Adding News Sources
1. Add RSS feed configuration to `app/news.py`
2. Update feed parsing logic if needed
3. Test source reliability

## 📊 System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: 16GB VRAM minimum
- **RAM**: 32GB recommended
- **Storage**: 50GB free space (for models)
- **Network**: Stable internet connection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**Models not downloading:**
- Check internet connection
- Verify sufficient disk space
- Try manual download from HuggingFace

**GPU memory errors:**
- Ensure no other GPU processes are running
- Check GPU memory with `nvidia-smi`
- Restart system if needed

**News feed failures:**
- Check internet connectivity
- Verify RSS feed URLs are accessible
- Check GDELT API status

**Slow response times:**
- News fetching may take 10-20 seconds
- Model loading adds 5-10 seconds per model
- Consider caching for repeated queries

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review system logs
3. Verify all dependencies are installed
4. Ensure models are properly downloaded

---

**Built with ❤️ for advanced AI reasoning and real-time analysis**
