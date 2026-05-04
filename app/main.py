import asyncio
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from .model_manager import ModelManager
from .news import NewsRetriever
from .neurons import NeuronSystem
from .aggregator import AggregationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Neuron AI Chat System",
    description="Real-time news reasoning with 15 specialized neurons",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global components
model_manager = None
news_retriever = None
neuron_system = None
aggregator = None

class ChatRequest(BaseModel):
    query: str
    max_news_articles: int = 5

class ChatResponse(BaseModel):
    response: str
    confidence: float
    processing_time: float
    sources: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    components_ready: Dict[str, bool]

async def initialize_components():
    """Initialize all system components."""
    global model_manager, news_retriever, neuron_system, aggregator
    
    try:
        logger.info("Initializing system components...")
        
        # Initialize components
        model_manager = ModelManager()
        news_retriever = NewsRetriever()
        neuron_system = NeuronSystem(model_manager)
        aggregator = AggregationEngine()
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    success = await initialize_components()
    if not success:
        logger.error("System initialization failed")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main chat interface."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Multi-Neuron AI Chat System</h1><p>Frontend not found. Please check static/index.html</p>")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health status."""
    components = {
        "model_manager": model_manager is not None,
        "news_retriever": news_retriever is not None,
        "neuron_system": neuron_system is not None,
        "aggregator": aggregator is not None
    }
    
    all_ready = all(components.values())
    status = "healthy" if all_ready else "unhealthy"
    
    return HealthResponse(
        status=status,
        models_loaded=model_manager is not None,
        components_ready=components
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Process user query through the multi-neuron reasoning system."""
    
    if not all([model_manager, news_retriever, neuron_system, aggregator]):
        raise HTTPException(status_code=503, detail="System components not initialized")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Step 1: Fetch relevant news
        logger.info("Fetching news...")
        news_articles = await news_retriever.get_news(request.query, request.max_news_articles)
        news_context = news_retriever.format_news_context(news_articles)
        
        # Step 2: Process all neurons sequentially
        logger.info("Processing neurons...")
        neuron_results = await neuron_system.process_all_neurons(request.query, news_context)
        
        # Step 3: Aggregate results
        logger.info("Aggregating results...")
        aggregated_data = aggregator.aggregate_results(neuron_results)
        
        # Step 4: Format final response
        final_response = aggregator.format_final_response(aggregated_data, news_articles)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        return ChatResponse(
            response=final_response,
            confidence=aggregated_data["overall_confidence"],
            processing_time=processing_time,
            sources=[{
                "title": article["title"],
                "link": article["link"],
                "source": article["source"]
            } for article in news_articles]
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models and their status."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    models_info = {}
    for model_name, config in model_manager.models.items():
        models_info[model_name] = {
            "path": str(config["path"]),
            "neurons": config["neurons"],
            "loaded": model_manager.current_model_name == model_name,
            "file_exists": config["path"].exists()
        }
    
    return {"models": models_info}

@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a specific model."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    success = model_manager.load_model(model_name)
    if success:
        return {"message": f"Model {model_name} loaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}")

@app.post("/models/unload")
async def unload_model():
    """Unload current model."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    model_manager.unload_model()
    return {"message": "Model unloaded successfully"}

@app.get("/neurons")
async def list_neurons():
    """List all available neurons."""
    if not neuron_system:
        raise HTTPException(status_code=503, detail="Neuron system not initialized")
    
    return {"neurons": neuron_system.neurons}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
