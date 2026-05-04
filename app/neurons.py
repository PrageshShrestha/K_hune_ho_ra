import json
import logging
from typing import Dict, Any, List
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

class NeuronSystem:
    """Implements 15 specialized reasoning neurons for multi-domain analysis."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.neurons = {
            "sentiment": {
                "name": "Sentiment Analysis",
                "description": "Analyzes emotional tone and sentiment patterns"
            },
            "financial": {
                "name": "Financial Analysis", 
                "description": "Evaluates financial implications and market impacts"
            },
            "geopolitical": {
                "name": "Geopolitical Analysis",
                "description": "Assesses international relations and political dynamics"
            },
            "legal": {
                "name": "Legal Analysis",
                "description": "Examines legal implications and regulatory considerations"
            },
            "technological": {
                "name": "Technological Analysis",
                "description": "Evaluates technological aspects and innovation impacts"
            },
            "social": {
                "name": "Social Analysis",
                "description": "Analyzes social implications and community impacts"
            },
            "environmental": {
                "name": "Environmental Analysis",
                "description": "Assesses environmental impacts and sustainability"
            },
            "health": {
                "name": "Health Analysis",
                "description": "Evaluates health implications and medical aspects"
            },
            "military": {
                "name": "Military Analysis",
                "description": "Analyzes defense and security implications"
            },
            "economic": {
                "name": "Economic Analysis",
                "description": "Assesses economic impacts and market dynamics"
            },
            "cultural": {
                "name": "Cultural Analysis",
                "description": "Evaluates cultural implications and societal values"
            },
            "ethical": {
                "name": "Ethical Analysis",
                "description": "Examines ethical considerations and moral implications"
            },
            "strategic": {
                "name": "Strategic Analysis",
                "description": "Assesses strategic implications and long-term planning"
            },
            "historical": {
                "name": "Historical Analysis",
                "description": "Provides historical context and precedent analysis"
            },
            "predictive": {
                "name": "Predictive Analysis",
                "description": "Forecasts potential outcomes and future trends"
            }
        }
    
    def create_neuron_prompt(self, neuron: str, query: str, news_context: str) -> str:
        """Create a specialized prompt for each neuron."""
        neuron_info = self.neurons.get(neuron, {})
        
        prompt = f"""You are a {neuron_info.get('name', neuron)} expert.

Your task is to analyze the following query using your domain expertise and the provided real-time news context.

QUERY: {query}

REAL-TIME NEWS CONTEXT:
{news_context}

ANALYSIS REQUIREMENTS:
1. Use logical reasoning and your domain expertise
2. Consider the provided news context for current relevance
3. Identify key insights, risks, and opportunities
4. Assess confidence in your analysis

Return your analysis in this exact JSON format:
{{
    "summary": "Brief summary of your analysis",
    "key_points": ["Key point 1", "Key point 2", "Key point 3"],
    "risks": ["Risk 1", "Risk 2"],
    "opportunities": ["Opportunity 1", "Opportunity 2"],
    "confidence": 0.85
}}

Ensure your response is valid JSON that can be parsed directly."""
        
        return prompt
    
    def parse_neuron_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate neuron response."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["summary", "key_points", "risks", "opportunities", "confidence"]
                for field in required_fields:
                    if field not in parsed:
                        parsed[field] = [] if field in ["key_points", "risks", "opportunities"] else ""
                        if field == "confidence":
                            parsed[field] = 0.5
                
                # Ensure confidence is between 0 and 1
                parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
                
                return parsed
            else:
                logger.warning(f"No JSON found in response for neuron")
                return self._create_fallback_response(response)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for neuron: {e}")
            return self._create_fallback_response(response)
        except Exception as e:
            logger.error(f"Error parsing neuron response: {e}")
            return self._create_fallback_response(response)
    
    def _create_fallback_response(self, response: str) -> Dict[str, Any]:
        """Create fallback response when parsing fails."""
        return {
            "summary": response[:200] + "..." if len(response) > 200 else response,
            "key_points": ["Analysis completed with limited structure"],
            "risks": ["Potential analysis limitations"],
            "opportunities": ["Further investigation recommended"],
            "confidence": 0.3
        }
    
    async def process_neuron(self, neuron: str, query: str, news_context: str) -> Dict[str, Any]:
        """Process a single neuron analysis."""
        try:
            logger.info(f"Processing neuron: {neuron}")
            
            # Create specialized prompt
            prompt = self.create_neuron_prompt(neuron, query, news_context)
            
            # Get model response
            response = self.model_manager.process_neuron(neuron, prompt)
            
            # Parse response
            parsed_response = self.parse_neuron_response(response)
            parsed_response["neuron"] = neuron
            parsed_response["neuron_name"] = self.neurons.get(neuron, {}).get("name", neuron)
            
            logger.info(f"Completed neuron: {neuron} with confidence: {parsed_response['confidence']}")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error processing neuron {neuron}: {e}")
            return {
                "neuron": neuron,
                "neuron_name": self.neurons.get(neuron, {}).get("name", neuron),
                "summary": f"Error processing analysis: {str(e)}",
                "key_points": ["Analysis failed"],
                "risks": ["System error occurred"],
                "opportunities": ["Retry analysis"],
                "confidence": 0.0
            }
    
    async def process_all_neurons(self, query: str, news_context: str) -> List[Dict[str, Any]]:
        """Process all 15 neurons sequentially."""
        results = []
        
        for neuron in self.neurons.keys():
            result = await self.process_neuron(neuron, query, news_context)
            results.append(result)
        
        logger.info(f"Processed all {len(results)} neurons")
        return results
    
    def get_neuron_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics of neuron processing."""
        total_neurons = len(results)
        successful_neurons = len([r for r in results if r["confidence"] > 0])
        avg_confidence = sum(r["confidence"] for r in results) / total_neurons if total_neurons > 0 else 0
        
        return {
            "total_neurons": total_neurons,
            "successful_neurons": successful_neurons,
            "average_confidence": avg_confidence,
            "processing_complete": successful_neurons == total_neurons
        }
