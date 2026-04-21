"""
All 15 Specialized Neurons for KHUNEHO? Neural Analysis System
"""
from .base_neuron import BaseNeuron, NeuronReport

class SentimentNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="sentiment")
    
    def forward(self, text: str, context: str = "") -> NeuronReport:
        report = super().forward(text, context)
        # Map star rating to sentiment
        star_to_sentiment = {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}
        prediction = star_to_sentiment.get(report.predicted_class, 'unknown')
        report.reasoning = f"Sentiment: {prediction} (confidence: {report.confidence:.3f})"
        return report

class FinancialNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="financial")

class GeopoliticalNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="geopolitical")

class LegalNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="legal")

class TechnologicalNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="technological")

class SocialNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="social")

class EnvironmentalNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="environmental")

class HealthNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="health")

class MilitaryNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="military")

class EconomicNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="economic")

class CulturalNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="cultural")

class EthicalNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="ethical")

class StrategicNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="strategic")

class HistoricalNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="historical")

class PredictiveNeuron(BaseNeuron):
    def __init__(self):
        super().__init__(neuron_id="predictive")

# Registry
NEURON_REGISTRY = {
    "sentiment": SentimentNeuron,
    "financial": FinancialNeuron,
    "geopolitical": GeopoliticalNeuron,
    "legal": LegalNeuron,
    "technological": TechnologicalNeuron,
    "social": SocialNeuron,
    "environmental": EnvironmentalNeuron,
    "health": HealthNeuron,
    "military": MilitaryNeuron,
    "economic": EconomicNeuron,
    "cultural": CulturalNeuron,
    "ethical": EthicalNeuron,
    "strategic": StrategicNeuron,
    "historical": HistoricalNeuron,
    "predictive": PredictiveNeuron,
}
