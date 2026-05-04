import json
import logging
from typing import Dict, Any, List, Set
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class AggregationEngine:
    """Aggregates and synthesizes outputs from multiple neuron analyses."""
    
    def __init__(self):
        self.domain_emojis = {
            "economic": "📊",
            "geopolitical": "🌍", 
            "technological": "⚙️",
            "ethical": "⚖️",
            "predictive": "🔮",
            "financial": "💰",
            "social": "👥",
            "environmental": "🌱",
            "health": "🏥",
            "military": "⚔️",
            "cultural": "🎭",
            "sentiment": "😊",
            "legal": "⚖️",
            "strategic": "🎯",
            "historical": "📚"
        }
    
    def aggregate_results(self, neuron_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all neurons into a comprehensive analysis."""
        
        # Group by domain
        domain_groups = self._group_by_domain(neuron_results)
        
        # Detect contradictions
        contradictions = self._detect_contradictions(neuron_results)
        
        # Calculate weighted insights
        weighted_insights = self._calculate_weighted_insights(neuron_results)
        
        # Extract risks and opportunities
        all_risks = self._extract_risks(neuron_results)
        all_opportunities = self._extract_opportunities(neuron_results)
        
        # Generate overall confidence
        overall_confidence = self._calculate_overall_confidence(neuron_results)
        
        # Create final response
        final_response = {
            "header": "🧠 Deep Analysis",
            "domains": {},
            "contradictions": contradictions,
            "risks": all_risks,
            "opportunities": all_opportunities,
            "overall_confidence": overall_confidence,
            "summary": self._generate_summary(neuron_results, weighted_insights)
        }
        
        # Format domain analyses
        for domain, results in domain_groups.items():
            emoji = self.domain_emojis.get(domain, "🔍")
            domain_analysis = self._format_domain_analysis(domain, results, emoji)
            final_response["domains"][domain] = domain_analysis
        
        return final_response
    
    def _group_by_domain(self, neuron_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group neuron results by domain."""
        domains = defaultdict(list)
        
        for result in neuron_results:
            neuron = result.get("neuron", "")
            domains[neuron].append(result)
        
        return dict(domains)
    
    def _detect_contradictions(self, neuron_results: List[Dict[str, Any]]) -> List[str]:
        """Detect contradictions between different neuron analyses."""
        contradictions = []
        
        # Extract key themes and sentiments
        positive_themes = []
        negative_themes = []
        
        for result in neuron_results:
            summary = result.get("summary", "").lower()
            key_points = result.get("key_points", [])
            
            # Simple sentiment detection
            positive_words = ["opportunity", "growth", "positive", "benefit", "advantage", "success"]
            negative_words = ["risk", "threat", "challenge", "negative", "danger", "problem", "decline"]
            
            for word in positive_words:
                if word in summary:
                    positive_themes.append(f"{result.get('neuron_name', '')}: {word}")
            
            for word in negative_words:
                if word in summary:
                    negative_themes.append(f"{result.get('neuron_name', '')}: {word}")
        
        # Check for direct contradictions
        for pos in positive_themes:
            for neg in negative_themes:
                if pos.split(":")[0] == neg.split(":")[0]:  # Same domain
                    contradictions.append(f"Contradictory signals in {pos.split(':')[0]}: {pos} vs {neg}")
        
        return contradictions[:5]  # Limit to top 5 contradictions
    
    def _calculate_weighted_insights(self, neuron_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate weighted insights based on confidence scores."""
        insights = {}
        
        for result in neuron_results:
            neuron = result.get("neuron", "")
            confidence = result.get("confidence", 0.0)
            insights[neuron] = confidence
        
        return insights
    
    def _extract_risks(self, neuron_results: List[Dict[str, Any]]) -> List[str]:
        """Extract and rank risks from all neuron analyses."""
        all_risks = []
        
        for result in neuron_results:
            risks = result.get("risks", [])
            confidence = result.get("confidence", 0.0)
            
            for risk in risks:
                # Weight risk by confidence
                weighted_risk = f"{risk} (confidence: {confidence:.2f})"
                all_risks.append(weighted_risk)
        
        # Remove duplicates and rank
        unique_risks = list(set(all_risks))
        return unique_risks[:10]  # Top 10 risks
    
    def _extract_opportunities(self, neuron_results: List[Dict[str, Any]]) -> List[str]:
        """Extract and rank opportunities from all neuron analyses."""
        all_opportunities = []
        
        for result in neuron_results:
            opportunities = result.get("opportunities", [])
            confidence = result.get("confidence", 0.0)
            
            for opportunity in opportunities:
                # Weight opportunity by confidence
                weighted_opportunity = f"{opportunity} (confidence: {confidence:.2f})"
                all_opportunities.append(weighted_opportunity)
        
        # Remove duplicates and rank
        unique_opportunities = list(set(all_opportunities))
        return unique_opportunities[:10]  # Top 10 opportunities
    
    def _calculate_overall_confidence(self, neuron_results: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence across all analyses."""
        if not neuron_results:
            return 0.0
        
        total_confidence = sum(result.get("confidence", 0.0) for result in neuron_results)
        average_confidence = total_confidence / len(neuron_results)
        
        # Adjust for contradictions
        contradictions = self._detect_contradictions(neuron_results)
        contradiction_penalty = len(contradictions) * 0.05
        
        final_confidence = max(0.0, min(1.0, average_confidence - contradiction_penalty))
        return final_confidence
    
    def _format_domain_analysis(self, domain: str, results: List[Dict[str, Any]], emoji: str) -> Dict[str, Any]:
        """Format analysis for a specific domain."""
        if not results:
            return {"emoji": emoji, "title": f"{domain.title()} Analysis", "content": "No analysis available."}
        
        # Take the highest confidence result for this domain
        best_result = max(results, key=lambda x: x.get("confidence", 0.0))
        
        content = f"**Summary:** {best_result.get('summary', '')}\n\n"
        
        if best_result.get("key_points"):
            content += "**Key Points:**\n"
            for point in best_result.get("key_points", [])[:3]:  # Top 3 points
                content += f"• {point}\n"
            content += "\n"
        
        content += f"**Confidence:** {best_result.get('confidence', 0.0):.2f}"
        
        return {
            "emoji": emoji,
            "title": f"{domain.title()} Analysis",
            "content": content,
            "confidence": best_result.get("confidence", 0.0)
        }
    
    def _generate_summary(self, neuron_results: List[Dict[str, Any]], weighted_insights: Dict[str, float]) -> str:
        """Generate an overall summary of the analysis."""
        # Get highest confidence domains
        top_domains = sorted(weighted_insights.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Count overall sentiment
        positive_count = 0
        negative_count = 0
        
        for result in neuron_results:
            summary = result.get("summary", "").lower()
            if any(word in summary for word in ["positive", "opportunity", "growth", "benefit"]):
                positive_count += 1
            elif any(word in summary for word in ["negative", "risk", "threat", "challenge"]):
                negative_count += 1
        
        sentiment = "balanced"
        if positive_count > negative_count:
            sentiment = "generally positive"
        elif negative_count > positive_count:
            sentiment = "cautiously negative"
        
        summary = f"Multi-domain analysis reveals a {sentiment} outlook. "
        
        if top_domains:
            top_domain_names = [domain.replace("_", " ").title() for domain, _ in top_domains]
            summary += f"Key insights from {', '.join(top_domain_names)} domains. "
        
        total_risks = sum(len(result.get("risks", [])) for result in neuron_results)
        total_opportunities = sum(len(result.get("opportunities", [])) for result in neuron_results)
        
        summary += f"Analysis identified {total_risks} potential risks and {total_opportunities} opportunities."
        
        return summary
    
    def format_final_response(self, aggregated_data: Dict[str, Any], sources: List[Dict[str, Any]] = None) -> str:
        """Format the final response for user display."""
        response = f"# {aggregated_data['header']}\n\n"
        response += f"**Overall Confidence:** {aggregated_data['overall_confidence']:.2f}\n\n"
        
        response += "## Summary\n"
        response += f"{aggregated_data['summary']}\n\n"
        
        # Domain analyses
        response += "## Domain Analyses\n\n"
        for domain, analysis in aggregated_data.get("domains", {}).items():
            response += f"### {analysis['emoji']} {analysis['title']}\n"
            response += f"{analysis['content']}\n\n"
        
        # Contradictions
        if aggregated_data.get("contradictions"):
            response += "## ⚠️ Detected Contradictions\n\n"
            for contradiction in aggregated_data["contradictions"]:
                response += f"• {contradiction}\n"
            response += "\n"
        
        # Risks
        if aggregated_data.get("risks"):
            response += "## ⚠️ Risks\n\n"
            for risk in aggregated_data["risks"][:5]:  # Top 5 risks
                response += f"• {risk}\n"
            response += "\n"
        
        # Opportunities
        if aggregated_data.get("opportunities"):
            response += "## 🎯 Opportunities\n\n"
            for opportunity in aggregated_data["opportunities"][:5]:  # Top 5 opportunities
                response += f"• {opportunity}\n"
            response += "\n"
        
        # Sources
        if sources:
            response += "## 📚 Sources\n\n"
            for source in sources:
                response += f"• [{source['title']}]({source['link']}) - {source['source']}\n"
            response += "\n"
        
        return response
