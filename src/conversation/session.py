"""
Conversation Interface for KHUNEHO? Neural Analysis System
Clean terminal interface without emojis
"""
import sys
from datetime import datetime
from typing import Dict, List
import asyncio
from ..core.config import config

class ConversationSession:
    """Clean terminal interface without emojis"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.system_name = config.system_name
        self.show_system_info = config.show_system_info
    
    def run(self):
        """Main conversation loop"""
        self._print_header()
        
        while True:
            try:
                user_input = self._get_input()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    self._print_footer()
                    break
                
                if user_input.lower() in ['status', 'info']:
                    self._print_status()
                    continue
                
                if user_input.lower() in ['help', '?']:
                    self._print_help()
                    continue
                
                if not user_input.strip():
                    continue
                
                # Run analysis
                result = asyncio.run(self.orchestrator.analyze(user_input))
                
                # Display results
                self._display_result(result)
                
            except KeyboardInterrupt:
                self._print_footer()
                break
            except Exception as e:
                print(f"\nError: {e}\n")
    
    def _print_header(self):
        print("="*50)
        print("KHUNEHO? Neural Analysis System")
        print("="*50)
        print(f"Specialized Neurons: {len(self.orchestrator.get_system_status()['available_neurons'])}")
        print("Mode: Sequential VRAM Loading")
        print("Web Search: RSS Feed Parser")
        print("="*50 + "\n")
        print("="*70)
        print("\nDescribe an event for analysis. Type 'help' for commands.\n")
    
    def _get_input(self) -> str:
        print("> ", end="", flush=True)
        return sys.stdin.readline().strip()
    
    def _display_result(self, result: Dict):
        print("\n" + "-"*70)
        print("ANALYSIS REPORT")
        print("-"*70)
        
        # Event and timestamp
        print(f"\nEvent: {result.get('event', 'Unknown')}")
        print(f"Time: {result.get('timestamp', 'Unknown')}")
        
        # All neural agent reports with reasoning
        print("\nNeural Agent Analysis:")
        print("-" * 50)
        
        # Sort by weight (most influential first)
        all_reports = result.get('all_agent_reports', [])
        all_reports.sort(key=lambda x: x.get('weight', 0), reverse=True)
        
        for agent in all_reports:
            print(f"\n[{agent['neuron'].upper()}] Analysis:")
            print(f"  Prediction: {agent['prediction']}")
            print(f"  Confidence: {agent['confidence']}")
            print(f"  Weight: {agent['weight']}")
            print(f"  Dynamic Prompt: {agent.get('dynamic_prompt', 'Standard prompt')}")
            print(f"  Reasoning: {agent['reasoning']}")
            print(f"  Web Sources: {agent['web_sources']}")
        
        # Top influencers summary
        print(f"\nTop 3 Influencing Neurons:")
        for influencer in result.get('top_influencers', [])[:3]:
            print(f"  - {influencer['neuron']}: weight={influencer['weight']}, "
                  f"prediction={influencer['prediction']}, confidence={influencer['confidence']}")
        
        # Dynamic Categories Analysis
        dynamic_categories = result.get('dynamic_categories', {})
        if dynamic_categories:
            print("\n" + "="*70)
            print("AI-GENERATED DYNAMIC CATEGORIES")
            print("="*70)
            
            print(f"\nTotal Dynamic Categories Generated: {len(dynamic_categories)}")
            
            # Show top categories by relevance
            sorted_categories = sorted(
                dynamic_categories.items(), 
                key=lambda x: x[1].get('relevance_score', 0), 
                reverse=True
            )
            
            print(f"\nTop 5 Categories by Relevance:")
            for category_name, category_data in sorted_categories[:5]:
                relevance = category_data.get('relevance_score', 0)
                description = category_data.get('description', '')
                param_count = len(category_data.get('dynamic_parameters', []))
                
                print(f"  - {category_name.replace('_', ' ').title()}")
                print(f"    Relevance: {relevance:.1%}")
                print(f"    Parameters: {param_count}")
                print(f"    Description: {description[:100]}{'...' if len(description) > 100 else ''}")
                print()
            
            print("See detailed report for complete category analysis...")
        
        # Professional Research Analysis
        research = result.get('research_analysis', {})
        if research:
            print("\n" + "="*70)
            print("PROFESSIONAL RESEARCH ANALYSIS")
            print("="*70)
            
            # Executive Summary
            print(f"\nEXECUTIVE SUMMARY:")
            print(f"  {research.get('executive_summary', 'Analysis processing...')}")
            
            # Sector Impacts
            sector_impacts = research.get('sector_impacts', {})
            if sector_impacts:
                print(f"\nSECTOR IMPACT ANALYSIS:")
                for sector, impact in sector_impacts.items():
                    if isinstance(impact, dict):
                        print(f"  {sector.upper()}: {impact.get('impact_level', 'Medium')} Impact")
                        print(f"    Market Sentiment: {impact.get('market_sentiment', 'Neutral')}")
                        print(f"    Short-term: {', '.join(impact.get('short_term_effects', ['Monitoring']))}")
                        print(f"    Long-term: {', '.join(impact.get('long_term_effects', ['Analysis needed']))}")
            
            # Future Predictions
            predictions = research.get('future_predictions', {})
            if predictions:
                print(f"\nFUTURE PREDICTIONS:")
                for timeframe, prediction in predictions.items():
                    print(f"  {timeframe.replace('_', ' ').title()}: {prediction}")
            
            # Risk Assessment
            risk = research.get('risk_assessment', {})
            if risk and isinstance(risk, dict):
                print(f"\nRISK ASSESSMENT:")
                print(f"  Overall Risk Level: {risk.get('overall_risk_level', 'Medium')}")
                key_risks = risk.get('key_risks', [])
                if key_risks:
                    print(f"  Key Risks: {', '.join(key_risks)}")
                mitigation = risk.get('mitigation_strategies', [])
                if mitigation:
                    print(f"  Mitigation: {', '.join(mitigation)}")
            
            # Investment Outlook
            investment = research.get('investment_outlook', {})
            if investment and isinstance(investment, dict):
                print(f"\nINVESTMENT OUTLOOK:")
                print(f"  Market Outlook: {investment.get('market_outlook', 'Analysis in progress...')}")
                recommendations = investment.get('recommendations', [])
                if recommendations:
                    print(f"  Recommendations:")
                    for rec in recommendations:
                        print(f"    * {rec}")
                print(f"  Risk Level: {investment.get('risk_level', 'Medium')}")
            
            # Analysis Confidence
            print(f"\nAnalysis Confidence: {research.get('confidence_level', 'Medium')}")
            print(f"Data Sources: {research.get('data_sources', {}).get('news_articles_analyzed', 0)} articles analyzed")
            
        # Course of action
        print("\nRecommended Course of Action:")
        for action in result.get('course_of_action', []):
            print(f"  * {action}")
        
        # Report Information
        report_path = result.get('report_path', '')
        if report_path:
            print("\n" + "="*70)
            print("COMPREHENSIVE REPORT GENERATED")
            print("="*70)
            print(f"\nMain Report: {report_path}")
            print("Additional files generated:")
            print("  - Category-specific detailed reports")
            print("  - Resource usage analysis")
            print("  - Analysis index with all links")
        
        # Resource Usage Summary
        resource_usage = result.get('resource_usage', {})
        if resource_usage:
            metadata = resource_usage.get('analysis_metadata', {})
            total_time = metadata.get('total_analysis_time', 0)
            device_info = metadata.get('device_info', {})
            
            print(f"\nResource Usage Summary:")
            print(f"  Total Analysis Time: {total_time:.2f} seconds")
            print(f"  Device: {device_info.get('device', 'Unknown')}")
            print(f"  CPU Cores: {device_info.get('cpu_count', 'Unknown')}")
            print(f"  Total Memory: {device_info.get('total_memory_gb', 0):.1f} GB")
            print(f"  GPU Available: {device_info.get('gpu_available', False)}")
        
        # System info
        if self.show_system_info:
            system_info = result.get('system_info', {})
            print(f"\nDetailed System Information:")
            print(f"  Device: {system_info.get('device', 'Unknown')}")
            if system_info.get('device') == 'cuda':
                print(f"  VRAM Usage: {system_info.get('vram_allocated_gb', 0):.2f} GB")
                print(f"  Peak VRAM: {system_info.get('peak_vram_gb', 0):.2f} GB")
            print(f"  Web Sources: {result.get('web_sources_summary', {}).get('total_sources', 0)}")
        
        print("\n" + "-"*70 + "\n")
    
    def _print_status(self):
        status = self.orchestrator.get_system_status()
        print("\n" + "="*50)
        print("SYSTEM STATUS")
        print("="*50)
        print(f"System: {status['system_name']} v{status['version']}")
        print(f"Total Analyses: {status['total_analyses']}")
        print(f"Available Neurons: {len(status['available_neurons'])}")
        
        memory_stats = status['memory_stats']
        print(f"Device: {memory_stats['device']}")
        if memory_stats['device'] == 'cuda':
            print(f"VRAM Allocated: {memory_stats.get('vram_allocated_gb', 0):.2f} GB")
            print(f"VRAM Free: {memory_stats.get('vram_free_gb', 0):.2f} GB")
        
        print("="*50 + "\n")
    
    def _print_help(self):
        print("\n" + "="*50)
        print("HELP")
        print("="*50)
        print("Commands:")
        print("  help, ?      - Show this help")
        print("  status, info - Show system status")
        print("  exit, quit   - Exit the program")
        print("\nUsage:")
        print("  Type any event description for analysis")
        print("  Example: 'Federal Reserve raises interest rates'")
        print("="*50 + "\n")
    
    def _print_footer(self):
        print("\nSession ended.\n")
