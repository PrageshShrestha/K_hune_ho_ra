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
        print("\n" + "="*70)
        print(f"{self.system_name} Neural News Analysis System")
        print("="*70)
        print(f"Specialized Neurons: {len(self.orchestrator.get_system_status()['available_neurons'])}")
        print(f"Mode: Sequential VRAM Loading")
        print(f"Web Search: {config.search_engine.title()}")
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
        
        # Top influencers
        print("\nTop Influencing Neurons:")
        for influencer in result.get('top_influencers', []):
            print(f"  - {influencer['neuron']}: weight={influencer['weight']}, "
                  f"prediction={influencer['prediction']}, confidence={influencer['confidence']}")
        
        # Course of action
        print("\nRecommended Course of Action:")
        for action in result.get('course_of_action', []):
            print(f"  * {action}")
        
        # System info
        if self.show_system_info:
            system_info = result.get('system_info', {})
            print(f"\nSystem Information:")
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
