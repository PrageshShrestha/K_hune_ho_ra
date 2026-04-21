"""
Main Entry Point for KHUNEHO? Neural Analysis System
"""
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.orchestrator import KhunehoOrchestrator
from src.conversation.session import ConversationSession
from src.core.config import config

def main():
    """Main entry point"""
    try:
        # Initialize orchestrator
        orchestrator = KhunehoOrchestrator()
        
        # Start conversation session
        session = ConversationSession(orchestrator)
        session.run()
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nFatal error: {e}")
        if config.debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
