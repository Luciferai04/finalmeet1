#!/usr/bin/env python3
"""
Real-Time Translator - Gradio UI Launcher
=========================================

Launcher script for the Gradio-based user interface.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import UI class
from src.ui.live_camera_enhanced_ui import LiveCameraEnhancedUI

if __name__ == "__main__":
    print("=" * 60)
    print("Real-Time Translator - Gradio Interface")
    print("=" * 60)
    print("\n[IMPORTANT] Before starting:")
    print("1. Make sure WhisperLive server is running:")
    print("   python -m whisper_live.server --port 9090")
    print("\n2. Set your Google API key:")
    print("   export GOOGLE_API_KEY='your_api_key_here'")
    print("\n3. Ensure Redis is running (optional, for advanced features):")
    print("   redis-server")
    print("\nStarting Gradio interface...\n")
    
    try:
        # Create and launch UI
        ui = LiveCameraEnhancedUI()
        interface = ui.create_interface()
        
        # Launch with appropriate settings
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Set to True if you want to share publicly
            debug=os.environ.get('DEBUG', 'false').lower() == 'true',
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
    except Exception as e:
        print(f"\nError starting application: {e}")
        sys.exit(1)
