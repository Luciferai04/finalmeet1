#!/usr/bin/env python3
"""
WhisperLive Server Launcher
===========================

Script to start the WhisperLive server for real-time transcription.
"""

import sys
import os
import argparse
from pathlib import Path

# Add specific paths to Python path to avoid import conflicts
sys.path.insert(0, str(Path(__file__).parent / "src" / "services" / "whisper_live"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import server class directly
from server import TranscriptionServer

def main():
    parser = argparse.ArgumentParser(description="WhisperLive Server")
    parser.add_argument("--host", default="localhost", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9090, help="Port to bind the server")
    parser.add_argument("--backend", default="faster_whisper", 
                        choices=["faster_whisper", "tensorrt", "openvino"],
                        help="Backend to use for transcription")
    parser.add_argument("--model", default="base", help="Whisper model to use")
    parser.add_argument("--single-model", action="store_true", 
                        help="Use single model mode")
    parser.add_argument("--cache-path", default="~/.cache/whisper-live/",
                        help="Cache path for models")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WhisperLive Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print("=" * 60)
    print("\nStarting server...")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        server = TranscriptionServer()
        server.run(
            host=args.host,
            port=args.port,
            backend=args.backend,
            single_model=args.single_model,
            cache_path=args.cache_path
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
