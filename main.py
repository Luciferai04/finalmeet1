#!/usr/bin/env python3
"""
Real-Time Translator - Main Application Entry Point
=================================================

Production-ready entry point for the real-time translator application.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.app_factory import create_app
from src.core.config import Config

if __name__ == "__main__":
    # Get configuration from environment
    config_name = os.environ.get('FLASK_ENV', 'development')
    
    # Create application instance
    app = create_app(config_name)
    
    # Get configuration
    config = Config.get_config(config_name)
    
    # Run application
    if config_name == 'development':
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=True,
            threaded=True
        )
    else:
        # In production, this will be handled by gunicorn/wsgi
        print("Production mode - use gunicorn or wsgi server")
        print("Example: gunicorn --config gunicorn.conf.py main:app")
