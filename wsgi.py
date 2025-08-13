"""
WSGI Config
===========

WSGI configuration for running the application under a WSGI server.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.app_factory import create_app
from src.core.config import Config

# Get configuration from environment
config_name = os.environ.get('FLASK_ENV', 'production')

# Create application instance
app = create_app(config_name)

"""
This `app` object is used by any WSGI server configured to use this file.
e.g. `gunicorn --config gunicorn.conf.py wsgi:app`
"""
