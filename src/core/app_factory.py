"""
Application Factory
==================

Creates and configures Flask application instances with proper setup.
"""

import os
from flask import Flask
from flask_cors import CORS

from .config import Config
from ..api.routes import api_bp
from ..services.websocket_service import socketio


def create_app(config_class=None):
    """Create and configure Flask application."""
    
    # Create Flask app instance
    app = Flask(__name__)
    
    # Load configuration
    if config_class is None:
        config_name = os.environ.get('FLASK_CONFIG', 'development')
        config_class = Config.get_config(config_name)
    
    app.config.from_object(config_class)
    
    # Initialize extensions
    CORS(app, origins=["*"])
    socketio.init_app(app, cors_allowed_origins="*")
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'live-camera-translator'}
    
    @app.route('/')
    def index():
        return {
            'service': 'Live Camera Enhanced Translator',
            'version': '1.0.0',
            'status': 'running',
            'endpoints': {
                'api': '/api',
                'health': '/health',
                'websocket': '/socket.io'
            }
        }
    
    return app
