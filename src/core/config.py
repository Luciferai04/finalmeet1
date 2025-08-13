"""
Application Configuration
========================

Configuration classes for different deployment environments.
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration class."""
    
    # Core Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Google API Settings
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    
    # Redis Settings
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    
    # Session Settings
    SESSION_TIMEOUT = int(os.environ.get('SESSION_TIMEOUT', 3600))
    
    # File Upload Settings
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE', 100)) * 1024 * 1024  # MB
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', '../data/uploads')
    
    # Processing Settings
    MAX_CONCURRENT_SESSIONS = int(os.environ.get('MAX_CONCURRENT_SESSIONS', 10))
    
    # AI Model Settings
    WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base')
    TRANSLATION_MODEL = os.environ.get('TRANSLATION_MODEL', 'gemini-pro')
    
    # Logging Settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', '../data/logs/app.log')
    
    # Database Settings (if using database)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    @staticmethod
    def get_config(config_name):
        """Get configuration class by name."""
        config_map = {
            'development': DevelopmentConfig,
            'testing': TestingConfig,
            'production': ProductionConfig,
            'default': DevelopmentConfig
        }
        return config_map.get(config_name, DevelopmentConfig)


class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG = True
    TESTING = False
    
    # Development-specific settings
    LOG_LEVEL = 'DEBUG'
    MAX_CONCURRENT_SESSIONS = 5


class TestingConfig(Config):
    """Testing configuration."""
    
    DEBUG = False
    TESTING = True
    
    # Test-specific settings
    WTF_CSRF_ENABLED = False
    MAX_CONCURRENT_SESSIONS = 2
    
    # Use in-memory Redis for testing
    REDIS_URL = 'redis://localhost:6379/1'


class ProductionConfig(Config):
    """Production configuration."""
    
    DEBUG = False
    TESTING = False
    
    # Production-specific settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Security headers
    SEND_FILE_MAX_AGE = timedelta(hours=1)
    
    # Performance settings
    MAX_CONCURRENT_SESSIONS = 50
    
    # Logging
    LOG_LEVEL = 'WARNING'
