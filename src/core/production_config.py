"""
Production configuration management system
"""
import os
import secrets
from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ProductionConfig(BaseSettings):
    """Production configuration with validation and security"""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    
    # Application
    app_name: str = Field(default="Live Camera Translator")
    app_version: str = Field(default="1.0.0")
    app_host: str = Field(default="127.0.0.1")
    app_port: int = Field(default=7860)
    
    # Security
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    api_key_encryption_key: Optional[str] = Field(default=None)
    max_request_size: int = Field(default=16 * 1024 * 1024)  # 16MB
    
    # API Keys (secured)
    google_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    
    # Redis Configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)
    redis_ssl: bool = Field(default=False)
    redis_connection_pool_size: int = Field(default=10)
    
    # WhisperLive Configuration
    whisper_host: str = Field(default="localhost")
    whisper_port: int = Field(default=9090)
    whisper_model: str = Field(default="base")
    whisper_language: str = Field(default="en")
    whisper_use_vad: bool = Field(default=True)
    
    # Resource Management
    max_queue_size: int = Field(default=1000)
    max_session_duration: int = Field(default=7200)  # 2 hours
    max_concurrent_sessions: int = Field(default=10)
    memory_limit_mb: int = Field(default=2048)  # 2GB
    
    # Performance
    translation_timeout: int = Field(default=30)
    transcription_timeout: int = Field(default=15)
    max_translation_length: int = Field(default=5000)
    max_file_size_mb: int = Field(default=100)
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: str = Field(default="json")
    log_file: Optional[str] = Field(default=None)
    log_rotation: str = Field(default="1 day")
    log_retention: str = Field(default="30 days")
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=8000)
    health_check_interval: int = Field(default=30)
    
    # Database/Storage
    data_dir: str = Field(default="./data")
    transcript_dir: str = Field(default="./transcripts")
    schema_dir: str = Field(default="./schemas")
    reports_dir: str = Field(default="./reports")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_burst: int = Field(default=10)
    
    @validator('google_api_key', 'openai_api_key', pre=True)
    def validate_api_keys(cls, v):
        """Validate API keys without exposing them in logs"""
        if v and len(v) < 10:
            raise ValueError("API key appears to be invalid (too short)")
        return v
    
    @validator('environment')
    def validate_environment_settings(cls, v, values):
        """Validate environment-specific settings"""
        if v == Environment.PRODUCTION:
            if not values.get('secret_key') or len(values.get('secret_key', '')) < 32:
                raise ValueError("Production environment requires strong secret key")
        return v
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        protocol = "rediss" if self.redis_ssl else "redis"
        return f"{protocol}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def get_directories(self) -> List[str]:
        """Get all data directories that need to be created"""
        return [
            self.data_dir,
            self.transcript_dir,
            self.schema_dir,
            self.reports_dir,
        ]
    
    def is_api_key_configured(self, service: str) -> bool:
        """Check if API key is configured for a service without exposing it"""
        if service.lower() == "google":
            return bool(self.google_api_key and len(self.google_api_key) > 10)
        elif service.lower() == "openai":
            return bool(self.openai_api_key and len(self.openai_api_key) > 10)
        return False
    
    def get_safe_config(self) -> Dict[str, Any]:
        """Get configuration dict with sensitive data masked"""
        config = self.dict()
        # Mask sensitive fields
        sensitive_fields = ['google_api_key', 'openai_api_key', 'secret_key', 'redis_password']
        for field in sensitive_fields:
            if config.get(field):
                config[field] = f"***{config[field][-4:]}" if len(config[field]) > 4 else "***"
        return config
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from env without validation errors
        
        # Environment variable mapping
        fields = {
            'google_api_key': {'env': 'GOOGLE_API_KEY'},
            'openai_api_key': {'env': 'OPENAI_API_KEY'},
            'redis_password': {'env': 'REDIS_PASSWORD'},
            'secret_key': {'env': 'SECRET_KEY'},
        }

# Global configuration instance
config = ProductionConfig()

def get_config() -> ProductionConfig:
    """Get the global configuration instance"""
    return config

def create_directories():
    """Create all required directories"""
    for directory in config.get_directories():
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")
