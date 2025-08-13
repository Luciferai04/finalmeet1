"""
Configuration Manager for Fallback Modes
=========================================
Centralized configuration management for enabling/disabling advanced features
and managing fallback behavior.
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration for fallback modes and advanced features"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/fallback_config.json"
        self.config = self._load_default_config()
        self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "advanced_rl": False,
            "use_prometheus": True,
            "enable_egoschema": False,
            "redis_url": "redis://localhost:6379",
            "metrics_port": 8000,
            "enable_schema_checker": True,
            "fallback_modes": {
                "rl_coordinator": "basic",
                "performance_monitor": "prometheus",
                "schema_checker": "enabled"
            },
            "logging": {
                "level": "INFO",
                "enable_file_logging": True
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    self.config.update(file_config)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.info(f"Config file not found at {self.config_path}, using defaults")
                self.save_config()  # Save defaults for future reference
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
        
        return self.config
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            # Ensure config directory exists
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logger.info(f"Configuration updated: {key} = {value}")
        return True
    
    def is_advanced_rl_enabled(self) -> bool:
        """Check if advanced RL coordinator is enabled"""
        return self.get("advanced_rl", False)
    
    def is_egoschema_enabled(self) -> bool:
        """Check if EgoSchema integration is enabled"""
        return self.get("enable_egoschema", False)
    
    def is_prometheus_enabled(self) -> bool:
        """Check if Prometheus monitoring is enabled"""
        return self.get("use_prometheus", True)
    
    def get_fallback_mode(self, component: str) -> str:
        """Get fallback mode for a specific component"""
        return self.get(f"fallback_modes.{component}", "basic")
    
    def enable_advanced_feature(self, feature: str) -> bool:
        """Enable an advanced feature"""
        if feature == "rl":
            self.set("advanced_rl", True)
        elif feature == "egoschema":
            self.set("enable_egoschema", True)
        elif feature == "prometheus":
            self.set("use_prometheus", True)
        else:
            logger.warning(f"Unknown feature: {feature}")
            return False
        
        return self.save_config()
    
    def disable_advanced_feature(self, feature: str) -> bool:
        """Disable an advanced feature"""
        if feature == "rl":
            self.set("advanced_rl", False)
        elif feature == "egoschema":
            self.set("enable_egoschema", False)
        elif feature == "prometheus":
            self.set("use_prometheus", False)
        else:
            logger.warning(f"Unknown feature: {feature}")
            return False
        
        return self.save_config()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current configuration status"""
        return {
            "config_path": self.config_path,
            "advanced_rl": self.is_advanced_rl_enabled(),
            "egoschema": self.is_egoschema_enabled(),
            "prometheus": self.is_prometheus_enabled(),
            "fallback_modes": self.get("fallback_modes", {}),
            "last_updated": os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else None
        }


# Global configuration manager instance
config_manager = ConfigManager()


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager"""
    global config_manager
    if config_path and config_path != config_manager.config_path:
        config_manager = ConfigManager(config_path)
    return config_manager


# Convenience functions
def is_advanced_rl_enabled() -> bool:
    """Check if advanced RL is enabled"""
    return config_manager.is_advanced_rl_enabled()


def is_egoschema_enabled() -> bool:
    """Check if EgoSchema is enabled"""
    return config_manager.is_egoschema_enabled()


def is_prometheus_enabled() -> bool:
    """Check if Prometheus is enabled"""
    return config_manager.is_prometheus_enabled()


def get_fallback_mode(component: str) -> str:
    """Get fallback mode for a component"""
    return config_manager.get_fallback_mode(component)


# Example usage and testing
if __name__ == "__main__":
    # Test configuration manager
    config = ConfigManager("test_config.json")
    
    print("Initial config:", config.get_status())
    
    # Test setting values
    config.enable_advanced_feature("rl")
    config.enable_advanced_feature("egoschema")
    
    print("After enabling features:", config.get_status())
    
    # Test getting specific values
    print("Advanced RL enabled:", config.is_advanced_rl_enabled())
    print("EgoSchema enabled:", config.is_egoschema_enabled())
    print("Fallback mode for RL:", config.get_fallback_mode("rl_coordinator"))
