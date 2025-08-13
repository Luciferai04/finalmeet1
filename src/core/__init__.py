"""
Core Application Module
=======================

Contains the main application factory and core initialization logic.
"""

from .app_factory import create_app
from .config import Config, DevelopmentConfig, ProductionConfig, TestingConfig

__all__ = ["create_app", "Config", "DevelopmentConfig", "ProductionConfig", "TestingConfig"]
