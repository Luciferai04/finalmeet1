"""
Fallback RL Coordinator
=======================
This module provides a basic fallback implementation when the advanced 
RL coordinator fails, ensuring system reliability through graceful degradation.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import json

# Setup logging
logger = logging.getLogger(__name__)


class BasicTranslationModel:
    """Simple fallback translation model"""
    
    def __init__(self):
        self.available = True
        
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Basic translation - returns original text with language indication"""
        if not text:
            return ""
        
        logger.info(f"Using basic translation fallback: {source_lang} -> {target_lang}")
        
        # In a real implementation, this would use a simple translation service
        # For now, we'll return the text with a simple transformation
        return f"[{target_lang.upper()}] {text}"


class BasicRLCoordinator:
    """Basic RL coordinator with minimal functionality"""
    
    def __init__(self):
        self.translation_model = BasicTranslationModel()
        self.metrics = {
            "total_translations": 0,
            "successful_translations": 0,
            "failed_translations": 0
        }
        
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Perform basic translation"""
        try:
            self.metrics["total_translations"] += 1
            result = self.translation_model.translate(text, source_lang, target_lang)
            self.metrics["successful_translations"] += 1
            return result
        except Exception as e:
            logger.error(f"Basic translation failed: {e}")
            self.metrics["failed_translations"] += 1
            return text  # Return original text as ultimate fallback
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get basic performance metrics"""
        return self.metrics.copy()


class FallbackRLCoordinator:
    """
    Fallback coordinator that gracefully degrades when advanced features fail
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.advanced_mode = False
        self.basic_coordinator = BasicRLCoordinator()
        self.enhanced_coordinator = None
        
        # Try to initialize enhanced coordinator
        self._initialize_enhanced_mode()
        
    def load_config(self, config_path):
        """Load configuration for enabling/disabling features"""
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
            logger.info(f"Configuration loaded: {self.config}")

    def _initialize_enhanced_mode(self):
        """Try to initialize enhanced RL coordinator"""
        try:
            # Try to import and initialize enhanced coordinator
            from .enhanced_rl_coordinator import EnhancedRLCoordinator
            self.enhanced_coordinator = EnhancedRLCoordinator()
            self.advanced_mode = True
            logger.info("Enhanced RL coordinator initialized successfully")
        except Exception as e:
            logger.warning(f"Enhanced RL coordinator initialization failed: {e}")
            logger.info("Falling back to basic RL coordinator")
            self.advanced_mode = False
    
    async def translate(self, text: str, source_lang: str, target_lang: str, 
                       user_id: str = "default") -> str:
        """
        Translate text using the best available method
        """
        # Try enhanced mode first
        if self.advanced_mode and self.enhanced_coordinator:
            try:
                from .enhanced_rl_coordinator import TranslationRequest
                request = TranslationRequest(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    user_id=user_id,
                    timestamp=datetime.now(),
                    request_id=f"req_{hash(text + user_id)}"
                )
                result = await self.enhanced_coordinator.translate(request)
                return result
            except Exception as e:
                logger.error(f"Enhanced translation failed, falling back to basic: {e}")
                self.advanced_mode = False  # Disable advanced mode on failure
        
        # Fall back to basic translation
        return self.basic_coordinator.translate(text, source_lang, target_lang)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current coordinator status"""
        status = {
            "mode": "enhanced" if self.advanced_mode else "basic",
            "basic_metrics": self.basic_coordinator.get_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.advanced_mode and self.enhanced_coordinator:
            try:
                # Get enhanced metrics if available
                status["enhanced_available"] = True
            except:
                status["enhanced_available"] = False
        else:
            status["enhanced_available"] = False
            
        return status
    
    def force_basic_mode(self):
        """Force coordinator to use basic mode only"""
        self.advanced_mode = False
        logger.info("Forced to basic mode")
    
    def retry_enhanced_mode(self):
        """Try to re-enable enhanced mode"""
        self._initialize_enhanced_mode()


# Factory function for easy instantiation
def create_rl_coordinator(config: Dict[str, Any] = None) -> FallbackRLCoordinator:
    """Create an RL coordinator with fallback capabilities"""
    return FallbackRLCoordinator(config)


# Example usage
async def main():
    """Example usage of fallback coordinator"""
    coordinator = create_rl_coordinator()
    
    # Test translation
    result = await coordinator.translate(
        "Hello world", 
        "en", 
        "es", 
        "test_user"
    )
    print(f"Translation result: {result}")
    
    # Check status
    status = coordinator.get_status()
    print(f"Coordinator status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
