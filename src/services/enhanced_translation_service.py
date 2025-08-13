"""
Enhanced Translation Service
===========================

Enhanced service layer that integrates the AdvancedTranslationEngine
with backward compatibility for existing API.
"""

import uuid
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime

from .translation_service import TranslationService
from .advanced_translation_engine import AdvancedTranslationEngine, TranslationResult


class EnhancedTranslationService(TranslationService):
    """Enhanced translation service with advanced features."""
    
    def __init__(self, api_key: Optional[str] = None, use_advanced_engine: bool = True):
        """Initialize the enhanced translation service."""
        super().__init__()
        self.use_advanced_engine = use_advanced_engine
        
        if use_advanced_engine:
            try:
                self.advanced_engine = AdvancedTranslationEngine(api_key)
                self.logger.info("Advanced translation engine initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize advanced engine: {e}")
                self.use_advanced_engine = False
                self.advanced_engine = None
        else:
            self.advanced_engine = None
    
    def translate(self, text: str, target_language: str = 'Bengali', 
                  session_id: Optional[str] = None, 
                  user_context: str = "",
                  use_advanced: bool = None) -> Dict:
        """
        Enhanced translate method with advanced features.
        
        Args:
            text: Text to translate
            target_language: Target language for translation
            session_id: Optional session ID for tracking
            user_context: Additional context for translation
            use_advanced: Override to use/not use advanced engine
            
        Returns:
            Dict containing translation results with quality metrics
        """
        # Determine which engine to use
        should_use_advanced = use_advanced if use_advanced is not None else self.use_advanced_engine
        
        if should_use_advanced and self.advanced_engine:
            return self._translate_with_advanced_engine(
                text, target_language, session_id, user_context
            )
        else:
            # Fallback to basic translation service
            return super().translate(text, target_language, session_id)
    
    def _translate_with_advanced_engine(self, text: str, target_language: str, 
                                      session_id: Optional[str], 
                                      user_context: str) -> Dict:
        """Translate using the advanced engine."""
        try:
            # Use the multi-stage pipeline for best quality
            result = self.advanced_engine.multi_stage_translation_pipeline(
                text=text,
                target_language=target_language,
                session_id=session_id,
                user_context=user_context,
                use_history=True
            )
            
            # Convert to dict format for backward compatibility
            response_dict = {
                'session_id': result.session_id,
                'original_text': result.original_text,
                'translated_text': result.translated_text,
                'target_language': result.target_language,
                'latency': result.processing_time,
                'quality_score': result.quality_score,
                'domain': result.domain,
                'register': result.register,
                'model_used': result.model_used,
                'quality_metrics': result.quality_metrics,
                'timestamp': result.timestamp
            }
            
            # Track in sessions for compatibility
            if not session_id:
                session_id = str(uuid.uuid4())
                response_dict['session_id'] = session_id
            
            self._track_session(session_id, text, target_language)
            self._update_session(session_id, {
                'translated_text': result.translated_text,
                'latency': result.processing_time,
                'quality_score': result.quality_score,
                'timestamp': result.timestamp
            })
            
            return response_dict
            
        except Exception as e:
            self.logger.error(f"Advanced translation failed: {e}")
            # Fallback to basic translation
            return super().translate(text, target_language, session_id)
    
    def get_translation_stats(self) -> Dict:
        """Get comprehensive translation statistics."""
        basic_stats = {}
        advanced_stats = {}
        
        # Get basic service stats
        total_sessions = len(self.sessions)
        if total_sessions > 0:
            total_translations = sum(s['total_translations'] for s in self.sessions.values())
            avg_latency = sum(
                sum(t.get('latency', 0) for t in s['translations']) 
                for s in self.sessions.values()
            ) / max(total_translations, 1)
            
            basic_stats = {
                'total_sessions': total_sessions,
                'total_translations': total_translations,
                'average_latency': avg_latency
            }
        
        # Get advanced engine stats
        if self.advanced_engine:
            advanced_stats = self.advanced_engine.get_translation_stats()
        
        return {
            'basic_service': basic_stats,
            'advanced_engine': advanced_stats,
            'engine_in_use': 'advanced' if self.use_advanced_engine else 'basic'
        }
    
    def reset_conversation_history(self, session_id: Optional[str] = None):
        """Reset conversation history."""
        if self.advanced_engine:
            self.advanced_engine.reset_conversation_history()
        
        if session_id and session_id in self.sessions:
            del self.sessions[session_id]
    
    def export_translation_history(self) -> List[Dict]:
        """Export translation history from advanced engine."""
        if self.advanced_engine:
            return self.advanced_engine.export_translation_history()
        return []
    
    def set_engine_mode(self, use_advanced: bool):
        """Switch between basic and advanced engines."""
        if use_advanced and not self.advanced_engine:
            try:
                self.advanced_engine = AdvancedTranslationEngine()
                self.use_advanced_engine = True
                self.logger.info("Switched to advanced translation engine")
            except Exception as e:
                self.logger.error(f"Failed to initialize advanced engine: {e}")
                self.use_advanced_engine = False
        else:
            self.use_advanced_engine = use_advanced
            self.logger.info(f"Engine mode set to: {'advanced' if use_advanced else 'basic'}")
