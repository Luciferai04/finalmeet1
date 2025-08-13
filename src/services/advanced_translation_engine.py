"""
Advanced Translation Engine
===========================

Comprehensive translation system with context awareness, quality assessment,
adaptive parameters, and post-processing improvements.
"""

import os
import re
import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

import google.generativeai as genai
import numpy as np

from .enhanced_translation_prompts import (
    EnhancedTranslationPrompts, TranslationContext, DomainType, 
    RegisterType, QualityMetrics
)


@dataclass
class TranslationResult:
    """Comprehensive translation result."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    quality_score: float
    processing_time: float
    model_used: str
    domain: str
    register: str
    context_used: bool
    quality_metrics: Dict[str, Any]
    timestamp: str
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GenerationParams:
    """Gemini generation parameters."""
    temperature: float = 0.3
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 1024
    candidate_count: int = 1


class ConversationHistory:
    """Manages conversation history for context-aware translation."""
    
    def __init__(self, max_history_length: int = 10):
        self.max_history_length = max_history_length
        self.history: List[Dict] = []
        self.domain_context: Optional[DomainType] = None
        self.register_context: Optional[RegisterType] = None
    
    def add_exchange(self, original: str, translated: str, 
                    domain: DomainType, register: RegisterType):
        """Add a translation exchange to history."""
        exchange = {
            'original': original,
            'translated': translated,
            'domain': domain.value,
            'register': register.value,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(exchange)
        
        # Maintain max length
        if len(self.history) > self.max_history_length:
            self.history.pop(0)
        
        # Update context
        self._update_context(domain, register)
    
    def _update_context(self, domain: DomainType, register: RegisterType):
        """Update contextual understanding based on recent history."""
        if len(self.history) >= 3:
            recent_domains = [h['domain'] for h in self.history[-3:]]
            recent_registers = [h['register'] for h in self.history[-3:]]
            
            # Use most common domain/register in recent history
            self.domain_context = DomainType(max(set(recent_domains), key=recent_domains.count))
            self.register_context = RegisterType(max(set(recent_registers), key=recent_registers.count))
    
    def get_context_summary(self) -> str:
        """Get a summary of recent conversation context."""
        if not self.history:
            return ""
        
        recent_exchanges = self.history[-3:]
        context_items = []
        
        for exchange in recent_exchanges:
            context_items.append(f"'{exchange['original']}' → '{exchange['translated']}'")
        
        return " | ".join(context_items)
    
    def get_consistency_terms(self) -> Dict[str, str]:
        """Extract terms that should be translated consistently."""
        term_map = {}
        
        for exchange in self.history:
            # Simple keyword extraction (can be enhanced with NLP)
            original_words = re.findall(r'\b[A-Z][a-z]+\b', exchange['original'])
            translated_words = re.findall(r'\b\w+\b', exchange['translated'])
            
            # Map proper nouns and technical terms
            for word in original_words:
                if len(word) > 3 and word not in term_map:
                    # This is a simplified mapping - in practice, would use alignment
                    term_map[word] = exchange['translated']
        
        return term_map


class ParameterOptimizer:
    """Optimizes generation parameters based on content and feedback."""
    
    def __init__(self):
        self.performance_history: List[Dict] = []
        self.domain_params: Dict[DomainType, GenerationParams] = self._initialize_domain_params()
        self.learning_rate = 0.1
    
    def _initialize_domain_params(self) -> Dict[DomainType, GenerationParams]:
        """Initialize domain-specific parameters."""
        return {
            DomainType.BUSINESS: GenerationParams(temperature=0.2, top_p=0.8, top_k=30),
            DomainType.TECHNICAL: GenerationParams(temperature=0.1, top_p=0.7, top_k=25),
            DomainType.MEDICAL: GenerationParams(temperature=0.1, top_p=0.6, top_k=20),
            DomainType.LEGAL: GenerationParams(temperature=0.1, top_p=0.6, top_k=20),
            DomainType.CONVERSATION: GenerationParams(temperature=0.4, top_p=0.9, top_k=50),
            DomainType.EDUCATION: GenerationParams(temperature=0.3, top_p=0.8, top_k=35),
            DomainType.NEWS: GenerationParams(temperature=0.2, top_p=0.8, top_k=30),
            DomainType.GENERAL: GenerationParams(temperature=0.3, top_p=0.8, top_k=40)
        }
    
    def get_optimal_params(self, domain: DomainType, 
                          text_length: int = 0) -> GenerationParams:
        """Get optimal parameters for given domain and text characteristics."""
        base_params = self.domain_params.get(domain, self.domain_params[DomainType.GENERAL])
        
        # Adjust for text length
        if text_length > 100:
            # Longer texts need more tokens
            base_params.max_output_tokens = min(2048, int(text_length * 1.5))
        elif text_length < 20:
            # Short texts can be more creative
            base_params.temperature = min(0.5, base_params.temperature + 0.1)
        
        return base_params
    
    def update_params_from_feedback(self, domain: DomainType, 
                                   quality_score: float, 
                                   params_used: GenerationParams):
        """Update parameters based on quality feedback."""
        self.performance_history.append({
            'domain': domain.value,
            'quality_score': quality_score,
            'params': asdict(params_used),
            'timestamp': datetime.now().isoformat()
        })
        
        # Simple adaptive learning
        if quality_score < 0.7:  # Poor quality
            current_params = self.domain_params[domain]
            # Decrease temperature for more consistent output
            current_params.temperature = max(0.1, current_params.temperature - self.learning_rate * 0.1)
            current_params.top_p = max(0.5, current_params.top_p - self.learning_rate * 0.1)
        elif quality_score > 0.9:  # High quality
            current_params = self.domain_params[domain]
            # Allow slightly more creativity
            current_params.temperature = min(0.5, current_params.temperature + self.learning_rate * 0.05)


class TextProcessor:
    """Handles text cleaning and post-processing."""
    
    @staticmethod
    def clean_input_text(text: str) -> str:
        """Clean and normalize input text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common formatting issues
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Fix spacing after punctuation
        
        # Normalize quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        return text
    
    def post_process_translation(self, translated_text: str, 
                               target_language: str,
                               preserve_formatting: bool = True) -> str:
        """Post-process translated text for quality and formatting."""
        if not translated_text:
            return ""
        
        # Remove common translation artifacts
        cleaned = translated_text.strip()
        
        # Remove potential model artifacts
        artifacts = [
            "Translation:", "translation:", "Here is the translation:",
            "The translation is:", "In Bengali:", "In Hindi:",
            "bengali:", "hindi:", "Translated text:"
        ]
        
        for artifact in artifacts:
            if cleaned.lower().startswith(artifact.lower()):
                cleaned = cleaned[len(artifact):].strip()
        
        # Remove extra quotes that models sometimes add
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]
        
        # Fix spacing issues
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Perform grammar check
        cleaned = self._perform_grammar_check(cleaned, target_language)
        
        # Language-specific post-processing
        if target_language.lower() in ['bengali', 'বাংলা']:
            cleaned = TextProcessor._post_process_bengali(cleaned)
        elif target_language.lower() in ['hindi', 'हिंदी']:
            cleaned = TextProcessor._post_process_hindi(cleaned)

        return cleaned.strip()

    def _perform_grammar_check(self, text: str, language: str) -> str:
        """Perform grammar checking on the translated text."""
        # This is a placeholder for actual grammar checking logic
        # For now, it's a simple pass-through
        # You may integrate a grammar checking library or API here
        return text

    @staticmethod
    def _post_process_bengali(text: str) -> str:
        """Bengali-specific post-processing."""
        # Fix common spacing issues in Bengali
        text = re.sub(r'\s+([।,])', r'\1', text)  # Remove space before Bengali punctuation
        text = re.sub(r'([।])\s*([।])', r'\1', text)  # Remove duplicate danda
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([।])\s*([^\s])', r'\1 \2', text)
        
        return text
    
    @staticmethod
    def _post_process_hindi(text: str) -> str:
        """Hindi-specific post-processing."""
        # Fix common spacing issues in Hindi
        text = re.sub(r'\s+([।,])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([।])\s*([।])', r'\1', text)  # Remove duplicate danda
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([।])\s*([^\s])', r'\1 \2', text)
        
        return text


class AdvancedTranslationEngine:
    """Advanced translation engine with all quality improvements."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the advanced translation engine."""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Initialize components
        self.prompt_manager = EnhancedTranslationPrompts()
        self.conversation_history = ConversationHistory()
        self.parameter_optimizer = ParameterOptimizer()
        self.text_processor = TextProcessor()
        self.quality_metrics = QualityMetrics()
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.available_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        
        # Performance tracking
        self.translation_stats = {
            'total_translations': 0,
            'average_quality': 0.0,
            'average_latency': 0.0,
            'model_usage': {}
        }
        
        # Translation caching and memory system
        self.translation_memory = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _get_working_model(self, preferred_model: str = None) -> Optional[genai.GenerativeModel]:
        """Get a working Gemini model with fallback."""
        models_to_try = [preferred_model] if preferred_model else self.available_models
        
        for model_name in models_to_try:
            if model_name is None:
                continue
            try:
                model = genai.GenerativeModel(model_name)
                # Test the model with a simple query
                test_response = model.generate_content("Test")
                if test_response and test_response.text:
                    self.logger.info(f"Using model: {model_name}")
                    return model
            except Exception as e:
                self.logger.warning(f"Model {model_name} not available: {str(e)}")
                continue
        
        raise RuntimeError("No working Gemini models available")
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the advanced translation engine."""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Initialize components
        self.prompt_manager = EnhancedTranslationPrompts()
        self.conversation_history = ConversationHistory()
        self.parameter_optimizer = ParameterOptimizer()
        self.text_processor = TextProcessor()
        self.quality_metrics = QualityMetrics()
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.available_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        
        # Performance tracking
        self.translation_stats = {
            'total_translations': 0,
            'average_quality': 0.0,
            'average_latency': 0.0,
            'model_usage': {}
        }
        
        # Caching
        self.translation_memory = {}
        
        self.logger = logging.getLogger(__name__)

    def _cache_translation(self, original_text: str, translated_text: str, target_language: str):
        """Cache a translated text for future reference."""
        cache_key = (original_text, target_language)
        self.translation_memory[cache_key] = translated_text

    def _get_cached_translation(self, original_text: str, target_language: str) -> Optional[str]:
        """Retrieve cached translation if available."""
        cache_key = (original_text, target_language)
        return self.translation_memory.get(cache_key)
        """
        Multi-stage translation pipeline for improved translation quality.
        
        Stages:
        1. Initial Translation
        2. Quality Assessment
        3. Iterative Improvement
        """
        # Stage 1: Initial Translation
        initial_result = self.perform_initial_translation(text, target_language, source_language, session_id, user_context, use_history)

        # Stage 2: Quality Assessment
        quality_metrics = self.quality_metrics.assess_translation_quality(initial_result.original_text, initial_result.translated_text, initial_result)
        
        # Stage 3: Iterative Improvement if necessary
        if quality_metrics['overall_quality_score'] < 0.9:
            improved_result = self.iterative_improvement(initial_result, quality_metrics)
        else:
            improved_result = initial_result

        return improved_result

    def perform_initial_translation(self, text: str, target_language: str, source_language: str, session_id: str, user_context: str, use_history: bool) -> TranslationResult:
        """Perform the initial translation using advanced context-aware techniques."""
        return asyncio.run(self.translate(text, target_language, source_language, session_id, user_context, use_history))

    def iterative_improvement(self, result: TranslationResult, quality_metrics: Dict[str, float]) -> TranslationResult:
        """Iteratively improve translation based on quality metrics."""
        # Example logic for improvement
        if quality_metrics['untranslated_terms'] > 0:
            # Adjust prompt and retry translation
            print(f"[INFO] Improving translation for untranslated terms: {quality_metrics['untranslated_terms']}")
            return self.perform_initial_translation(result.original_text, result.target_language, result.source_language, result.session_id, f"Untranslated terms: {quality_metrics['untranslated_terms']}", result.context_used)
        return result

    async def translate(self, 
                       text: str,
                       target_language: str = "Bengali",
                       source_language: str = "English",
                       session_id: Optional[str] = None,
                       user_context: str = "",
                       use_history: bool = True) -> TranslationResult:
        """
        Perform advanced translation with all quality improvements.
        
        Args:
            text: Text to translate
            target_language: Target language
            source_language: Source language
            session_id: Optional session ID for tracking
            user_context: Additional context provided by user
            use_history: Whether to use conversation history
            
        Returns:
            TranslationResult with comprehensive information
        """
        start_time = time.time()
        
        try:
            # Clean input text
            cleaned_text = self.text_processor.clean_input_text(text)
            if not cleaned_text:
                raise ValueError("Empty or invalid input text")
            
            # Detect domain and register
            detected_domain = self.prompt_manager.detect_domain(cleaned_text)
            detected_register = self.prompt_manager.detect_register(cleaned_text)
            
            # Build translation context
            context = TranslationContext(
                domain=detected_domain,
                register=detected_register,
                source_language=source_language,
                target_language=target_language,
                cultural_context=user_context,
                conversation_history=self.conversation_history.history if use_history else [],
                previous_translations=self.conversation_history.history[-5:] if use_history else []
            )
            
            # Get optimal parameters
            optimal_params = self.parameter_optimizer.get_optimal_params(
                detected_domain, len(cleaned_text)
            )
            
            # Build context-aware prompt
            prompt = self.prompt_manager.build_context_aware_prompt(cleaned_text, context)
            
            # Get working model
            model = self._get_working_model()
            model_name = model.model_name if hasattr(model, 'model_name') else 'gemini-unknown'
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=optimal_params.temperature,
                top_p=optimal_params.top_p,
                top_k=optimal_params.top_k,
                max_output_tokens=optimal_params.max_output_tokens,
                candidate_count=optimal_params.candidate_count
            )
            
            # Check if translation is cached
            cached_translation = self._get_cached_translation(cleaned_text, target_language)
            if cached_translation:
                translated_text = cached_translation
                self.logger.info("Using cached translation.")
            else:
                # Generate translation
                response = await asyncio.to_thread(
                    model.generate_content, 
                    prompt, 
                    generation_config=generation_config
                )

                if not response or not response.text:
                    raise RuntimeError("No response from translation model")

                # Post-process translation
                translated_text = self.text_processor.post_process_translation(
                    response.text, target_language
                )

                # Cache translation
                self._cache_translation(cleaned_text, translated_text, target_language)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Assess quality
            quality_metrics = self.quality_metrics.assess_translation_quality(
                cleaned_text, translated_text, context
            )
            
            # Estimate translation confidence
            confidence_score = self.estimate_translation_confidence(quality_metrics, cleaned_text)
            quality_metrics['confidence_score'] = confidence_score
            
            # Update conversation history
            if use_history:
                self.conversation_history.add_exchange(
                    cleaned_text, translated_text, detected_domain, detected_register
                )
                # Perform terminology consistency check
                glossary = self.maintain_terminology_consistency(context.previous_translations)
                if glossary:
                    print(f"[INFO] Updating glossary for context: {len(glossary)} terms")
            
            # Update parameter optimizer
            self.parameter_optimizer.update_params_from_feedback(
                detected_domain, quality_metrics['overall_quality_score'], optimal_params
            )
            
            # Update statistics
            self._update_stats(model_name, quality_metrics['overall_quality_score'], processing_time)
            
            # Create result
            result = TranslationResult(
                original_text=cleaned_text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                quality_score=quality_metrics['overall_quality_score'],
                processing_time=processing_time,
                model_used=model_name,
                domain=detected_domain.value,
                register=detected_register.value,
                context_used=use_history,
                quality_metrics=quality_metrics,
                timestamp=datetime.now().isoformat(),
                session_id=session_id
            )
            
            self.logger.info(f"Translation completed: quality={result.quality_score:.2f}, "
                           f"time={result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            # Return error result
            return TranslationResult(
                original_text=text,
                translated_text=f"[Translation Error: {str(e)}]",
                source_language=source_language,
                target_language=target_language,
                quality_score=0.0,
                processing_time=time.time() - start_time,
                model_used="error",
                domain="unknown",
                register="unknown",
                context_used=False,
                quality_metrics={},
                timestamp=datetime.now().isoformat(),
                session_id=session_id
            )
    
    def translate_sync(self, text: str, target_language: str = "Bengali", **kwargs) -> TranslationResult:
        """Synchronous wrapper for translation."""
        return asyncio.run(self.translate(text, target_language, **kwargs))
    
    def _update_stats(self, model_name: str, quality_score: float, processing_time: float):
        """Update translation statistics."""
        self.translation_stats['total_translations'] += 1
        
        # Update averages
        total = self.translation_stats['total_translations']
        self.translation_stats['average_quality'] = (
            (self.translation_stats['average_quality'] * (total - 1) + quality_score) / total
        )
        self.translation_stats['average_latency'] = (
            (self.translation_stats['average_latency'] * (total - 1) + processing_time) / total
        )
        
        # Update model usage
        if model_name not in self.translation_stats['model_usage']:
            self.translation_stats['model_usage'][model_name] = 0
        self.translation_stats['model_usage'][model_name] += 1
    
    def get_translation_stats(self) -> Dict:
        """Get translation statistics."""
        return self.translation_stats.copy()
    
    def reset_conversation_history(self):
        """Reset conversation history."""
        self.conversation_history = ConversationHistory()
    
    def estimate_translation_confidence(self, quality_metrics: Dict[str, float], text: str) -> float:
        """Estimate the translation confidence based on quality metrics and text complexity."""
        # Basic confidence estimation logic
        complexity_factor = max(0.5, min(1.5, len(text.split()) / 100))  # More words, lower confidence
        score = quality_metrics['overall_quality_score'] * complexity_factor
        return min(1.0, max(0.0, score))

    def export_translation_history(self) -> List[Dict]:
        """Export translation history for analysis."""
        return [exchange for exchange in self.conversation_history.history]
    
    def maintain_terminology_consistency(self, previous_translations: List[Dict]) -> Dict[str, str]:
        """Maintain context-aware terminology consistency by creating a glossary."""
        term_glossary = {}
        for translation in previous_translations:
            if 'original' in translation and 'translated' in translation:
                original_terms = translation['original'].split()
                translated_terms = translation['translated'].split()
                # Simple alignment - in practice, would use proper alignment algorithms
                for i, term in enumerate(original_terms):
                    if i < len(translated_terms) and len(term) > 3:
                        term_glossary[term] = translated_terms[i]
        return term_glossary
