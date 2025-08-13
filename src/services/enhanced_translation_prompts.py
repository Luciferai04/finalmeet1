"""
Enhanced Translation Prompt Manager
===================================

Advanced prompt engineering for high-quality translations with cultural sensitivity,
domain awareness, and contextual adaptation.
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class DomainType(Enum):
    """Domain classification for specialized translation."""
    GENERAL = "general"
    BUSINESS = "business"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    LEGAL = "legal"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    CONVERSATION = "conversation"
    FORMAL = "formal"
    INFORMAL = "informal"


class RegisterType(Enum):
    """Text register for style adaptation."""
    FORMAL = "formal"
    INFORMAL = "informal"
    ACADEMIC = "academic"
    COLLOQUIAL = "colloquial"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"


@dataclass
class TranslationContext:
    """Context information for enhanced translation."""
    domain: DomainType = DomainType.GENERAL
    register: RegisterType = RegisterType.INFORMAL
    conversation_history: List[str] = None
    source_language: str = "English"
    target_language: str = "Bengali"
    cultural_context: str = ""
    user_preferences: Dict = None
    previous_translations: List[Dict] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.previous_translations is None:
            self.previous_translations = []


class EnhancedTranslationPrompts:
    """Advanced prompt manager for high-quality translations."""
    
    def __init__(self):
        self.domain_keywords = self._initialize_domain_keywords()
        self.cultural_guidelines = self._initialize_cultural_guidelines()
        self.language_specific_rules = self._initialize_language_rules()
        
    def _initialize_domain_keywords(self) -> Dict[DomainType, List[str]]:
        """Initialize domain-specific keywords for classification."""
        return {
            DomainType.BUSINESS: [
                "revenue", "profit", "market", "sales", "company", "corporation",
                "business", "meeting", "presentation", "proposal", "contract",
                "client", "customer", "service", "product", "strategy"
            ],
            DomainType.TECHNICAL: [
                "software", "hardware", "system", "algorithm", "database",
                "server", "network", "programming", "code", "technical",
                "specification", "configuration", "installation", "API"
            ],
            DomainType.MEDICAL: [
                "patient", "doctor", "hospital", "medicine", "treatment",
                "diagnosis", "symptom", "disease", "health", "medical",
                "prescription", "clinic", "surgery", "therapy"
            ],
            DomainType.LEGAL: [
                "law", "legal", "court", "judge", "lawyer", "contract",
                "agreement", "regulation", "compliance", "rights",
                "liability", "jurisdiction", "statute", "litigation"
            ],
            DomainType.EDUCATION: [
                "student", "teacher", "school", "university", "education",
                "learning", "course", "curriculum", "exam", "grade",
                "assignment", "research", "study", "academic"
            ],
            DomainType.NEWS: [
                "news", "report", "journalist", "breaking", "update",
                "announcement", "press", "media", "headline", "story",
                "coverage", "investigation", "politics", "economy"
            ]
        }
    
    def _initialize_cultural_guidelines(self) -> Dict[str, Dict]:
        """Initialize cultural adaptation guidelines."""
        return {
            "Bengali": {
                "honorifics": {
                    "formal": ["আপনি", "স্যার", "ম্যাডাম"],
                    "respectful_addressing": True,
                    "age_hierarchy": True
                },
                "cultural_context": {
                    "festivals": ["দুর্গাপূজা", "কালীপূজা", "পয়লা বৈশাখ"],
                    "food_terms": ["ভাত", "মাছ", "দাল", "তরকারি"],
                    "regional_variations": ["কলকাতা", "ঢাকা", "চট্টগ্রাম"]
                },
                "script_handling": {
                    "numbers": "bengali_numerals_optional",
                    "punctuation": "bangla_punctuation"
                }
            },
            "Hindi": {
                "honorifics": {
                    "formal": ["आप", "जी", "साहब", "मैडम"],
                    "respectful_addressing": True,
                    "age_hierarchy": True
                },
                "cultural_context": {
                    "festivals": ["दीवाली", "होली", "दशहरा"],
                    "food_terms": ["रोटी", "चावल", "दाल", "सब्जी"],
                    "regional_variations": ["दिल्ली", "मुंबई", "कोलकाता"]
                },
                "script_handling": {
                    "numbers": "devanagari_numerals_optional",
                    "punctuation": "hindi_punctuation"
                }
            }
        }
    
    def _initialize_language_rules(self) -> Dict[str, Dict]:
        """Initialize language-specific translation rules."""
        return {
            "Bengali": {
                "word_order": "SOV",  # Subject-Object-Verb
                "formal_pronouns": ["আপনি", "তিনি"],
                "informal_pronouns": ["তুমি", "তোমার"],
                "compound_verbs": True,
                "postpositions": True,
                "conjunct_consonants": True
            },
            "Hindi": {
                "word_order": "SOV",
                "formal_pronouns": ["आप", "वे"],
                "informal_pronouns": ["तुम", "तू"],
                "compound_verbs": True,
                "postpositions": True,
                "gender_agreement": True
            }
        }
    
    def detect_domain(self, text: str) -> DomainType:
        """Detect the domain of the input text."""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        # Fallback domain detection based on text characteristics
        if any(word in text_lower for word in ["hello", "hi", "how are you", "thank you"]):
            return DomainType.CONVERSATION
        elif len(text.split()) > 50:
            return DomainType.FORMAL
        else:
            return DomainType.GENERAL
    
    def detect_register(self, text: str) -> RegisterType:
        """Detect the register/formality level of the text."""
        formal_indicators = [
            "please", "kindly", "would you", "could you", "thank you",
            "sincerely", "regards", "respectfully", "sir", "madam"
        ]
        informal_indicators = [
            "hey", "hi", "yeah", "ok", "gonna", "wanna", "stuff",
            "thing", "cool", "awesome", "lol"
        ]
        
        text_lower = text.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
        
        if formal_count > informal_count:
            return RegisterType.FORMAL
        elif informal_count > 0:
            return RegisterType.INFORMAL
        else:
            return RegisterType.PROFESSIONAL
    
    def build_context_aware_prompt(self, text: str, context: TranslationContext) -> str:
        """Build an advanced context-aware translation prompt with few-shot examples and chain-of-thought reasoning."""

        # Use improved prompting techniques based on text complexity and domain
        if len(text.split()) > 30 or context.domain in [DomainType.TECHNICAL, DomainType.MEDICAL, DomainType.LEGAL, DomainType.BUSINESS]:
            return self._build_chain_of_thought_prompt(text, context)
        else:
            return self._build_enhanced_few_shot_prompt(text, context)

    
    def _build_chain_of_thought_prompt(self, text: str, context: TranslationContext) -> str:
        """Build a chain-of-thought reasoning prompt for complex translations."""
        
        base_prompt = f"""You are a professional {context.source_language}-{context.target_language} translator with expertise in {context.domain.value} domain.

TRANSLATION METHODOLOGY:
1. First, analyze the source text for key concepts, cultural references, and technical terms
2. Consider the target audience and appropriate register in {context.target_language}
3. Identify any ambiguities or context-dependent meanings
4. Choose the most appropriate translation strategy
5. Produce a natural, fluent translation that preserves meaning and tone

SOURCE TEXT: "{text}"

DOMAIN: {context.domain.value.title()}
REGISTER: {context.register.value.title()}
TARGET LANGUAGE: {context.target_language}

ANALYSIS:
Let me analyze this step by step:

1. KEY CONCEPTS: [Identify main concepts and technical terms]
2. CULTURAL CONTEXT: [Note any cultural references or context-specific meanings]
3. TONE AND REGISTER: [Determine appropriate formality and style]
4. TRANSLATION STRATEGY: [Choose best approach for this specific text]

"""
        
        # Add domain-specific guidance
        domain_guidance = self._get_advanced_domain_guidance(context.domain, context.target_language)
        base_prompt += f"DOMAIN-SPECIFIC GUIDANCE:\n{domain_guidance}\n\n"
        
        # Add few-shot examples if available
        examples = self._get_few_shot_examples(context.domain, context.source_language, context.target_language)
        if examples:
            base_prompt += "HIGH-QUALITY EXAMPLES:\n"
            for i, example in enumerate(examples[:2], 1):
                base_prompt += f"Example {i}:\n{example}\n\n"
        
        # Add consistency context
        if context.previous_translations:
            consistency_terms = self._extract_consistency_terms(context.previous_translations)
            if consistency_terms:
                base_prompt += f"TERMINOLOGY CONSISTENCY:\n{consistency_terms}\n\n"
        
        base_prompt += f"""Now, following this methodology, provide your {context.target_language} translation:

FINAL TRANSLATION:"""
        
        return base_prompt
    
    def _build_enhanced_few_shot_prompt(self, text: str, context: TranslationContext) -> str:
        """Build an enhanced few-shot prompt for standard translations."""
        
        # Role-based prompting
        role_description = self._get_role_description(context.domain, context.target_language)
        
        base_prompt = f"""You are {role_description}

Your task is to translate {context.source_language} text to {context.target_language} with the highest quality and cultural appropriateness.

"""
        
        # Add few-shot examples
        examples = self._get_few_shot_examples(context.domain, context.source_language, context.target_language)
        if examples:
            base_prompt += "Here are examples of high-quality translations in this domain:\n\n"
            for i, example in enumerate(examples[:3], 1):
                base_prompt += f"Example {i}:\n{example}\n\n"
        
        # Add specific guidelines
        guidelines = self._get_enhanced_guidelines(context)
        base_prompt += f"TRANSLATION GUIDELINES:\n{guidelines}\n\n"
        
        # Add the actual translation task
        base_prompt += f"""Now translate this {context.source_language} text to {context.target_language}:

"{text}"

Provide ONLY the {context.target_language} translation:"""
        
        return base_prompt
    
    def _get_domain_guidelines(self, domain: DomainType, target_language: str) -> str:
        """Get domain-specific translation guidelines."""
        guidelines = {
            DomainType.BUSINESS: f"Use professional {target_language} business terminology. Maintain formal tone suitable for corporate communication.",
            DomainType.TECHNICAL: f"Preserve technical terms where appropriate. Use established {target_language} technical vocabulary when available.",
            DomainType.MEDICAL: f"Use precise medical terminology in {target_language}. Maintain clinical accuracy and professional tone.",
            DomainType.LEGAL: f"Use formal legal language in {target_language}. Maintain precision and avoid ambiguity.",
            DomainType.EDUCATION: f"Use clear, educational {target_language} suitable for learning contexts.",
            DomainType.CONVERSATION: f"Use natural, conversational {target_language} appropriate for casual interaction.",
            DomainType.NEWS: f"Use clear, informative {target_language} suitable for news reporting.",
            DomainType.FORMAL: f"Maintain formal register and respectful tone in {target_language}.",
            DomainType.GENERAL: f"Use clear, natural {target_language} appropriate for general communication."
        }
        return guidelines.get(domain, guidelines[DomainType.GENERAL])
    
    def _get_register_guidelines(self, register: RegisterType, target_language: str) -> str:
        """Get register-specific guidelines."""
        guidelines = {
            RegisterType.FORMAL: f"Use formal pronouns, honorifics, and polite expressions in {target_language}.",
            RegisterType.INFORMAL: f"Use casual, friendly tone while maintaining respect in {target_language}.",
            RegisterType.ACADEMIC: f"Use scholarly vocabulary and formal academic style in {target_language}.",
            RegisterType.PROFESSIONAL: f"Use professional business language appropriate for workplace communication in {target_language}.",
            RegisterType.TECHNICAL: f"Use precise technical language and terminology in {target_language}.",
            RegisterType.COLLOQUIAL: f"Use natural, everyday language with appropriate colloquialisms in {target_language}."
        }
        return guidelines.get(register, guidelines[RegisterType.PROFESSIONAL])
    
    def _get_cultural_guidelines(self, target_language: str, cultural_context: str = "") -> str:
        """Get cultural adaptation guidelines."""
        if target_language not in self.cultural_guidelines:
            return ""
        
        culture_info = self.cultural_guidelines[target_language]
        guidelines = []
        
        if culture_info.get("honorifics", {}).get("respectful_addressing"):
            guidelines.append("Use appropriate honorifics and respectful addressing")
        
        if cultural_context:
            guidelines.append(f"Consider cultural context: {cultural_context}")
        
        return "; ".join(guidelines) if guidelines else ""
    
    def _build_history_context(self, history: List[str]) -> str:
        """Build context from conversation history."""
        if not history:
            return ""
        
        recent_history = history[-3:]  # Last 3 exchanges
        return f"Previous context: {' | '.join(recent_history)}"
    
    def _build_consistency_guidelines(self, previous_translations: List[Dict]) -> str:
        """Build guidelines for translation consistency."""
        if not previous_translations:
            return ""
        
        # Extract common terms and their translations
        term_consistency = {}
        for trans in previous_translations[-5:]:  # Last 5 translations
            original = trans.get('original_text', '')
            translated = trans.get('translated_text', '')
            # Simple term extraction (can be enhanced with NLP)
            # This is a simplified version
            
        return "Maintain consistency with previous translations of similar terms"
    
    def _get_advanced_domain_guidance(self, domain: DomainType, target_language: str) -> str:
        """Get advanced domain-specific guidance for complex translations."""
        guidance = {
            DomainType.TECHNICAL: f"""For technical {target_language} translation:
- Preserve technical terminology precisely
- Use established {target_language} technical vocabulary
- Maintain consistent technical term translations
- Consider target technical audience expertise level""",
            
            DomainType.MEDICAL: f"""For medical {target_language} translation:
- Use precise medical terminology
- Maintain clinical accuracy and safety
- Follow medical translation standards
- Consider patient safety implications""",
            
            DomainType.LEGAL: f"""For legal {target_language} translation:
- Maintain legal precision and avoid ambiguity
- Use formal legal language conventions
- Preserve legal concepts accurately
- Consider jurisdictional differences""",
            
            DomainType.BUSINESS: f"""For business {target_language} translation:
- Use professional business terminology
- Maintain corporate communication standards
- Consider cultural business practices
- Preserve formal business tone"""
        }
        return guidance.get(domain, f"Use appropriate {target_language} terminology for {domain.value} domain.")
    
    def _get_few_shot_examples(self, domain: DomainType, source_lang: str, target_lang: str) -> List[str]:
        """Get few-shot examples for the specific domain and language pair."""
        examples = []
        
        if source_lang == "English" and target_lang == "Bengali":
            if domain == DomainType.BUSINESS:
                examples = [
                    "English: 'Please review the quarterly financial report.'\nBengali: 'অনুগ্রহ করে ত্রৈমাসিক আর্থিক প্রতিবেদনটি পর্যালোচনা করুন।'",
                    "English: 'The meeting is scheduled for tomorrow at 2 PM.'\nBengali: 'সভাটি আগামীকাল দুপুর ২টায় নির্ধারিত।'"
                ]
            elif domain == DomainType.CONVERSATION:
                examples = [
                    "English: 'How are you doing today?'\nBengali: 'আজ আপনার কেমন লাগছে?'",
                    "English: 'Thank you so much for your help.'\nBengali: 'আপনার সাহায্যের জন্য অনেক ধন্যবাদ।'"
                ]
        elif source_lang == "English" and target_lang == "Hindi":
            if domain == DomainType.BUSINESS:
                examples = [
                    "English: 'Please review the quarterly financial report.'\nHindi: 'कृपया त्रैमासिक वित्तीय रिपोर्ट की समीक्षा करें।'",
                    "English: 'The meeting is scheduled for tomorrow at 2 PM.'\nHindi: 'बैठक कल दोपहर 2 बजे निर्धारित है।'"
                ]
            elif domain == DomainType.CONVERSATION:
                examples = [
                    "English: 'How are you doing today?'\nHindi: 'आज आप कैसे हैं?'",
                    "English: 'Thank you so much for your help.'\nHindi: 'आपकी सहायता के लिए बहुत धन्यवाद।'"
                ]
        
        return examples
    
    def _get_role_description(self, domain: DomainType, target_language: str) -> str:
        """Get role description for the translator based on domain."""
        roles = {
            DomainType.TECHNICAL: f"a specialized technical translator with expertise in {target_language} technical documentation",
            DomainType.MEDICAL: f"a certified medical translator specializing in {target_language} healthcare communication",
            DomainType.LEGAL: f"a professional legal translator with expertise in {target_language} legal documents",
            DomainType.BUSINESS: f"a professional business translator specializing in {target_language} corporate communication",
            DomainType.EDUCATION: f"an educational content translator specializing in {target_language} learning materials",
            DomainType.CONVERSATION: f"a conversational {target_language} translator focused on natural dialogue"
        }
        return roles.get(domain, f"a professional {target_language} translator")
    
    def _get_enhanced_guidelines(self, context: TranslationContext) -> str:
        """Get enhanced translation guidelines based on context."""
        guidelines = []
        
        # Domain-specific guidelines
        domain_guide = self._get_domain_guidelines(context.domain, context.target_language)
        guidelines.append(f"Domain: {domain_guide}")
        
        # Register guidelines
        register_guide = self._get_register_guidelines(context.register, context.target_language)
        guidelines.append(f"Register: {register_guide}")
        
        # Cultural guidelines
        cultural_guide = self._get_cultural_guidelines(context.target_language, context.cultural_context)
        if cultural_guide:
            guidelines.append(f"Cultural: {cultural_guide}")
        
        # Language-specific rules
        if context.target_language in self.language_specific_rules:
            rules = self.language_specific_rules[context.target_language]
            guidelines.append(f"Language structure: Follow {context.target_language} {rules['word_order']} word order")
        
        return "\n- ".join(["GUIDELINES:"] + guidelines)
    
    def _extract_consistency_terms(self, previous_translations: List[Dict]) -> str:
        """Extract terminology consistency guidelines from previous translations."""
        if not previous_translations:
            return ""
        
        consistency_terms = []
        
        # Extract key terms from recent translations
        for trans in previous_translations[-3:]:
            original = trans.get('original_text', '')
            translated = trans.get('translated_text', '')
            
            # Extract technical terms, proper nouns, etc.
            # This is a simplified implementation
            original_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', original)
            
            for term in original_terms:
                if len(term) > 3:  # Filter short terms
                    consistency_terms.append(f"'{term}' → maintain consistent translation")
        
        if consistency_terms:
            return "\n- ".join(["Terminology consistency:"] + consistency_terms[:5])  # Show top 5
        
        return "Maintain consistent terminology with previous translations"


class QualityMetrics:
    """Translation quality assessment metrics with semantic similarity and BLEU score."""
    
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except ImportError as e:
            print("[WARNING] Sentence transformer model not available: {e}")
        
    @staticmethod
    def calculate_bleu_score(reference: List[List[str]], hypothesis: List[str]) -> float:
        """Calculate BLEU score for the given translation.
        Using nltk library for BLEU computation.
        """
        from nltk.translate.bleu_score import sentence_bleu
        return sentence_bleu(reference, hypothesis)

    def calculate_semantic_similarity(self, original: str, translated: str) -> float:
        """Calculate semantic similarity between original and translated text."""
        if not hasattr(self, 'similarity_model'):
            return 0.0  # Model not loaded
        
        try:
            from sentence_transformers import util
            original_embedding = self.similarity_model.encode([original], convert_to_tensor=True)
            translated_embedding = self.similarity_model.encode([translated], convert_to_tensor=True)
            
            similarity_score = util.pytorch_cos_sim(original_embedding, translated_embedding)
            return similarity_score.item()
        except Exception as e:
            print(f"[WARNING] Semantic similarity calculation failed: {e}")
            return 0.0

    @staticmethod
    def calculate_length_ratio(original: str, translated: str) -> float:
        """Calculate length ratio between original and translated text."""
        if not original or not translated:
            return 0.0
        return len(translated) / len(original)

    @staticmethod
    def detect_untranslated_terms(original: str, translated: str, 
                                 source_lang: str = "en") -> List[str]:
        """Detect potentially untranslated terms."""
        if source_lang != "en":
            return []  # Only works for English source for now
        
        original_words = set(re.findall(r'\b[A-Za-z]+\b', original.lower()))
        translated_words = set(re.findall(r'\b[A-Za-z]+\b', translated.lower()))
        
        # Common words that might legitimately appear in translations
        common_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'from', 'with', 'by'}
        
        potentially_untranslated = []
        for word in original_words:
            if word in translated_words and word not in common_words and len(word) >= 3:
                potentially_untranslated.append(word)
        
        return potentially_untranslated
    
    def assess_translation_quality(self, original: str, translated: str, 
                                  context: TranslationContext) -> Dict[str, float]:
        """Comprehensive translation quality assessment with advanced metrics."""
        metrics = {}
        
        # Length ratio
        length_ratio = self.calculate_length_ratio(original, translated)
        metrics['length_ratio'] = length_ratio
        metrics['length_score'] = 1.0 if 0.5 <= length_ratio <= 2.0 else max(0.1, 1.0 - abs(length_ratio - 1.0))
        
        # BLEU score
        reference = [original.split()]
        hypothesis = translated.split()
        metrics['bleu_score'] = self.calculate_bleu_score(reference, hypothesis)
        
        # Semantic similarity
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(original, translated)

        # Untranslated terms detection
        untranslated = self.detect_untranslated_terms(original, translated)
        metrics['untranslated_terms'] = len(untranslated)
        metrics['translation_completeness'] = max(0.0, 1.0 - len(untranslated) * 0.1)
        
        # Basic structural assessment
        original_sentences = len(re.findall(r'[.!?]+', original))
        translated_sentences = len(re.findall(r'[.!?।]+', translated))  # Includes Bengali/Hindi punctuation
        
        sentence_ratio = translated_sentences / max(1, original_sentences)
        metrics['sentence_structure_score'] = 1.0 if 0.8 <= sentence_ratio <= 1.2 else max(0.3, 1.0 - abs(sentence_ratio - 1.0))
        
        # Overall quality score (weighted average with new metrics)
        weights = {
            'length_score': 0.2,
            'translation_completeness': 0.3,
            'sentence_structure_score': 0.2,
            'bleu_score': 0.1,
            'semantic_similarity': 0.2
        }
        
        overall_score = sum(metrics[key] * weight for key, weight in weights.items())
        metrics['overall_quality_score'] = overall_score
        
        return metrics
