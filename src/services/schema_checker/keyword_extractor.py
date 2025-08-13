import re
import nltk
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional
from rake_nltk import Rake
import spacy
from datetime import datetime
import logging

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class KeywordExtractor:
    """
    Advanced keyword extraction using multiple NLP techniques including RAKE and spaCy.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        # Setup logging first
        self.logger = logging.getLogger(__name__)

        self.spacy_model_name = spacy_model
        self.nlp = None
        self._load_spacy_model()

        # Initialize RAKE
        self.rake = Rake()

        # Educational stopwords (domain-specific)
        self.education_stopwords = {
            "class",
            "lesson",
            "today",
            "tomorrow",
            "yesterday",
            "morning",
            "afternoon",
            "students",
            "teacher",
            "professor",
            "instructor",
            "course",
            "subject",
            "homework",
            "assignment",
            "test",
            "exam",
            "quiz",
            "grade",
            "score",
            "chapter",
            "page",
            "book",
            "textbook",
            "material",
            "resources",
        }

    def _load_spacy_model(self):
        """Load spaCy model with error handling."""
        try:
            self.nlp = spacy.load(self.spacy_model_name)
        except OSError:
            self.logger.warning(
                f"spaCy model '{
                    self.spacy_model_name}' not found. spaCy extraction will be disabled."
            )
            self.nlp = None

    def extract_keywords_rake(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract keywords using RAKE (Rapid Automatic Keyword Extraction).
        """
        if not text.strip():
            return []

        try:
            self.rake.extract_keywords_from_text(text)
            ranked_phrases = self.rake.get_ranked_phrases()

            # Filter and clean keywords
            filtered_keywords = []
            for phrase in ranked_phrases[: max_keywords * 2]:  # Get more to filter
                cleaned = self._clean_keyword(phrase)
                if self._is_valid_keyword(cleaned):
                    filtered_keywords.append(cleaned)
                if len(filtered_keywords) >= max_keywords:
                    break

            return filtered_keywords
        except Exception as e:
            self.logger.error(f"Error in RAKE extraction: {e}")
            return []

    def extract_keywords_spacy(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract keywords using spaCy NER and linguistic features.
        """
        if not self.nlp or not text.strip():
            return []

        try:
            doc = self.nlp(text)
            keywords = set()

            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in [
                    "PERSON",
                    "ORG",
                    "PRODUCT",
                    "EVENT",
                    "WORK_OF_ART",
                    "LANGUAGE",
                ]:
                    cleaned = self._clean_keyword(ent.text)
                    if self._is_valid_keyword(cleaned):
                        keywords.add(cleaned)

            # Extract important nouns and adjectives
            for token in doc:
                if (
                    token.pos_ in ["NOUN", "PROPN"]
                    and not token.is_stop
                    and not token.is_punct
                    and len(token.text) > 2
                ):

                    cleaned = self._clean_keyword(token.lemma_)
                    if self._is_valid_keyword(cleaned):
                        keywords.add(cleaned)

            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Keep phrases short
                    cleaned = self._clean_keyword(chunk.text)
                    if self._is_valid_keyword(cleaned):
                        keywords.add(cleaned)

            # Convert to list and sort by relevance (simplified scoring)
            keyword_list = list(keywords)
            keyword_scores = self._score_keywords(keyword_list, text)
            sorted_keywords = sorted(
                keyword_scores.items(), key=lambda x: x[1], reverse=True
            )

            return [kw for kw, score in sorted_keywords[:max_keywords]]

        except Exception as e:
            self.logger.error(f"Error in spaCy extraction: {e}")
            return []

    def extract_keywords_hybrid(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        Combine RAKE and spaCy methods for better keyword extraction.
        """
        rake_keywords = set(self.extract_keywords_rake(text, max_keywords))
        spacy_keywords = set(self.extract_keywords_spacy(text, max_keywords))

        # Combine and deduplicate
        all_keywords = rake_keywords.union(spacy_keywords)

        # Score combined keywords
        keyword_scores = self._score_keywords(list(all_keywords), text)
        sorted_keywords = sorted(
            keyword_scores.items(), key=lambda x: x[1], reverse=True
        )

        return [kw for kw, score in sorted_keywords[:max_keywords]]

    def _clean_keyword(self, keyword: str) -> str:
        """Clean and normalize keyword."""
        # Remove extra whitespace and convert to lowercase
        cleaned = re.sub(r"\s+", " ", keyword.strip().lower())

        # Remove common prefixes/suffixes
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
        cleaned = re.sub(r"\s+(is|are|was|were|be|been|being)$", "", cleaned)

        return cleaned

    def _is_valid_keyword(self, keyword: str) -> bool:
        """Check if keyword is valid for educational content."""
        if not keyword or len(keyword) < 2:
            return False

        # Skip pure numbers or single characters
        if keyword.isdigit() or len(keyword) == 1:
            return False

        # Skip common educational stopwords
        if keyword.lower() in self.education_stopwords:
            return False

        # Skip very common words
        common_words = {
            "this",
            "that",
            "these",
            "those",
            "here",
            "there",
            "when",
            "where",
            "how",
            "why",
            "what",
        }
        if keyword.lower() in common_words:
            return False

        return True

    def _score_keywords(self, keywords: List[str], text: str) -> Dict[str, float]:
        """Score keywords based on frequency and position in text."""
        scores = {}
        text_lower = text.lower()

        for keyword in keywords:
            score = 0.0

            # Frequency score
            frequency = text_lower.count(keyword.lower())
            score += frequency * 2

            # Length bonus (longer phrases often more specific)
            word_count = len(keyword.split())
            score += word_count * 0.5

            # Position bonus (keywords near beginning often more important)
            first_occurrence = text_lower.find(keyword.lower())
            if first_occurrence >= 0:
                position_score = max(0, 1 - (first_occurrence / len(text_lower)))
                score += position_score

            scores[keyword] = score

        return scores

    def extract_keywords(
        self, text: str, method: str = "hybrid", max_keywords: int = 20
    ) -> List[str]:
        """
        Main entry point for keyword extraction.

        Args:
            text: Input text to extract keywords from
            method: Extraction method ('rake', 'spacy', 'hybrid')
            max_keywords: Maximum number of keywords to return

        Returns:
            List of extracted keywords
        """
        if not text or not text.strip():
            return []

        if method == "rake":
            return self.extract_keywords_rake(text, max_keywords)
        elif method == "spacy":
            return self.extract_keywords_spacy(text, max_keywords)
        elif method == "hybrid":
            return self.extract_keywords_hybrid(text, max_keywords)
        else:
            raise ValueError(f"Unsupported keyword extraction method: {method}")

    def extract_with_metadata(
        self, text: str, method: str = "hybrid", max_keywords: int = 20
    ) -> Dict:
        """
        Extract keywords with additional metadata.

        Returns:
            Dictionary with keywords and metadata
        """
        keywords = self.extract_keywords(text, method, max_keywords)

        metadata = {
            "extraction_method": method,
            "text_length": len(text),
            "word_count": len(text.split()),
            "keywords_count": len(keywords),
            "extraction_timestamp": datetime.now().isoformat(),
            "spacy_available": self.nlp is not None,
        }

        return {"keywords": keywords, "metadata": metadata}


# Global instance for backward compatibility
_default_extractor = None


def get_default_extractor() -> KeywordExtractor:
    """Get or create default keyword extractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = KeywordExtractor()
    return _default_extractor


# Convenience functions for backward compatibility


def extract_keywords_rake(text: str) -> List[str]:
    """Extract keywords using RAKE method."""
    extractor = get_default_extractor()
    return extractor.extract_keywords_rake(text)


def extract_keywords(text: str, method: str = "hybrid") -> List[str]:
    """Extract keywords using specified method."""
    extractor = get_default_extractor()
    return extractor.extract_keywords(text, method)


def extract_keywords_with_metadata(text: str, method: str = "hybrid") -> Dict:
    """Extract keywords with metadata."""
    extractor = get_default_extractor()
    return extractor.extract_with_metadata(text, method)
