import re
import difflib
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
import logging

# Try to import advanced NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TopicComparator:
    """
    Advanced topic comparison using multiple matching algorithms including
    exact matching, fuzzy matching, semantic similarity, and NLP-based comparison.
    """

    def __init__(self, similarity_threshold: float = 0.6,
                 spacy_model: str = "en_core_web_sm"):
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)

        # Initialize spaCy model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                self.logger.warning(
                    f"spaCy model '{spacy_model}' not found. Semantic similarity disabled.")

        # Initialize TF-IDF vectorizer if available
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 3),
                max_features=1000
            )

    def compare_topics(
            self, expected_topics: List[str], extracted_keywords: List[str]) -> Dict[str, Any]:
        """
        Main comparison function that uses multiple matching strategies.

        Args:
            expected_topics: List of expected topics from schema
            extracted_keywords: List of keywords extracted from transcript

        Returns:
            Dictionary with detailed comparison results
        """
        # Normalize inputs
        expected_topics = [self._normalize_text(
            topic) for topic in expected_topics if topic.strip()]
        extracted_keywords = [self._normalize_text(
            kw) for kw in extracted_keywords if kw.strip()]

        # Different matching strategies
        exact_matches = self._find_exact_matches(
            expected_topics, extracted_keywords)
        fuzzy_matches = self._find_fuzzy_matches(
            expected_topics, extracted_keywords)
        semantic_matches = self._find_semantic_matches(
            expected_topics, extracted_keywords)

        # Combine all matches
        all_matches = self._combine_matches(
            exact_matches, fuzzy_matches, semantic_matches)

        # Categorize results
        matched_topics = []
        missed_topics = []
        unexpected_topics = []

        # Track which expected topics were matched
        matched_expected_topics = set()
        for match in all_matches:
            matched_expected_topics.add(match['expected_topic'])
            matched_topics.append(match)

        # Find missed topics
        for topic in expected_topics:
            if topic not in matched_expected_topics:
                missed_topics.append({
                    'topic': topic,
                    'reason': 'No matching keywords found',
                    'suggestions': self._get_suggestions(topic, extracted_keywords)
                })

        # Find unexpected topics (keywords not matching any expected topic)
        matched_keywords = set()
        for match in all_matches:
            matched_keywords.update(match['matched_keywords'])

        for keyword in extracted_keywords:
            if keyword not in matched_keywords:
                unexpected_topics.append({
                    'keyword': keyword,
                    'relevance_score': self._calculate_relevance_score(keyword, expected_topics)
                })

        # Calculate overall statistics
        coverage_score = len(matched_expected_topics) / \
            len(expected_topics) if expected_topics else 0
        precision = len(matched_keywords) / \
            len(extracted_keywords) if extracted_keywords else 0

        return {
            'matched_topics': matched_topics,
            'missed_topics': missed_topics,
            'unexpected_topics': unexpected_topics,
            'statistics': {
                'total_expected_topics': len(expected_topics),
                'total_extracted_keywords': len(extracted_keywords),
                'topics_covered': len(matched_expected_topics),
                'topics_missed': len(missed_topics),
                'unexpected_keywords': len(unexpected_topics),
                'coverage_score': round(coverage_score, 3),
                'precision_score': round(precision, 3),
                'coverage_percentage': round(coverage_score * 100, 1),
                'quality_score': round((coverage_score + precision) * 50, 1),
                'comparison_timestamp': datetime.now().isoformat()
            },
            'metadata': {
                'similarity_threshold': self.similarity_threshold,
                'spacy_available': self.nlp is not None,
                'sklearn_available': SKLEARN_AVAILABLE,
                'matching_methods_used': ['exact', 'fuzzy'] + (['semantic'] if self.nlp else [])
            }
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text.lower().strip())

        # Remove common articles and prepositions
        normalized = re.sub(
            r'\b(the|a|an|in|on|at|to|for|of|with|by)\b',
            '',
            normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def _find_exact_matches(
            self, expected_topics: List[str], extracted_keywords: List[str]) -> List[Dict]:
        """Find exact string matches."""
        matches = []

        for topic in expected_topics:
            matched_keywords = []
            for keyword in extracted_keywords:
                if topic == keyword or topic in keyword or keyword in topic:
                    matched_keywords.append(keyword)

            if matched_keywords:
                matches.append({
                    'expected_topic': topic,
                    'matched_keywords': matched_keywords,
                    'match_type': 'exact',
                    'confidence_score': 1.0
                })

        return matches

    def _find_fuzzy_matches(
            self, expected_topics: List[str], extracted_keywords: List[str]) -> List[Dict]:
        """Find fuzzy string matches using sequence matching."""
        matches = []

        for topic in expected_topics:
            topic_matches = []

            for keyword in extracted_keywords:
                # Calculate similarity ratio
                similarity = difflib.SequenceMatcher(None, topic, keyword).ratio()

                if similarity >= self.similarity_threshold:
                    topic_matches.append({
                        'keyword': keyword,
                        'similarity': similarity
                    })

            if topic_matches:
                # Sort by similarity and take the best matches
                topic_matches.sort(key=lambda x: x['similarity'], reverse=True)
                matched_keywords = [match['keyword'] for match in topic_matches]
                avg_confidence = sum(match['similarity']
                                     for match in topic_matches) / len(topic_matches)

                matches.append({
                    'expected_topic': topic,
                    'matched_keywords': matched_keywords,
                    'match_type': 'fuzzy',
                    'confidence_score': round(avg_confidence, 3),
                    'similarity_details': topic_matches
                })

        return matches

    def _find_semantic_matches(
            self, expected_topics: List[str], extracted_keywords: List[str]) -> List[Dict]:
        """Find semantic matches using spaCy word vectors."""
        if not self.nlp:
            return []

        matches = []

        try:
            for topic in expected_topics:
                topic_doc = self.nlp(topic)
                topic_matches = []

                for keyword in extracted_keywords:
                    keyword_doc = self.nlp(keyword)

                    # Calculate semantic similarity
                    if topic_doc.has_vector and keyword_doc.has_vector:
                        similarity = topic_doc.similarity(keyword_doc)

                        if similarity >= self.similarity_threshold:
                            topic_matches.append({
                                'keyword': keyword,
                                'similarity': similarity
                            })

                if topic_matches:
                    # Sort by similarity
                    topic_matches.sort(key=lambda x: x['similarity'], reverse=True)
                    matched_keywords = [match['keyword'] for match in topic_matches]
                    avg_confidence = sum(match['similarity']
                                         for match in topic_matches) / len(topic_matches)

                    matches.append({
                        'expected_topic': topic,
                        'matched_keywords': matched_keywords,
                        'match_type': 'semantic',
                        'confidence_score': round(avg_confidence, 3),
                        'similarity_details': topic_matches
                    })

        except Exception as e:
            self.logger.error(f"Error in semantic matching: {e}")

        return matches

    def _combine_matches(self, *match_lists) -> List[Dict]:
        """Combine matches from different strategies, avoiding duplicates."""
        combined = {}

        for match_list in match_lists:
            for match in match_list:
                topic = match['expected_topic']

                if topic not in combined:
                    combined[topic] = match
                else:
                    # Merge keywords and update confidence
                    existing = combined[topic]
                    all_keywords = set(
                        existing['matched_keywords'] +
                        match['matched_keywords'])
                    existing['matched_keywords'] = list(all_keywords)

                    # Use the highest confidence score
                    if match['confidence_score'] > existing['confidence_score']:
                        existing['confidence_score'] = match['confidence_score']
                    existing['match_type'] = f"{existing['match_type']},{match['match_type']}"

        return list(combined.values())

    def _get_suggestions(self, missed_topic: str,
                         extracted_keywords: List[str]) -> List[str]:
        """Get suggestions for missed topics based on partial matches."""
        suggestions = []

        # Find keywords with partial similarity
        for keyword in extracted_keywords:
            similarity = difflib.SequenceMatcher(None, missed_topic, keyword).ratio()
            if 0.3 <= similarity < self.similarity_threshold:  # Partial match
                suggestions.append(keyword)

        # Sort by similarity and return top suggestions
        suggestions.sort(
            key=lambda kw: difflib.SequenceMatcher(
                None,
                missed_topic,
                kw).ratio(),
            reverse=True)
        return suggestions[:3]

    def _calculate_relevance_score(
            self, keyword: str, expected_topics: List[str]) -> float:
        """Calculate how relevant an unexpected keyword is to the expected topics."""
        if not expected_topics:
            return 0.0

        max_similarity = 0.0

        for topic in expected_topics:
            # String similarity
            string_sim = difflib.SequenceMatcher(None, topic, keyword).ratio()
            max_similarity = max(max_similarity, string_sim)

            # Semantic similarity if available
            if self.nlp:
                try:
                    topic_doc = self.nlp(topic)
                    keyword_doc = self.nlp(keyword)
                    if topic_doc.has_vector and keyword_doc.has_vector:
                        semantic_sim = topic_doc.similarity(keyword_doc)
                        max_similarity = max(max_similarity, semantic_sim)
                except Exception:
                    pass

        return round(max_similarity, 3)


# Global instance for backward compatibility
_default_comparator = None


def get_default_comparator() -> TopicComparator:
    """Get or create default topic comparator instance."""
    global _default_comparator
    if _default_comparator is None:
        _default_comparator = TopicComparator()
    return _default_comparator


# Convenience function for backward compatibility

def compare_topics(
        expected_topics: List[str], extracted_keywords: List[str]) -> Dict[str, Any]:
    """Compare expected topics with extracted keywords."""
    comparator = get_default_comparator()
    return comparator.compare_topics(expected_topics, extracted_keywords)
