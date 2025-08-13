import re
from collections import Counter


class KeywordExtractor:
    def __init__(self, language="en"):
        self.language = language
        # In a real-world scenario, you would use a more comprehensive list of
        # stopwords.
        self.stopwords = set(
            ["a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "with"]
        )

    def _get_keywords_rake(self, text):
        # Basic RAKE (Rapid Automatic Keyword Extraction) implementation
        # Clean the text and extract words
        words = re.findall(r"\w+", text.lower())
        phrases = re.split(r"[.!?;,]+", text)

        keywords = []
        for phrase in phrases:
            # Extract only alphabetic words and filter stopwords
            words_in_phrase = re.findall(r"\b[a-zA-Z]+\b", phrase.lower())
            words_in_phrase = [
                word
                for word in words_in_phrase
                if word not in self.stopwords and len(word) > 2
            ]
            keywords.extend(words_in_phrase)

        return [keyword for keyword, _ in Counter(keywords).most_common(20)]

    def extract_keywords(self, transcript_text, method="rake"):
        if method == "rake":
            return self._get_keywords_rake(transcript_text)
        elif method == "spacy":
            import spacy

            nlp = spacy.load("en_core_web_sm")
            doc = nlp(transcript_text)
            return [chunk.text for chunk in doc.noun_chunks]
        else:
            raise ValueError(f"Unknown keyword extraction method: {method}")


if __name__ == "__main__":
    # Example Usage
    extractor = KeywordExtractor()
    sample_transcript = """
 Good morning, class. Today, we're going to discuss the solar system.
 Our solar system consists of the sun and everything that orbits it, including planets, moons, asteroids, and comets.
 The sun is the center of our solar system. There are eight planets in our solar system.
 The planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.
 Each planet has unique characteristics. For example, Jupiter is the largest planet, and Mars is known as the Red Planet.
 """

    print(
        "Keywords (RAKE):", extractor.extract_keywords(sample_transcript, method="rake")
    )
