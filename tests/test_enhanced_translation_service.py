import unittest
from services.enhanced_translation_service import EnhancedTranslationService

class TestEnhancedTranslationService(unittest.TestCase):

    def setUp(self):
        self.service = EnhancedTranslationService(api_key="test-key")

    def test_basic_translation(self):
        result = self.service.translate("Hello world", "Spanish")
        self.assertIn('translated_text', result)
        self.assertIn('quality_score', result)

    def test_advanced_translation(self):
        result = self.service.translate("Advanced test", "Bengali", use_advanced=True)
        self.assertIn('translated_text', result)
        self.assertGreater(result.get('quality_score', 0), 0.8)

    def test_switch_engine_mode(self):
        self.service.set_engine_mode(use_advanced=False)
        result = self.service.translate("Engine mode test", "French")
        self.assertIn('translated_text', result)

    def test_export_translation_history(self):
        history = self.service.export_translation_history()
        self.assertIsInstance(history, list)

    def test_statistics(self):
        stats = self.service.get_translation_stats()
        self.assertIn('basic_service', stats)
        self.assertIn('advanced_engine', stats)

if __name__ == '__main__':
    unittest.main()
