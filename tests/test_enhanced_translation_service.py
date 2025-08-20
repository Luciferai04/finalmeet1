import unittest
import sys
from pathlib import Path
from flask import Flask

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.services.enhanced_translation_service import EnhancedTranslationService

class TestEnhancedTranslationService(unittest.TestCase):

    def setUp(self):
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.config['GOOGLE_API_KEY'] = "test-key"
        self.ctx = self.app.app_context()
        self.ctx.push()
        self.service = EnhancedTranslationService(api_key="test-key")

    def tearDown(self):
        self.ctx.pop()

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
