#!/usr/bin/env python3
"""
Translation Quality Validation Script
====================================

This script validates the translation quality improvements by testing
various scenarios and comparing basic vs advanced translation engines.
"""

import os
import sys
import json
import time
from typing import List, Dict
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.enhanced_translation_service import EnhancedTranslationService


class TranslationQualityValidator:
    """Validates translation quality improvements."""
    
    def __init__(self, api_key: str = None):
        """Initialize the validator."""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable required")
        
        self.service = EnhancedTranslationService(api_key=self.api_key)
        self.test_cases = self._load_test_cases()
    
    def _load_test_cases(self) -> List[Dict]:
        """Load test cases for validation."""
        return [
            {
                "text": "Hello, how are you today?",
                "target_language": "Bengali",
                "expected_domain": "conversation",
                "context": "casual greeting"
            },
            {
                "text": "Please review the quarterly financial report by end of day.",
                "target_language": "Hindi", 
                "expected_domain": "business",
                "context": "professional workplace communication"
            },
            {
                "text": "The patient exhibits symptoms of acute respiratory distress.",
                "target_language": "Bengali",
                "expected_domain": "medical",
                "context": "clinical medical record"
            },
            {
                "text": "The algorithm optimizes database query performance using indexing.",
                "target_language": "Hindi",
                "expected_domain": "technical",
                "context": "software documentation"
            },
            {
                "text": "Thanks for your help! You're awesome.",
                "target_language": "Bengali",
                "expected_domain": "conversation",
                "context": "informal appreciation"
            }
        ]
    
    def validate_basic_vs_advanced(self) -> Dict:
        """Compare basic vs advanced translation quality."""
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "total_tests": len(self.test_cases),
            "basic_engine_results": [],
            "advanced_engine_results": [],
            "comparison_summary": {}
        }
        
        print("🔍 Running translation quality validation...")
        print(f"Testing {len(self.test_cases)} test cases\n")
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"Test {i}/{len(self.test_cases)}: {test_case['text'][:50]}...")
            
            # Test with basic engine
            basic_result = self._test_with_engine(test_case, use_advanced=False)
            results["basic_engine_results"].append(basic_result)
            
            # Test with advanced engine
            advanced_result = self._test_with_engine(test_case, use_advanced=True)
            results["advanced_engine_results"].append(advanced_result)
            
            # Compare results
            self._compare_results(basic_result, advanced_result, i)
            
            print()
        
        # Generate summary
        results["comparison_summary"] = self._generate_summary(results)
        
        return results
    
    def _test_with_engine(self, test_case: Dict, use_advanced: bool) -> Dict:
        """Test translation with specified engine."""
        start_time = time.time()
        
        try:
            result = self.service.translate(
                text=test_case["text"],
                target_language=test_case["target_language"],
                user_context=test_case.get("context", ""),
                use_advanced=use_advanced
            )
            
            processing_time = time.time() - start_time
            
            return {
                "engine_type": "advanced" if use_advanced else "basic",
                "original_text": test_case["text"],
                "translated_text": result.get("translated_text", ""),
                "target_language": test_case["target_language"],
                "processing_time": processing_time,
                "quality_score": result.get("quality_score", 0.0),
                "domain_detected": result.get("domain", "unknown"),
                "expected_domain": test_case.get("expected_domain", "unknown"),
                "model_used": result.get("model_used", "unknown"),
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "engine_type": "advanced" if use_advanced else "basic",
                "original_text": test_case["text"],
                "translated_text": "",
                "processing_time": time.time() - start_time,
                "quality_score": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def _compare_results(self, basic_result: Dict, advanced_result: Dict, test_num: int):
        """Compare and display results from both engines."""
        print(f"  📊 Basic Engine:")
        print(f"    Translation: {basic_result['translated_text']}")
        print(f"    Quality: {basic_result['quality_score']:.2f}")
        print(f"    Time: {basic_result['processing_time']:.2f}s")
        
        print(f"  🚀 Advanced Engine:")
        print(f"    Translation: {advanced_result['translated_text']}")
        print(f"    Quality: {advanced_result['quality_score']:.2f}")
        print(f"    Domain: {advanced_result.get('domain_detected', 'N/A')}")
        print(f"    Time: {advanced_result['processing_time']:.2f}s")
        
        # Quality improvement
        quality_improvement = advanced_result['quality_score'] - basic_result['quality_score']
        if quality_improvement > 0:
            print(f"  ✅ Quality improved by {quality_improvement:.2f}")
        elif quality_improvement < 0:
            print(f"  ⚠️  Quality decreased by {abs(quality_improvement):.2f}")
        else:
            print(f"  ➡️  Quality unchanged")
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate validation summary."""
        basic_results = results["basic_engine_results"]
        advanced_results = results["advanced_engine_results"]
        
        basic_avg_quality = sum(r['quality_score'] for r in basic_results) / len(basic_results)
        advanced_avg_quality = sum(r['quality_score'] for r in advanced_results) / len(advanced_results)
        
        basic_avg_time = sum(r['processing_time'] for r in basic_results) / len(basic_results)
        advanced_avg_time = sum(r['processing_time'] for r in advanced_results) / len(advanced_results)
        
        basic_success_rate = sum(1 for r in basic_results if r['success']) / len(basic_results)
        advanced_success_rate = sum(1 for r in advanced_results if r['success']) / len(advanced_results)
        
        return {
            "basic_engine": {
                "average_quality_score": basic_avg_quality,
                "average_processing_time": basic_avg_time,
                "success_rate": basic_success_rate
            },
            "advanced_engine": {
                "average_quality_score": advanced_avg_quality,
                "average_processing_time": advanced_avg_time,
                "success_rate": advanced_success_rate
            },
            "improvements": {
                "quality_improvement": advanced_avg_quality - basic_avg_quality,
                "time_difference": advanced_avg_time - basic_avg_time,
                "success_rate_improvement": advanced_success_rate - basic_success_rate
            }
        }
    
    def print_final_summary(self, results: Dict):
        """Print final validation summary."""
        summary = results["comparison_summary"]
        
        print("\n" + "="*60)
        print("🎯 TRANSLATION QUALITY VALIDATION SUMMARY")
        print("="*60)
        
        print(f"\n📈 BASIC ENGINE PERFORMANCE:")
        print(f"  Average Quality Score: {summary['basic_engine']['average_quality_score']:.3f}")
        print(f"  Average Processing Time: {summary['basic_engine']['average_processing_time']:.3f}s")
        print(f"  Success Rate: {summary['basic_engine']['success_rate']:.1%}")
        
        print(f"\n🚀 ADVANCED ENGINE PERFORMANCE:")
        print(f"  Average Quality Score: {summary['advanced_engine']['average_quality_score']:.3f}")
        print(f"  Average Processing Time: {summary['advanced_engine']['average_processing_time']:.3f}s")
        print(f"  Success Rate: {summary['advanced_engine']['success_rate']:.1%}")
        
        print(f"\n📊 IMPROVEMENTS:")
        quality_imp = summary['improvements']['quality_improvement']
        time_diff = summary['improvements']['time_difference']
        success_imp = summary['improvements']['success_rate_improvement']
        
        print(f"  Quality Score: {'+' if quality_imp >= 0 else ''}{quality_imp:.3f}")
        print(f"  Processing Time: {'+' if time_diff >= 0 else ''}{time_diff:.3f}s")
        print(f"  Success Rate: {'+' if success_imp >= 0 else ''}{success_imp:.1%}")
        
        # Overall assessment
        if quality_imp > 0.1:
            print(f"\n✅ VALIDATION PASSED: Significant quality improvements detected!")
        elif quality_imp > 0:
            print(f"\n✅ VALIDATION PASSED: Quality improvements detected.")
        else:
            print(f"\n⚠️  VALIDATION WARNING: No significant quality improvements detected.")


def main():
    """Main validation function."""
    try:
        print("🚀 Starting Translation Quality Validation")
        print("-" * 50)
        
        # Initialize validator
        validator = TranslationQualityValidator()
        
        # Run validation
        results = validator.validate_basic_vs_advanced()
        
        # Print summary
        validator.print_final_summary(results)
        
        # Save results
        results_file = f"translation_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
