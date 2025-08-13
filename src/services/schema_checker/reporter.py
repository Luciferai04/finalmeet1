import json
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

class ReportGenerator:
    """
    Comprehensive report generator for classroom session analysis.
    Generates structured reports in JSON, CSV, and HTML formats.
    """

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_report(self, class_id: str, date: str, comparison_result: Dict[str, Any],
                        schema_data: Optional[Dict] = None,
                        extraction_metadata: Optional[Dict] = None) -> str:
        """
        Generate comprehensive analysis report.

        Args:
            class_id: Unique identifier for the class
            date: Date of the class session
            comparison_result: Result from topic comparison
            schema_data: Original schema data (optional)
            extraction_metadata: Keyword extraction metadata (optional)

        Returns:
            Path to generated report file
        """
        report_data = self._create_comprehensive_report(
            class_id, date, comparison_result, schema_data, extraction_metadata
        )

        # Generate JSON report (primary format)
        json_path = self._save_json_report(class_id, date, report_data)

        # Generate CSV summary (for easy analysis)
        csv_path = self._save_csv_summary(class_id, date, report_data)

        # Generate HTML report (for human readability)
        html_path = self._save_html_report(class_id, date, report_data)

        self.logger.info(
            f"Generated reports for {class_id}_{date}: JSON, CSV, HTML")

        return json_path

    def _create_comprehensive_report(self, class_id: str, date: str, comparison_result: Dict[str, Any],
                                     schema_data: Optional[Dict] = None,
                                     extraction_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create comprehensive report structure."""
        # Calculate additional metrics
        statistics = comparison_result.get('statistics', {})
        matched_topics = comparison_result.get('matched_topics', [])
        missed_topics = comparison_result.get('missed_topics', [])
        unexpected_topics = comparison_result.get('unexpected_topics', [])

        # Enhanced statistics
        coverage_percentage = statistics.get('coverage_score', 0) * 100
        precision_percentage = statistics.get('precision_score', 0) * 100

        # Quality assessment
        quality_score = self._calculate_quality_score(comparison_result)
        session_assessment = self._assess_session_quality(comparison_result)

        report_data = {
            # Session metadata
            "session_info": {
                "class_id": class_id,
                "date": date,
                "report_generated_at": datetime.now().isoformat(),
                "report_version": "2.0"
            },

            # Original schema information
            "schema_info": schema_data or {},

            # Extraction metadata
            "extraction_info": extraction_metadata or {},

            # Core comparison results
            "analysis_results": {
                "matched_topics": matched_topics,
                "missed_topics": missed_topics,
                "unexpected_topics": unexpected_topics
            },

            # Enhanced statistics
            "statistics": {
                **statistics,
                "coverage_percentage": round(coverage_percentage, 1),
                "precision_percentage": round(precision_percentage, 1),
                "quality_score": quality_score,
                "session_assessment": session_assessment
            },

            # Detailed insights
            "insights": {
                "topic_coverage_analysis": self._analyze_topic_coverage(matched_topics, missed_topics),
                "keyword_relevance_analysis": self._analyze_keyword_relevance(unexpected_topics),
                "session_focus_analysis": self._analyze_session_focus(comparison_result),
                "recommendations": self._generate_recommendations(comparison_result)
            },

            # Summary for quick overview
            "summary": {
                "total_topics_planned": len(matched_topics) + len(missed_topics),
                "topics_covered": len(matched_topics),
                "topics_missed": len(missed_topics),
                "additional_topics_discussed": len(unexpected_topics),
                "overall_assessment": session_assessment["overall"],
                "key_achievements": self._extract_key_achievements(comparison_result),
                "areas_for_improvement": self._extract_improvement_areas(comparison_result)
            },

            # Metadata about the analysis
            "metadata": comparison_result.get('metadata', {})
        }

        return report_data

    def _calculate_quality_score(self, comparison_result: Dict[str, Any]) -> float:
        """Calculate overall session quality score (0-100)."""
        stats = comparison_result.get('statistics', {})

        coverage_score = stats.get('coverage_score', 0)
        precision_score = stats.get('precision_score', 0)

        # Weighted combination of coverage and precision
        quality_score = (coverage_score * 0.7 + precision_score * 0.3) * 100

        return round(quality_score, 1)

    def _assess_session_quality(self, comparison_result: Dict[str, Any]) -> Dict[str, str]:
        """Assess session quality in different dimensions."""
        stats = comparison_result.get('statistics', {})
        coverage = stats.get('coverage_score', 0)
        precision = stats.get('precision_score', 0)

        # Coverage assessment
        if coverage >= 0.9:
            coverage_assessment = "Excellent"
        elif coverage >= 0.7:
            coverage_assessment = "Good"
        elif coverage >= 0.5:
            coverage_assessment = "Fair"
        else:
            coverage_assessment = "Needs Improvement"

        # Focus assessment (based on precision)
        if precision >= 0.8:
            focus_assessment = "Highly Focused"
        elif precision >= 0.6:
            focus_assessment = "Well Focused"
        elif precision >= 0.4:
            focus_assessment = "Moderately Focused"
        else:
            focus_assessment = "Needs Better Focus"

        # Overall assessment
        quality_score = self._calculate_quality_score(comparison_result)
        if quality_score >= 85:
            overall = "Excellent"
        elif quality_score >= 70:
            overall = "Good"
        elif quality_score >= 55:
            overall = "Satisfactory"
        else:
            overall = "Needs Improvement"

        return {
            "coverage": coverage_assessment,
            "focus": focus_assessment,
            "overall": overall
        }

    # Placeholder methods for additional logic

    def _analyze_topic_coverage(self, matched_topics: List[Dict], missed_topics: List[Dict]) -> Dict:
        # Analyze topic coverage
        return {}

    def _analyze_keyword_relevance(self, unexpected_topics: List[Dict]) -> Dict:
        # Analyze keyword relevance
        return {}

    def _analyze_session_focus(self, comparison_result: Dict[str, Any]) -> Dict:
        # Analyze session focus
        return {}

    def _generate_recommendations(self, comparison_result: Dict[str, Any]) -> List[Dict]:
        # Generate recommendations for improvement
        return []

    def _extract_key_achievements(self, comparison_result: Dict[str, Any]) -> List[str]:
        # Extract key achievements from the session
        return []

    def _extract_improvement_areas(self, comparison_result: Dict[str, Any]) -> List[str]:
        # Extract areas for improvement
        return []

    def _save_json_report(self, class_id: str, date: str, report_data: Dict[str, Any]) -> str:
        """Save report data to JSON format."""
        json_file = self.reports_dir / f"{class_id}_{date}_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        return str(json_file)

    def _save_csv_summary(self, class_id: str, date: str, report_data: Dict[str, Any]) -> str:
        """Save report summary to CSV format."""
        csv_file = self.reports_dir / f"{class_id}_{date}_summary.csv"
        # Implement CSV saving logic here
        return str(csv_file)

    def _save_html_report(self, class_id: str, date: str, report_data: Dict[str, Any]) -> str:
        """Save report data to HTML format."""
        html_file = self.reports_dir / f"{class_id}_{date}_report.html"
        # Implement HTML saving logic here
        return str(html_file)

# Sample usage
if __name__ == "__main__":
    generator = ReportGenerator()
    # Assuming comparison_result is provided
    # json_path = generator.generate_report("class1", "2023-10-16", comparison_result)
