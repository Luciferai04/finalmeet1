"""
Report Generator Module for Schema Checker Pipeline

This module generates detailed JSON reports from topic comparison results,
including match statistics, coverage metrics, and actionable insights.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import os


class ReportGenerator:
    """Generates comprehensive JSON reports from topic comparison results."""

    def __init__(self, output_dir: str = "reports"):
        pass
    """
 Initialize the report generator.

 Args:
 output_dir: Directory to save generated reports
 """
    self.output_dir = output_dir
    self._ensure_output_dir()

    def _ensure_output_dir(self):
        pass
    """Create output directory if it doesn't exist."""
    if not os.path.exists(self.output_dir):
        pass
    os.makedirs(self.output_dir)

    def generate_report(
        self,
        comparison_result: Dict[str, Any],
        class_info: Dict[str, Any],
        transcript_data: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> Dict[str, Any]:
    """
 Generate a comprehensive JSON report from comparison results, including meta-cognitive insights.

 Args:
 comparison_result: Result from TopicComparator.compare_topics()
 class_info: Information about the class (id, date, etc.)
 transcript_data: Original transcript and extracted keywords
 output_filename: Optional custom filename for the report

 Returns:
 Generated report as dictionary
 """
    # Generate report data
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "report_version": "1.0",
            "pipeline_version": "1.0"
        },
        "class_information": {
            "class_id": class_info.get("class_id", "unknown"),
            "date": class_info.get("date", datetime.now().strftime("%Y-%m-%d")),
            "duration_minutes": class_info.get("duration", 0),
            "instructor": class_info.get("instructor", "N/A"),
            "subject": class_info.get("subject", "N/A")
        },
        "transcript_summary": {
            "total_words": len(transcript_data.get("transcript", "").split()),
            "total_keywords_extracted": len(transcript_data.get("keywords", [])),
            "extraction_method": transcript_data.get("extraction_method", "RAKE"),
            "language": transcript_data.get("language", "en")
        },
        "topic_analysis": {
            "expected_topics": comparison_result.get("expected_topics", []),
            "covered_topics": comparison_result.get("covered_topics", []),
            "missed_topics": comparison_result.get("missed_topics", []),
            "unexpected_topics": comparison_result.get("unexpected_topics", [])
        },
        "coverage_metrics": self._calculate_coverage_metrics(comparison_result),
        "detailed_matches": comparison_result.get("detailed_matches", []),
        "recommendations": self._generate_recommendations(comparison_result),
        "quality_indicators": self._calculate_quality_indicators(comparison_result, transcript_data),
        "meta_cognitive_insights": self._generate_meta_cognitive_insights(comparison_result, transcript_data)
    }

    # Save report if filename provided
    if output_filename:
        pass
    self._save_report(report, output_filename)

    return report

    def _calculate_coverage_metrics(
            self, comparison_result: Dict[str, Any]) -> Dict[str, float]:
    """Calculate topic coverage metrics."""
    expected_count = len(comparison_result.get("expected_topics", []))
    covered_count = len(comparison_result.get("covered_topics", []))
    missed_count = len(comparison_result.get("missed_topics", []))
    unexpected_count = len(comparison_result.get("unexpected_topics", []))

    coverage_percentage = (
        covered_count /
        expected_count *
        100) if expected_count > 0 else 0

    return {
        "coverage_percentage": round(coverage_percentage, 2),
        "topics_covered": covered_count,
        "topics_missed": missed_count,
        "topics_expected": expected_count,
        "unexpected_topics_found": unexpected_count,
        "coverage_score": round(coverage_percentage / 100, 3)
    }

    def _generate_recommendations(
            self, comparison_result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []

    missed_topics = comparison_result.get("missed_topics", [])
    unexpected_topics = comparison_result.get("unexpected_topics", [])
    coverage_metrics = self._calculate_coverage_metrics(comparison_result)

    # Coverage-based recommendations
    if coverage_metrics["coverage_percentage"] < 70:
        pass
    recommendations.append({
        "type": "coverage_improvement",
        "priority": "high",
        "message": f"Low topic coverage ({coverage_metrics['coverage_percentage']:.1f}%). Consider reviewing lesson plan or extending class time.",
        "action": "Review curriculum alignment and pacing"
    })

    # Missed topics recommendations
    if missed_topics:
        pass
    recommendations.append({
        "type": "missed_content",
        "priority": "medium",
        "message": f"Missed {len(missed_topics)} expected topics: {', '.join(missed_topics[:3])}{'...' if len(missed_topics) > 3 else ''}",
        "action": "Plan follow-up session or allocate additional time"
    })

    # Unexpected topics analysis
    if unexpected_topics:
        pass
    recommendations.append({
        "type": "content_deviation",
        "priority": "low",
        "message": f"Covered {len(unexpected_topics)} unplanned topics. May indicate student questions or tangential discussions.",
        "action": "Review if these topics should be added to curriculum"
    })

    # Positive feedback
    if coverage_metrics["coverage_percentage"] >= 90:
        pass
    recommendations.append({
        "type": "positive_feedback",
        "priority": "info",
        "message": "Excellent topic coverage! Class objectives were well met.",
        "action": "Continue current teaching approach"
    })

    return recommendations

    def _calculate_quality_indicators(
        self,
        comparison_result: Dict[str, Any],
        transcript_data: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Calculate quality indicators for the class session."""
    keywords = transcript_data.get("keywords", [])
    transcript_length = len(transcript_data.get("transcript", "").split())

    # Keyword density
    keyword_density = len(keywords) / \
        transcript_length if transcript_length > 0 else 0

    # Topic distribution score
    covered_topics = comparison_result.get("covered_topics", [])
    expected_topics = comparison_result.get("expected_topics", [])
    distribution_score = len(set(covered_topics)) / \
        len(set(expected_topics)) if expected_topics else 0

    return {
        "keyword_density": round(keyword_density * 100, 2),
        "topic_distribution_score": round(distribution_score, 3),
        "content_richness": "high" if keyword_density > 0.05 else "medium" if keyword_density > 0.02 else "low",
        "session_effectiveness": "excellent" if distribution_score > 0.9 else "good" if distribution_score > 0.7 else "needs_improvement"
    }

    def _generate_meta_cognitive_insights(
        self,
        comparison_result: Dict[str, Any],
        transcript_data: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Generate meta-cognitive insights for the session."""
    covered_topics = comparison_result.get("covered_topics", [])
    missed_topics = comparison_result.get("missed_topics", [])

    recurring_missed_topics = [
        topic for topic in missed_topics if missed_topics.count(topic) > 1]

    insights = {
        "confidence_assessment": {
            "average_confidence": sum(match.get("confidence", 0) for match in comparison_result.get("detailed_matches", [])) / len(comparison_result.get("detailed_matches", [])),
            "confidence_trend": "improving" if sum(match.get("confidence", 0) for match in comparison_result.get("detailed_matches", [])) / len(comparison_result.get("detailed_matches", [])) > 0.8 else "needs_attention"
        },
        "teaching_progress_notes": {
            "recurring_missed_topics": recurring_missed_topics,
            "coverage_improvement_advice": "Focus more on these topics in upcoming sessions"
        }
    }

    return insights

    def _save_report(self, report: Dict[str, Any], filename: str):
        pass
    """Save report to file."""
    filepath = os.path.join(self.output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        pass
    json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {filepath}")

    def generate_summary_report(
            self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary report from multiple class reports."""
    if not reports:
        pass
    return {"error": "No reports provided"}

    # Aggregate metrics
    total_classes = len(reports)
    avg_coverage = sum(r["coverage_metrics"]["coverage_percentage"]
                       for r in reports) / total_classes

    # Find common missed topics
    all_missed = []
    for report in reports:
        pass
    all_missed.extend(report["topic_analysis"]["missed_topics"])

    from collections import Counter
    common_missed = Counter(all_missed).most_common(5)

    summary = {
        "summary_metadata": {
            "generated_at": datetime.now().isoformat(),
            "classes_analyzed": total_classes,
            "date_range": {
                "start": min(r["class_information"]["date"] for r in reports),
                "end": max(r["class_information"]["date"] for r in reports)
            }
        },
        "aggregate_metrics": {
            "average_coverage_percentage": round(avg_coverage, 2),
            "total_classes": total_classes,
            "classes_above_80_percent": sum(1 for r in reports if r["coverage_metrics"]["coverage_percentage"] >= 80),
            "classes_below_50_percent": sum(1 for r in reports if r["coverage_metrics"]["coverage_percentage"] < 50)
        },
        "common_issues": {
            "frequently_missed_topics": [{"topic": topic, "frequency": count} for topic, count in common_missed],
            "average_missed_per_class": sum(len(r["topic_analysis"]["missed_topics"]) for r in reports) / total_classes
        },
        "trends": self._analyze_trends(reports)
    }

    return summary

    def _analyze_trends(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass
    """Analyze trends across multiple reports."""
    # Sort reports by date
    sorted_reports = sorted(
        reports, key=lambda x: x["class_information"]["date"])

    if len(sorted_reports) < 2:
        pass
    return {"message": "Insufficient data for trend analysis"}

    # Calculate coverage trend
    coverages = [r["coverage_metrics"]["coverage_percentage"]
                 for r in sorted_reports]
    trend_direction = "improving" if coverages[-1] > coverages[0] else "declining" if coverages[-1] < coverages[0] else "stable"

    return {
        "coverage_trend": trend_direction,
        "coverage_change": round(coverages[-1] - coverages[0], 2),
        "most_consistent_topics": self._find_consistent_topics(sorted_reports),
        "most_problematic_topics": self._find_problematic_topics(sorted_reports)
    }

    def _find_consistent_topics(
            self, reports: List[Dict[str, Any]]) -> List[str]:
    """Find topics that are consistently covered."""
    topic_coverage = {}
    for report in reports:
        pass
    for topic in report["topic_analysis"]["covered_topics"]:
        pass
    topic_coverage[topic] = topic_coverage.get(topic, 0) + 1

    total_classes = len(reports)
    consistent_topics = [topic for topic, count in topic_coverage.items()
                         if count >= total_classes * 0.8]

    return consistent_topics

    def _find_problematic_topics(
            self, reports: List[Dict[str, Any]]) -> List[str]:
    """Find topics that are frequently missed."""
    topic_misses = {}
    for report in reports:
        pass
    for topic in report["topic_analysis"]["missed_topics"]:
        pass
    topic_misses[topic] = topic_misses.get(topic, 0) + 1

    total_classes = len(reports)
    problematic_topics = [topic for topic, count in topic_misses.items()
                          if count >= total_classes * 0.5]

    return problematic_topics


# Example usage and testing
if __name__ == "__main__":
    pass
    # Example usage
    generator = ReportGenerator()

    # Sample data
    sample_comparison = {
        "expected_topics": ["python basics", "functions", "loops", "data structures"],
        "covered_topics": ["python basics", "functions", "loops"],
        "missed_topics": ["data structures"],
        "unexpected_topics": ["debugging", "best practices"],
        "detailed_matches": [
            {"expected": "python basics",
             "matched": "python basics",
             "confidence": 0.95},
            {"expected": "functions", "matched": "functions", "confidence": 0.88},
            {"expected": "loops", "matched": "loops", "confidence": 0.92}
        ]
    }

    sample_class_info = {
        "class_id": "CS101_001",
        "date": "2024-01-15",
        "duration": 90,
        "instructor": "Dr. Smith",
        "subject": "Introduction to Programming"
    }

    sample_transcript = {
        "transcript": "Today we covered python basics including variables and data types. We also discussed functions and how to define them. We spent time on loops and iteration. Students asked about debugging techniques.",
        "keywords": ["python", "basics", "variables", "data types", "functions", "define", "loops", "iteration", "debugging", "techniques"],
        "extraction_method": "RAKE",
        "language": "en"
    }

    # Generate report
    report = generator.generate_report(
        sample_comparison,
        sample_class_info,
        sample_transcript,
        "sample_class_report.json"
    )

    print("Sample report generated successfully!")
    print(
        f"Coverage: {
            report['coverage_metrics']['coverage_percentage']:.1f}%")
    print(f"Recommendations: {len(report['recommendations'])}")
