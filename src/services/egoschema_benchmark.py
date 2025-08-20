"""
EgoSchema Benchmark Implementation
Based on "EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding"

This module implements the core concepts from the EgoSchema paper to evaluate
very long-form video understanding capabilities in the real-time translator system.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from pathlib import Path
import numpy as np
from datetime import datetime
import uuid
import re


@dataclass
class TemporalCertificate:
    """
    Represents a temporal certificate set - minimum subclips necessary and sufficient
    to verify the correctness of an annotation without watching the rest of the clip.
    """
    start_time: float
    end_time: float
    confidence: float
    description: str

    @property
    def length(self) -> float:
        """Certificate length in seconds"""
        return max(0.0, self.end_time - self.start_time)


@dataclass
class VideoQuestionAnswer:
    """
    Represents a question-answer triplet for very long-form video understanding
    """
    question_id: str
    question: str
    correct_answer: str
    wrong_answers: List[str]
    video_clip_path: str
    temporal_certificates: List[TemporalCertificate]
    difficulty_level: str  # "short", "long-form", "very-long-form"

    @property
    def certificate_length(self) -> float:
        """Total certificate length in seconds"""
        return float(sum(cert.length for cert in self.temporal_certificates))

    @property
    def all_options(self) -> List[str]:
        """All answer options (correct + wrong)."""
        return [self.correct_answer] + list(self.wrong_answers)


class EgoSchemaDataProcessor:
    """
    Processes video data to create EgoSchema-style question-answer pairs
    with temporal certificates for very long-form video understanding evaluation.
    """

    def __init__(self, llm_model: str = "gpt-4", min_certificate_length: float = 30.0):
        self.llm_model = llm_model
        # Minimum 30 seconds as per paper
        self.min_certificate_length = float(min_certificate_length)
        self.logger = logging.getLogger(__name__)

        # Certificate length taxonomy from paper
        self.temporal_taxonomy: Dict[str, Tuple[float, float]] = {
            "short": (0.0, 10.0),  # ~1 second order
            "long-form": (10.0, 60.0),  # ~10 second order
            "very-long-form": (60.0, float("inf")),  # ~100 second order
        }

    def extract_temporal_certificates(
        self,
        video_transcript: str,
        video_duration: float,
        question: str,
        correct_answer: str,
    ) -> List[TemporalCertificate]:
        """
        Extract temporal certificate sets for a given question-answer pair.

        This implements the core concept from the paper - finding the minimum
        set of subclips necessary and sufficient to verify the answer.
        """
        certificates: List[TemporalCertificate] = []

        # Parse transcript to find temporal markers like: [00:00 - 00:30] Some text
        lines = video_transcript.splitlines()
        relevant_segments: List[Tuple[float, float, str]] = []

        ts_pattern = re.compile(r"\[(?P<start>[^\]]+?)\s*-\s*(?P<end>[^\]]+?)\]")

        for line in lines:
            try:
                m = ts_pattern.search(line)
                if not m:
                    continue
                timestamp_part = m.group(0)
                content = line.split("]", 1)[1].strip() if "]" in line else ""

                # Check if content is relevant to question/answer
                if self._is_content_relevant(content, question, correct_answer):
                    start_time, end_time = self._parse_timestamp(timestamp_part)
                    # clip to video duration bounds
                    start_time = max(0.0, min(start_time, float(video_duration)))
                    end_time = max(0.0, min(end_time, float(video_duration)))
                    if end_time > start_time:
                        relevant_segments.append((start_time, end_time, content))
            except Exception as e:
                self.logger.warning(f"Failed to parse timestamp: {line!r}, error: {e}")
                continue

        # Merge nearby segments and create certificates
        merged_segments = self._merge_nearby_segments(relevant_segments)

        for start, end, description in merged_segments:
            cert_length = end - start
            if cert_length >= 0.1:  # Minimum 0.1 second as a sanity threshold
                certificates.append(
                    TemporalCertificate(
                        start_time=start,
                        end_time=end,
                        confidence=0.8,  # Placeholder; can be improved with a model
                        description=description,
                    )
                )

        return certificates

    def _is_content_relevant(self, content: str, question: str, answer: str) -> bool:
        """
        Determine if transcript content is relevant to the question-answer pair.
        Uses simple keyword overlap as a baseline.
        """
        content_lower = content.lower()
        question_lower = question.lower()
        answer_lower = answer.lower()

        # Extract key words from question and answer
        question_words = set(w for w in re.findall(r"\w+", question_lower) if len(w) > 2)
        answer_words = set(w for w in re.findall(r"\w+", answer_lower) if len(w) > 2)
        content_words = set(w for w in re.findall(r"\w+", content_lower) if len(w) > 2)

        # Check for overlap
        question_overlap = len(question_words.intersection(content_words)) > 0
        answer_overlap = len(answer_words.intersection(content_words)) > 0

        return question_overlap or answer_overlap

    def _parse_timestamp(self, timestamp_str: str) -> Tuple[float, float]:
        """Parse timestamp string like [00:30 - 01:00] to seconds"""
        ts = timestamp_str.strip().strip("[]")
        if "-" not in ts:
            raise ValueError(f"Invalid timestamp: {timestamp_str}")
        start_str, end_str = [s.strip() for s in ts.split("-", 1)]

        def time_to_seconds(time_str: str) -> float:
            parts = [p.strip() for p in time_str.split(":")]
            if len(parts) == 1:  # SS
                return float(parts[0])
            if len(parts) == 2:  # MM:SS
                return float(parts[0]) * 60.0 + float(parts[1])
            if len(parts) == 3:  # HH:MM:SS
                return (
                    float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
                )
            raise ValueError(f"Invalid time component: {time_str}")

        return time_to_seconds(start_str), time_to_seconds(end_str)

    def _merge_nearby_segments(
        self, segments: List[Tuple[float, float, str]], merge_threshold: float = 5.0
    ) -> List[Tuple[float, float, str]]:
        """
        Merge segments that are less than merge_threshold seconds apart
        """
        if not segments:
            return []

        # Sort by start time
        segments.sort(key=lambda x: x[0])
        merged: List[Tuple[float, float, str]] = [segments[0]]

        for current in segments[1:]:
            last = merged[-1]
            # If segments are close, merge them
            if current[0] - last[1] <= merge_threshold:
                merged[-1] = (
                    last[0],
                    max(last[1], current[1]),
                    (last[2] + " " + current[2]).strip(),
                )
            else:
                merged.append(current)

        return merged

    def classify_temporal_difficulty(self, certificate_length: float) -> str:
        """
        Classify the temporal difficulty based on certificate length
        Following the taxonomy from the paper
        """
        if certificate_length <= self.temporal_taxonomy["short"][1]:
            return "short"
        if certificate_length <= self.temporal_taxonomy["long-form"][1]:
            return "long-form"
        return "very-long-form"


class EgoSchemaBenchmark:
    """
    Main benchmark class implementing EgoSchema evaluation methodology
    for very long-form video language understanding
    """

    def __init__(self, data_processor: EgoSchemaDataProcessor):
        self.data_processor = data_processor
        self.logger = logging.getLogger(__name__)
        self.benchmark_data: List[VideoQuestionAnswer] = []

    def create_question_answer_pairs(
        self,
        video_transcript: str,
        video_duration: float,
        video_path: str,
        num_questions: int = 3,
    ) -> List[VideoQuestionAnswer]:
        """
        Generate question-answer pairs for a video clip following EgoSchema methodology.

        Args:
            video_transcript: Dense manual narration of the video
            video_duration: Duration in seconds (should be ~180s for 3-minute clips)
            video_path: Path to the video file
            num_questions: Number of questions to generate (default 3 as per paper)
        """
        qa_pairs: List[VideoQuestionAnswer] = []

        # Generate questions using LLM-based approach (simulated)
        questions = self._generate_questions_from_transcript(video_transcript, num_questions)

        for i, (question, correct_answer, wrong_answers) in enumerate(questions):
            question_id = f"{Path(video_path).stem}_q{i + 1}_{uuid.uuid4().hex[:8]}"

            # Extract temporal certificates
            certificates = self.data_processor.extract_temporal_certificates(
                video_transcript, video_duration, question, correct_answer
            )

            # Only include questions with sufficient certificate length
            total_cert_length = sum(cert.length for cert in certificates)
            if total_cert_length >= self.data_processor.min_certificate_length:
                difficulty = self.data_processor.classify_temporal_difficulty(total_cert_length)

                qa_pair = VideoQuestionAnswer(
                    question_id=question_id,
                    question=question,
                    correct_answer=correct_answer,
                    wrong_answers=wrong_answers,
                    video_clip_path=video_path,
                    temporal_certificates=certificates,
                    difficulty_level=difficulty,
                )
                qa_pairs.append(qa_pair)

        return qa_pairs

    def _generate_questions_from_transcript(
        self, transcript: str, num_questions: int
    ) -> List[Tuple[str, str, List[str]]]:
        """
        Generate questions from transcript using LLM-based approach.
        This is a simplified version - in practice would use an LLM API.
        """
        sample_questions: List[Tuple[str, str, List[str]]] = [
            (
                "What is the overarching behavior of the main character in the video?",
                "The person is systematically organizing and categorizing items in their workspace",
                [
                    "The person is randomly moving objects around without purpose",
                    "The person is searching for a specific lost item throughout",
                    "The person is cleaning and tidying up the entire space",
                    "The person is demonstrating how to use various tools",
                ],
            ),
            (
                "How do the character's actions and interactions evolve throughout the video?",
                "Actions become more focused and methodical as the video progresses",
                [
                    "Actions remain consistently chaotic throughout the entire video",
                    "Actions start methodical but become increasingly disorganized",
                    "Actions show no clear pattern or progression over time",
                    "Actions are entirely reactive to external interruptions",
                ],
            ),
            (
                "What long-term goal is the character working towards in this video?",
                "Establishing an efficient workflow system for future activities",
                [
                    "Completing a single immediate task as quickly as possible",
                    "Impressing observers with their organizational skills",
                    "Following a strict predetermined schedule without deviation",
                    "Experimenting with different approaches without commitment",
                ],
            ),
        ]

        return sample_questions[: max(0, min(num_questions, len(sample_questions)))]

    def evaluate_model_performance(
        self, model_predictions: Dict[str, str], ground_truth: List[VideoQuestionAnswer]
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on EgoSchema benchmark

        Args:
            model_predictions: Dict mapping question_id to predicted answer
            ground_truth: List of VideoQuestionAnswer objects with correct answers

        Returns:
            Dictionary with evaluation metrics
        """
        total_questions = len(ground_truth)
        correct_predictions = 0
        results_by_difficulty = {
            "short": {"correct": 0, "total": 0},
            "long-form": {"correct": 0, "total": 0},
            "very-long-form": {"correct": 0, "total": 0},
        }

        certificate_lengths: List[float] = []

        for qa in ground_truth:
            question_id = qa.question_id
            correct_answer = qa.correct_answer
            difficulty = qa.difficulty_level
            cert_length = qa.certificate_length

            certificate_lengths.append(cert_length)
            results_by_difficulty[difficulty]["total"] += 1

            if question_id in model_predictions:
                predicted_answer = model_predictions[question_id]
                if predicted_answer.strip().lower() == correct_answer.strip().lower():
                    correct_predictions += 1
                    results_by_difficulty[difficulty]["correct"] += 1

        # Calculate metrics
        overall_accuracy = (
            correct_predictions / total_questions if total_questions > 0 else 0
        )

        difficulty_accuracies: Dict[str, float] = {}
        for difficulty, stats in results_by_difficulty.items():
            if stats["total"] > 0:
                difficulty_accuracies[difficulty] = stats["correct"] / stats["total"]
            else:
                difficulty_accuracies[difficulty] = 0.0

        return {
            "overall_accuracy": overall_accuracy,
            "accuracy_by_difficulty": difficulty_accuracies,
            "total_questions": total_questions,
            "correct_predictions": correct_predictions,
            "average_certificate_length": float(np.mean(certificate_lengths))
            if certificate_lengths
            else 0.0,
            "median_certificate_length": float(np.median(certificate_lengths))
            if certificate_lengths
            else 0.0,
            "certificate_length_std": float(np.std(certificate_lengths))
            if certificate_lengths
            else 0.0,
            "random_baseline": 0.2,  # 20% for 5-choice questions
            "human_performance": 0.76,  # 76% as reported in paper
            "evaluation_timestamp": datetime.now().isoformat(),
        }

    def save_benchmark_data(self, filepath: str) -> None:
        """Save benchmark data to JSON file"""
        data: Dict[str, Any] = {
            "benchmark_metadata": {
                "creation_timestamp": datetime.now().isoformat(),
                "total_questions": len(self.benchmark_data),
                "min_certificate_length": self.data_processor.min_certificate_length,
            },
            "questions": [],
        }

        for qa in self.benchmark_data:
            data["questions"].append(
                {
                    "question_id": qa.question_id,
                    "question": qa.question,
                    "correct_answer": qa.correct_answer,
                    "wrong_answers": qa.wrong_answers,
                    "video_clip_path": qa.video_clip_path,
                    "difficulty_level": qa.difficulty_level,
                    "certificate_length": qa.certificate_length,
                    "temporal_certificates": [
                        {
                            "start_time": cert.start_time,
                            "end_time": cert.end_time,
                            "confidence": cert.confidence,
                            "description": cert.description,
                        }
                        for cert in qa.temporal_certificates
                    ],
                }
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved benchmark data to {filepath}")

    def load_benchmark_data(self, filepath: str) -> None:
        """Load benchmark data from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.benchmark_data = []
        for q_data in data.get("questions", []):
            certificates = [
                TemporalCertificate(
                    start_time=cert["start_time"],
                    end_time=cert["end_time"],
                    confidence=cert["confidence"],
                    description=cert["description"],
                )
                for cert in q_data.get("temporal_certificates", [])
            ]

            qa = VideoQuestionAnswer(
                question_id=q_data["question_id"],
                question=q_data["question"],
                correct_answer=q_data["correct_answer"],
                wrong_answers=list(q_data["wrong_answers"]),
                video_clip_path=q_data["video_clip_path"],
                temporal_certificates=certificates,
                difficulty_level=q_data["difficulty_level"],
            )
            self.benchmark_data.append(qa)

        self.logger.info(f"Loaded {len(self.benchmark_data)} questions from {filepath}")


# Example usage and integration helper

def integrate_with_translation_system(
    translator_system, video_path: str, transcript: str
) -> Dict[str, Any]:
    """
    Integrate EgoSchema benchmark with the existing translation system
    """
    # Initialize EgoSchema components
    data_processor = EgoSchemaDataProcessor(min_certificate_length=30.0)
    benchmark = EgoSchemaBenchmark(data_processor)

    # Create QA pairs for the video
    qa_pairs = benchmark.create_question_answer_pairs(
        video_transcript=transcript,
        video_duration=180.0,  # 3 minutes as per EgoSchema
        video_path=video_path,
        num_questions=3,
    )

    # Add to benchmark
    benchmark.benchmark_data.extend(qa_pairs)

    # Evaluate current system performance (placeholder)
    # Simulate poor performance
    dummy_predictions = {qa.question_id: qa.wrong_answers[0] for qa in qa_pairs}

    evaluation_results = benchmark.evaluate_model_performance(dummy_predictions, qa_pairs)
    return evaluation_results
    return {
        "qa_pairs": qa_pairs,
        "evaluation_results": evaluation_results,
        "benchmark": benchmark
    }
