"""
EgoSchema LLM-based Question Generation
Implements the chained LLM prompting approach from the EgoSchema paper
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
import time
import random


class EgoSchemaLLMGenerator:
    """
    Implements the LLM-based question generation pipeline from EgoSchema paper.
    Uses chained prompting: Q(AW)-shot approach for optimal quality and cost.
    """

    def __init__(self, llm_model: str = "gpt-4", temperature: float = 0.7):
        self.llm_model = llm_model
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

        # Prompts based on the paper's methodology
        self.question_generation_prompt = self._get_question_generation_prompt()
        self.answer_generation_prompts = self._get_answer_generation_prompts()
        self.blind_filtering_prompt = self._get_blind_filtering_prompt()

    def _get_question_generation_prompt(self) -> str:
        """
        Question generation prompt based on the paper's approach.
        Modified from Figure 5 in the paper.
        """
        return (
            "I want you to act as a teacher in the class called \"Long-term video understanding\". "
            "I will provide video action narrations and their timestamps and you will generate three highly difficult "
            "and diverse questions that test students' ability to understand very long-term temporal patterns, "
            "overarching behaviors, and abstract reasoning about video content.\n\n"
            "Your questions should:\n"
            "1. Require understanding of the ENTIRE video duration (not just short segments)\n"
            "2. Test abstract reasoning about overarching behaviors and intentions\n"
            "3. Focus on long-term patterns, progressions, and character development\n"
            "4. Require synthesis of information across multiple time segments\n"
            "5. Be impossible to answer correctly using only a few frames or short clips\n\n"
            "The questions should assess:\n"
            "- Overarching behavior and intentions of characters\n"
            "- How actions and interactions evolve throughout the video\n"
            "- Long-term goals and planning\n"
            "- Temporal progressions and developments\n"
            "- Abstract reasoning about the video narrative\n\n"
            "Please label each question as \"Question 1:\", \"Question 2:\", \"Question 3:\" followed by the full question.\n\n"
            "Timestamps and narrations:\n{narrations}\n\n"
            "Generate exactly 3 questions that require very long-term video understanding:"
        )

    def _get_answer_generation_prompts(self) -> List[str]:
        """
        Multiple answer generation prompts to avoid bias (as mentioned in paper).
        Uses Q(AW)-shot approach.
        """
        prompts = [
            (
                "I want you to create a difficult multiple-choice exam that tests the above student abilities based on the three questions I just provided. "
                "Each question should have five similar open-ended but short answers where:\n\n"
                "1. Only ONE answer is completely correct\n"
                "2. The other four answers are plausible but wrong (hard negatives)\n"
                "3. Wrong answers should be semantically similar to the correct answer\n"
                "4. All options should be concise (1-2 sentences maximum)\n"
                "5. Wrong answers should not be obviously incorrect\n\n"
                "For each question, provide:\n"
                "- The correct answer\n"
                "- Four wrong but plausible answers\n\n"
                "Format your response as:\n"
                "Question 1: [question text]\n"
                "Correct answer: [correct answer text]\n"
                "Wrong answer 1: [wrong answer text]\n"
                "Wrong answer 2: [wrong answer text]\n"
                "Wrong answer 3: [wrong answer text]\n"
                "Wrong answer 4: [wrong answer text]\n\n"
                "[Repeat for Questions 2 and 3]\n\n"
                "Timestamps and narrations:\n{narrations}\n\n"
                "Questions:\n{questions}"
            ),
            (
                "Based on the provided questions about long-term video understanding, create five answer choices for each question where one is correct and four are challenging distractors.\n\n"
                "Requirements:\n"
                "- Correct answer must accurately reflect the video content\n"
                "- Wrong answers should be believable but factually incorrect\n"
                "- Avoid obviously wrong options\n"
                "- Keep answers concise and focused\n"
                "- Ensure distractors test different aspects of understanding\n\n"
                "Format each question's answers as:\n"
                "Question [N]: [question]\n"
                "A) [option A]\n"
                "B) [option B]\n"
                "C) [option C]\n"
                "D) [option D]\n"
                "E) [option E]\n"
                "Correct: [letter of correct answer]\n\n"
                "Video content:\n{narrations}\n\n"
                "Questions to answer:\n{questions}"
            ),
            (
                "Create multiple choice answers for the given long-term video understanding questions. Each question needs exactly 5 options with only one being correct.\n\n"
                "Design principles:\n"
                "- Correct answer demonstrates deep video understanding\n"
                "- Wrong answers are plausible misconceptions\n"
                "- Options should require careful video analysis to distinguish\n"
                "- Avoid trivial or obviously incorrect choices\n\n"
                "Structure:\n"
                "For each question, provide the correct answer and 4 wrong answers.\n"
                "Clearly mark which is correct.\n\n"
                "Content reference:\n{narrations}\n\n"
                "Questions requiring answers:\n{questions}"
            ),
        ]
        return prompts

    def _get_blind_filtering_prompt(self) -> str:
        """
        Blind filtering prompt to ensure questions require visual grounding.
        Based on the paper's filtering methodology.
        """
        return (
            "Given a question and five answer choices, please try to provide the most probable answer in your opinion without any additional context. "
            "Your output should be just one of A, B, C, D, E and nothing else.\n\n"
            "This is a test to see if the question can be answered without watching the video. If you can confidently choose an answer, the question may not require visual understanding.\n\n"
            "Question: {question}\n"
            "Option A: {option_a}\n"
            "Option B: {option_b}\n"
            "Option C: {option_c}\n"
            "Option D: {option_d}\n"
            "Option E: {option_e}\n\n"
            "Answer (just the letter):"
        )

    def generate_questions_from_narrations(self, narrations: str, num_questions: int = 3) -> List[Dict[str, str]]:
        """
        Generate questions from video narrations using LLM (simulated).
        Implements the Q(AW)-shot approach from the paper.
        """
        try:
            # Step 1: Generate questions using the question generation prompt
            question_prompt = self.question_generation_prompt.format(narrations=narrations)
            questions_response = self._call_llm(question_prompt)

            # Parse questions from response
            questions = self._parse_questions_from_response(questions_response)
            if not questions:
                self.logger.warning("No questions generated from LLM response")
                return []

            # Step 2: Generate answers for each question using Q(AW)-shot
            qa_pairs: List[Dict[str, str]] = []
            for question in questions[: num_questions]:
                # Use random prompt to avoid bias
                answer_prompt_template = random.choice(self.answer_generation_prompts)
                answer_prompt = answer_prompt_template.format(narrations=narrations, questions=question)
                answers_response = self._call_llm(answer_prompt)
                parsed_answers = self._parse_answers_from_response(answers_response, question)
                if parsed_answers:
                    qa_pairs.append(parsed_answers)

            # Step 3: Apply blind filtering
            filtered_qa_pairs: List[Dict[str, str]] = []
            for qa in qa_pairs:
                if self._passes_blind_filter(qa):
                    filtered_qa_pairs.append(qa)
                else:
                    self.logger.info(
                        f"Question filtered out by blind test: {qa['question'][:50]}..."
                    )

            return filtered_qa_pairs
        except Exception as e:
            self.logger.error(f"Error generating questions: {e}")
            return []

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with the given prompt. Placeholder implementation.
        """
        pl = prompt.lower()
        if "generate three highly difficult" in pl or "generate exactly 3 questions" in pl:
            return self._generate_sample_questions()
        if "multiple-choice exam" in pl or "answer choices" in pl or "five answer choices" in pl:
            return self._generate_sample_answers()
        # Blind filtering case
        return "A"

    def _generate_sample_questions(self) -> str:
        """Generate realistic sample questions for demonstration"""
        return (
            "Question 1: What is the overarching behavioral pattern that characterizes the main person's approach throughout the entire video sequence?\n\n"
            "Question 2: How do the person's interactions with objects and environment evolve from the beginning to the end of the video, and what does this reveal about their underlying intentions?\n\n"
            "Question 3: What long-term goal or objective drives the person's actions across the full duration of the video, and how do their strategies adapt to achieve this goal?"
        )

    def _generate_sample_answers(self) -> str:
        """Generate realistic sample answers for demonstration"""
        return (
            "Question 1: What is the overarching behavioral pattern that characterizes the main person's approach throughout the entire video sequence?\n"
            "Correct answer: The person demonstrates systematic problem-solving by methodically analyzing situations before taking action\n"
            "Wrong answer 1: The person shows impulsive decision-making and reacts quickly to immediate stimuli\n"
            "Wrong answer 2: The person exhibits repetitive behaviors without clear purpose or direction\n"
            "Wrong answer 3: The person alternates between focused work and distracted exploration randomly\n"
            "Wrong answer 4: The person follows a rigid predetermined sequence without adaptation\n\n"
            "Question 2: How do the person's interactions with objects and environment evolve from the beginning to the end of the video?\n"
            "Correct answer: Interactions become increasingly efficient and purposeful as the person develops better understanding\n"
            "Wrong answer 1: Interactions remain consistently chaotic and disorganized throughout the video\n"
            "Wrong answer 2: Interactions start purposeful but become increasingly random and unfocused\n"
            "Wrong answer 3: Interactions show no clear pattern or development over the time period\n"
            "Wrong answer 4: Interactions are entirely reactive to environmental changes without planning\n\n"
            "Question 3: What long-term goal drives the person's actions across the full duration of the video?\n"
            "Correct answer: Establishing an organized system that will facilitate future similar activities\n"
            "Wrong answer 1: Completing a single immediate task as quickly as possible without regard for future\n"
            "Wrong answer 2: Demonstrating competence to impress potential observers or evaluators\n"
            "Wrong answer 3: Following strict external instructions without personal judgment or adaptation\n"
            "Wrong answer 4: Experimenting with different approaches without commitment to any particular outcome"
        )

    def _parse_questions_from_response(self, response: str) -> List[str]:
        """Parse questions from LLM response"""
        questions: List[str] = []
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("question"):
                # Extract question text after the label
                m = re.search(r"Question\s*\d+:?\s*(.*)", line, re.IGNORECASE)
                if m:
                    q = m.group(1).strip()
                    if q:
                        questions.append(q)
        return questions

    def _parse_answers_from_response(self, response: str, question: str) -> Optional[Dict[str, str]]:
        """Parse answers from LLM response"""
        try:
            correct_answer: Optional[str] = None
            wrong_answers: List[str] = []

            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("correct answer:"):
                    correct_answer = line.split(":", 1)[1].strip()
                elif line.lower().startswith("wrong answer"):
                    wa = re.sub(r"Wrong answer\s*\d+:\s*", "", line, flags=re.IGNORECASE).strip()
                    if wa:
                        wrong_answers.append(wa)

            if correct_answer and len(wrong_answers) >= 4:
                return {
                    "question": question,
                    "correct_answer": correct_answer,
                    "wrong_answers": wrong_answers[:4],
                }
            return None
        except Exception as e:
            self.logger.error(f"Error parsing answers: {e}")
            return None

    def _passes_blind_filter(self, qa_pair: Dict[str, str]) -> bool:
        """
        Apply blind filtering to ensure questions require visual grounding.
        Returns True if question passes the filter (requires video watching).
        """
        try:
            question = qa_pair["question"]
            correct_answer = qa_pair["correct_answer"]
            wrong_answers = qa_pair["wrong_answers"]

            # Shuffle options
            all_options = [correct_answer] + list(wrong_answers)
            random.shuffle(all_options)

            # Create blind filtering prompt
            prompt = self.blind_filtering_prompt.format(
                question=question,
                option_a=all_options[0],
                option_b=all_options[1],
                option_c=all_options[2],
                option_d=all_options[3],
                option_e=all_options[4],
            )

            # Get LLM prediction without video context
            blind_response = self._call_llm(prompt).strip().upper()

            # Find which option the LLM chose
            chosen_option = None
            if blind_response in ["A", "B", "C", "D", "E"]:
                chosen_index = ord(blind_response) - ord("A")
                chosen_option = all_options[chosen_index]

            # If LLM correctly guessed the answer without video, filter out the question
            if chosen_option == correct_answer:
                self.logger.info("Question filtered: can be answered without video")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error in blind filtering: {e}")
            return True  # Default to keeping the question if filtering fails


class EgoSchemaDataPipeline:
    """
    Complete data pipeline implementing the 4-stage process from the EgoSchema paper:
    Stage I: Raw Data Filtering
    Stage II: QA Generation
    Stage III: QA Filtering
    Stage IV: Manual QA Curation
    """

    def __init__(self, llm_generator: EgoSchemaLLMGenerator):
        self.llm_generator = llm_generator
        self.logger = logging.getLogger(__name__)

    def stage_1_filter_raw_data(
        self, video_data: List[Dict], min_duration: float = 180.0, min_narrations: int = 30
    ) -> List[Dict]:
        """
        Stage I: Filter video data for 3-minute clips with sufficient narration density.
        """
        filtered_data: List[Dict] = []
        for video in video_data:
            duration = video.get("duration", 0)
            narration_count = len(video.get("narrations", []))
            if duration >= min_duration and narration_count >= min_narrations:
                filtered_data.append(video)
        self.logger.info(
            f"Stage I: Filtered {len(filtered_data)} videos from {len(video_data)} total"
        )
        return filtered_data

    def stage_2_generate_qa(self, filtered_videos: List[Dict]) -> List[Dict]:
        """
        Stage II: Generate QA pairs using LLM-based approach.
        """
        all_qa_pairs: List[Dict] = []
        for video in filtered_videos:
            narrations = video.get("narrations_text", "")
            video_path = video.get("path", "")
            qa_pairs = self.llm_generator.generate_questions_from_narrations(narrations)
            for qa in qa_pairs:
                qa = dict(qa)
                qa["video_path"] = video_path
                qa["video_duration"] = video.get("duration", 180.0)
                all_qa_pairs.append(qa)
        self.logger.info(f"Stage II: Generated {len(all_qa_pairs)} QA pairs")
        return all_qa_pairs

    def stage_3_filter_qa(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Stage III: Apply rule-based and LLM-based filtering.
        """
        filtered_pairs: List[Dict] = []
        for qa in qa_pairs:
            # Rule-based filtering
            if self._passes_rule_based_filter(qa):
                # LLM-based filtering (blind filter) already applied in generation
                filtered_pairs.append(qa)
        self.logger.info(f"Stage III: Filtered to {len(filtered_pairs)} QA pairs")
        return filtered_pairs

    def _passes_rule_based_filter(self, qa_pair: Dict) -> bool:
        """
        Rule-based filtering to remove malformed or problematic QA pairs.
        """
        question = qa_pair.get("question", "")
        correct_answer = qa_pair.get("correct_answer", "")
        wrong_answers = qa_pair.get("wrong_answers", [])

        # Check for minimum content length
        if len(question) < 10 or len(correct_answer) < 5:
            return False

        # Check for sufficient wrong answers
        if len(wrong_answers) < 4:
            return False

        # Check for keyword bleeding from prompts
        bad_keywords = ["long-term", "narrations", "timestamp", "LLM", "AI"]
        text_to_check = (question + " " + correct_answer + " " + " ".join(wrong_answers)).lower()
        for keyword in bad_keywords:
            if keyword.lower() in text_to_check:
                return False

        return True

    def stage_4_prepare_for_curation(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Stage IV: Prepare data for manual curation.
        Add metadata and structure for human annotators.
        """
        curation_data: List[Dict] = []
        for qa in qa_pairs:
            curation_item = {
                "question_id": f"ego_{len(curation_data) + 1:04d}",
                "question": qa["question"],
                "correct_answer": qa["correct_answer"],
                "wrong_answers": qa["wrong_answers"],
                "video_path": qa.get("video_path", ""),
                "video_duration": qa.get("video_duration", 180.0),
                "curation_status": "pending",
                "curation_notes": "",
                "estimated_certificate_length": 0,  # To be filled by annotators
                "created_timestamp": time.time(),
            }
            curation_data.append(curation_item)
        self.logger.info(f"Stage IV: Prepared {len(curation_data)} items for curation")
        return curation_data

    def run_complete_pipeline(self, video_data: List[Dict]) -> List[Dict]:
        """
        Run the complete 4-stage EgoSchema data pipeline.
        """
        self.logger.info("Starting EgoSchema data pipeline...")
        # Stage I: Raw Data Filtering
        filtered_videos = self.stage_1_filter_raw_data(video_data)
        # Stage II: QA Generation
        qa_pairs = self.stage_2_generate_qa(filtered_videos)
        # Stage III: QA Filtering
        filtered_qa = self.stage_3_filter_qa(qa_pairs)
        # Stage IV: Prepare for Manual Curation
        curation_ready = self.stage_4_prepare_for_curation(filtered_qa)
        self.logger.info(
            f"Pipeline complete: {len(curation_ready)} QA pairs ready for curation"
        )
        return curation_ready
