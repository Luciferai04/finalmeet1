from .reporter import ReportGenerator as Reporter
from .comparator import TopicComparator as Comparator
import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .schema_parser import SchemaParser
from .keyword_extractor import KeywordExtractor
from .comparator import TopicComparator
from .reporter import ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schema_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SchemaCheckerPipeline:
    """
    Complete automated pipeline for classroom session evaluation.
    Handles schema parsing, keyword extraction, topic comparison, and reporting.
    """

    def __init__(self,
                 similarity_threshold: float = 0.6,
                 extraction_method: str = "hybrid",
                 reports_dir: str = "reports",
                 schemas_dir: str = "schemas/normalized"):

        self.similarity_threshold = similarity_threshold
        self.extraction_method = extraction_method

        # Initialize components
        self.schema_parser = SchemaParser(schemas_dir)
        self.keyword_extractor = KeywordExtractor()
        self.comparator = Comparator(similarity_threshold)
        self.reporter = Reporter(reports_dir)

        # Create directories
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        Path(schemas_dir).mkdir(parents=True, exist_ok=True)
        Path("transcripts").mkdir(parents=True, exist_ok=True)

        logger.info("Schema Checker Pipeline initialized")

    def process_session(self,
                        schema_file: str,
                        transcript_file: str,
                        class_id: Optional[str] = None,
                        date: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a complete classroom session with error handling.
        
        Args:
            schema_file: Path to schema file (JSON, YAML, or CSV)
            transcript_file: Path to transcript text file
            class_id: Override class ID (optional)
            date: Override date (optional)

        Returns:
            Dictionary with processing results and paths
        """
        logger.info(f"Processing session: {schema_file} -> {transcript_file}")

        try:
            # Step 1: Parse and normalize schema
            logger.info("Step 1: Parsing schema file")
            try:
                schema_data = self.schema_parser.parse_schema(schema_file)
            except Exception as e:
                logger.error(f"Failed to parse schema file: {e}")
                raise

            # Override class_id and date if provided
            if class_id:
                schema_data['class_id'] = class_id
            if date:
                schema_data['date'] = date

            # Validate required fields
            if not schema_data.get('class_id') or not schema_data.get('date'):
                raise ValueError("Schema must contain class_id and date")

            expected_topics = schema_data.get('expected_topics', [])
            if not expected_topics:
                logger.warning("No expected topics found in schema")

            # Save normalized schema
            normalized_schema_path = self.schema_parser.save_normalized_schema(schema_data)
            logger.info(f"Normalized schema saved: {normalized_schema_path}")

            # Step 2: Read transcript
            logger.info("Step 2: Reading transcript")
            try:
                if not os.path.exists(transcript_file):
                    raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
                
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_text = f.read().strip()

                if not transcript_text:
                    raise ValueError("Transcript file is empty")
                    
            except Exception as e:
                logger.error(f"Error opening transcript file: {e}")
                raise

            # Step 3: Extract keywords
            logger.info("Step 3: Extracting keywords from transcript")
            extraction_result = self.keyword_extractor.extract_keywords(transcript_text)
            extracted_keywords = extraction_result

            logger.info(f"Extracted {len(extracted_keywords)} keywords using {self.extraction_method} method")

            # Step 4: Compare topics
            logger.info("Step 4: Comparing expected topics with extracted keywords")
            comparison_result = self.comparator.compare_topics(
                expected_topics,
                extracted_keywords
            )

            # Step 5: Generate comprehensive report
            logger.info("Step 5: Generating comprehensive report")
            report_path = self.reporter.generate_report(
                schema_data['class_id'],
                schema_data['date'],
                comparison_result,
                schema_data,
                {}
            )

            # Prepare result summary
            result = {
                'success': True,
                'class_id': schema_data['class_id'],
                'date': schema_data['date'],
                'schema_file': schema_file,
                'transcript_file': transcript_file,
                'normalized_schema_path': normalized_schema_path,
                'report_path': report_path,
                'statistics': comparison_result.get('statistics', {}),
                'processing_time': datetime.now().isoformat()
            }

            # Log summary
            stats = comparison_result.get('statistics', {})
            logger.info(f"Session processed successfully:")
            logger.info(f" - Class: {schema_data['class_id']} ({schema_data['date']})")
            logger.info(f" - Topics covered: {stats.get('topics_covered', 0)}/{stats.get('total_expected_topics', 0)}")
            logger.info(f" - Coverage: {stats.get('coverage_percentage', 0):.1f}%")
            logger.info(f" - Quality score: {stats.get('quality_score', 0):.1f}/100")
            logger.info(f" - Report saved: {report_path}")

            return result

        except Exception as e:
            logger.error(f"Error processing session: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'schema_file': schema_file,
                'transcript_file': transcript_file,
                'processing_time': datetime.now().isoformat()
            }

    def batch_process(self,
                      schema_dir: str,
                      transcript_dir: str,
                      pattern_matching: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple sessions in batch.

        Args:
            schema_dir: Directory containing schema files
            transcript_dir: Directory containing transcript files
            pattern_matching: Whether to match files by name patterns

        Returns:
            List of processing results
        """
        logger.info(f"Starting batch processing: {schema_dir} -> {transcript_dir}")
        results = []

        try:
            schema_files = list(Path(schema_dir).glob("*.json")) + \
                          list(Path(schema_dir).glob("*.yaml")) + \
                          list(Path(schema_dir).glob("*.yml")) + \
                          list(Path(schema_dir).glob("*.csv"))

            transcript_files = list(Path(transcript_dir).glob("*.txt"))

            if not schema_files:
                logger.warning(f"No schema files found in {schema_dir}")
                return results

            if not transcript_files:
                logger.warning(f"No transcript files found in {transcript_dir}")
                return results

            # Match files by pattern if requested
            if pattern_matching:
                pairs = self._match_files_by_pattern(schema_files, transcript_files)
            else:
                # Simple pairing by order
                pairs = list(zip(schema_files, transcript_files[:len(schema_files)]))

            logger.info(f"Found {len(pairs)} file pairs to process")

            for schema_file, transcript_file in pairs:
                logger.info(f"Processing pair: {schema_file.name} + {transcript_file.name}")
                result = self.process_session(str(schema_file), str(transcript_file))
                results.append(result)

            # Log batch summary
            successful = len([r for r in results if r.get('success', False)])
            failed = len(results) - successful
            
            logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
            
            return results

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return results

    def _match_files_by_pattern(self, schema_files, transcript_files):
        """Match schema and transcript files by similar names"""
        pairs = []
        
        for schema_file in schema_files:
            schema_stem = schema_file.stem.lower()
            
            # Find best matching transcript
            best_match = None
            best_score = 0
            
            for transcript_file in transcript_files:
                transcript_stem = transcript_file.stem.lower()
                
                # Simple matching - common words
                schema_words = set(schema_stem.split('_'))
                transcript_words = set(transcript_stem.split('_'))
                
                common_words = schema_words.intersection(transcript_words)
                score = len(common_words)
                
                if score > best_score:
                    best_score = score
                    best_match = transcript_file
            
            if best_match:
                pairs.append((schema_file, best_match))
                transcript_files.remove(best_match)  # Avoid duplicate matching
        
        return pairs

    def create_sample_files(self):
        """Create sample schema and transcript files for testing"""
        # Create sample schema
        sample_schema = {
            "class_id": "SAMPLE_CLASS_001",
            "date": "2024-01-15",
            "expected_topics": [
                "introduction",
                "basic concepts",
                "examples",
                "homework assignment"
            ]
        }
        
        schema_path = Path("schemas") / "sample_schema.json"
        schema_path.parent.mkdir(exist_ok=True)
        
        with open(schema_path, 'w') as f:
            json.dump(sample_schema, f, indent=2)
        
        # Create sample transcript
        sample_transcript = """
        Hello everyone, welcome to today's class. Let me start with a brief introduction
        to the topic we'll be covering today. We'll go through some basic concepts first,
        and then I'll show you some examples to illustrate these concepts.
        
        The basic concepts include understanding the fundamentals and key principles.
        Let me give you some examples to make this clearer.
        
        For your homework assignment, please review chapters 1-3 and complete the exercises
        at the end of each chapter.
        """
        
        transcript_path = Path("transcripts") / "sample_transcript.txt"
        transcript_path.parent.mkdir(exist_ok=True)
        
        with open(transcript_path, 'w') as f:
            f.write(sample_transcript.strip())
        
        logger.info(f"Sample files created: {schema_path}, {transcript_path}")
        return str(schema_path), str(transcript_path)


def main():
    """CLI interface for the Schema Checker Pipeline"""
    parser = argparse.ArgumentParser(description="Schema Checker Pipeline")
    parser.add_argument("--schema", help="Path to schema file")
    parser.add_argument("--transcript", help="Path to transcript file")
    parser.add_argument("--schema-dir", help="Directory containing schema files (batch mode)")
    parser.add_argument("--transcript-dir", help="Directory containing transcript files (batch mode)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")
    parser.add_argument("--method", default="hybrid", help="Keyword extraction method")
    parser.add_argument("--create-samples", action="store_true", help="Create sample files")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SchemaCheckerPipeline(
        similarity_threshold=args.threshold,
        extraction_method=args.method
    )
    
    # Create samples if requested
    if args.create_samples:
        schema_file, transcript_file = pipeline.create_sample_files()
        print(f"Sample files created:")
        print(f"  Schema: {schema_file}")
        print(f"  Transcript: {transcript_file}")
        return
    
    # Batch processing
    if args.schema_dir and args.transcript_dir:
        results = pipeline.batch_process(args.schema_dir, args.transcript_dir)
        successful = len([r for r in results if r.get('success', False)])
        print(f"Batch processing complete: {successful}/{len(results)} successful")
        return
    
    # Single file processing
    if args.schema and args.transcript:
        result = pipeline.process_session(args.schema, args.transcript)
        if result['success']:
            print(f"Processing successful! Report saved to: {result['report_path']}")
        else:
            print(f"Processing failed: {result['error']}")
        return
    
    # Show help if no arguments
    parser.print_help()


if __name__ == "__main__":
    main()
