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
                        date: Optional[str] = None) -e Dict[str, Any]:
    """
    Process a complete classroom session with error handling.
    """
        """
        Process a complete classroom session.

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
            normalized_schema_path = self.schema_parser.save_normalized_schema(
                schema_data)
            logger.info(f"Normalized schema saved: {normalized_schema_path}")

            # Step 2: Read transcript
logger.info("Step 2: Reading transcript")
try:
    if not os.path.exists(transcript_file):
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
except Exception as e:
    logger.error(f"Error opening transcript file: {e}")
    raise

            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read().strip()

            if not transcript_text:
                raise ValueError("Transcript file is empty")

            # Step 3: Extract keywords
            logger.info("Step 3: Extracting keywords from transcript")
            extraction_result = self.keyword_extractor.extract_keywords(
                transcript_text)

            extracted_keywords = extraction_result

            logger.info(
                f"Extracted {
                    len(extracted_keywords)} keywords using {
                    self.extraction_method} method")

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
            logger.info(
                f" - Topics covered: {stats.get('topics_covered', 0)}/{stats.get('total_expected_topics', 0)}")
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
        logger.info(f"Starting batch processing: {schema_dir} + {transcript_dir}")

        schema_dir = Path(schema_dir)
        transcript_dir = Path(transcript_dir)

        if not schema_dir.exists():
            raise FileNotFoundError(f"Schema directory not found: {schema_dir}")
        if not transcript_dir.exists():
            raise FileNotFoundError(
                f"Transcript directory not found: {transcript_dir}")

        # Find schema files
        schema_files = list(schema_dir.glob('*.json')) + \
            list(schema_dir.glob('*.yaml')) + \
            list(schema_dir.glob('*.yml')) + \
            list(schema_dir.glob('*.csv'))

        results = []

        for schema_file in schema_files:
            # Find corresponding transcript
            transcript_file = None

            if pattern_matching:
                # Try to match by filename pattern
                base_name = schema_file.stem
                possible_transcripts = [
                    transcript_dir / f"{base_name}.txt",
                    transcript_dir / f"{base_name}_transcript.txt",
                    transcript_dir / f"{base_name}_transcription.txt"
                ]

                for possible_transcript in possible_transcripts:
                    if possible_transcript.exists():
                        transcript_file = possible_transcript
                        break

            if not transcript_file:
                logger.warning(f"No matching transcript found for {schema_file}")
                continue

            # Process the session
            result = self.process_session(str(schema_file), str(transcript_file))
            results.append(result)

        # Summary
        successful = len([r for r in results if r.get('success', False)])
        failed = len(results) - successful

        logger.info(
            f"Batch processing completed: {successful} successful, {failed} failed")

        return results

    def create_sample_files(self):
        """Create sample schema and transcript files for testing."""
        logger.info("Creating sample files for testing")

        # Sample schema
        sample_schema = {
            "class_id": "MATH101",
            "date": "2025-01-02",
            "course_name": "Introduction to Calculus",
            "instructor": "Dr. Smith",
            "expected_topics": [
                "derivatives",
                "chain rule",
                "product rule",
                "quotient rule",
                "implicit differentiation"
            ]
        }

        schema_file = Path("sample_schema.json")
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f, indent=2)

        # Sample transcript
        sample_transcript = """
Today we're going to cover derivatives in calculus. A derivative represents the rate of change
of a function. We'll start with the basic definition and then move on to some important rules.

The chain rule is fundamental when dealing with composite functions. If you have a function
within another function, you need to use the chain rule to find the derivative.

Next, we have the product rule. When you're multiplying two functions together, you can't
just multiply their derivatives. You need to use the product rule formula.

We'll also discuss integration briefly, though that's not our main focus today.
The fundamental theorem of calculus connects derivatives and integrals.

Finally, we'll look at some practical applications of derivatives in physics and engineering.
        """

        transcript_file = Path("sample_transcript.txt")
        with open(transcript_file, 'w') as f:
            f.write(sample_transcript.strip())

        logger.info(f"Sample files created: {schema_file}, {transcript_file}")
        return str(schema_file), str(transcript_file)


class TranscriptWatcher(FileSystemEventHandler):
    """File system watcher for automatic processing of new transcripts."""

    def __init__(self, pipeline: SchemaCheckerPipeline, schema_dir: str):
        self.pipeline = pipeline
        self.schema_dir = Path(schema_dir)

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.endswith('.txt'):
            logger.info(f"New transcript detected: {event.src_path}")

            # Try to find corresponding schema
            transcript_file = Path(event.src_path)
            base_name = transcript_file.stem.replace(
                '_transcript', '').replace(
                '_transcription', '')

            # Look for schema file
            possible_schemas = [
                self.schema_dir / f"{base_name}.json",
                self.schema_dir / f"{base_name}.yaml",
                self.schema_dir / f"{base_name}.csv"
            ]

            schema_file = None
            for possible_schema in possible_schemas:
                if possible_schema.exists():
                    schema_file = possible_schema
                    break

            if schema_file:
                logger.info(f"Auto-processing: {schema_file} + {transcript_file}")
                result = self.pipeline.process_session(
                    str(schema_file), str(transcript_file))

                if result.get('success'):
                    logger.info(f"Auto-processing successful: {result.get('report_path')}")
                else:
                    logger.error(f"Auto-processing failed: {result.get('error')}")


def start_file_watcher(pipeline: SchemaCheckerPipeline,
                       transcript_dir: str = "transcripts",
                       schema_dir: str = "schemas"):
    """Start file system watcher for automatic processing."""
    logger.info(
        f"Starting file watcher: {transcript_dir} (schemas: {schema_dir})")

    event_handler = TranscriptWatcher(pipeline, schema_dir)
    observer = Observer()
    observer.schedule(event_handler, transcript_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("File watcher stopped")

    observer.join()


def main(argv=None):
    """
    Main function with comprehensive CLI for the schema checker pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Automated schema checker pipeline for classroom sessions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
 # Process single session
 python -m schema_checker.main schema.json transcript.txt

 # Batch process directory
 python -m schema_checker.main --batch schemas/ transcripts/

 # Start file watcher
 python -m schema_checker.main --watch transcripts/ --schemas schemas/

 # Create sample files for testing
 python -m schema_checker.main --create-samples
 """
    )

    # Main arguments
    parser.add_argument(
        'schema_file',
        nargs='?',
        help='Path to schema file (JSON, YAML, or CSV)')
    parser.add_argument(
        'transcript_file',
        nargs='?',
        help='Path to transcript text file')

    # Processing options
    parser.add_argument('--method', default='hybrid',
                        choices=['rake', 'spacy', 'hybrid'],
                        help='Keyword extraction method (default: hybrid)')
    parser.add_argument('--similarity', type=float, default=0.6,
                        help='Similarity threshold for topic matching (default: 0.6)')
    parser.add_argument('--class-id', help='Override class ID from schema')
    parser.add_argument('--date', help='Override date from schema')

    # Batch processing
    parser.add_argument('--batch', action='store_true',
                        help='Batch process: schema_file=schema_dir, transcript_file=transcript_dir')

    # File watching
    parser.add_argument('--watch', metavar='TRANSCRIPT_DIR',
                        help='Watch directory for new transcripts and auto-process')
    parser.add_argument('--schemas', default='schemas',
                        help='Schema directory for file watching (default: schemas)')

    # Utilities
    parser.add_argument('--create-samples', action='store_true',
                        help='Create sample schema and transcript files for testing')
    parser.add_argument('--reports-dir', default='reports',
                        help='Output directory for reports (default: reports)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args(argv)

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize pipeline
    pipeline = SchemaCheckerPipeline(
        similarity_threshold=args.similarity,
        extraction_method=args.method,
        reports_dir=args.reports_dir
    )

    try:
        # Create samples
        if args.create_samples:
            schema_file, transcript_file = pipeline.create_sample_files()
            print(f"[PASS] Sample files created:")
            print(f" Schema: {schema_file}")
            print(f" Transcript: {transcript_file}")
            print(f"\n[TIP] Test the pipeline with:")
            print(f" python -m schema_checker.main {schema_file} {transcript_file}")
            return 0

        # File watching
        if args.watch:
            start_file_watcher(pipeline, args.watch, args.schemas)
            return 0

        # Batch processing
        if args.batch:
            if not args.schema_file or not args.transcript_file:
                parser.error("Batch mode requires schema_dir and transcript_dir arguments")

            results = pipeline.batch_process(args.schema_file, args.transcript_file)

            successful = len([r for r in results if r.get('success', False)])
            total = len(results)

            print(f"\nBatch Processing Results:")
            print(f" [PASS] Successful: {successful}/{total}")
            print(f" [FAIL] Failed: {total - successful}/{total}")

            # Show failed sessions
            failed_sessions = [r for r in results if not r.get('success', False)]
            if failed_sessions:
                print(f"\n[FAIL] Failed Sessions:")
                for session in failed_sessions:
                    print(f" - {session.get('schema_file', 'Unknown')}: {session.get('error', 'Unknown error')}")

            return 0 if successful == total else 1

        # Single session processing
        if not args.schema_file or not args.transcript_file:
            parser.error(
                "Schema file and transcript file are required (or use --batch/--watch/--create-samples)")

        result = pipeline.process_session(
            args.schema_file,
            args.transcript_file,
            args.class_id,
            args.date
        )

        if result.get('success'):
            stats = result.get('statistics', {})
            print(f"\n[PASS] Session processed successfully!")
            print(f" Class: {result.get('class_id')} ({result.get('date')})")
            print(f" Coverage: {stats.get('coverage_percentage', 0):.1f}%")
            print(f" Quality Score: {stats.get('quality_score', 0):.1f}/100")
            print(f" Report: {result.get('report_path')}")
            return 0
        else:
            print(f"\n[FAIL] Processing failed: {result.get('error')}")
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
