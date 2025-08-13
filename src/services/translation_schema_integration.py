"""
Translation Schema Integration

This module provides integration between the real-time translation system
and the schema checker pipeline, enabling automatic topic analysis after
translation sessions complete.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import the schema pipeline
from .schema_checker_pipeline import SchemaCheckerPipeline

# Configure logging
logger = logging.getLogger(__name__)


class TranslationSchemaIntegrator:
    """
    Integrates the schema checker pipeline with the translation system.

    This class handles the automatic triggering of schema checking after
    translation sessions complete, managing data flow between systems.
    """

    def __init__(self, pipeline_config: Optional[Dict[str, Any]] = None):
        pass
    """
 Initialize the integrator.

 Args:
 pipeline_config: Configuration for the schema checker pipeline
 """
    self.pipeline = SchemaCheckerPipeline(pipeline_config)
    self.active_sessions = {}
    self.completed_sessions = {}

    # Default paths for schema integration
    self.default_schema_dir = "schemas"
    self.transcript_cache_dir = "transcript_cache"
    self.integration_log_file = "translation_schema_integration.log"

    # Ensure directories exist
    os.makedirs(self.default_schema_dir, exist_ok=True)
    os.makedirs(self.transcript_cache_dir, exist_ok=True)

    logger.info("Translation Schema Integrator initialized")

    async def start_translation_session(
        self,
        session_id: str,
        class_info: Optional[Dict[str, Any]] = None,
        schema_file: Optional[str] = None
    ) -> Dict[str, Any]:
    """
 Start tracking a translation session for future schema analysis.

 Args:
 session_id: Unique identifier for the session
 class_info: Information about the class/session
 schema_file: Path to schema file, or None to use default

 Returns:
 Session tracking information
 """
    logger.info(f"Starting translation session tracking: {session_id}")

    # Generate session info
    session_info = {
        'session_id': session_id,
        'start_time': datetime.now().isoformat(),
        'class_info': class_info or {},
        'schema_file': schema_file or self._get_default_schema_file(class_info),
        'transcript_chunks': [],
        'status': 'active',
        'total_transcript_length': 0
    }

    self.active_sessions[session_id] = session_info

    return {
        'status': 'session_started',
        'session_id': session_id,
        'schema_file': session_info['schema_file']
    }

    def add_transcript_chunk(
        self,
        session_id: str,
        transcript_chunk: str,
        timestamp: Optional[str] = None,
        language: str = 'en'
    ) -> Dict[str, Any]:
    """
 Add a transcript chunk to an active session.

 Args:
 session_id: Session identifier
 transcript_chunk: Text chunk from the translation
 timestamp: Timestamp of the chunk
 language: Language of the transcript

 Returns:
 Status information
 """
    if session_id not in self.active_sessions:
        pass
    logger.warning(f"Session {session_id} not found in active sessions")
    return {'status': 'error', 'message': 'Session not found'}

    session = self.active_sessions[session_id]

    # Add chunk to session
    chunk_info = {
        'text': transcript_chunk,
        'timestamp': timestamp or datetime.now().isoformat(),
        'language': language,
        'length': len(transcript_chunk)
    }

    session['transcript_chunks'].append(chunk_info)
    session['total_transcript_length'] += len(transcript_chunk)

    logger.debug(
        f"Added transcript chunk to session {session_id}: {
            len(transcript_chunk)} chars")

    return {
        'status': 'chunk_added',
        'session_id': session_id,
        'total_length': session['total_transcript_length'],
        'chunks_count': len(session['transcript_chunks'])
    }

    async def complete_translation_session(
        self,
        session_id: str,
        final_class_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
    """
 Complete a translation session and trigger schema analysis.

 Args:
 session_id: Session identifier
 final_class_info: Final class information to merge

 Returns:
 Schema analysis result
 """
    if session_id not in self.active_sessions:
        pass
    logger.error(f"Session {session_id} not found in active sessions")
    return {'status': 'error', 'message': 'Session not found'}

    logger.info(f"Completing translation session: {session_id}")

    session = self.active_sessions[session_id]

    # Update session info
    session['end_time'] = datetime.now().isoformat()
    session['status'] = 'completed'

    if final_class_info:
        pass
    session['class_info'].update(final_class_info)

    # Combine all transcript chunks
    full_transcript = self._combine_transcript_chunks(
        session['transcript_chunks'])

    # Save transcript for future reference
    transcript_file = await self._save_transcript_cache(session_id, full_transcript, session['class_info'])
    session['cached_transcript_file'] = transcript_file

    # Trigger schema analysis
    try:
        pass
    logger.info(f"Starting schema analysis for session {session_id}")

    analysis_result = await self.pipeline.process_session(
        session_id=session_id,
        transcript_text=full_transcript,
        schema_file=session['schema_file'],
        class_info=session['class_info']
    )

    # Update session with analysis results
    session['analysis_result'] = analysis_result
    session['analysis_completed'] = True

    # Move to completed sessions
    self.completed_sessions[session_id] = session
    del self.active_sessions[session_id]

    logger.info(f"Session {session_id} completed successfully")
    logger.info(
        f"Schema coverage: {
            analysis_result.get(
                'coverage_percentage',
                0):.1f}%")

    return {
        'status': 'success',
        'session_id': session_id,
        'transcript_length': len(full_transcript),
        'schema_analysis': analysis_result,
        'coverage_percentage': analysis_result.get('coverage_percentage', 0),
        'report_available': True
    }

    except Exception as e:
        pass
    logger.error(
        f"Error during schema analysis for session {session_id}: {
            str(e)}")

    session['analysis_error'] = str(e)
    session['analysis_completed'] = False
    self.completed_sessions[session_id] = session
    del self.active_sessions[session_id]

    return {
        'status': 'error',
        'session_id': session_id,
        'error': str(e),
        'transcript_length': len(full_transcript),
        'report_available': False
    }

    def _combine_transcript_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        pass
    """Combine transcript chunks into a single text."""
    # Sort chunks by timestamp if available
    try:
        pass
    sorted_chunks = sorted(chunks, key=lambda x: x.get('timestamp', ''))
    except BaseException:
        pass
    sorted_chunks = chunks

    # Combine text with proper spacing
    combined_text = []
    for chunk in sorted_chunks:
        pass
    text = chunk.get('text', '').strip()
    if text:
        pass
    combined_text.append(text)

    return ' '.join(combined_text)

    async def _save_transcript_cache(
        self,
        session_id: str,
        transcript: str,
        class_info: Dict[str, Any]
    ) -> str:
    """Save transcript to cache for future reference."""

    # Generate filename
    date_str = class_info.get('date', datetime.now().strftime('%Y-%m-%d'))
    filename = f"{session_id}_{date_str}_transcript.txt"
    filepath = os.path.join(self.transcript_cache_dir, filename)

    # Save transcript with metadata
    metadata = {
        'session_id': session_id,
        'date': date_str,
        'class_info': class_info,
        'transcript_length': len(transcript),
        'saved_at': datetime.now().isoformat()
    }

    # Write files
    async def write_files():
        # Write transcript
    with open(filepath, 'w', encoding='utf-8') as f:
        pass
    f.write(transcript)

    # Write metadata
    meta_filepath = filepath.replace('.txt', '_metadata.json')
    with open(meta_filepath, 'w', encoding='utf-8') as f:
        pass
    json.dump(metadata, f, indent=2)

    await asyncio.to_thread(write_files)

    logger.info(f"Transcript cached: {filepath}")
    return filepath

    def _get_default_schema_file(
            self, class_info: Optional[Dict[str, Any]]) -> str:
    """Get default schema file path based on class info."""
    if not class_info:
        pass
    return os.path.join(self.default_schema_dir, "default_schema.json")

    # Try to determine schema file from class info
    subject = class_info.get('subject', 'general').lower().replace(' ', '_')
    schema_file = os.path.join(
        self.default_schema_dir,
        f"{subject}_schema.json")

    # Fall back to default if subject-specific schema doesn't exist
    if not os.path.exists(schema_file):
        pass
    default_schema = os.path.join(
        self.default_schema_dir,
        "default_schema.json")
    self._create_default_schema_if_missing(default_schema)
    return default_schema

    return schema_file

    def _create_default_schema_if_missing(self, schema_path: str):
        pass
    """Create a default schema file if it doesn't exist."""
    if os.path.exists(schema_path):
        pass
    return

    default_schema = {
        "class_id": "default",
        "date": datetime.now().strftime('%Y-%m-%d'),
        "expected_topics": [
            "introduction",
            "main concepts",
            "examples",
            "practice",
            "questions",
            "summary",
            "homework",
            "next class"
        ]
    }

    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
    with open(schema_path, 'w', encoding='utf-8') as f:
        pass
    json.dump(default_schema, f, indent=2)

    logger.info(f"Created default schema file: {schema_path}")

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        pass
    """Get status of a session."""
    # Check active sessions
    if session_id in self.active_sessions:
        pass
    session = self.active_sessions[session_id]
    return {
        'status': 'active',
        'session_info': session,
        'transcript_chunks': len(session['transcript_chunks']),
        'total_length': session['total_transcript_length']
    }

    # Check completed sessions
    if session_id in self.completed_sessions:
        pass
    session = self.completed_sessions[session_id]
    return {
        'status': 'completed',
        'session_info': session,
        'analysis_completed': session.get('analysis_completed', False),
        'coverage_percentage': session.get('analysis_result', {}).get('coverage_percentage', 0)
    }

    return {'status': 'not_found', 'message': 'Session not found'}

    def get_integration_stats(self) -> Dict[str, Any]:
        pass
    """Get integration statistics."""
    active_count = len(self.active_sessions)
    completed_count = len(self.completed_sessions)

    # Calculate success rate
    successful_analyses = sum(
        1 for session in self.completed_sessions.values()
        if session.get('analysis_completed', False)
    )

    success_rate = (
        successful_analyses /
        completed_count *
        100) if completed_count > 0 else 0

    # Get pipeline stats
    pipeline_stats = self.pipeline.get_pipeline_stats()

    return {
        'active_sessions': active_count,
        'completed_sessions': completed_count,
        'successful_analyses': successful_analyses,
        'success_rate': success_rate,
        'pipeline_stats': pipeline_stats,
        # Last 5
        'recent_completions': list(self.completed_sessions.keys())[-5:]
    }

    async def batch_reprocess_sessions(
        self,
        session_ids: List[str]
    ) -> Dict[str, Any]:
    """
 Reprocess completed sessions with updated schemas or settings.

 Args:
 session_ids: List of session IDs to reprocess

 Returns:
 Batch processing results
 """
    logger.info(f"Starting batch reprocessing of {len(session_ids)} sessions")

    sessions_to_process = []

    for session_id in session_ids:
        pass
    if session_id in self.completed_sessions:
        pass
    session = self.completed_sessions[session_id]

    # Read cached transcript
    transcript_file = session.get('cached_transcript_file')
    if transcript_file and os.path.exists(transcript_file):
        pass
    with open(transcript_file, 'r', encoding='utf-8') as f:
        pass
    transcript = f.read()

    sessions_to_process.append({
        'session_id': session_id,
        'transcript': transcript,
        'schema_file': session['schema_file'],
        'class_info': session['class_info']
    })
    else:
        pass
    logger.warning(f"Transcript cache not found for session {session_id}")

    if sessions_to_process:
        pass
    results = await self.pipeline.batch_process_sessions(sessions_to_process)

    # Update completed sessions with new results
    for i, result in enumerate(results):
        pass
    if isinstance(result, dict) and result.get('status') == 'success':
        pass
    session_id = sessions_to_process[i]['session_id']
    self.completed_sessions[session_id]['analysis_result'] = result
    self.completed_sessions[session_id]['reprocessed_at'] = datetime.now(
    ).isoformat()

    return {
        'status': 'completed',
        'processed_sessions': len(sessions_to_process),
        'results': results
    }
    else:
        pass
    return {
        'status': 'no_sessions_to_process',
        'message': 'No valid sessions found for reprocessing'
    }

    def cleanup_old_sessions(self, days_old: int = 30) -> Dict[str, Any]:
        pass
    """
 Clean up old session data to free memory.

 Args:
 days_old: Number of days after which to clean up sessions

 Returns:
 Cleanup statistics
 """
    from datetime import timedelta

    cutoff_date = datetime.now() - timedelta(days=days_old)

    # Clean completed sessions
    sessions_to_remove = []
    for session_id, session in self.completed_sessions.items():
        pass
    try:
        pass
    session_date = datetime.fromisoformat(session.get('end_time', ''))
    if session_date < cutoff_date:
        pass
    sessions_to_remove.append(session_id)
    except BaseException:
        pass
        # If we can't parse the date, keep the session
    continue

    removed_count = 0
    for session_id in sessions_to_remove:
        pass
    del self.completed_sessions[session_id]
    removed_count += 1

    logger.info(
        f"Cleaned up {removed_count} old sessions (older than {days_old} days)")

    return {
        'cleaned_sessions': removed_count,
        'remaining_active': len(self.active_sessions),
        'remaining_completed': len(self.completed_sessions)
    }

# Factory function for easy integration


def create_translation_schema_integrator(
        config_file: Optional[str] = None) -> TranslationSchemaIntegrator:
    """
    Factory function to create a TranslationSchemaIntegrator instance.

    Args:
    config_file: Optional path to configuration file

    Returns:
    Configured integrator instance
    """
    config = None
    if config_file and os.path.exists(config_file):
        pass
    with open(config_file, 'r') as f:
        pass
    config = json.load(f)

    return TranslationSchemaIntegrator(config)

# Example usage and testing


async def example_integration():
    """Example of how to use the integration system."""

    # Create integrator
    integrator = TranslationSchemaIntegrator()

    # Start a session
    session_result = await integrator.start_translation_session(
        session_id="example_session_001",
        class_info={
            'subject': 'Mathematics',
            'instructor': 'Dr. Smith',
            'duration': 60
        }
    )
    print(f"Session started: {session_result}")

    # Add some transcript chunks (simulating real-time translation)
    integrator.add_transcript_chunk(
        "example_session_001",
        "Welcome to today's mathematics class")
    integrator.add_transcript_chunk(
        "example_session_001",
        "We will cover quadratic equations")
    integrator.add_transcript_chunk(
        "example_session_001",
        "Let's start with basic examples")

    # Complete the session
    completion_result = await integrator.complete_translation_session(
        "example_session_001",
        final_class_info={'actual_duration': 65}
    )
    print(f"Session completed: {completion_result}")

    # Get stats
    stats = integrator.get_integration_stats()
    print(f"Integration stats: {stats}")

if __name__ == "__main__":
    pass
    # Run example
    asyncio.run(example_integration())
