"""
Validation utilities for the application
"""

from typing import Dict, Any, List, Optional
import re


def validate_translation_request(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate translation request data
    
    Args:
        data: The request data to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Request data must be a dictionary"
    
    # Check required fields
    required_fields = ['text', 'target_language']
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        if not data[field] or not isinstance(data[field], str):
            return False, f"Field '{field}' must be a non-empty string"
    
    # Validate target language format
    if not re.match(r'^[a-z]{2}$', data['target_language']):
        return False, "Target language must be a 2-letter language code (e.g., 'en', 'bn', 'hi')"
    
    # Validate text length
    if len(data['text']) > 10000:
        return False, "Text is too long (maximum 10,000 characters)"
    
    return True, ""


def validate_schema_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate schema data
    
    Args:
        data: The schema data to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Schema data must be a dictionary"
    
    # Check required fields
    required_fields = ['class_id', 'expected_topics']
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate class_id
    if not isinstance(data['class_id'], str) or not data['class_id'].strip():
        return False, "class_id must be a non-empty string"
    
    # Validate expected_topics
    if not isinstance(data['expected_topics'], list):
        return False, "expected_topics must be a list"
    
    if not data['expected_topics']:
        return False, "expected_topics cannot be empty"
    
    for topic in data['expected_topics']:
        if not isinstance(topic, str) or not topic.strip():
            return False, "All topics must be non-empty strings"
    
    return True, ""


def validate_file_upload(file_path: str, allowed_extensions: List[str] = None) -> tuple[bool, str]:
    """
    Validate uploaded file
    
    Args:
        file_path: Path to the uploaded file
        allowed_extensions: List of allowed file extensions
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not file_path:
        return False, "No file provided"
    
    if allowed_extensions is None:
        allowed_extensions = ['.json', '.csv', '.pdf', '.txt', '.yaml', '.yml']
    
    # Check file extension
    file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
    if f'.{file_ext}' not in allowed_extensions:
        return False, f"File type '.{file_ext}' not allowed. Allowed types: {', '.join(allowed_extensions)}"
    
    return True, ""


def validate_language_code(lang_code: str) -> bool:
    """
    Validate language code format
    
    Args:
        lang_code: Language code to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(lang_code, str):
        return False
    
    # Check if it's a 2-letter language code
    return bool(re.match(r'^[a-z]{2}$', lang_code))


def validate_session_id(session_id: str) -> tuple[bool, str]:
    """
    Validate session ID format
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(session_id, str):
        return False, "Session ID must be a string"
    
    if not session_id.strip():
        return False, "Session ID cannot be empty"
    
    # Check format: should be alphanumeric with underscores and dashes
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return False, "Session ID can only contain letters, numbers, underscores, and dashes"
    
    if len(session_id) > 100:
        return False, "Session ID is too long (maximum 100 characters)"
    
    return True, ""
