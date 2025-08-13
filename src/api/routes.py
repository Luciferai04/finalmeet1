"""
API Routes
==========

Flask Blueprint containing all API endpoints.
"""

from flask import Blueprint, request, jsonify, current_app
from ..services.translation_service import TranslationService
from ..services.schema_service import SchemaService
from ..utils.validators import validate_translation_request

# Create API Blueprint
api_bp = Blueprint('api', __name__)

# Initialize services
translation_service = TranslationService()
schema_service = SchemaService()


@api_bp.route('/health')
def health():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'api_version': '1.0.0',
        'service': 'live-camera-translator-api'
    })


@api_bp.route('/translate', methods=['POST'])
def translate():
    """Translate text from one language to another."""
    try:
        data = request.get_json()
        
        # Validate request
        is_valid, error_msg = validate_translation_request(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Perform translation
        result = translation_service.translate(
            text=data['text'],
            target_language=data.get('target_language', 'Bengali'),
            session_id=data.get('session_id')
        )
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        current_app.logger.error(f"Translation error: {str(e)}")
        return jsonify({'error': 'Translation failed'}), 500


@api_bp.route('/schemas', methods=['GET'])
def list_schemas():
    """List all available schemas."""
    try:
        schemas = schema_service.list_schemas()
        return jsonify({
            'status': 'success',
            'data': schemas
        })
    except Exception as e:
        current_app.logger.error(f"Schema listing error: {str(e)}")
        return jsonify({'error': 'Failed to list schemas'}), 500


@api_bp.route('/schemas', methods=['POST'])
def upload_schema():
    """Upload a new schema."""
    try:
        if 'schema_file' not in request.files:
            return jsonify({'error': 'No schema file provided'}), 400
        
        file = request.files['schema_file']
        result = schema_service.upload_schema(file)
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        current_app.logger.error(f"Schema upload error: {str(e)}")
        return jsonify({'error': 'Schema upload failed'}), 500


@api_bp.route('/schemas/<schema_id>', methods=['GET'])
def get_schema(schema_id):
    """Get a specific schema by ID."""
    try:
        schema = schema_service.get_schema(schema_id)
        if not schema:
            return jsonify({'error': 'Schema not found'}), 404
        
        return jsonify({
            'status': 'success',
            'data': schema
        })
        
    except Exception as e:
        current_app.logger.error(f"Schema retrieval error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve schema'}), 500


@api_bp.route('/schemas/<schema_id>', methods=['DELETE'])
def delete_schema(schema_id):
    """Delete a schema by ID."""
    try:
        result = schema_service.delete_schema(schema_id)
        if not result:
            return jsonify({'error': 'Schema not found'}), 404
        
        return jsonify({
            'status': 'success',
            'message': 'Schema deleted successfully'
        })
        
    except Exception as e:
        current_app.logger.error(f"Schema deletion error: {str(e)}")
        return jsonify({'error': 'Failed to delete schema'}), 500


@api_bp.route('/process', methods=['POST'])
def process_session():
    """Process a session against a schema."""
    try:
        data = request.get_json()
        
        if not all(k in data for k in ['session_id', 'schema_id']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        result = schema_service.process_session(
            session_id=data['session_id'],
            schema_id=data['schema_id'],
            transcript=data.get('transcript')
        )
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        current_app.logger.error(f"Session processing error: {str(e)}")
        return jsonify({'error': 'Session processing failed'}), 500


@api_bp.route('/reports', methods=['GET'])
def list_reports():
    """List all processing reports."""
    try:
        reports = schema_service.list_reports()
        return jsonify({
            'status': 'success',
            'data': reports
        })
    except Exception as e:
        current_app.logger.error(f"Reports listing error: {str(e)}")
        return jsonify({'error': 'Failed to list reports'}), 500


@api_bp.route('/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions."""
    try:
        sessions = translation_service.list_sessions()
        return jsonify({
            'status': 'success',
            'data': sessions
        })
    except Exception as e:
        current_app.logger.error(f"Sessions listing error: {str(e)}")
        return jsonify({'error': 'Failed to list sessions'}), 500
