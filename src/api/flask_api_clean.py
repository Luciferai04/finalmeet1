from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from typing import Dict, List, Any, Optional
import tempfile
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json', 'yaml', 'yml', 'csv', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def handle_error(error_message: str, status_code: int = 400):
    """Standard error response handler"""
    return jsonify({
        'success': False,
        'error': error_message,
        'timestamp': datetime.now().isoformat()
    }), status_code


def handle_success(data: Any, message: str = "Operation successful"):
    """Standard success response handler"""
    return jsonify({
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    })


# API Routes

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API welcome message"""
    return handle_success({
        'name': 'Real-Time Translator API',
        'version': '1.0.0',
        'description': 'REST API for real-time translation and transcription',
        'endpoints': {
            'health': '/api/health',
            'translate': '/api/translate',
            'transcribe': '/api/transcribe',
            'status': '/api/status'
        }
    }, "Welcome to Real-Time Translator API")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return handle_success({
        'status': 'healthy',
        'version': '1.0.0',
        'components': {
            'translation_service': 'active',
            'transcription_service': 'active'
        }
    })


@app.route('/api/translate', methods=['POST'])
def translate_text():
    """Translate text endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return handle_error("No JSON data provided")
        
        text = data.get('text')
        target_language = data.get('target_language', 'en')
        
        if not text:
            return handle_error("Text is required")
        
        # Mock translation for now
        result = {
            'original_text': text,
            'translated_text': f"[TRANSLATED TO {target_language}] {text}",
            'target_language': target_language,
            'confidence': 0.95
        }
        
        return handle_success(result, "Text translated successfully")
        
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return handle_error(f"Failed to translate text: {str(e)}")


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio endpoint"""
    try:
        if 'audio' not in request.files:
            return handle_error("No audio file provided")
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return handle_error("No file selected")
        
        # Mock transcription for now
        result = {
            'transcribed_text': "This is a mock transcription",
            'confidence': 0.98,
            'duration': 5.2,
            'language': 'en'
        }
        
        return handle_success(result, "Audio transcribed successfully")
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return handle_error(f"Failed to transcribe audio: {str(e)}")


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        status_info = {
            'server': {
                'uptime': 'Unknown',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            },
            'services': {
                'translation': 'active',
                'transcription': 'active',
                'websocket': 'active'
            },
            'statistics': {
                'total_translations': 0,
                'total_transcriptions': 0,
                'active_sessions': 0
            }
        }
        
        return handle_success(status_info, "Status retrieved successfully")
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return handle_error(f"Failed to get status: {str(e)}")


# Error handlers

@app.errorhandler(404)
def not_found(error):
    return handle_error("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(error):
    return handle_error("Method not allowed", 405)


@app.errorhandler(413)
def too_large(error):
    return handle_error("File too large", 413)


@app.errorhandler(500)
def internal_error(error):
    return handle_error("Internal server error", 500)


if __name__ == '__main__':
    logger.info("Starting Flask API server...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Max file size: {MAX_CONTENT_LENGTH} bytes")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
