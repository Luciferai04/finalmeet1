from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = '../data/uploads'
ALLOWED_EXTENSIONS = {'json', 'yaml', 'yml', 'csv', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_error(error_message: str, status_code: int = 400):
    """Standard error response handler"""
    return jsonify({
        'success': False,
        'error': error_message,
        'timestamp': datetime.now().isoformat()
    }), status_code

def handle_success(data, message: str = "Operation successful"):
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
        'description': 'REST API for real-time translation and analysis',
        'endpoints': {
            'health': '/api/health',
            'translate': '/api/translate',
            'schemas': '/api/schemas',
            'reports': '/api/reports'
        }
    }, "Welcome to Real-Time Translator API")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return handle_success({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """Translate text endpoint"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("No JSON data provided")
        
        text = data.get('text', '')
        target_language = data.get('target_language', 'bn')
        
        if not text:
            return handle_error("Text is required")
        
        # Mock translation for testing (replace with actual translation logic)
        translated_text = f"[{target_language.upper()}] {text}"
        
        return handle_success({
            'original_text': text,
            'translated_text': translated_text,
            'target_language': target_language,
            'processing_time': 0.1
        }, "Translation completed successfully")
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return handle_error(f"Translation failed: {str(e)}")

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        return handle_success({
            'system': 'operational',
            'components': {
                'api': 'active',
                'translation': 'available',
                'storage': 'ready'
            },
            'uptime': '24/7',
            'version': '1.0.0'
        }, "System status retrieved successfully")
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return handle_error(f"Status check failed: {str(e)}")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return handle_error("Endpoint not found", 404)

@app.errorhandler(405)
def method_not_allowed(error):
    return handle_error("Method not allowed", 405)

@app.errorhandler(413)
def too_large(error):
    return handle_error("Request too large", 413)

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
        port=5001,
        debug=True,
        threaded=True
    )
