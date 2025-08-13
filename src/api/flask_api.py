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

# Import existing components
from admin_interface import AdminInterface
from src.schema_checker_pipeline import SchemaCheckerPipeline

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

# Initialize components
admin_interface = AdminInterface()
schema_pipeline = SchemaCheckerPipeline()


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
        'name': 'Schema Checker API',
        'version': '1.0.0',
        'description': 'REST API for schema validation and transcript analysis',
        'endpoints': {
            'health': '/api/health',
            'schemas': '/api/schemas',
            'reports': '/api/reports',
            'transcripts': '/api/transcripts',
            'process': '/api/process',
            'system_info': '/api/system-info'
        },
        'admin_interface': 'http://localhost:7861'
    }, "Welcome to Schema Checker API")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return handle_success({
        'status': 'healthy',
        'version': '1.0.0',
        'components': {
            'admin_interface': 'active',
            'schema_pipeline': 'active'
        }
    })


@app.route('/api/schemas', methods=['GET'])
def list_schemas():
    """List all schema files"""
    try:
        schemas = admin_interface.list_schema_files()
        return handle_success(schemas, "Schema files retrieved successfully")
    except Exception as e:
        logger.error(f"Error listing schemas: {e}")
        return handle_error(f"Failed to list schemas: {str(e)}")


@app.route('/api/schemas', methods=['POST'])
def upload_schema():
    """Upload a new schema file"""
    try:
        if 'file' not in request.files:
            return handle_error("No file part in request")

        file = request.files['file']
        if file.filename == '':
            return handle_error("No file selected")

        if not allowed_file(file.filename):
            return handle_error(
                "File type not allowed. Allowed types: json, yaml, yml, csv")

        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        # Use admin interface to upload
        result = admin_interface.upload_schema_file(temp_path)

        # Clean up temporary file
        os.remove(temp_path)

        if result[0]:  # Success
            return handle_success({
                'filename': filename,
                'message': result[1]
            }, "Schema uploaded successfully")
        else:
            return handle_error(result[1])

    except Exception as e:
        logger.error(f"Error uploading schema: {e}")
        return handle_error(f"Failed to upload schema: {str(e)}")


@app.route('/api/schemas/<filename>', methods=['GET'])
def get_schema(filename):
    """Get a specific schema file content"""
    try:
        pass
    schema_path = Path("schemas") / filename
    if not schema_path.exists():
        pass
    return handle_error("Schema file not found", 404)

    with open(schema_path, 'r', encoding='utf-8') as f:
        pass
    content = f.read()

    # Try to parse as JSON/YAML for better response
    try:
        pass
    if filename.endswith('.json'):
        pass
    parsed_content = json.loads(content)
    return handle_success({
        'filename': filename,
        'content': parsed_content,
        'raw_content': content
    })
    else:
        pass
    return handle_success({
        'filename': filename,
        'raw_content': content
    })
    except BaseException:
        pass
    return handle_success({
        'filename': filename,
        'raw_content': content
    })

    except Exception as e:
        pass
    logger.error(f"Error getting schema: {e}")
    return handle_error(f"Failed to get schema: {str(e)}")


@app.route('/api/schemas/<filename>', methods=['DELETE'])
def delete_schema(filename):
    """Delete a schema file"""
    try:
        pass
    schema_path = Path("schemas") / filename
    if not schema_path.exists():
        pass
    return handle_error("Schema file not found", 404)

    os.remove(schema_path)
    return handle_success({
        'filename': filename
    }, "Schema deleted successfully")

    except Exception as e:
        pass
    logger.error(f"Error deleting schema: {e}")
    return handle_error(f"Failed to delete schema: {str(e)}")


@app.route('/api/reports', methods=['GET'])
def list_reports():
    """List all analysis reports"""
    try:
        pass
    reports = admin_interface.list_reports()
    return handle_success(reports, "Reports retrieved successfully")
    except Exception as e:
        pass
    logger.error(f"Error listing reports: {e}")
    return handle_error(f"Failed to list reports: {str(e)}")


@app.route('/api/reports/<filename>', methods=['GET'])
def get_report(filename):
    """Get a specific report content"""
    try:
        pass
    report_path = Path("reports") / filename
    if not report_path.exists():
        pass
    return handle_error("Report file not found", 404)

    with open(report_path, 'r', encoding='utf-8') as f:
        pass
    content = f.read()

    # Try to parse as JSON for better response
    try:
        pass
    parsed_content = json.loads(content)
    return handle_success({
        'filename': filename,
        'content': parsed_content,
        'raw_content': content
    })
    except BaseException:
        pass
    return handle_success({
        'filename': filename,
        'raw_content': content
    })

    except Exception as e:
        pass
    logger.error(f"Error getting report: {e}")
    return handle_error(f"Failed to get report: {str(e)}")


@app.route('/api/reports/<filename>/download', methods=['GET'])
def download_report(filename):
    """Download a report file"""
    try:
        pass
    report_path = Path("reports") / filename
    if not report_path.exists():
        pass
    return handle_error("Report file not found", 404)

    return send_file(report_path, as_attachment=True, download_name=filename)

    except Exception as e:
        pass
    logger.error(f"Error downloading report: {e}")
    return handle_error(f"Failed to download report: {str(e)}")


@app.route('/api/transcripts', methods=['POST'])
def upload_transcript():
    """Upload a transcript file"""
    try:
        pass
    if 'file' not in request.files:
        pass
    return handle_error("No file part in request")

    file = request.files['file']
    if file.filename == '':
        pass
    return handle_error("No file selected")

    if not file.filename.endswith('.txt'):
        pass
    return handle_error("Only .txt files are allowed for transcripts")

    # Save file to transcripts directory
    filename = secure_filename(file.filename)
    transcript_path = Path("transcripts") / filename
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    file.save(transcript_path)

    return handle_success({
        'filename': filename,
        'path': str(transcript_path)
    }, "Transcript uploaded successfully")

    except Exception as e:
        pass
    logger.error(f"Error uploading transcript: {e}")
    return handle_error(f"Failed to upload transcript: {str(e)}")


@app.route('/api/process', methods=['POST'])
def process_session():
    """Process a session with schema and transcript"""
    try:
        pass
    data = request.get_json()

    if not data:
        pass
    return handle_error("No JSON data provided")

    schema_file = data.get('schema_file')
    transcript_file = data.get('transcript_file')
    class_id = data.get('class_id')
    date = data.get('date')

    if not schema_file or not transcript_file:
        pass
    return handle_error("Both schema_file and transcript_file are required")

    # Process the session
    result = schema_pipeline.process_session(
        schema_file,
        transcript_file,
        class_id,
        date
    )

    return handle_success(result, "Session processed successfully" if result.get(
        'success') else "Session processing failed")

    except Exception as e:
        pass
    logger.error(f"Error processing session: {e}")
    return handle_error(f"Failed to process session: {str(e)}")


@app.route('/api/test-session', methods=['POST'])
def test_session():
    """Run a test analysis session"""
    try:
        pass
    data = request.get_json() or {}

    schema_file = data.get('schema_file')
    transcript_text = data.get('transcript_text', '')
    class_id = data.get('class_id', 'TEST_CLASS')
    date = data.get('date', datetime.now().strftime('%Y-%m-%d'))

    if not schema_file:
        pass
    return handle_error("schema_file is required")

    if not transcript_text:
        pass
    return handle_error("transcript_text is required")

    # Create temporary transcript file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        pass
    temp_file.write(transcript_text)
    temp_transcript_path = temp_file.name

    try:
        pass
        # Process the session
    result = schema_pipeline.process_session(
        schema_file,
        temp_transcript_path,
        class_id,
        date
    )

    return handle_success(result, "Test session completed")

    finally:
        pass
        # Clean up temporary file
    if os.path.exists(temp_transcript_path):
        pass
    os.remove(temp_transcript_path)

    except Exception as e:
        pass
    logger.error(f"Error in test session: {e}")
    return handle_error(f"Failed to run test session: {str(e)}")


@app.route('/api/batch-process', methods=['POST'])
def batch_process():
    """Process multiple sessions in batch"""
    try:
        pass
    data = request.get_json()

    if not data:
        pass
    return handle_error("No JSON data provided")

    schema_dir = data.get('schema_dir', 'schemas')
    transcript_dir = data.get('transcript_dir', 'transcripts')
    pattern_matching = data.get('pattern_matching', True)

    # Process batch
    results = schema_pipeline.batch_process(
        schema_dir, transcript_dir, pattern_matching)

    # Calculate statistics
    successful = len([r for r in results if r.get('success', False)])
    failed = len(results) - successful

    return handle_success({
        'results': results,
        'statistics': {
            'total': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / len(results) * 100) if results else 0
        }
    }, "Batch processing completed")

    except Exception as e:
        pass
    logger.error(f"Error in batch processing: {e}")
    return handle_error(f"Failed to process batch: {str(e)}")


@app.route('/api/system-info', methods=['GET'])
def get_system_info():
    """Get system information and statistics"""
    try:
        pass
        # Get directory statistics
    schemas_dir = Path("schemas")
    reports_dir = Path("reports")
    transcripts_dir = Path("transcripts")

    schema_count = len(list(schemas_dir.glob('*'))
                       ) if schemas_dir.exists() else 0
    report_count = len(list(reports_dir.glob('*'))
                       ) if reports_dir.exists() else 0
    transcript_count = len(list(transcripts_dir.glob('*'))
                           ) if transcripts_dir.exists() else 0

    # Get disk usage
    def get_directory_size(path):
        pass
    if not path.exists():
        pass
    return 0
    return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())

    system_info = {
        'directories': {
            'schemas': {
                'path': str(schemas_dir),
                'exists': schemas_dir.exists(),
                'file_count': schema_count,
                'size_bytes': get_directory_size(schemas_dir)
            },
            'reports': {
                'path': str(reports_dir),
                'exists': reports_dir.exists(),
                'file_count': report_count,
                'size_bytes': get_directory_size(reports_dir)
            },
            'transcripts': {
                'path': str(transcripts_dir),
                'exists': transcripts_dir.exists(),
                'file_count': transcript_count,
                'size_bytes': get_directory_size(transcripts_dir)
            }
        },
        'pipeline_config': {
            'similarity_threshold': schema_pipeline.similarity_threshold,
            'extraction_method': schema_pipeline.extraction_method,
            'reports_dir': schema_pipeline.reporter.reports_dir,
            'schemas_dir': schema_pipeline.schema_parser.schemas_dir
        },
        'server_info': {
            'timestamp': datetime.now().isoformat(),
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'max_content_length': app.config['MAX_CONTENT_LENGTH']
        }
    }

    return handle_success(
        system_info, "System information retrieved successfully")

    except Exception as e:
        pass
    logger.error(f"Error getting system info: {e}")
    return handle_error(f"Failed to get system info: {str(e)}")


@app.route('/api/create-samples', methods=['POST'])
def create_samples():
    """Create sample schema and transcript files for testing"""
    try:
        pass
    schema_file, transcript_file = schema_pipeline.create_sample_files()

    return handle_success({
        'schema_file': schema_file,
        'transcript_file': transcript_file
    }, "Sample files created successfully")

    except Exception as e:
        pass
    logger.error(f"Error creating samples: {e}")
    return handle_error(f"Failed to create samples: {str(e)}")


@app.route('/api/export-data', methods=['GET'])
def export_data():
    """Export all data (schemas, reports, transcripts) as a ZIP file"""
    try:
        pass
        # Create a temporary ZIP file
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        pass
    with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        pass

        # Add schemas
    schemas_dir = Path("schemas")
    if schemas_dir.exists():
        pass
    for schema_file in schemas_dir.glob('*'):
        pass
    if schema_file.is_file():
        pass
    zipf.write(schema_file, f"schemas/{schema_file.name}")

    # Add reports
    reports_dir = Path("reports")
    if reports_dir.exists():
        pass
    for report_file in reports_dir.glob('*'):
        pass
    if report_file.is_file():
        pass
    zipf.write(report_file, f"reports/{report_file.name}")

    # Add transcripts
    transcripts_dir = Path("transcripts")
    if transcripts_dir.exists():
        pass
    for transcript_file in transcripts_dir.glob('*'):
        pass
    if transcript_file.is_file():
        pass
    zipf.write(transcript_file, f"transcripts/{transcript_file.name}")

    # Send the ZIP file
    return send_file(
        temp_zip.name,
        as_attachment=True,
        download_name=f"schema_checker_data_{
            datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mimetype='application/zip'
    )

    except Exception as e:
        pass
    logger.error(f"Error exporting data: {e}")
    return handle_error(f"Failed to export data: {str(e)}")

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
