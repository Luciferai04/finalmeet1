"""
WebSocket service for real-time communication
"""

try:
    from flask_socketio import SocketIO, emit, disconnect
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False
    # Minimal no-op stubs so the app can run without flask_socketio in CI
    class SocketIO:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def init_app(self, *args, **kwargs):
            pass
        def emit(self, *args, **kwargs):
            pass
        def on(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    def emit(*args, **kwargs):  # type: ignore
        pass
    def disconnect(*args, **kwargs):  # type: ignore
        pass

from flask import request
import logging
from datetime import datetime

# Initialize SocketIO (real or dummy)
socketio = SocketIO(cors_allowed_origins="*", logger=True, engineio_logger=True)

logger = logging.getLogger(__name__)


def handle_connect(auth):
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to translation server'})


def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


def handle_start_translation(data):
    """Handle start translation request"""
    logger.info(f"Starting translation session for client: {request.sid}")
    session_id = data.get('session_id', f'session_{request.sid}')
    target_language = data.get('target_language', 'bn')
    
    emit('translation_started', {
        'session_id': session_id,
        'target_language': target_language,
        'message': 'Translation session started'
    })


def handle_stop_translation(data):
    """Handle stop translation request"""
    logger.info(f"Stopping translation session for client: {request.sid}")
    session_id = data.get('session_id')
    
    emit('translation_stopped', {
        'session_id': session_id,
        'message': 'Translation session stopped'
    })


def handle_audio_chunk(data):
    """Handle incoming audio chunk for transcription"""
    logger.debug(f"Received audio chunk from client: {request.sid}")
    
    # In a real implementation, this would process the audio
    # For now, just acknowledge receipt
    emit('audio_received', {'status': 'received'})


# Attach event handlers only if real SocketIO is present
if HAS_SOCKETIO:
    socketio.on('connect')(handle_connect)
    socketio.on('disconnect')(handle_disconnect)
    socketio.on('start_translation')(handle_start_translation)
    socketio.on('stop_translation')(handle_stop_translation)
    socketio.on('audio_chunk')(handle_audio_chunk)


def broadcast_transcription(session_id, text, translation=None):
    """Broadcast transcription result to clients"""
    data = {
        'session_id': session_id,
        'transcription': text,
        'translation': translation,
        'timestamp': str(datetime.now())
    }
    socketio.emit('transcription_result', data, room=session_id)


def broadcast_system_status(status_data):
    """Broadcast system status to all connected clients"""
    socketio.emit('system_status', status_data)


# Error handlers
if HAS_SOCKETIO:
    @socketio.on_error()
    def error_handler(e):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {e}")
        emit('error', {'message': 'An error occurred'})

    @socketio.on_error_default
    def default_error_handler(e):
        """Default error handler"""
        logger.error(f"Unhandled WebSocket error: {e}")
        disconnect()


if __name__ == '__main__':
    # This is just for testing the module
    from flask import Flask
    from datetime import datetime
    
    app = Flask(__name__)
    socketio.init_app(app)
    
    print("WebSocket service initialized successfully")
