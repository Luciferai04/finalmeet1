"""
WebSocket service for real-time communication
"""

from flask_socketio import SocketIO, emit, disconnect
from flask import request
import logging

# Initialize SocketIO
socketio = SocketIO(cors_allowed_origins="*", logger=True, engineio_logger=True)

logger = logging.getLogger(__name__)


@socketio.on('connect')
def handle_connect(auth):
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to translation server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('start_translation')
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


@socketio.on('stop_translation')
def handle_stop_translation(data):
    """Handle stop translation request"""
    logger.info(f"Stopping translation session for client: {request.sid}")
    session_id = data.get('session_id')
    
    emit('translation_stopped', {
        'session_id': session_id,
        'message': 'Translation session stopped'
    })


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunk for transcription"""
    logger.debug(f"Received audio chunk from client: {request.sid}")
    
    # In a real implementation, this would process the audio
    # For now, just acknowledge receipt
    emit('audio_received', {'status': 'received'})


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
