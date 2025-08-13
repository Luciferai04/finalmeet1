"""
Console-Only Live Camera Real-Time Audio Translation System

This module provides a console-based live camera interface for systems
where GUI/tkinter is not available.
"""

import cv2
import pyaudio
import threading
import queue
import time
import numpy as np
from typing import Dict, Any, Optional
import asyncio
import logging
import json
from datetime import datetime
import google.generativeai as genai
import whisper

# Import existing components
try:
    from .rl_coordinator import RLCoordinator
    from .flask_api import RedisSessionManager
    from .schema_checker_pipeline import SchemaCheckerPipeline
except ImportError:
    pass
    # For direct execution
    from rl_coordinator import RLCoordinator
    from flask_api import RedisSessionManager
    from schema_checker_pipeline import SchemaCheckerPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsoleAudioProcessor:
    """Handles real-time audio capture and processing for console mode"""

    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        pass
    self.sample_rate = sample_rate
    self.chunk_size = chunk_size
    self.channels = channels
    self.is_recording = False
    self.audio_queue = queue.Queue()
    self.audio_thread = None

    # Initialize PyAudio
    self.audio = pyaudio.PyAudio()

    # Load Whisper model for ASR
    try:
        pass
    self.whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
    except Exception as e:
        pass
    logger.error(f"Failed to load Whisper model: {e}")
    self.whisper_model = None

    def start_recording(self):
        pass
    """Start real-time audio recording"""
    if self.is_recording:
        pass
    return

    self.is_recording = True
    self.audio_thread = threading.Thread(target=self._record_audio)
    self.audio_thread.daemon = True
    self.audio_thread.start()
    logger.info("Audio recording started")

    def stop_recording(self):
        pass
    """Stop audio recording"""
    self.is_recording = False
    if self.audio_thread:
        pass
    self.audio_thread.join()
    logger.info("Audio recording stopped")

    def _record_audio(self):
        pass
    """Internal method to record audio in a separate thread"""
    try:
        pass
    stream = self.audio.open(
        format=pyaudio.paFloat32,
        channels=self.channels,
        rate=self.sample_rate,
        input=True,
        frames_per_buffer=self.chunk_size
    )

    while self.is_recording:
        pass
    try:
        pass
    data = stream.read(self.chunk_size, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.float32)
    self.audio_queue.put(audio_data)
    except Exception as e:
        pass
    logger.error(f"Error reading audio: {e}")
    break

    stream.stop_stream()
    stream.close()

    except Exception as e:
        pass
    logger.error(f"Audio recording error: {e}")

    def get_audio_chunk(self, timeout=0.1):
        pass
    """Get the latest audio chunk"""
    try:
        pass
    return self.audio_queue.get(timeout=timeout)
    except queue.Empty:
        pass
    return None

    def transcribe_audio(self, audio_data, language="en"):
        pass
    """Transcribe audio data to text using Whisper"""
    if self.whisper_model is None:
        pass
    return ""

    try:
        pass
    if len(audio_data) < self.sample_rate * 0.5:  # Less than 0.5 seconds
    return ""

    result = self.whisper_model.transcribe(
        audio_data,
        language=language,
        task="transcribe"
    )
    return result.get("text", "").strip()
    except Exception as e:
        pass
    logger.error(f"Transcription error: {e}")
    return ""


class ConsoleVideoDisplay:
    """Simple console-based video status display"""

    def __init__(self):
        pass
    self.is_recording = False
    self.translation_count = 0
    self.start_time = None

    def start(self):
        pass
    """Start video status tracking"""
    self.is_recording = True
    self.start_time = time.time()
    logger.info("Video status tracking started")

    def stop(self):
        pass
    """Stop video status tracking"""
    self.is_recording = False
    logger.info("Video status tracking stopped")

    def update_status(self, transcript="", translation="", language=""):
        pass
    """Update console with latest translation"""
    self.translation_count += 1
    elapsed = time.time() - self.start_time if self.start_time else 0

    print("\n" + "=" * 60)
    print(f" LIVE TRANSLATION #{self.translation_count}")
    print(f"⏱ Elapsed: {elapsed:.1f}s | Language: {language}")
    print("-" * 60)
    if transcript:
        pass
    print(f" Original: {transcript}")
    if translation:
        pass
    print(f" Translated: {translation}")
    print("=" * 60)


class ConsoleLiveCameraTranslator:
    """Console-only version of live camera translator"""

    def __init__(self, target_language="Bengali", enable_video_check=True):
        pass
    self.target_language = target_language
    self.enable_video_check = enable_video_check

    # Initialize components
    self.audio_processor = ConsoleAudioProcessor()
    self.video_display = ConsoleVideoDisplay()
    self.rl_coordinator = RLCoordinator()
    self.session_manager = RedisSessionManager()
    self.schema_pipeline = SchemaCheckerPipeline()

    # Initialize Gemini
    try:
        pass
    import os
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        pass
    genai.configure(api_key=api_key)
    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini model initialized")
    else:
        pass
    logger.warning("GOOGLE_API_KEY not found, using mock translation")
    self.gemini_model = None
    except Exception as e:
        pass
    logger.error(f"Failed to initialize Gemini: {e}")
    self.gemini_model = None

    # Translation state
    self.is_running = False
    self.current_session_id = None
    self.audio_buffer = []
    self.translation_history = []
    self.last_translation_time = 0
    self.translation_interval = 3.0  # Translate every 3 seconds for console

    # Camera for optional video check
    self.camera_available = False
    if enable_video_check:
        pass
    self.check_camera_availability()

    def check_camera_availability(self):
        pass
    """Check if camera is available"""
    try:
        pass
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        pass
    self.camera_available = True
    cap.release()
    logger.info("Camera is available")
    else:
        pass
    logger.warning("Camera not available")
    except Exception as e:
        pass
    logger.warning(f"Camera check failed: {e}")

    def start_translation_session(self):
        pass
    """Start a new translation session"""
    try:
        pass
    self.current_session_id = self.session_manager.create_session()
    logger.info(f"Started translation session: {self.current_session_id}")
    return self.current_session_id
    except Exception as e:
        pass
    logger.error(f"Failed to start session: {e}")
    return None

    def stop_translation_session(self):
        pass
    """Stop the current translation session"""
    if self.current_session_id:
        pass
    try:
        pass
    self.session_manager.close_session(self.current_session_id)
    logger.info(f"Stopped translation session: {self.current_session_id}")
    except Exception as e:
        pass
    logger.error(f"Error stopping session: {e}")
    finally:
        pass
    self.current_session_id = None

    async def translate_text(self, text):
    """Translate text using the RL-optimized system"""
    if not text:
        pass
    return text

    if not self.gemini_model:
        pass
        # Mock translation for demo purposes
    if self.target_language.lower() == "bengali":
        pass
    return f"[Bengali translation of: {text}]"
    elif self.target_language.lower() == "hindi":
        pass
    return f"[Hindi translation of: {text}]"
    else:
        pass
    return f"[{self.target_language} translation of: {text}]"

    try:
        pass
    translation = await self.rl_coordinator.optimize_translation(
        text, self.target_language, self.gemini_model
    )
    return translation
    except Exception as e:
        pass
    logger.error(f"Translation error: {e}")
    return f"[Translation error: {text}]"

    def process_audio_chunk(self):
        pass
    """Process accumulated audio chunks for transcription"""
    if len(self.audio_buffer) < 32000:  # Need at least 2 seconds of audio for console
    return None

    # Combine audio chunks
    audio_data = np.concatenate(self.audio_buffer)
    self.audio_buffer = []  # Clear buffer

    # Transcribe audio
    transcript = self.audio_processor.transcribe_audio(audio_data)

    if transcript:
        pass
    logger.info(f"Transcribed: {transcript}")

    return transcript

    async def translation_loop(self):
    """Main translation processing loop"""
    print(
        f"\n Listening for audio... (Interval: {
            self.translation_interval}s)")
    print("[TIP] Speak clearly into your microphone")
    print(" Press Ctrl+C to stop\n")

    while self.is_running:
        pass
    try:
        pass
    current_time = time.time()

    # Get audio chunk
    audio_chunk = self.audio_processor.get_audio_chunk()
    if audio_chunk is not None:
        pass
    self.audio_buffer.append(audio_chunk)

    # Process translation if enough time has passed
    if current_time - self.last_translation_time >= self.translation_interval:
        pass
    transcript = self.process_audio_chunk()

    if transcript and len(
            transcript.strip()) > 5:  # Only process meaningful text
        # Translate the transcript
    translation = await self.translate_text(transcript)

    # Store in history
    translation_entry = {
        "timestamp": datetime.now().isoformat(),
        "original": transcript,
        "translated": translation,
        "language": self.target_language
    }
    self.translation_history.append(translation_entry)

    # Update console display
    self.video_display.update_status(
        transcript, translation, self.target_language
    )

    # Update session
    if self.current_session_id:
        pass
    try:
        pass
    self.session_manager.update_session(
        self.current_session_id,
        {"last_translation": translation_entry}
    )
    except BaseException:
        pass
    pass  # Continue even if session update fails

    self.last_translation_time = current_time

    await asyncio.sleep(0.1)  # Small delay to prevent high CPU usage

    except Exception as e:
        pass
    logger.error(f"Error in translation loop: {e}")
    await asyncio.sleep(1)  # Longer delay on error

    def save_session(self):
        pass
    """Save the current translation session"""
    if not self.translation_history:
        pass
    print("No translations to save")
    return

    try:
        pass
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"console_translation_session_{timestamp}.json"

    session_data = {
        "session_id": self.current_session_id,
        "target_language": self.target_language,
        "timestamp": datetime.now().isoformat(),
        "translations": self.translation_history,
        "camera_available": self.camera_available
    }

    with open(filename, 'w', encoding='utf-8') as f:
        pass
    json.dump(session_data, f, indent=2, ensure_ascii=False)

    print(f"[PASS] Session saved to {filename}")
    logger.info(f"Session saved to {filename}")

    except Exception as e:
        pass
    print(f"[FAIL] Error saving session: {e}")
    logger.error(f"Error saving session: {e}")

    def run(self):
        pass
    """Run the console live camera translator"""
    print(f"\n Console Live Camera Translator")
    print(f"Target Language: {self.target_language}")
    print(f"Camera Available: {'Yes' if self.camera_available else 'No'}")
    print(
        f"API Key: {
            'Configured' if self.gemini_model else 'Not Set (using mock)'}")
    print("=" * 50)

    try:
        pass
        # Start session
    self.start_translation_session()

    # Start audio recording
    self.audio_processor.start_recording()

    # Start video status
    self.video_display.start()

    # Set running flag
    self.is_running = True

    # Start translation loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(self.translation_loop())

    except KeyboardInterrupt:
        pass
    print("\n\n Translation stopped by user")
    except Exception as e:
        pass
    print(f"\n[FAIL] Error: {e}")
    logger.error(f"Runtime error: {e}")
    finally:
        pass
        # Cleanup
    self.is_running = False
    self.audio_processor.stop_recording()
    self.video_display.stop()

    # Save session
    if self.translation_history:
        pass
    print(f"\n Saving {len(self.translation_history)} translations...")
    self.save_session()

    self.stop_translation_session()
    print(" Translation session ended")


def main():
    """Main entry point for console translator"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Console Live Camera Real-Time Translator")
    parser.add_argument(
        "--language",
        default="Bengali",
        help="Target language for translation")
    parser.add_argument(
        "--no-camera-check",
        action="store_true",
        help="Skip camera availability check")

    args = parser.parse_args()

    # Create and run translator
    translator = ConsoleLiveCameraTranslator(
        target_language=args.language,
        enable_video_check=not args.no_camera_check
    )

    translator.run()


if __name__ == "__main__":
    main()
