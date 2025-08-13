import gradio as gr
import asyncio
import whisper
import google.generativeai as genai
from typing import Iterator
import redis
import uuid
import json
import os
from datetime import datetime
import numpy as np
import io
from pydub import AudioSegment
import tempfile
import threading
import queue
import time
import pyaudio
import cv2

# Import existing components
try:
    from .rl_coordinator import RLCoordinator
    from .schema_checker_pipeline import SchemaCheckerPipeline
    from .topic_comparator import TopicComparator
except ImportError:
    from rl_coordinator import RLCoordinator
    try:
        pass
    from src.schema_checker.pipeline import Pipeline as SchemaCheckerPipeline
    except ImportError:
        pass
    from schema_checker_pipeline import SchemaCheckerPipeline
    try:
        pass
    from topic_comparator import TopicComparator
    except ImportError:
        pass
        # Create a simple fallback

    class TopicComparator:
        pass
    def __init__(self, expected_topics=None):
        pass
    self.expected_topics = expected_topics or []
    def generate_report(self, text):
        pass
    return {"status": "Topic analysis available", "text_length": len(text)}


class RedisSessionManager:
    def __init__(self, host='localhost', port=6379):
        pass
    try:
        pass
    self.redis_client = redis.Redis(
        host=host, port=port, decode_responses=True)
    except BaseException:
        pass
    self.redis_client = None

    def create_session(self):
        pass
    session_id = str(uuid.uuid4())
    if self.redis_client:
        pass
    try:
        pass
    self.redis_client.setex(f"session:{session_id}", 3600, json.dumps({
        "created_at": datetime.now().isoformat(),
        "status": "active"
    }))
    except BaseException:
        pass
    pass
    return session_id

    def get_session(self, session_id):
        pass
    if not self.redis_client:
        pass
    return None
    try:
        pass
    data = self.redis_client.get(f"session:{session_id}")
    return json.loads(data) if data else None
    except BaseException:
        pass
    return None


class LiveAudioTranslator:
    """Handles live audio capture and real-time translation for Gradio"""

    def __init__(self, whisper_model, gemini_model, rl_coordinator):
        pass
    self.whisper_model = whisper_model
    self.gemini_model = gemini_model
    self.rl_coordinator = rl_coordinator

    # Audio settings
    self.sample_rate = 16000
    self.chunk_size = 1024
    self.channels = 1

    # Live translation state
    self.is_recording = False
    self.audio_queue = queue.Queue()
    self.audio_thread = None
    self.audio_buffer = []
    self.translation_history = []
    self.last_translation_time = 0
    self.translation_interval = 3.0  # Process every 3 seconds

    # Initialize PyAudio
    try:
        pass
    self.audio = pyaudio.PyAudio()
    self.audio_available = True
    except Exception as e:
        pass
    print(f"PyAudio not available: {e}")
    self.audio_available = False

    def start_recording(self):
        pass
    """Start live audio recording"""
    if not self.audio_available or self.is_recording:
        pass
    return False

    self.is_recording = True
    self.audio_thread = threading.Thread(target=self._record_audio)
    self.audio_thread.daemon = True
    self.audio_thread.start()
    return True

    def stop_recording(self):
        pass
    """Stop live audio recording"""
    self.is_recording = False
    if self.audio_thread:
        pass
    self.audio_thread.join(timeout=2)

    def _record_audio(self):
        pass
    """Background thread for audio recording"""
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
    print(f"Audio recording error: {e}")
    break

    stream.stop_stream()
    stream.close()

    except Exception as e:
        pass
    print(f"Failed to start audio stream: {e}")

    def get_audio_chunk(self):
        pass
    """Get audio chunk from queue"""
    try:
        pass
    return self.audio_queue.get(timeout=0.1)
    except queue.Empty:
        pass
    return None

    def transcribe_audio(self, audio_data):
        pass
    """Transcribe audio data using Whisper"""
    if self.whisper_model is None or len(audio_data) < self.sample_rate * 0.5:
        pass
    return ""

    try:
        pass
    result = self.whisper_model.transcribe(audio_data, task="transcribe")
    return result.get("text", "").strip()
    except Exception as e:
        pass
    print(f"Transcription error: {e}")
    return ""

    async def translate_text_async(self, text, target_language):
    """Translate text using RL coordinator"""
    if not text or not self.gemini_model:
        pass
    return text

    try:
        pass
    return await self.rl_coordinator.optimize_translation(
        text, target_language, self.gemini_model
    )
    except Exception as e:
        pass
    print(f"Translation error: {e}")
    return f"[Translation error: {text}]"

    def process_audio_buffer(self):
        pass
    """Process accumulated audio chunks"""
    if len(self.audio_buffer) < 32000:  # Need ~2 seconds of audio
    return None

    # Combine audio chunks
    audio_data = np.concatenate(self.audio_buffer)
    self.audio_buffer = []

    # Transcribe
    transcript = self.transcribe_audio(audio_data)
    return transcript if transcript else None

    def get_translation_status(self):
        pass
    """Get current translation status"""
    return {
        "recording": self.is_recording,
        "audio_available": self.audio_available,
        "translations_count": len(self.translation_history),
        "last_update": datetime.now().isoformat()
    }


class EnhancedGradioTranslator:
    def __init__(self):
        pass
    self.whisper_model = self.load_optimized_whisper()
    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    self.rl_coordinator = RLCoordinator()
    self.session_manager = RedisSessionManager()
    self.topic_comparator = TopicComparator(expected_topics=[
        "business", "technology", "finance", "strategy", "marketing"
    ])

    # Initialize live audio translator
    self.live_audio_translator = LiveAudioTranslator(
        self.whisper_model, self.gemini_model, self.rl_coordinator
    )

    # Set up Google API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        pass
    genai.configure(api_key=api_key)
    else:
        pass
        # For demo purposes, the API key is hardcoded.
        # It is recommended to use environment variables in production.
    genai.configure(api_key="AIzaSyCgfnxJ_kDJoFVPdJO-nmP8EkWJuLjKIyc")

    # Live translation state
    self.live_translation_active = False
    self.live_translation_thread = None

    def load_optimized_whisper(self):
        pass
    try:
        pass
    return whisper.load_model("base")
    except Exception as e:
        pass
    print(f"Error loading Whisper model: {e}")
    return None

    async def stream_audio_chunks(self, video_input, chunk_duration=2):
    """Stream audio chunks from video input"""
    if video_input is None:
        pass
    return

    try:
        pass
        # Convert video to audio chunks
    audio = AudioSegment.from_file(video_input)
    chunk_length_ms = chunk_duration * 1000

    for i in range(0, len(audio), chunk_length_ms):
        pass
    chunk = audio[i:i + chunk_length_ms]

    # Convert to temporary file for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        pass
    chunk.export(tmp_file.name, format="wav")
    yield tmp_file.name

    except Exception as e:
        pass
    print(f"Error processing audio: {e}")
    return

    async def process_video_stream(self, video_input, target_language):
    """Process video stream and return real-time translations with schema"""
    if video_input is None:
        pass
    yield "Please upload a video file", {}
    return

    session_id = self.session_manager.create_session()

    try:
        pass
    async for chunk_path in self.stream_audio_chunks(video_input):
    if self.whisper_model is None:
        pass
    transcription = "Whisper model not available"
    else:
        pass
        # RL-optimized audio processing
    transcription = await self.rl_coordinator.optimize_asr(
        chunk_path, self.whisper_model
    )

    if not transcription.strip():
        pass
    continue

    # RL-optimized translation
    translation = await self.rl_coordinator.optimize_translation(
        transcription, target_language, self.gemini_model
    )

    # Schema generation
    schema_data = await self.rl_coordinator.generate_schema(
        translation, session_id
    )

    # Add topic analysis to schema
    topic_report = self.topic_comparator.generate_report(transcription)
    schema_data['topic_analysis'] = topic_report

    # Clean up temporary file
    try:
        pass
    os.unlink(chunk_path)
    except BaseException:
        pass
    pass

    yield f"[{transcription}] → {translation}", schema_data

    except Exception as e:
        pass
    yield f"Error: {str(e)}", {"error": str(e)}

    def start_live_translation(self, target_language):
        pass
    """Start live audio translation"""
    if self.live_translation_active:
        pass
    return "Live translation already active", {"status": "already_active"}

    if not self.live_audio_translator.audio_available:
        pass
    return "Audio not available. Check microphone permissions.", {
        "status": "audio_error"}

    # Start recording
    if not self.live_audio_translator.start_recording():
        pass
    return "Failed to start audio recording", {"status": "recording_error"}

    self.live_translation_active = True

    # Start translation thread
    self.live_translation_thread = threading.Thread(
        target=self._live_translation_worker,
        args=(target_language,)
    )
    self.live_translation_thread.daemon = True
    self.live_translation_thread.start()

    return f"[PASS] Live translation started (Language: {target_language})", {
        "status": "recording",
        "language": target_language,
        "started_at": datetime.now().isoformat()
    }

    def stop_live_translation(self):
        pass
    """Stop live audio translation"""
    if not self.live_translation_active:
        pass
    return "No active live translation", {"status": "not_active"}

    self.live_translation_active = False
    self.live_audio_translator.stop_recording()

    # Save session
    translations = self.live_audio_translator.translation_history
    if translations:
        pass
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"live_translation_session_{timestamp}.json"

    session_data = {
        "timestamp": datetime.now().isoformat(),
        "translations": translations
    }

    try:
        pass
    with open(filename, 'w', encoding='utf-8') as f:
        pass
    json.dump(session_data, f, indent=2, ensure_ascii=False)
    save_message = f"Session saved to {filename}"
    except Exception as e:
        pass
    save_message = f"Failed to save session: {e}"
    else:
        pass
    save_message = "No translations to save"

    return f" Live translation stopped. {save_message}", {
        "status": "stopped",
        "translations_count": len(translations),
        "stopped_at": datetime.now().isoformat()
    }

    def _live_translation_worker(self, target_language):
        pass
    """Background worker for live translation"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        pass
    loop.run_until_complete(self._live_translation_loop(target_language))
    finally:
        pass
    loop.close()

    async def _live_translation_loop(self, target_language):
    """Main live translation processing loop"""
    translation_count = 0

    while self.live_translation_active:
        pass
    try:
        pass
    current_time = time.time()

    # Collect audio
    audio_chunk = self.live_audio_translator.get_audio_chunk()
    if audio_chunk is not None:
        pass
    self.live_audio_translator.audio_buffer.append(audio_chunk)

    # Process translation periodically
    if current_time - self.live_audio_translator.last_translation_time >= self.live_audio_translator.translation_interval:
        pass
    transcript = self.live_audio_translator.process_audio_buffer()

    if transcript and len(transcript.strip()) > 3:
        pass
    translation = await self.live_audio_translator.translate_text_async(
        transcript, target_language
    )
    translation_count += 1

    # Store in history
    entry = {
        "timestamp": datetime.now().isoformat(),
        "original": transcript,
        "translated": translation,
        "language": target_language
    }
    self.live_audio_translator.translation_history.append(entry)

    print(
        f"Live Translation #{translation_count}: {transcript} → {translation}")

    self.live_audio_translator.last_translation_time = current_time

    await asyncio.sleep(0.1)

    except Exception as e:
        pass
    print(f"Error in live translation loop: {e}")
    await asyncio.sleep(1)

    def get_live_translation_history(self):
        pass
    """Get current live translation history"""
    translations = self.live_audio_translator.translation_history
    if not translations:
        pass
    return "No translations yet", {}

    # Format for display
    display_text = "\n".join([
        f"[{i + 1}] {t['original']} → {t['translated']}"
        for i, t in enumerate(translations[-10:])  # Show last 10
    ])

    status = {
        "total_translations": len(translations),
        "recording": self.live_audio_translator.is_recording,
        "last_update": datetime.now().isoformat()
    }

    return display_text, status

    def create_interface(self):
        pass
    with gr.Blocks(title="Real-Time Translator with Live Camera Support", theme=gr.themes.Soft()) as interface:
        pass
    gr.Markdown(
        '''
 # Real-Time Translator with Live Camera Support

 Choose your preferred translation mode: Upload video files, or use live camera with real-time audio translation.
 Features: RL-optimized translation, progress tracking, and cultural context handling.
 '''
    )

    with gr.Tabs():
        pass
        # Tab 1: Live Camera Translation
    with gr.Tab(" Live Camera Translation", id="live_tab"):
        pass
    gr.Markdown("### Real-Time Audio Translation")
    gr.Markdown(
        "Capture audio in real-time and get instant translations using advanced AI models.")

    with gr.Row():
        pass
    with gr.Column(scale=1):
        pass
    live_language = gr.Dropdown(
        choices=[
            "Bengali",
            "Hindi",
            "Spanish",
            "French",
            "German",
            "Japanese"],
        value="Bengali",
        label=" Target Language"
    )

    with gr.Row():
        pass
    start_live_btn = gr.Button(" Start Live Translation", variant="primary")
    stop_live_btn = gr.Button(" Stop Translation", variant="stop")

    live_status = gr.Textbox(
        label="Status",
        value="Ready to start live translation",
        interactive=False
    )

    with gr.Column(scale=2):
        pass
    live_translation_output = gr.Textbox(
        label="Live Translations",
        lines=15,
        max_lines=25,
        placeholder="Live translations will appear here...",
        interactive=False
    )

    live_schema_output = gr.JSON(
        label="Live Translation Status",
        value={"status": "Ready to start"}
    )

    # Update button for refreshing live translations
    refresh_btn = gr.Button(" Refresh Translations")

    gr.Markdown(
        '''
 **Instructions for Live Translation:**
 1. Select your target language
 2. Click "Start Live Translation" (allow microphone access)
 3. Speak clearly into your microphone
 4. View real-time translations above
 5. Click "Stop Translation" when done

 **System Requirements:**
 - Microphone access permission
 - Stable internet connection for translation API
 - Modern browser with WebRTC support
 '''
    )

    # Tab 2: Video File Translation
    with gr.Tab(" Video File Translation", id="video_tab"):
        pass
    with gr.Row():
        pass
    with gr.Column(scale=1):
        pass
    gr.Markdown("### Video Input")
    video_source = gr.Radio(
        choices=["Upload File", "Webcam"],
        value="Upload File",
        label="Video Source"
    )

    video_file = gr.Video(
        label="Upload Video File",
        visible=True
    )

    webcam_input = gr.Video(
        label="Webcam Input",
        sources=["webcam"],
        visible=False
    )

    target_language = gr.Dropdown(
        choices=["Bengali", "Hindi"],
        value="Bengali",
        label=" Target Language"
    )

    translate_btn = gr.Button(" Start Translation", variant="primary")

    with gr.Column(scale=2):
        pass
    gr.Markdown("### Translation Output")
    translation_output = gr.Textbox(
        label="Real-time Translation",
        lines=10,
        max_lines=20,
        placeholder="Translation will appear here..."
    )

    gr.Markdown("### Progress Schema")
    schema_output = gr.JSON(
        label="Translation Progress & Metrics",
        value={"status": "Ready to translate"}
    )

    # Tab 3: Quick Demo
    with gr.Tab(" Quick Demo", id="demo_tab"):
        pass
    gr.Markdown("### Quick Text Translation Demo")

    with gr.Row():
        pass
    demo_text = gr.Textbox(
        label="Sample Text",
        value="Hello, how are you? Good morning!",
        placeholder="Enter text to translate..."
    )
    demo_lang = gr.Dropdown(
        choices=[
            "Bengali",
            "Hindi",
            "Spanish",
            "French",
            "German",
            "Japanese"],
        value="Bengali",
        label="Target Language"
    )
    demo_btn = gr.Button(" Quick Translate")

    demo_output = gr.Textbox(
        label="Demo Translation",
        lines=5
    )

    # Tab 4: Topic Analysis
    with gr.Tab(" Topic Analysis", id="analysis_tab"):
        pass
    gr.Markdown("## Automated Topic Analysis")
    gr.Markdown(
        "Upload a class schema (JSON, YAML, or CSV) and a session transcript (.txt) "
        "to automatically evaluate which topics were covered."
    )
    with gr.Row():
        pass
    with gr.Column():
        pass
    schema_file_upload = gr.File(
        label="Upload Class Schema (.json, .yaml, .csv)",
        file_types=[".json", ".yaml", ".yml", ".csv"],
    )
    transcript_file_upload = gr.File(
        label="Upload Session Transcript (.txt)", file_types=[".txt"]
    )
    analyze_button = gr.Button("Analyze Topics", variant="primary")
    with gr.Column():
        pass
    gr.Markdown("### Analysis Report")
    analysis_report_display = gr.JSON(label="Analysis Report")
    report_download_link = gr.File(label="Download Report")

    # Event handlers
    def handle_video_source_change(source):
        pass
    if source == "Upload File":
        pass
    return gr.Video(visible=True), gr.Video(visible=False)
    else:
        pass
    return gr.Video(visible=False), gr.Video(visible=True)

    def process_translation(
            video_file_input, webcam_input, target_lang, video_source):
    if video_source == 'Upload File':
        pass
    video_input = video_file_input
    else:
        pass
    video_input = webcam_input

    if video_input is None:
        pass
    yield "Please upload a video file", {}
    return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        pass
    async_gen = self.process_video_stream(video_input, target_lang)
    while True:
        pass
    try:
        pass
    yield loop.run_until_complete(async_gen.__anext__())
    except StopAsyncIteration:
        pass
    break
    except Exception as e:
        pass
    yield f"Error: {str(e)}", {"error": str(e)}
    finally:
        pass
    loop.close()

    def quick_translate(text, lang):
        pass
    if not text.strip():
        pass
    return "[WARNING] Please enter some text to translate."

    try:
        pass
        # Use RL coordinator for translation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        pass
    result = loop.run_until_complete(
        self.rl_coordinator.optimize_translation(text, lang, self.gemini_model)
    )
    return f" {lang} Translation:\n\n{result}"
    finally:
        pass
    loop.close()

    except Exception as e:
        pass
    return f"[FAIL] Translation Error: {str(e)}"

    def run_topic_analysis(schema_file, transcript_file):
        pass
    if schema_file is None or transcript_file is None:
        pass
    return None, None

    try:
        pass
    schema_path = schema_file.name
    transcript_path = transcript_file.name

    with open(transcript_path, 'r', encoding='utf-8') as f:
        pass
    transcript_text = f.read()

    pipeline = SchemaCheckerPipeline()
    report_data, report_path = pipeline.run(schema_path, transcript_text)

    return report_data, report_path
    except Exception as e:
        pass
    return {"error": str(e)}, None

    # Connect events
    video_source.change(
        handle_video_source_change,
        inputs=[video_source],
        outputs=[video_file, webcam_input]
    )

    translate_btn.click(
        process_translation,
        inputs=[video_file, webcam_input, target_language, video_source],
        outputs=[translation_output, schema_output]
    )

    demo_btn.click(
        quick_translate,
        inputs=[demo_text, demo_lang],
        outputs=[demo_output]
    )

    analyze_button.click(
        run_topic_analysis,
        inputs=[schema_file_upload, transcript_file_upload],
        outputs=[analysis_report_display, report_download_link],
    )

    # Live translation events
    start_live_btn.click(
        self.start_live_translation,
        inputs=[live_language],
        outputs=[live_status, live_schema_output]
    )

    stop_live_btn.click(
        self.stop_live_translation,
        outputs=[live_status, live_schema_output]
    )

    refresh_btn.click(
        self.get_live_translation_history,
        outputs=[live_translation_output, live_schema_output]
    )

    return interface


def create_app():
    translator = EnhancedGradioTranslator()
    return translator.create_interface()


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
