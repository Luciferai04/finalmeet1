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
        from src.schema_checker.pipeline import Pipeline as SchemaCheckerPipeline
    except ImportError:
        from schema_checker_pipeline import SchemaCheckerPipeline
    try:
        from topic_comparator import TopicComparator
    except ImportError:
        # Create a simple fallback
        class TopicComparator:
            def __init__(self, expected_topics=None):
                self.expected_topics = expected_topics or []
            
            def generate_report(self, text):
                return {"status": "Topic analysis available", "text_length": len(text)}


class RedisSessionManager:
    def __init__(self, host='localhost', port=6379):
        try:
            self.redis_client = redis.Redis(
                host=host, port=port, decode_responses=True)
        except BaseException as e:
            print(f"Redis connection failed: {e}")
            self.redis_client = None

    def create_session(self):
        session_id = str(uuid.uuid4())
        if self.redis_client:
            try:
                self.redis_client.setex(f"session:{session_id}", 3600, json.dumps({
                    "created_at": datetime.now().isoformat(),
                    "status": "active"
                }))
            except BaseException as e:
                print(f"Failed to create session: {e}")
        return session_id

    def get_session(self, session_id):
        if not self.redis_client:
            return None
        try:
            data = self.redis_client.get(f"session:{session_id}")
            return json.loads(data) if data else None
        except BaseException as e:
            print(f"Error retrieving session: {e}")
            return None


class LiveAudioTranslator:
    """Handles live audio capture and real-time translation for Gradio"""

    def __init__(self, whisper_model, gemini_model, rl_coordinator):
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
            self.audio = pyaudio.PyAudio()
            self.audio_available = True
        except Exception as e:
            print(f"PyAudio not available: {e}")
            self.audio_available = False

    def start_recording(self):
        """Start live audio recording"""
        if not self.audio_available or self.is_recording:
            return False

        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        return True

    def stop_recording(self):
        """Stop live audio recording"""
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2)

    def _record_audio(self):
        pass
    """Background thread for audio recording"""
try:
    stream = self.audio.open(
        format=pyaudio.paFloat32,
        channels=self.channels,
        rate=self.sample_rate,
        input=True,
        frames_per_buffer=self.chunk_size
    )

    while self.is_recording:
        try:
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        except Exception as e:
            print(f"Audio recording error: {e}")
            break

    stream.stop_stream()
    stream.close()
except Exception as e:
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


class GradioVideoTranslator:
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

    def create_interface(self):
        pass
    with gr.Blocks(title="Real-Time Video Translator with RL Optimization", theme=gr.themes.Soft()) as interface:
        pass
    gr.Markdown(
        '''
 # Real-Time Video Translator with RL Optimization

 Choose your preferred translation mode: Video upload processing, Live camera with real-time audio, or Topic analysis.
 Features: RL-optimized translation, progress tracking, and cultural context handling.
 '''
    )

    with gr.Tabs():
        pass
        # Tab 1: Video File Translation
    with gr.Tab(" Video Translation", id="video_tab"):
        pass
    with gr.Row():
        pass
    with gr.Column(scale=1):
        pass
        # Video input options
    gr.Markdown("### Video Input")
    video_source = gr.Radio(
        choices=["Upload File", "Webcam"],
        value="Upload File",
        label="Video Source"
    )

    # File upload
    video_file = gr.Video(
        label="Upload Video File",
        visible=True
    )

    # Webcam interface
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

    # Demo section
    with gr.Row():
        pass
    gr.Markdown(
        '''
 ### Quick Demo
 Try these sample texts for immediate translation:
 '''
    )

    with gr.Row():
        pass
    demo_text = gr.Textbox(
        label="Sample Text",
        value="Hello, how are you? Good morning!",
        placeholder="Enter text to translate..."
    )
    demo_lang = gr.Dropdown(
        choices=["Bengali", "Hindi"],
        value="Bengali",
        label="Target Language"
    )
    demo_btn = gr.Button(" Quick Translate")

    demo_output = gr.Textbox(
        label="Demo Translation",
        lines=5
    )

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
    import asyncio
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

    # Instructions
    with gr.Tab("Topic Analysis"):
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
        # Using gr.Warning for better visibility
    gr.Warning(f"An error occurred: {e}")
    return {"error": str(e)}, None

    analyze_button.click(
        run_topic_analysis,
        inputs=[schema_file_upload, transcript_file_upload],
        outputs=[analysis_report_display, report_download_link],
    )
    gr.Markdown(
        '''
 ### Instructions

 **For Video Translation:**
 1. Choose "Upload File" and select a video file, or
 2. Choose "Webcam" to use your camera (requires browser permissions)
 3. Select target language (Bengali/Hindi)
 4. Click "Start Translation"

 **For Quick Text Demo:**
 1. Enter any English text in the demo box
 2. Select target language
 3. Click "Quick Translate"

 **Features:**
 - RL-optimized translation quality
 - Kolkata business terminology support
 - Real-time progress tracking
 - Cultural context handling
 - Performance metrics
 '''
    )

    return interface


def create_app():
    translator = GradioVideoTranslator()
    return translator.create_interface()


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
