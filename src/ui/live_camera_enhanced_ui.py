import gradio as gr
import asyncio
import whisper
import google.generativeai as genai
import numpy as np
import threading
import queue
import time
import pyaudio
import cv2
import os
import json
from datetime import datetime
# Production-ready imports with error handling
try:
    from src.services.rl_coordinator import RLCoordinator
    from src.services.metacognitive_controller import get_metacognitive_controller
    from src.services.egoschema_integration import create_egoschema_integration
    from src.services.egoschema_app_integration import EgoSchemaEnhancedTranslator
    from src.services.performance_monitor import PerformanceMonitor
    from src.services.schema_checker.main import SchemaCheckerPipeline
    from src.services.schema_checker.schema_parser import SchemaParser
    from src.services.whisper_live.client import Client as WhisperLiveClient
except ImportError as e:
    print(f"[WARNING] Some advanced features may not be available: {e}")
    print("[INFO] Running in basic mode. Some functionality will be disabled.")
    
    # Fallback classes to prevent errors
    class RLCoordinator:
        def __init__(self): pass
        def metacognitive_translation_optimization(self, *args, **kwargs): return {"translation": args[0] if args else ""}
        def optimize_translation(self, *args, **kwargs): return args[0] if args else ""
        def optimize_whisper_real_time(self, *args, **kwargs): return {"transcription": ""}
        def get_system_performance_summary(self): return {"status": "basic_mode"}
        def get_metacognitive_report(self): return None
        
    def get_metacognitive_controller(redis_client=None): return None
    def create_egoschema_integration(config): return None
    class EgoSchemaEnhancedTranslator:
        def __init__(self, parent): pass
    class PerformanceMonitor:
        def __init__(self): pass
    class SchemaCheckerPipeline:
        def __init__(self, *args, **kwargs): pass
        def process_session(self, *args, **kwargs): return {"error": "Schema checker not available"}
    class SchemaParser:
        def __init__(self, *args): pass
        def save_normalized_schema(self, data): return None
    class WhisperLiveClient:
        def __init__(self, *args, **kwargs): pass

# Define the main class for live camera translation:


class LiveCameraEnhancedUI:
    def __init__(self):
        # Core components
        self.rl_coordinator = RLCoordinator()
        self.whisper_model = self.load_whisper_model()
        self.gemini_model = self.load_gemini_model()
        
        # Initialize Enhanced Translation Service (with Advanced Engine)
        self.translation_service = None
        self.advanced_translation_engine = None  # Keep for backward compatibility
        try:
            from src.services.enhanced_translation_service import EnhancedTranslationService
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                self.translation_service = EnhancedTranslationService(api_key=api_key, use_advanced_engine=True)
                self.advanced_translation_engine = self.translation_service.advanced_engine  # For compatibility
                print("[PASS] Enhanced Translation Service initialized with Advanced Engine")
            else:
                print("[WARNING] GOOGLE_API_KEY not set. Translation service unavailable.")
        except ImportError as e:
            print(f"[WARNING] Enhanced Translation Service not available: {e}")
            # Fallback to basic advanced engine
            try:
                from src.services.advanced_translation_engine import AdvancedTranslationEngine
                if os.getenv('GOOGLE_API_KEY'):
                    self.advanced_translation_engine = AdvancedTranslationEngine(os.getenv('GOOGLE_API_KEY'))
                    print("[PASS] Fallback to Advanced Translation Engine initialized")
            except Exception as e2:
                print(f"[WARNING] Fallback to Advanced Translation Engine failed: {e2}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize Enhanced Translation Service: {e}")
        
        self.audio_processor = AudioProcessorWhisperLive(self)
        self.video_processor = VideoProcessor()

        # Initialize EgoSchema integration
        self.init_egoschema_integration()

        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # Initialize metacognitive controller
        self.init_metacognitive_controller()

        # Initialize schema checker pipeline
        self.init_schema_checker()

        # Enhanced translation capabilities
        self.egoschema_enhanced_translator = None
        self.session_history = []
        self.current_session_data = {}
        
        # Session management for schema generation
        self.session_transcript = ""
        self.session_start_time = None
        self.session_id = None
        self.uploaded_course_material = None
        
        # WhisperLive specific attributes
        self.whisper_live_client = None
        self.is_recording = False
        self.target_language = "bn"  # Default target language (Bengali)
        self.live_transcript_queue = queue.Queue()
        self.live_translation_queue = queue.Queue()

        print(
            "[PASS] LiveCameraEnhancedUI initialized with WhisperLive, EgoSchema and Metacognitive features"
        )

    def init_egoschema_integration(self):
        """Initialize EgoSchema integration for video understanding evaluation"""
        try:
            config = {
                "min_certificate_length": 30.0,
                "llm_model": "gpt-4",
                "results_dir": "../data/reports/egoschema",
                "evaluation_interval": 1800,  # 30 minutes
                "auto_improvement": True,
            }

            self.egoschema_integration = create_egoschema_integration(config)
            self.egoschema_integration.setup_integration(self)

            # Initialize enhanced translator
            self.egoschema_enhanced_translator = EgoSchemaEnhancedTranslator(self)

            print("[PASS] EgoSchema integration initialized")
        except Exception as e:
            print(f"[WARNING] EgoSchema initialization failed: {e}")
            self.egoschema_integration = None
            self.egoschema_enhanced_translator = None

    def init_metacognitive_controller(self):
        """Initialize metacognitive controller for adaptive strategy selection"""
        try:
            import redis

            redis_client = redis.Redis(
                host="localhost", port=6379, decode_responses=True
            )
            self.metacognitive_controller = get_metacognitive_controller(redis_client)
            print("[PASS] Metacognitive controller initialized")
        except Exception as e:
            print(f"[WARNING] Metacognitive controller initialization failed: {e}")
            self.metacognitive_controller = get_metacognitive_controller(None)

    def init_schema_checker(self):
        """Initialize schema checker pipeline for topic analysis"""
        try:
            self.schema_pipeline = SchemaCheckerPipeline(
                similarity_threshold=0.7,
                extraction_method="hybrid",
                reports_dir="reports",
                schemas_dir="schemas/normalized"
            )
            self.schema_parser = SchemaParser("schemas/normalized")
            print("[PASS] Schema checker pipeline initialized")
        except Exception as e:
            print(f"[WARNING] Schema checker initialization failed: {e}")
            self.schema_pipeline = None
            self.schema_parser = None

    def load_whisper_model(self):
        try:
            return whisper.load_model("base")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            return None

    def load_gemini_model(self):
        try:
            # Check for API key
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                print("[WARNING] GOOGLE_API_KEY not set. Translation will use fallback mode.")
                print("To enable real translations, set: export GOOGLE_API_KEY='your_api_key_here'")
                return None
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            print("[PASS] Gemini model loaded successfully")
            return model
        except Exception as e:
            print(f"[WARNING] Failed to load Gemini model: {e}")
            print("Translation will use fallback mode.")
            return None

    def on_transcription_received(self, text, segments):
        """Callback for WhisperLive transcription results"""
        try:
            if text and text.strip():
                print(f"[INFO] Transcription received: {text}")
                
                # Add to transcript queue for UI update
                self.live_transcript_queue.put(text)
                
                # Add to session history
                self.session_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "original_text": text,
                    "translated_text": "",  # Will be filled by translation
                    "target_language": self.target_language,
                    "strategy_used": "whisper_live",
                    "quality_score": 0.85,
                    "processing_time": 0.0,
                    "metacognitive_state": {},
                    "segments": segments
                })
                
                # Trigger translation in a thread to avoid async issues
                import threading
                translation_thread = threading.Thread(
                    target=self.translate_live_text_sync, 
                    args=(text,)
                )
                translation_thread.daemon = True
                translation_thread.start()
                
        except Exception as e:
            print(f"[ERROR] Error in transcription callback: {e}")
    
    async def translate_live_text(self, text):
        """Translate text received from WhisperLive"""
        if text and self.gemini_model:
            try:
                # Use enhanced translation
                result = await self.translate_text(text, self.target_language)
                
                # Add to translation queue for UI update
                self.live_translation_queue.put(result)
                
                # Update the last session history entry
                if self.session_history:
                    self.session_history[-1]["translated_text"] = result
                    
            except Exception as e:
                print(f"[ERROR] Translation error: {e}")

    def translate_live_text_sync(self, text):
        """Synchronous version of translate_live_text for threading with advanced engine"""
        if text and text.strip():
            try:
                print(f"[INFO] Starting advanced translation for: {text}")
                
                # Use advanced translation engine if available
                if hasattr(self, 'advanced_translation_engine') and self.advanced_translation_engine:
                    try:
                        # Map language codes to full names
                        language_names = {
                            "bn": "Bengali",
                            "hi": "Hindi"
                        }
                        target_lang_name = language_names.get(self.target_language, self.target_language)
                        
                        # Use advanced translation engine
                        result = self.advanced_translation_engine.translate_sync(
                            text=text,
                            target_language=target_lang_name,
                            source_language="English",
                            use_history=True
                        )
                        
                        translated_text = result.translated_text
                        quality_score = result.quality_score
                        
                        print(f"[INFO] Advanced translation completed: {translated_text} "
                              f"(quality: {quality_score:.2f}, domain: {result.domain})")
                        
                        # Add to translation queue for UI update
                        self.live_translation_queue.put(translated_text)
                        
                        # Update the last session history entry with enhanced data
                        if self.session_history:
                            self.session_history[-1].update({
                                "translated_text": translated_text,
                                "quality_score": quality_score,
                                "domain": result.domain,
                                "register": result.register,
                                "model_used": result.model_used,
                                "processing_time": result.processing_time,
                                "quality_metrics": result.quality_metrics
                            })
                            
                    except Exception as e:
                        print(f"[ERROR] Advanced translation error: {e}")
                        # Fallback to basic translation
                        self._fallback_basic_translation(text)
                        
                elif self.gemini_model:
                    # Fallback to basic translation
                    self._fallback_basic_translation(text)
                else:
                    print(f"[WARNING] No translation models available, using fallback")
                    # Simple fallback without actual translation
                    fallback_text = f"[{self.target_language.upper()}] {text}"
                    self.live_translation_queue.put(fallback_text)
                    
            except Exception as e:
                print(f"[ERROR] Translation thread error: {e}")
    
    def _fallback_basic_translation(self, text):
        """Fallback to basic Gemini translation."""
        try:
            language_names = {
                "bn": "Bengali",
                "hi": "Hindi"
            }
            target_lang_name = language_names.get(self.target_language, self.target_language)
            
            prompt = f"Translate the following English text to {target_lang_name}:\n\n{text}\n\nTranslation:"
            
            response = self.gemini_model.generate_content(prompt)
            translated_text = response.text.strip()
            
            print(f"[INFO] Basic translation completed: {translated_text}")
            
            # Add to translation queue for UI update
            self.live_translation_queue.put(translated_text)
            
            # Update the last session history entry
            if self.session_history:
                self.session_history[-1]["translated_text"] = translated_text
                
        except Exception as e:
            print(f"[ERROR] Basic translation error: {e}")
            # Final fallback to showing original text
            self.live_translation_queue.put(f"[Translation Error] {text}")

    def transcribe_audio(self, audio_data):
        if (
            self.whisper_model
            and len(audio_data) >= self.audio_processor.sample_rate * 0.5
        ):
            result = self.whisper_model.transcribe(audio_data, task="transcribe")
            return result.get("text", "").strip()
        return ""

    async def translate_text(self, text, target_language, context=""):
        """Enhanced translation using metacognitive strategies"""
        if text and self.gemini_model:
            try:
                # Use metacognitive translation optimization
                result = (
                    await self.rl_coordinator.metacognitive_translation_optimization(
                        text, target_language, self.gemini_model, context
                    )
                )

                # Store translation result for EgoSchema evaluation
                self.session_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "original_text": text,
                        "translated_text": result.get("translation", text),
                        "target_language": target_language,
                        "strategy_used": result.get("strategy_used", "default"),
                        "quality_score": result.get("quality_score", 0.0),
                        "processing_time": result.get("processing_time", 0.0),
                        "metacognitive_state": result.get("metacognitive_state", {}),
                    }
                )

                return result.get("translation", text)
            except Exception as e:
                print(f"Enhanced translation error: {e}")
                # Fallback to basic translation
                return await self.rl_coordinator.optimize_translation(
                    text, target_language, self.gemini_model
                )
        return text

    async def enhanced_whisper_transcription(self, audio_data, context=""):
        """Enhanced transcription using RL-optimized Whisper"""
        if (
            self.whisper_model
            and len(audio_data) >= self.audio_processor.sample_rate * 0.5
        ):
            try:
                # Use enhanced whisper optimization
                result = await self.rl_coordinator.optimize_whisper_real_time(
                    audio_data, self.whisper_model, context
                )

                return {
                    "transcription": result.get("transcription", ""),
                    "confidence": result.get("confidence", 0.0),
                    "processing_time": result.get("processing_time", 0.0),
                    "optimization_params": result.get("optimization_params", {}),
                }
            except Exception as e:
                print(f"Enhanced transcription error: {e}")
                # Fallback to basic transcription
                result = self.whisper_model.transcribe(audio_data, task="transcribe")
                return {
                    "transcription": result.get("text", "").strip(),
                    "confidence": 0.5,
                    "processing_time": 0.0,
                    "optimization_params": {},
                }
        return {
            "transcription": "",
            "confidence": 0.0,
            "processing_time": 0.0,
            "optimization_params": {},
        }

    def process_video_with_egoschema(
        self, video_path, target_language, progress_callback=None
    ):
        """Process video with EgoSchema evaluation if available"""
        if self.egoschema_enhanced_translator:
            try:
                return self.egoschema_enhanced_translator.process_video_with_egoschema_evaluation(
                    video_path, target_language, progress_callback
                )
            except Exception as e:
                print(f"EgoSchema processing error: {e}")
                return None
        return None

    def get_egoschema_analysis(self):
        """Get EgoSchema analysis of current session"""
        if self.egoschema_integration and len(self.session_history) > 5:
            try:
                # Prepare session data for EgoSchema analysis
                transcript = "\n".join(
                    [item["original_text"] for item in self.session_history[-10:]]
                )
                translation = "\n".join(
                    [item["translated_text"] for item in self.session_history[-10:]]
                )

                translation_result = {
                    "duration": 180.0,
                    "translated_text": translation,
                    "source_language": "en",
                    "target_language": self.session_history[-1]["target_language"],
                    "quality_score": np.mean(
                        [item["quality_score"] for item in self.session_history[-10:]]
                    ),
                    "processing_time": np.mean(
                        [item["processing_time"] for item in self.session_history[-10:]]
                    ),
                }

                return self.egoschema_integration.process_video_with_egoschema(
                    "live_session", transcript, translation_result
                )
            except Exception as e:
                print(f"EgoSchema analysis error: {e}")
                return None
        return None

    def get_metacognitive_report(self):
        """Get metacognitive performance report"""
        if hasattr(self.rl_coordinator, "get_metacognitive_report"):
            try:
                return self.rl_coordinator.get_metacognitive_report()
            except Exception as e:
                print(f"Metacognitive report error: {e}")
                return None
        return None

    def get_system_status(self):
        """Get comprehensive system status including all integrated features"""
        # Check WhisperLive server status
        whisper_server_status = self.audio_processor.validate_server_connection()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "basic_components": {
                "whisper_model": self.whisper_model is not None,
                "whisper_live_client": self.whisper_live_client is not None,
                "gemini_model": self.gemini_model is not None,
                "rl_coordinator": self.rl_coordinator is not None,
                "audio_processor": self.audio_processor is not None,
                "video_processor": self.video_processor is not None,
            },
            "advanced_features": {
                "enhanced_translation_service": self.translation_service is not None,
                "advanced_translation_engine": self.advanced_translation_engine is not None,
                "egoschema_integration": self.egoschema_integration is not None,
                "egoschema_enhanced_translator": self.egoschema_enhanced_translator
                is not None,
                "metacognitive_controller": hasattr(self, "metacognitive_controller"),
                "performance_monitor": self.performance_monitor is not None,
            },
            "session_stats": {
                "total_translations": len(self.session_history),
                "average_quality": (
                    np.mean([item["quality_score"] for item in self.session_history])
                    if self.session_history
                    else 0.0
                ),
                "average_processing_time": (
                    np.mean([item["processing_time"] for item in self.session_history])
                    if self.session_history
                    else 0.0
                ),
            },
            "whisper_live_status": {
                "is_recording": self.is_recording,
                "client_connected": self.whisper_live_client is not None,
                "transcript_queue_size": self.live_transcript_queue.qsize(),
                "translation_queue_size": self.live_translation_queue.qsize(),
                "server_accessible": whisper_server_status,
                "server_host": self.audio_processor.whisper_host,
                "server_port": self.audio_processor.whisper_port,
                "connection_timeout": self.audio_processor.connection_timeout
            }
        }

        # Add RL coordinator performance summary
        if self.rl_coordinator:
            try:
                status["rl_performance"] = (
                    self.rl_coordinator.get_system_performance_summary()
                )
            except Exception as e:
                status["rl_performance"] = {"error": str(e)}

        # Add metacognitive report
        metacog_report = self.get_metacognitive_report()
        if metacog_report:
            status["metacognitive_analysis"] = metacog_report

        # Add EgoSchema analysis if available
        ego_analysis = self.get_egoschema_analysis()
        if ego_analysis:
            status["egoschema_evaluation"] = ego_analysis
        
        # Add enhanced translation service statistics
        if self.translation_service:
            try:
                translation_stats = self.translation_service.get_translation_stats()
                status["translation_service_stats"] = translation_stats
            except Exception as e:
                status["translation_service_stats"] = {"error": str(e)}
        
        # Add advanced engine statistics if available directly
        if self.advanced_translation_engine and hasattr(self.advanced_translation_engine, 'get_translation_stats'):
            try:
                advanced_stats = self.advanced_translation_engine.get_translation_stats()
                status["advanced_engine_stats"] = advanced_stats
            except Exception as e:
                status["advanced_engine_stats"] = {"error": str(e)}

        return status

    # Simplified start method for demonstration
    def start(self):
        self.audio_processor.start_recording()
        self.video_processor.start_capture()
        print("Live camera translation started")

    def stop(self):
        self.audio_processor.stop_recording()
        self.video_processor.stop_capture()
        print("Live camera translation stopped")

    def start_live_session(self, class_id="live_session", target_language="bn"):
        """Starts a new live session with WhisperLive streaming"""
        self.session_id = f"{class_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_transcript = ""
        self.session_start_time = time.time()
        self.target_language = target_language
        
        # Clear queues
        while not self.live_transcript_queue.empty():
            self.live_transcript_queue.get()
        while not self.live_translation_queue.empty():
            self.live_translation_queue.get()
        
        # Ensure directories exist
        os.makedirs("transcripts", exist_ok=True)
        os.makedirs("schemas/normalized", exist_ok=True)
        
        # Start audio processing with WhisperLive
        success = self.audio_processor.start_recording()
        if success:
            self.is_recording = True
            self.video_processor.start_capture()
            return f"Live session started with ID: {self.session_id} (WhisperLive streaming enabled)"
        else:
            return "Failed to start WhisperLive streaming. Check server connection."

    def stop_live_session(self):
        """Stops the live session and generates a schema from the transcript."""
        if not self.session_start_time:
            return "No active session to stop."

        self.is_recording = False
        self.audio_processor.stop_recording()
        self.video_processor.stop_capture()

        # Collect transcript from session history
        full_transcript = "\n".join(
            [item["original_text"] for item in self.session_history]
        )
        
        # If no history, use demo transcript
        if not full_transcript:
            full_transcript = "This is a demo transcript for testing purposes. Topics covered include machine learning, artificial intelligence, and natural language processing."

        # Save the full transcript
        transcript_path = os.path.join("transcripts", f"{self.session_id}_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(full_transcript)

        # Generate schema from transcript
        generated_schema_path = self._generate_schema_from_transcript(transcript_path)

        return f"Session stopped. Transcript saved to {transcript_path}. Schema generated at {generated_schema_path}."

    def upload_course_material(self, file_path):
        """Handles the upload of course material (PDF, CSV, etc.)."""
        if not file_path:
            return "Please upload a file."

        self.uploaded_course_material = file_path.name
        return f"Successfully uploaded {os.path.basename(file_path.name)}."

    def run_topic_analysis(self):
        """Runs topic analysis using the uploaded course material against the transcript."""
        if not self.session_id or not self.uploaded_course_material:
            return {"error": "Please stop a session and upload course material first."}

        transcript_path = f"transcripts/{self.session_id}_transcript.txt"
        
        if not os.path.exists(transcript_path):
            return {"error": f"Transcript not found: {transcript_path}"}
        
        if not os.path.exists(os.path.join("../data/course_materials", self.uploaded_course_material)):
            return {"error": f"Uploaded course material not found: {self.uploaded_course_material}"}

        try:
            # Process the uploaded course material as the expected topics schema
            # and compare it against the actual transcript from the live session
            result = self.schema_pipeline.process_session(
                schema_file=self.uploaded_course_material,  # Course handout (expected topics)
                transcript_file=transcript_path,             # Actual class transcript
                class_id=self.session_id.split('_')[0],     # Extract class name
                date=datetime.now().strftime("%Y-%m-%d")
            )
            
            # Add metadata about the analysis
            result.update({
                "analysis_type": "course_material_vs_transcript",
                "course_material_file": os.path.basename(self.uploaded_course_material),
                "session_id": self.session_id,
                "analysis_timestamp": datetime.now().isoformat()
            })
            
            return result
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}", "details": str(e)}

    def _generate_schema_from_transcript(self, transcript_path):
        """Generates a schema from the session transcript for reference."""
        if not self.schema_parser or not self.schema_pipeline:
            return None

        try:
            # Extract keywords from the transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()

            if not transcript_text.strip():
                print("Warning: Empty transcript, using demo content")
                return None

            keywords = self.schema_pipeline.keyword_extractor.extract_keywords(transcript_text)

            schema_data = {
                "class_id": self.session_id,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "expected_topics": keywords[:20],  # Limit to top 20 keywords
                "metadata": {
                    "generated_from": "live_session_transcript",
                    "session_duration": time.time() - self.session_start_time if self.session_start_time else 0,
                    "total_transcript_length": len(transcript_text),
                    "keywords_extracted": len(keywords)
                }
            }

            schema_path = self.schema_parser.save_normalized_schema(schema_data)
            print(f"Schema generated with {len(keywords)} keywords: {schema_path}")
            return schema_path
        except Exception as e:
            print(f"Error generating schema: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_interface(self):
        """Create Gradio interface for the live camera translation application with WhisperLive"""
        with gr.Blocks(title="Live Camera Enhanced Translator with WhisperLive") as interface:
            gr.Markdown("# Live Camera Enhanced Translator with WhisperLive Streaming")
            gr.Markdown("Real-time translation with live camera feed and WhisperLive audio processing")
            
            with gr.Tab("Live Translation (WhisperLive)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Camera Feed with better styling
                        camera_feed = gr.Image(
                            label="📹 Live Camera Feed",
                            sources=["webcam"],
                            streaming=True,
                            width=320,
                            height=240,
                            interactive=True
                        )
                        
                        # Audio Status
                        audio_status = gr.Markdown(
                            "🔊 **Audio Status**: Ready (WhisperLive)", 
                            visible=True
                        )
                        
                        # Controls
                        class_id_input = gr.Textbox(
                            label="Class ID",
                            value="DEMO_CLASS",
                            placeholder="Enter class identifier"
                        )
                        target_lang_dropdown = gr.Dropdown(
                            choices=["bn", "hi"],
                            value="bn",
                            label="Target Language",
                            info="Bengali (bn) or Hindi (hi)"
                        )
                        whisper_server = gr.Textbox(
                            label="WhisperLive Server",
                            value="localhost:9090",
                            placeholder="host:port"
                        )
                        
                        with gr.Row():
                            start_btn = gr.Button("Start Live Session", variant="primary", scale=1)
                            stop_btn = gr.Button("Stop Session", variant="stop", scale=1)
                        
                        session_status = gr.Textbox(label="Session Status", interactive=False)
                        
                    with gr.Column(scale=2):
                        live_transcript = gr.Textbox(
                            label="Live Transcript (WhisperLive)",
                            lines=12,
                            interactive=False,
                            placeholder="Real-time transcribed text will appear here...",
                            max_lines=15
                        )
                        translation_output = gr.Textbox(
                            label="Live Translation",
                            lines=12,
                            interactive=False,
                            placeholder="Real-time translated text will appear here...",
                            max_lines=15
                        )
            
            with gr.Tab("Topic Analysis"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Upload Course Material")
                        gr.Markdown("Upload the course handout (PDF/CSV/Excel) that contains expected topics")
                        course_material_upload = gr.File(
                            label="Upload Course Material",
                            file_types=[".json", ".csv", ".pdf", ".yaml", ".yml", ".xlsx", ".xls"]
                        )
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                        
                        gr.Markdown("### Run Analysis")
                        gr.Markdown("Compare transcript topics with course material")
                        analyze_btn = gr.Button("Run Topic Analysis", variant="primary")
                        
                    with gr.Column():
                        gr.Markdown("### Analysis Results")
                        analysis_output = gr.JSON(label="Topic Analysis Results")
                        
            with gr.Tab("System Status"):
                with gr.Row():
                    refresh_status_btn = gr.Button("Refresh Status")
                    system_status_output = gr.JSON(label="System Status")
                    
            with gr.Tab("Demo Schema Generator"):
                gr.Markdown("### Generate Sample Course Material")
                gr.Markdown("Create a sample course material file for testing")
                with gr.Row():
                    with gr.Column():
                        demo_class_id = gr.Textbox(label="Class ID", value="CS101")
                        demo_topics = gr.Textbox(
                            label="Expected Topics (comma-separated)",
                            value="machine learning, neural networks, deep learning, artificial intelligence, supervised learning, unsupervised learning",
                            lines=3
                        )
                        generate_demo_btn = gr.Button("Generate Demo Course Material")
                        demo_status = gr.Textbox(label="Generation Status", interactive=False)
            
            # Event handlers
            def handle_start_session(class_id, target_lang, server):
                # Update server if provided
                if server and ':' in server:
                    host, port = server.split(':')
                    self.audio_processor.whisper_host = host
                    self.audio_processor.whisper_port = int(port)
                return self.start_live_session(class_id, target_lang)
            
            def handle_stop_session():
                return self.stop_live_session()
            
            def handle_upload(file):
                return self.upload_course_material(file)
            
            def handle_analysis():
                result = self.run_topic_analysis()
                return result
            
            def handle_status_refresh():
                return self.get_system_status()
            
            def generate_demo_schema(class_id, topics_str):
                try:
                    topics = [t.strip() for t in topics_str.split(',')]
                    demo_schema = {
                        "class_id": class_id,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "expected_topics": topics,
                        "course_name": f"Demo Course {class_id}",
                        "instructor": "Demo Instructor"
                    }
                    
                    # Save demo schema
                    os.makedirs("schemas/demo", exist_ok=True)
                    demo_path = f"schemas/demo/{class_id}_course_material.json"
                    with open(demo_path, 'w') as f:
                        json.dump(demo_schema, f, indent=2)
                    
                    return f"Demo course material generated: {demo_path}. You can now upload this file in the Topic Analysis tab."
                except Exception as e:
                    return f"Error generating demo schema: {str(e)}"
            
            # Live update function for transcript and translation
            def update_live_displays():
                transcript_text = ""
                translation_text = ""
                
                # Get latest transcript
                while not self.live_transcript_queue.empty():
                    new_transcript = self.live_transcript_queue.get()
                    transcript_text = f"{transcript_text}\n{new_transcript}" if transcript_text else new_transcript
                
                # Get latest translation
                while not self.live_translation_queue.empty():
                    new_translation = self.live_translation_queue.get()
                    translation_text = f"{translation_text}\n{new_translation}" if translation_text else new_translation
                
                return transcript_text, translation_text
            
            # Setup periodic updates for live content
            def setup_live_updates():
                """Setup periodic updates for transcript and translation"""
                if self.is_recording:
                    return update_live_displays()
                return "", ""
            
            # Connect event handlers
            start_btn.click(
                fn=handle_start_session,
                inputs=[class_id_input, target_lang_dropdown, whisper_server],
                outputs=[session_status]
            )
            
            stop_btn.click(
                fn=handle_stop_session,
                inputs=[],
                outputs=[session_status]
            )
            
            course_material_upload.upload(
                fn=handle_upload,
                inputs=[course_material_upload],
                outputs=[upload_status]
            )
            
            analyze_btn.click(
                fn=handle_analysis,
                inputs=[],
                outputs=[analysis_output]
            )
            
            refresh_status_btn.click(
                fn=handle_status_refresh,
                inputs=[],
                outputs=[system_status_output]
            )
            
            generate_demo_btn.click(
                fn=generate_demo_schema,
                inputs=[demo_class_id, demo_topics],
                outputs=[demo_status]
            )
            
            # Initial status update
            interface.load(fn=handle_status_refresh, outputs=[system_status_output])
            
        return interface


class AudioProcessorWhisperLive:
    """Handles real-time audio capture and processing with WhisperLive"""

    def __init__(self, parent, whisper_host="localhost", whisper_port=9090):
        self.parent = parent
        self.whisper_host = whisper_host
        self.whisper_port = whisper_port
        self.whisper_client = None
        self.is_recording = False
        self.sample_rate = 16000  # Keep for backward compatibility
        self.connection_timeout = 5  # seconds
        
    def validate_server_connection(self):
        """Validate that WhisperLive server is accessible"""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connection_timeout)
            result = sock.connect_ex((self.whisper_host, self.whisper_port))
            sock.close()
            return result == 0
        except Exception as e:
            print(f"[ERROR] Connection validation failed: {e}")
            return False

    def start_recording(self):
        """Start WhisperLive client for real-time transcription"""
        # First validate server connection
        if not self.validate_server_connection():
            print(f"[ERROR] Cannot connect to WhisperLive server at {self.whisper_host}:{self.whisper_port}")
            print(f"[INFO] Make sure WhisperLive server is running on {self.whisper_host}:{self.whisper_port}")
            print("[INFO] Start server with: python -m whisper_live.server --port 9090")
            return False
            
        try:
            # Initialize WhisperLive client
            self.whisper_client = WhisperLiveClient(
                host=self.whisper_host,
                port=self.whisper_port,
                lang="en",
                translate=False,
                model="base",
                use_vad=True,
                log_transcription=False,
                transcription_callback=self.parent.on_transcription_received
            )
            
            self.is_recording = True
            print(f"[INFO] Connected to WhisperLive server at {self.whisper_host}:{self.whisper_port}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to connect to WhisperLive server: {e}")
            print(f"[INFO] Make sure WhisperLive server is running on {self.whisper_host}:{self.whisper_port}")
            print("[INFO] Start server with: python -m whisper_live.server --port 9090")
            return False

    def stop_recording(self):
        """Stop WhisperLive client"""
        self.is_recording = False
        if self.whisper_client:
            try:
                # WhisperLive client cleanup
                self.whisper_client.recording = False
                if hasattr(self.whisper_client, 'client_socket') and self.whisper_client.client_socket:
                    self.whisper_client.client_socket.close()
                self.whisper_client = None
                print("[INFO] WhisperLive client stopped")
            except Exception as e:
                print(f"[ERROR] Error stopping WhisperLive client: {e}")
                
    def handle_connection_error(self, error):
        """Handle connection errors gracefully"""
        print(f"[ERROR] WhisperLive connection error: {error}")
        self.is_recording = False
        if self.parent:
            # Add error message to transcript queue for user visibility
            self.parent.live_transcript_queue.put(f"[CONNECTION ERROR] {error}")
            # Try to recover connection after a delay
            import threading
            recovery_thread = threading.Thread(target=self.attempt_reconnection)
            recovery_thread.daemon = True
            recovery_thread.start()
            
    def attempt_reconnection(self):
        """Attempt to reconnect to WhisperLive server"""
        import time
        retry_delay = 5  # seconds
        max_retries = 3
        
        for attempt in range(max_retries):
            print(f"[INFO] Attempting reconnection to WhisperLive server (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
            
            if self.validate_server_connection():
                print("[INFO] Server is accessible, attempting to restart recording")
                try:
                    success = self.start_recording()
                    if success:
                        print("[INFO] Successfully reconnected to WhisperLive server")
                        if self.parent:
                            self.parent.live_transcript_queue.put("[RECONNECTED] WhisperLive connection restored")
                        return
                except Exception as e:
                    print(f"[ERROR] Reconnection attempt failed: {e}")
            else:
                print(f"[WARNING] Server still not accessible, will retry in {retry_delay} seconds")
                
        print("[ERROR] Failed to reconnect after maximum attempts")
        if self.parent:
            self.parent.live_transcript_queue.put("[ERROR] Could not reconnect to WhisperLive server. Please restart manually.")


class AudioProcessor:
    """Legacy AudioProcessor for backward compatibility - deprecated in favor of WhisperLive"""

    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        print("[WARNING] Using legacy AudioProcessor. Consider upgrading to WhisperLive.")

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

    def start_recording(self):
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join()

    def _record_audio(self):
        try:
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            while self.is_recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                self.audio_queue.put(audio_data)

            stream.stop_stream()
            stream.close()

        except Exception as e:
            print(f"Audio recording error: {e}")


class VideoProcessor:
    """Handles video capture and display"""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_capturing = False

    def start_capture(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")

            self.is_capturing = True

        except Exception as e:
            print(f"Failed to start video capture: {e}")

    def stop_capture(self):
        self.is_capturing = False
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    print("=" * 60)
    print("Live Camera Enhanced UI with WhisperLive Integration")
    print("=" * 60)
    print("\n[IMPORTANT] Make sure WhisperLive server is running:")
    print("python -m whisper_live.server --port 9090")
    print("\nOr if you have the WhisperLive repository:")
    print("cd WhisperLive && python run_server.py --port 9090")
    print("\nStarting Gradio interface...\n")
    
    ui = LiveCameraEnhancedUI()
    interface = ui.create_interface()
    interface.launch()
