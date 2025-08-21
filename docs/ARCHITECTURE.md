## Architecture Overview

This document outlines the core architecture of the Real-Time Translator, including service boundaries, entry points, data flows, and runtime components.

### High-Level Components

- **HTTP API (Flask)**
  - Application factory: `src/core/app_factory.py`
  - Blueprints: `src/api/routes.py` mounted at `/api`
  - WSGI entry: `wsgi.py` (for Gunicorn)
  - Dev entry: `main.py` (Flask dev server)

- **Realtime I/O**
  - WebSockets: `src/services/websocket_service.py` (Socket.IO events)
  - WhisperLive server: `src/services/whisper_live/server.py` and `run_server.py`
  - WhisperLive client: `src/services/whisper_live/client.py`
  - WebRTC (experimental): `src/services/webrtc_handler.py`

- **UI**
  - Gradio interface: `src/ui/live_camera_enhanced_ui.py`
  - Launch script: `run_ui.py`

### Running the Gradio App

The main Gradio application is `src/ui/live_camera_enhanced_ui.py`. To run it:

1. **Start the WhisperLive server first**:
   ```bash
   python run_server.py --port 9090
   ```

2. **Launch the Gradio UI**:
   ```bash
   python run_ui.py
   ```

3. **Access the interface** at `http://localhost:7860`

**Note**: The Gradio app requires:
- A running WhisperLive server
- Valid `GOOGLE_API_KEY` environment variable
- Optional: Redis server for advanced features

- **Core Services**
  - Translation pipeline: `src/services/translation_service.py`, `src/services/enhanced_translation_service.py`, `src/services/advanced_translation_engine.py`
  - Schema operations: `src/services/schema_service.py` and `src/services/schema_checker/*`
  - RL coordination: `src/services/enhanced_rl_coordinator.py`, `src/services/fallback_rl_coordinator.py`, `src/services/rl_coordinator.py`
  - Monitoring: `src/services/performance_monitor.py`

- **Configuration & Logging**
  - App config: `src/core/config.py`, `src/core/production_config.py`
  - Env templates: `config/environments/*.env.example`
  - Fallback feature flags: `config/fallback_config.json` (managed by `src/services/config_manager.py`)
  - Logging setup: `src/core/logging_config.py`

### Entry Points and Process Roles

- `main.py`
  - Sets up `PYTHONPATH`, creates app via `create_app()`, and runs the Flask dev server when `FLASK_ENV=development`.
  - In production, start via Gunicorn: `gunicorn --config config/gunicorn.conf.py wsgi:app`.

- `wsgi.py`
  - Exposes `app = create_app(config_name)` for WSGI servers (Gunicorn/UWSGI).

- `run_ui.py`
  - Boots the Gradio UI (`LiveCameraEnhancedUI`). Requires WhisperLive server and a valid `GOOGLE_API_KEY`.

- `run_server.py`
  - Convenience wrapper to run the WhisperLive `TranscriptionServer` outside of the module’s CLI.

### HTTP API Mounting

- `src/core/app_factory.py`
  - Registers `api_bp` at `/api`.
  - Exposes `/health` and `/` service info endpoints.
  - Initializes CORS and Socket.IO.

### Realtime Data Flow

1) Audio capture (browser, client app, or UI) →
2) WebSocket stream to WhisperLive server (`whisper_live.server`) →
3) Transcription chunks produced (optionally VAD-gated) →
4) Translation service consumes transcript and produces target-language text →
5) Socket.IO emits `transcription_result`/custom events to clients.

Optional: WebRTC path (experimental) for direct audio/video tracks via `aiortc` handled in `webrtc_handler.py`.

### Translation Pipeline (Conceptual)

- Preprocessing: normalization, domain/topic detection
- Core translation: model invocation (e.g., Gemini, Whisper for STT)
- Postprocessing: terminology consistency, cultural adaptation, quality scoring
- Iteration & feedback: RL signals and heuristics influence prompt/strategy

Implementations live across:
- `src/services/advanced_translation_engine.py`
- `src/services/enhanced_translation_service.py`
- `src/services/enhanced_translation_prompts.py`

### Configuration Layers

- Environment variables via `.env` and the OS (see `config/environments/*.env.example`).
- Flask config classes (`src/core/config.py`) selected by `FLASK_CONFIG`/`FLASK_ENV`.
- Production Pydantic config in `src/core/production_config.py` (safe serialization, masking helpers).
- Feature flag/fallback configuration in `config/fallback_config.json` managed by `ConfigManager`.

### Observability

- Prometheus metrics exposed by `src/services/performance_monitor.py` (default port 8000).
- Docker Compose bundle with Prometheus and Grafana (see `deploy/docker/docker-compose.prod.yml`).
- Logs written to `data/logs` (configurable via env/`Config`).

### Data & Artifacts

- `data/`
  - `uploads/`, `transcripts/`, `reports/`, `monitoring/` (Prometheus/Grafana configs)
- `exported_models/`
  - Model metadata and exported formats (`onnx/`, `pytorch/`, `torchscript/`).

### Networking Ports (defaults)

- Flask API: 5000 (dev), as configured behind Nginx in prod
- WhisperLive server: 9090
- Gradio UI: 7860
- Prometheus: 9090 (container), app metrics server: 8000
- Grafana: 3000

### Key External Dependencies

- Flask, Flask-CORS, Flask-SocketIO
- Prometheus client
- aiortc, OpenCV, webrtcvad (WebRTC path)
- Whisper/WhisperLive for STT
- Gradio for UI


