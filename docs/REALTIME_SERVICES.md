## Realtime Services

This project supports realtime transcription and translation via WhisperLive, WebSockets (Socket.IO), and an experimental WebRTC path.

### WhisperLive

- Server implementation: `src/services/whisper_live/server.py`
- Client: `src/services/whisper_live/client.py`
- Quick start:
  ```bash
  # Option 1: module entry
  python -m whisper_live.server --port 9090

  # Option 2: repository helper
  python run_server.py --port 9090
  ```

The server receives audio over raw websockets, applies VAD when configured, and streams transcriptions.

### WebSockets (Socket.IO)

- Implementation: `src/services/websocket_service.py`
- Initialization: wired in `src/core/app_factory.py` via `socketio.init_app(app)`
- Events:
  - `connect` / `disconnect`
  - `start_translation` → emits `translation_started`
  - `stop_translation` → emits `translation_stopped`
  - `audio_chunk` → acknowledges with `audio_received`
  - Broadcast `transcription_result`, `system_status`, `error`

See the event payloads in `docs/API_REFERENCE.md`.

### WebRTC (Experimental)

- Handler: `src/services/webrtc_handler.py`
- Uses `aiortc` to build `VideoStreamTrack` and `AudioStreamTrack` with optional VAD and real-time processing hooks.
- Current file includes scaffolding for track creation and processing; treat as a starter for deeper integration.

### Gradio UI

- UI class: `src/ui/live_camera_enhanced_ui.py`
- Launcher: `run_ui.py`
- Expects:
  - WhisperLive server reachable (`WHISPER_HOST`, `WHISPER_PORT`)
  - Valid `GOOGLE_API_KEY`

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


