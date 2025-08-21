## API Reference

This reference lists HTTP endpoints exposed by the Flask API and Socket.IO events for realtime interactions.

Base URL: `/api`

### Health

- GET `/api/health`
  - Response: `{ status, api_version, service }`

### Translation

- POST `/api/translate`
  - Body (JSON):
    - `text` (string, required): Source text
    - `target_language` (string, optional, default "Bengali")
    - `session_id` (string, optional)
  - Responses:
    - 200: `{ status: "success", data: { ...translationResult } }`
    - 400: `{ error }` on validation error
    - 500: `{ error: "Translation failed" }`

### Schemas

- GET `/api/schemas`
  - List available schemas

- POST `/api/schemas`
  - Multipart form with `schema_file`

- GET `/api/schemas/{schema_id}`
  - Retrieve specific schema

- DELETE `/api/schemas/{schema_id}`
  - Delete schema by id

### Session Processing

- POST `/api/process`
  - Body (JSON): `{ session_id, schema_id, transcript? }`
  - Runs processing against a schema

### Sessions

- GET `/api/sessions`
  - List active translation sessions

---

## Socket.IO Events

Namespace: default (`/`)

### Connection Lifecycle

- `connect`
  - Server emits `status`: `{ message: "Connected to translation server" }`

- `disconnect`

### Translation Control

- `start_translation`
  - Client → Server payload:
    - `session_id` (string, optional)
    - `target_language` (string, optional, default `bn`)
  - Server → Client emit `translation_started`:
    - `{ session_id, target_language, message }`

- `stop_translation`
  - Client → Server: `{ session_id }`
  - Server → Client `translation_stopped`: `{ session_id, message }`

### Audio Streaming

- Event: `audio_chunk`
  - Client → Server: audio buffer payload
  - Server → Client `audio_received`: `{ status: "received" }`

### Results Broadcast

- `transcription_result`
  - Server → Client: `{ session_id, transcription, translation?, timestamp }`

### Errors

- `error`
  - Server → Client on handler errors: `{ message }`

### Notes

- Socket.IO server is initialized in `src/services/websocket_service.py` and registered in the Flask app at startup (`src/core/app_factory.py`).
- For production, serve via Gunicorn with eventlet/gevent workers compatible with Socket.IO.

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


