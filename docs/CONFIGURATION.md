## Configuration

Configuration is layered across environment variables, Flask config classes, a production Pydantic model, and a JSON feature-flag file.

### Sources

- Environment files: `config/environments/*.env.example` (copy to active `.env`)
- Flask config classes: `src/core/config.py`
- Production config (Pydantic): `src/core/production_config.py`
- Fallback/feature flags JSON: `config/fallback_config.json` managed by `src/services/config_manager.py`

### Important Environment Variables

- `SECRET_KEY`
- `GOOGLE_API_KEY`
- `REDIS_URL` (default `redis://localhost:6379`)
- `SESSION_TIMEOUT` (seconds)
- `MAX_FILE_SIZE` (MB) → `MAX_CONTENT_LENGTH`
- `UPLOAD_FOLDER` (default `../data/uploads`)
- `MAX_CONCURRENT_SESSIONS`
- `WHISPER_MODEL` (default `base`)
- `TRANSLATION_MODEL` (default `gemini-pro`)
- `LOG_LEVEL` (default `INFO`), `LOG_FILE`

See `config/environments/production.env.example` for additional production-oriented variables (SSL paths, Prometheus/Grafana flags, Gunicorn worker settings, GPU toggles, etc.).

### Selecting Config

- Dev run (`main.py`): uses `FLASK_ENV` / `FLASK_CONFIG` to pick from `DevelopmentConfig`, `TestingConfig`, or `ProductionConfig` in `src/core/config.py`.
- WSGI (`wsgi.py`): defaults to production, can be influenced by env.

### Feature Flags (Fallback Config)

- File: `config/fallback_config.json`
- Manager: `ConfigManager` in `src/services/config_manager.py`
- Defaults include:
  - `advanced_rl` (bool)
  - `use_prometheus` (bool)
  - `enable_egoschema` (bool)
  - `redis_url`, `metrics_port`
  - Nested `fallback_modes` for components

`ConfigManager` provides `.get(key)`, `.set(key, value)`, `.save_config()` and a global accessor `get_config_manager()`.

### Production Pydantic Config

`src/core/production_config.py` exposes `ProductionConfig` with helpers:

- `get_directories()` to ensure critical directories exist
- `is_api_key_configured(service)` safe checks
- `get_safe_config()` masks secrets for logs/diagnostics

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


