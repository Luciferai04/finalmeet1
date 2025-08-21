## Monitoring and Observability

This project uses Prometheus for metrics and Grafana for dashboards.

### Application Metrics

Emitted by `src/services/performance_monitor.py` (Prometheus client):

- Counters/Histograms/Gauges
  - `translation_requests_total{language}` (Counter)
  - `translation_latency_seconds` (Histogram)
  - `translation_quality_score{language}` (Gauge)
  - `rl_agent_reward{agent_type}` (Gauge)
  - `rl_episode_length{agent_type}` (Histogram)
  - `active_sessions_count` (Gauge)
  - `websocket_connections_count` (Gauge)
  - `system_cpu_usage_percent` (Gauge)
  - `system_memory_usage_bytes` (Gauge)
  - `system_disk_usage_bytes` (Gauge)

The metrics HTTP server typically runs on port `8000` (configurable) via `start_http_server(metrics_port)`.

### Prometheus

- Docker Compose service defined in `deploy/docker/docker-compose.prod.yml`
- App scrape config lives at `data/monitoring/prometheus.yml`
- Example jobs include `flask-api`, `gradio-app`, and infra exporters.

Run via Docker Compose:
```bash
docker-compose -f deploy/docker/docker-compose.prod.yml up -d prometheus grafana
```

Prometheus UI: `http://localhost:9090`

### Grafana

- Provisioning mounted from `data/monitoring/grafana/*`
- Default admin password via env in compose (`GF_SECURITY_ADMIN_PASSWORD`)
- Access at `http://localhost:3000`

### Logging

- Central logging configuration via `src/core/logging_config.py`
- Log file path defaults to `data/logs/app.log` (configurable)

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


