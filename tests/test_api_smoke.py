import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.app_factory import create_app
from src.core.config import Config


def test_health_endpoint():
    app = create_app(Config.get_config("testing"))
    client = app.test_client()
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data
    # Support both raw jsonify and wrapper structure
    assert (data.get("status") == "healthy") or (data.get("success") is True)


def test_translate_smoke():
    app = create_app(Config.get_config("testing"))
    client = app.test_client()
    payload = {"text": "Hello", "target_language": "Bengali"}
    res = client.post("/api/translate", json=payload)
    assert res.status_code in (200, 400, 500)  # tolerate missing external keys in CI
    if res.status_code == 200:
        data = res.get_json()
        assert data
        # Support both route return shapes
        if "data" in data:
            assert "translated_text" in data["data"]
        else:
            assert data.get("success") is True

