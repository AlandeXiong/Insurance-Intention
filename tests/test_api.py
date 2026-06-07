"""FastAPI endpoint tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.server import app

client = TestClient(app)


class TestAPI:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_predict_sync(self):
        resp = client.post("/v1/intent/predict/sync", json={
            "utterance": "How much is Anxin Critical Illness premium?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "premium_inquiry"
        assert data["confidence"] >= 0.75
        assert data["total_latency_ms"] <= 600
        session_id = data["session_id"]

        # Multi-turn: reference resolution
        resp2 = client.post("/v1/intent/predict/sync", json={
            "utterance": "How long is its waiting period?",
            "session_id": session_id,
        })
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["category"] == "coverage_terms"
        assert "Anxin" in data2["resolved_utterance"]

    def test_reset_session(self):
        resp = client.post("/v1/intent/predict/sync", json={"utterance": "Hello"})
        sid = resp.json()["session_id"]
        del_resp = client.delete(f"/v1/session/{sid}")
        assert del_resp.status_code == 200
