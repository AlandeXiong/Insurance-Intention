"""FastAPI 接口测试。"""

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
            "utterance": "安心保重疾险保费多少？",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["primary_intent"] == "query_critical_illness_premium"
        assert data["confidence"] >= 0.75
        assert data["total_latency_ms"] <= 600
        session_id = data["session_id"]

        # 多轮：指代消解
        resp2 = client.post("/v1/intent/predict/sync", json={
            "utterance": "它的等待期呢？",
            "session_id": session_id,
        })
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["primary_intent"] == "query_waiting_period"
        assert "安心保" in data2["resolved_utterance"]

    def test_reset_session(self):
        resp = client.post("/v1/intent/predict/sync", json={"utterance": "你好"})
        sid = resp.json()["session_id"]
        del_resp = client.delete(f"/v1/session/{sid}")
        assert del_resp.status_code == 200
