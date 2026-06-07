"""FastAPI 服务入口 — 动态意图识别。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.insurance_domain import INSURANCE_INTENT_CATEGORIES
from src.pipeline import IntentPipeline, PipelineResponse

app = FastAPI(title="Dynamic Intent Recognition API", version="2.0.0")
pipeline = IntentPipeline()


class PredictRequest(BaseModel):
    utterance: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class PredictResponse(BaseModel):
    session_id: str
    utterance: str
    resolved_utterance: str
    intent_label: str
    category: str
    category_name: str
    confidence: float
    reasoning: str
    sub_intents: list
    implicit_intents: list
    slots: dict
    missing_info: list
    drift_detected: bool
    drift_type: str
    drift_reason: str
    should_clarify: bool
    clarification: dict
    total_latency_ms: float
    engine_path: str


def _to_api(resp: PipelineResponse) -> PredictResponse:
    intent = resp.intent
    return PredictResponse(
        session_id=resp.session_id,
        utterance=resp.utterance,
        resolved_utterance=resp.resolved_utterance,
        intent_label=intent.intent_label,
        category=intent.category,
        category_name=resp.metadata.get("category_name", intent.category),
        confidence=intent.confidence,
        reasoning=intent.reasoning,
        sub_intents=[s.model_dump() for s in intent.sub_intents],
        implicit_intents=[i.model_dump() for i in intent.implicit_intents],
        slots={k: v.model_dump() for k, v in intent.slots.items()},
        missing_info=resp.missing_slots,
        drift_detected=intent.drift_detected,
        drift_type=intent.drift_type.value,
        drift_reason=intent.drift_reason,
        should_clarify=resp.should_clarify,
        clarification=resp.clarification.model_dump(),
        total_latency_ms=resp.total_latency_ms,
        engine_path=resp.engine_path,
    )


@app.get("/v1/intent/categories")
def list_categories() -> dict:
    return {
        "categories": [
            {"code": c.code, "name": c.name, "description": c.description, "examples": c.examples}
            for c in INSURANCE_INTENT_CATEGORIES
        ]
    }


@app.post("/v1/intent/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    resp = await pipeline.process_async(req.utterance, req.session_id)
    return _to_api(resp)


@app.post("/v1/intent/predict/sync", response_model=PredictResponse)
def predict_sync(req: PredictRequest) -> PredictResponse:
    resp = pipeline.process(req.utterance, req.session_id)
    return _to_api(resp)


@app.delete("/v1/session/{session_id}")
def reset_session(session_id: str) -> dict:
    pipeline.reset_session(session_id)
    return {"status": "ok", "session_id": session_id}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}
