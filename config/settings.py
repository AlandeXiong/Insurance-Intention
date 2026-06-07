"""系统配置 — 2026 工业级意图识别基准参数。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

from config.llm_providers import get_provider

load_dotenv()


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class LatencyBudget:
    """端到端延迟预算 (ms)。"""
    frontend_engine_ms: int = 150
    backend_engine_ms: int = 500
    total_ms: int = 600


@dataclass
class QualityThresholds:
    """行业基准指标。"""
    intent_accuracy: float = 0.95
    drift_detection_rate: float = 0.92
    multi_intent_accuracy: float = 0.88
    fusion_confidence_min: float = 0.75
    clarification_confidence_threshold: float = field(
        default_factory=lambda: _env_float("CLARIFICATION_CONFIDENCE_THRESHOLD", 0.72)
    )


def _resolve_llm_fields() -> dict:
    """按 LLM_PROVIDER 解析对应厂商的环境变量。"""
    provider_name = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    spec = get_provider(provider_name)
    api_key = os.getenv(spec.api_key_env) or os.getenv("LLM_API_KEY", "")
    api_base = os.getenv(spec.api_base_env) or os.getenv("LLM_API_BASE") or spec.api_base
    model = os.getenv(spec.model_env) or os.getenv("LLM_MODEL") or spec.default_model
    return {
        "provider": spec.name,
        "display_name": spec.display_name,
        "api_base": api_base,
        "api_key": api_key,
        "model": model,
        "docs_url": spec.docs_url,
        "api_key_env": spec.api_key_env,
    }


@dataclass
class LLMConfig:
    """
    大模型 API 配置（OpenAI 兼容协议）。
    支持 DeepSeek、阿里云千问（DashScope），通过 LLM_PROVIDER 切换。
    """
    provider: str = field(default_factory=lambda: _resolve_llm_fields()["provider"])
    display_name: str = field(default_factory=lambda: _resolve_llm_fields()["display_name"])
    api_base: str = field(default_factory=lambda: _resolve_llm_fields()["api_base"])
    api_key: str = field(default_factory=lambda: _resolve_llm_fields()["api_key"])
    model: str = field(default_factory=lambda: _resolve_llm_fields()["model"])
    docs_url: str = field(default_factory=lambda: _resolve_llm_fields()["docs_url"])
    api_key_env: str = field(default_factory=lambda: _resolve_llm_fields()["api_key_env"])
    timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 30.0))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    enabled: bool = field(default_factory=lambda: os.getenv("LLM_ENABLED", "true").lower() == "true")
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024")))

    @property
    def chat_completions_url(self) -> str:
        base = self.api_base.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.api_key.strip())


@dataclass
class Settings:
    latency: LatencyBudget = field(default_factory=LatencyBudget)
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    llm: LLMConfig = field(default_factory=LLMConfig)
    max_history_turns: int = 10
    drift_similarity_threshold: float = 0.35
    drift_fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        "category_distance": 0.25,
        "utterance_semantic_shift": 0.25,
        "intent_label_shift": 0.15,
        "topic_stack_divergence": 0.10,
        "product_focus_change": 0.10,
        "explicit_marker_boost": 0.10,
        "llm_drift_signal": 0.15,
    })
    lightweight_confidence_threshold: float = 0.82
    embedding_dim: int = 384


settings = Settings()
