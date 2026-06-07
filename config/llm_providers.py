"""LLM 提供商注册表 — DeepSeek / 阿里云千问等 OpenAI 兼容 API。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class LLMProviderSpec:
    name: str
    display_name: str
    api_base: str
    default_model: str
    api_key_env: str
    api_base_env: str
    model_env: str
    docs_url: str


PROVIDERS: Dict[str, LLMProviderSpec] = {
    "deepseek": LLMProviderSpec(
        name="deepseek",
        display_name="DeepSeek",
        api_base="https://api.deepseek.com",
        default_model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        api_base_env="DEEPSEEK_API_BASE",
        model_env="DEEPSEEK_MODEL",
        docs_url="https://api-docs.deepseek.com/",
    ),
    "qwen": LLMProviderSpec(
        name="qwen",
        display_name="阿里云千问",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        api_base_env="QWEN_API_BASE",
        model_env="QWEN_MODEL",
        docs_url="https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api",
    ),
}


def get_provider(name: str) -> LLMProviderSpec:
    key = (name or "deepseek").strip().lower()
    return PROVIDERS.get(key, PROVIDERS["deepseek"])


def list_provider_names() -> list[str]:
    return list(PROVIDERS.keys())
