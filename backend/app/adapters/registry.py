"""Adapter selection by `config.model_type`."""
from __future__ import annotations

from app.adapters.base import BaseModelAdapter
from app.adapters.gpt2 import GPT2Adapter
from app.adapters.llama import LlamaAdapter
from app.adapters.qwen import QwenAdapter

ADAPTERS: list[type[BaseModelAdapter]] = [GPT2Adapter, LlamaAdapter, QwenAdapter]


def supported_model_types() -> list[str]:
    return [t for a in ADAPTERS for t in a.model_types]


def adapter_class_for(model_type: str) -> type[BaseModelAdapter]:
    for a in ADAPTERS:
        if model_type in a.model_types:
            return a
    raise ValueError(
        f"No adapter for model_type '{model_type}'. Supported: {', '.join(supported_model_types())}. "
        "Add a BaseModelAdapter subclass to support another architecture."
    )


def build_adapter(model, tokenizer) -> BaseModelAdapter:
    return adapter_class_for(model.config.model_type)(model, tokenizer)
