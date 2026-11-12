"""Llama-architecture adapter (Llama 2/3, SmolLM2, TinyLlama, …)."""
from app.adapters.rope_decoder import RopeDecoderAdapter


class LlamaAdapter(RopeDecoderAdapter):
    family = "llama"
    model_types = ("llama",)
