"""Qwen2 / Qwen2.5 adapter.

Structurally a Llama-style decoder; the differences that matter for inspection are
biases on the q/k/v projections and Qwen's own RMSNorm / rotary classes, both of
which are read from the loaded modules rather than assumed.
"""
from app.adapters.rope_decoder import RopeDecoderAdapter


class QwenAdapter(RopeDecoderAdapter):
    family = "qwen"
    model_types = ("qwen2",)
