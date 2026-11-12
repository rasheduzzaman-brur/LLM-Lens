"""LLM Lens backend: real transformer inference with inspectable intermediate computations."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api import inference, misc, models
from app.core.config import settings

app = FastAPI(
    title="LLM Lens API",
    version="0.1.0",
    description="Runs causal Transformer models under forward hooks and serves every intermediate "
                "computation (tokens, embeddings, normalization, Q/K/V, attention, MLP, logits, KV cache) as "
                "sliceable tensors. All values come from actual model execution.",
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(misc.router)
