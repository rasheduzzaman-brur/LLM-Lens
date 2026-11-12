from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings
from app.inference.session import runs
from app.services.model_manager import models
from app.services.sysinfo import system_info

router = APIRouter(prefix="/api", tags=["system"])


class TokenizeRequest(BaseModel):
    model_id: str
    text: str


@router.post("/tokenize")
def tokenize(req: TokenizeRequest) -> dict:
    lm = models.get(req.model_id)
    tok = lm.tokenizer
    fast = bool(getattr(tok, "is_fast", False))
    enc = tok(req.text, return_offsets_mapping=fast)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"] if fast else [None] * len(ids)
    raw = tok.convert_ids_to_tokens(ids)
    special = set(tok.all_special_ids)
    tokens = []
    for i, (tid, r, off) in enumerate(zip(ids, raw, offsets)):
        text = tok.decode([tid], clean_up_tokenization_spaces=False)
        tokens.append({
            "index": i, "token_id": tid, "text": text, "raw": r,
            "start": off[0] if off else None, "end": off[1] if off else None,
            "bytes": len(text.encode("utf-8")), "special": tid in special,
            "leading_space": text.startswith(" "),
        })
    return {
        "text": req.text, "tokens": tokens, "count": len(ids), "characters": len(req.text),
        "decoded": tok.decode(ids, clean_up_tokenization_spaces=False),
        "round_trip": tok.decode(ids, clean_up_tokenization_spaces=False) == req.text,
        "vocabulary": {
            "tokenizer_class": type(tok).__name__, "vocab_size": len(tok), "base_vocab_size": tok.vocab_size,
            "model_max_length": tok.model_max_length if tok.model_max_length < 10**12 else None,
            "special_tokens": {k: v for k, v in tok.special_tokens_map.items() if isinstance(v, str)},
            "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id, "is_fast": fast,
            "algorithm": (type(tok.backend_tokenizer.model).__name__ if fast else None),
        },
    }


@router.get("/system")
def system() -> dict:
    return {**system_info(), "trace_store": runs.usage(), "loaded_models": list(models.loaded)}


class SettingsPatch(BaseModel):
    max_runs: Optional[int] = Field(None, ge=1, le=32)
    max_store_bytes: Optional[int] = Field(None, ge=10_000_000)
    retain_steps: Optional[int] = Field(None, ge=1, le=256)
    max_prompt_tokens: Optional[int] = Field(None, ge=1, le=4096)
    max_new_tokens: Optional[int] = Field(None, ge=1, le=1024)
    max_cells: Optional[int] = Field(None, ge=1_000, le=2_000_000)


def _settings() -> dict:
    return {k: getattr(settings, k) for k in SettingsPatch.model_fields} | {"models_dir": str(settings.models_dir)}


@router.get("/settings")
def get_settings() -> dict:
    return _settings()


@router.patch("/settings")
def patch_settings(p: SettingsPatch) -> dict:
    for k, v in p.model_dump(exclude_none=True).items():
        setattr(settings, k, v)
    return _settings()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
