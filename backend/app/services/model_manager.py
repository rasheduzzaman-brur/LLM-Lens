"""Model loading. Local directories first; nothing is downloaded without explicit confirmation."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from fastapi import HTTPException

from app.adapters.base import BaseModelAdapter
from app.adapters.registry import adapter_class_for, build_adapter, supported_model_types
from app.core.config import settings

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "float64": torch.float64}
WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]


@dataclass
class LoadedModel:
    model_id: str
    source: str
    path: str
    adapter: BaseModelAdapter
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def model(self):
        return self.adapter.model

    @property
    def tokenizer(self):
        return self.adapter.tokenizer

    def describe(self) -> dict:
        tok = self.tokenizer
        return {
            "model_id": self.model_id,
            "name": self.model_id,
            "source": self.source,
            "path": self.path,
            **self.adapter.info(),
            "tokenizer": {
                "class": type(tok).__name__,
                "vocab_size": len(tok),
                "bos_token": tok.bos_token,
                "eos_token": tok.eos_token,
                "pad_token": tok.pad_token,
                "is_fast": bool(getattr(tok, "is_fast", False)),
            },
            "precision_note": (
                None if self.adapter.dtype == torch.float32 else
                f"Model runs in {self.adapter.dtype}; displayed values are that reduced-precision computation, "
                "upcast losslessly to float32 for transfer."
            ),
        }


def slug(source: str) -> str:
    return source.strip().rstrip("/").split("/")[-1] if Path(source).exists() else source.strip().replace("/", "--")


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class ModelManager:
    def __init__(self) -> None:
        self.loaded: "OrderedDict[str, LoadedModel]" = OrderedDict()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- discovery
    def local_models(self) -> list[dict]:
        out = []
        root = settings.models_dir
        if root.exists():
            for d in sorted(root.iterdir()):
                cfg = d / "config.json"
                if not cfg.exists():
                    continue
                try:
                    c = json.loads(cfg.read_text())
                except Exception:
                    continue
                mt = c.get("model_type")
                out.append({
                    "name": d.name, "path": str(d), "model_type": mt, "supported": mt in supported_model_types(),
                    "size_bytes": _dir_size(d), "loaded": d.name in self.loaded,
                    "has_weights": any(d.glob("*.safetensors")) or any(d.glob("*.bin")),
                })
        return out

    def get(self, model_id: str) -> LoadedModel:
        lm = self.loaded.get(model_id)
        if lm is None:
            raise HTTPException(404, f"model '{model_id}' is not loaded")
        return lm

    # ---------------------------------------------------------------- loading
    def _resolve(self, source: str, allow_download: bool) -> Path:
        p = Path(source).expanduser()
        if p.is_dir() and (p / "config.json").exists():
            return p
        cand = settings.models_dir / source
        if cand.is_dir() and (cand / "config.json").exists():
            return cand
        cand = settings.models_dir / source.replace("/", "--")
        if cand.is_dir() and (cand / "config.json").exists():
            return cand
        if p.is_absolute() or source.startswith("."):
            raise HTTPException(404, f"'{source}' is not a model directory (no config.json found)")
        from huggingface_hub import snapshot_download
        try:  # already in the Hugging Face cache?
            return Path(snapshot_download(source, local_files_only=True, allow_patterns=WEIGHT_PATTERNS))
        except Exception:
            pass
        if not allow_download:
            size = None
            try:
                from huggingface_hub import HfApi
                info = HfApi().model_info(source, files_metadata=True, timeout=10)
                size = sum((s.size or 0) for s in info.siblings
                           if s.rfilename.endswith((".safetensors", ".json", ".txt", ".model")) and "/" not in s.rfilename)
            except Exception:
                pass
            raise HTTPException(409, {
                "code": "download_required", "repo_id": source, "size_bytes": size,
                "message": f"'{source}' is not available locally. Downloading it requires explicit confirmation.",
            })
        try:
            return Path(snapshot_download(source, local_dir=str(cand), allow_patterns=WEIGHT_PATTERNS,
                                          ignore_patterns=["onnx/*", "*.onnx", "runs/*"]))
        except Exception as e:
            raise HTTPException(502, f"download of '{source}' failed: {e}") from e

    def load(self, source: str, device: str = "auto", dtype: str = "auto", allow_download: bool = False) -> LoadedModel:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        path = self._resolve(source, allow_download)
        model_id = path.name if path.parent == settings.models_dir or Path(source).exists() else slug(source)
        with self._lock:
            if model_id in self.loaded:
                self.loaded.move_to_end(model_id)
                return self.loaded[model_id]
            try:
                cfg = AutoConfig.from_pretrained(path, local_files_only=True)
                adapter_class_for(cfg.model_type)
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise HTTPException(422, "CUDA was requested but no CUDA device is available")
            if dtype == "auto":
                dtype = "float32"
            if dtype not in DTYPES:
                raise HTTPException(422, f"dtype must be one of {list(DTYPES)} or 'auto'")
            if device == "cpu" and dtype == "float16":
                raise HTTPException(422, "float16 inference is not supported on CPU; use float32 or bfloat16")
            while len(self.loaded) >= settings.max_loaded_models:
                old, _ = self.loaded.popitem(last=False)
                self._after_unload(old)
            try:
                tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    path, local_files_only=True, dtype=DTYPES[dtype], attn_implementation="eager"
                ).to(device).eval()
                adapter = build_adapter(model, tok)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(422, f"failed to load '{source}': {e}") from e
            lm = LoadedModel(model_id=model_id, source=source, path=str(path), adapter=adapter)
            self.loaded[model_id] = lm
            return lm

    def _after_unload(self, model_id: str) -> None:
        from app.inference.session import runs
        runs.drop_model(model_id)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self, model_id: str) -> None:
        with self._lock:
            if self.loaded.pop(model_id, None) is None:
                raise HTTPException(404, f"model '{model_id}' is not loaded")
            self._after_unload(model_id)


models = ModelManager()
