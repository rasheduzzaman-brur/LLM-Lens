from __future__ import annotations

from typing import Literal, Optional

import torch
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.adapters.registry import supported_model_types
from app.services import analysis
from app.services import tensor_view as tv
from app.services.model_manager import models

router = APIRouter(prefix="/api/models", tags=["models"])

SUGGESTED = [
    {"repo_id": "openai-community/gpt2", "note": "GPT-2 small, 124M parameters (reference implementation)"},
    {"repo_id": "Qwen/Qwen2.5-0.5B-Instruct", "note": "Qwen2.5, 494M parameters"},
    {"repo_id": "HuggingFaceTB/SmolLM2-135M", "note": "Llama architecture, 135M parameters, ungated"},
]


class LoadRequest(BaseModel):
    source: str = Field(description="Local directory, a name inside the models folder, or a Hugging Face model id")
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16", "float64"] = "auto"
    allow_download: bool = Field(False, description="Must be true to download weights that are not available locally")


@router.get("")
def list_models() -> dict:
    return {"loaded": [m.describe() for m in models.loaded.values()], "local": models.local_models(),
            "supported_model_types": supported_model_types(), "suggested": SUGGESTED}


@router.post("/load")
def load_model(req: LoadRequest) -> dict:
    return models.load(req.source, req.device, req.dtype, req.allow_download).describe()


@router.delete("/{model_id}")
def unload_model(model_id: str) -> dict:
    models.unload(model_id)
    return {"unloaded": model_id}


@router.get("/{model_id}")
def get_model(model_id: str) -> dict:
    return models.get(model_id).describe()


@router.get("/{model_id}/architecture")
def architecture(model_id: str) -> dict:
    return models.get(model_id).adapter.architecture()


@router.get("/{model_id}/weights")
def weights(model_id: str, layer: Optional[int] = None) -> dict:
    """Parameter listing (shapes only). All matrices use the `y = x @ W + b` convention."""
    ad = models.get(model_id).adapter
    if layer is not None and not 0 <= layer < ad.n_layers:
        raise HTTPException(422, "layer out of range")
    ws = ad.global_weights() if layer is None else ad.layer_weights(layer)
    return {"layer": layer, "convention": "y = x @ W + b  (nn.Linear weights are shown transposed to [in, out])",
            "weights": [{"name": n, "shape": list(w.shape), "dtype": str(w.dtype).replace("torch.", "")} for n, w in ws.items()]}


@router.get("/{model_id}/weights/tensor")
def weight_tensor(model_id: str, name: str, layer: Optional[int] = None, index: Optional[str] = None) -> dict:
    ad = models.get(model_id).adapter
    ws = ad.global_weights() if layer is None else ad.layer_weights(layer)
    if name not in ws:
        raise HTTPException(404, f"no parameter '{name}' (available: {list(ws)})")
    out = tv.view(ws[name].detach(), index)
    out.update(name=name, layer=layer, provenance="hooked", source="model parameter")
    return out


@router.get("/{model_id}/embedding")
def embedding(model_id: str, token_id: int, neighbors: int = Query(8, ge=0, le=50)) -> dict:
    lm = models.get(model_id)
    w = lm.adapter.embed_tokens.weight.detach()
    if not 0 <= token_id < w.shape[0]:
        raise HTTPException(422, f"token_id out of range (0…{w.shape[0] - 1})")
    vec = w[token_id]
    out = {"token_id": token_id, "token": lm.tokenizer.decode([token_id]), "shape": [int(w.shape[1])],
           "dtype": str(w.dtype).replace("torch.", ""), "values": tv.to_jsonable(vec), "stats": tv.stats(vec).model_dump(),
           "matrix_shape": list(w.shape)}
    if neighbors:
        with torch.inference_mode():
            sim = torch.nn.functional.cosine_similarity(w.float(), vec.float()[None], dim=-1)
            top = torch.topk(sim, min(neighbors + 1, sim.numel()))
        out["neighbors"] = [{"token_id": int(i), "token": lm.tokenizer.decode([int(i)]), "cosine": float(s)}
                            for s, i in zip(top.values, top.indices) if int(i) != token_id][:neighbors]
        out["neighbors_note"] = "Visualization transform: nearest rows of the embedding matrix by cosine similarity."
    return out


class PcaRequest(BaseModel):
    token_ids: list[int] = Field(min_length=1, max_length=2048)


@router.post("/{model_id}/embeddings/pca")
def embeddings_pca(model_id: str, req: PcaRequest) -> dict:
    lm = models.get(model_id)
    w = lm.adapter.embed_tokens.weight.detach()
    if any(not 0 <= t < w.shape[0] for t in req.token_ids):
        raise HTTPException(422, "token id out of range")
    pts, ratio = analysis.pca2(w[torch.tensor(req.token_ids, device=w.device)])
    return {"points": tv.to_jsonable(pts), "explained_variance_ratio": ratio,
            "note": "Visualization transform: PCA fitted on the token-embedding rows of these tokens only."}


@router.get("/{model_id}/activation")
def activation(model_id: str, lo: float = -6.0, hi: float = 6.0, n: int = Query(241, ge=11, le=2001)) -> dict:
    """The loaded model's real activation module evaluated on a grid, plus reference curves."""
    ad = models.get(model_id).adapter
    x = torch.linspace(lo, hi, n)
    with torch.inference_mode():
        y = ad.activation_module()(x.to(ad.device, ad.dtype)).float().cpu()
        refs = {"relu": torch.relu(x), "gelu": torch.nn.functional.gelu(x), "silu": torch.nn.functional.silu(x)}
    caps = ad.capabilities()
    return {"name": ad.activation_name, "module": type(ad.activation_module()).__name__, "mlp": caps["mlp"],
            "x": x.tolist(), "y": y.tolist(), "references": {k: v.tolist() for k, v in refs.items()},
            "note": "`y` is the loaded model's activation module applied to x. Reference curves are standard torch functions "
                    "shown for comparison only; they are not used by this model unless named above."}
