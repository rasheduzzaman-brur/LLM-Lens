"""Checks on real downloaded weights (skipped when ../models/<name> is absent)."""
from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_manager import models
from tests.conftest import MODELS_DIR

pytestmark = pytest.mark.real
TEXT = "The Eiffel Tower is located in the city of"


def _load(name):
    if not (MODELS_DIR / name / "config.json").exists():
        pytest.skip(f"{name} not downloaded")
    return models.load(name)


@pytest.mark.parametrize("name,family,norm", [("gpt2", "gpt2", "layernorm"), ("SmolLM2-135M", "llama", "rmsnorm"),
                                              ("Qwen2.5-0.5B-Instruct", "qwen", "rmsnorm")])
def test_real_model_trace(name, family, norm):
    lm = _load(name)
    assert lm.adapter.family == family and lm.adapter.norm_type == norm
    c = TestClient(app)
    s = c.post("/api/inference/run", json={"model_id": lm.model_id, "prompt": TEXT, "max_new_tokens": 4}).json()
    rid = s["run_id"]
    for _ in range(3):
        s = c.post(f"/api/inference/{rid}/step", json={}).json()
    assert s["checks"]["failed"] == 0 and s["checks"]["total"] > 100
    ids = torch.tensor([[t["token_id"] for t in s["tokens"]][: s["n_prompt_tokens"]]])
    with torch.inference_mode():
        out = lm.model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=lm.tokenizer.eos_token_id)
    assert [t["token_id"] for t in s["tokens"]] == out[0].tolist()
    a = c.get(f"/api/inference/{rid}/attention", params={"layer": lm.adapter.n_layers - 1, "head": 0}).json()
    assert a["verified"]
    assert c.get(f"/api/inference/{rid}/kv-cache/compare").json()["same_argmax"]
    c.delete(f"/api/inference/{rid}")


def test_gpt2_tokenization():
    lm = _load("gpt2")
    c = TestClient(app)
    t = c.post("/api/tokenize", json={"model_id": "gpt2", "text": "Hello world, tokenization!"}).json()
    assert [x["token_id"] for x in t["tokens"]] == lm.tokenizer.encode("Hello world, tokenization!")
    assert [x["text"] for x in t["tokens"]] == ["Hello", " world", ",", " token", "ization", "!"]
    assert t["round_trip"] and t["count"] == 6 and t["vocabulary"]["vocab_size"] == 50257
    assert [(x["start"], x["end"]) for x in t["tokens"]][:2] == [(0, 5), (5, 11)]
    assert "".join("Hello world, tokenization!"[x["start"]:x["end"]] for x in t["tokens"]) == "Hello world, tokenization!"
    e = c.get("/api/models/gpt2/embedding", params={"token_id": 15496, "neighbors": 0}).json()
    assert e["values"] == lm.model.transformer.wte.weight[15496].tolist() and e["matrix_shape"] == [50257, 768]
