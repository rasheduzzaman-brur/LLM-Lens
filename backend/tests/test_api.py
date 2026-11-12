"""REST API over a tiny model, plus tensor slicing / pagination."""
from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.services import tensor_view as tv
from app.services.model_manager import models
from tests.conftest import PROMPT


@pytest.fixture(scope="module")
def client(gpt2):
    models.loaded[gpt2.model_id] = gpt2
    yield TestClient(app)
    models.loaded.pop(gpt2.model_id, None)


@pytest.fixture(scope="module")
def rid(client):
    r = client.post("/api/inference/run", json={"model_id": "tiny-gpt2", "prompt": PROMPT, "max_new_tokens": 5})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["checks"]["failed"] == 0 and s["checks"]["total"] > 0 and len(s["steps"]) == 1
    client.post(f"/api/inference/{s['run_id']}/step", json={})
    return s["run_id"]


def test_models_and_architecture(client):
    m = client.get("/api/models").json()
    assert m["loaded"][0]["model_id"] == "tiny-gpt2" and "gpt2" in m["supported_model_types"]
    info = m["loaded"][0]
    assert (info["n_layers"], info["n_heads"], info["hidden_size"], info["intermediate_size"], info["vocab_size"]) == (3, 4, 32, 128, 101)
    assert info["parameter_count"] == sum(p.numel() for p in models.get("tiny-gpt2").model.parameters())
    arch = client.get("/api/models/tiny-gpt2/architecture").json()
    assert [n["id"] for n in arch["nodes"] if n["kind"] == "layer"] == ["layer0", "layer1", "layer2"]
    assert arch["nodes"][0]["id"] == "input" and arch["nodes"][-1]["id"] == "output"
    assert client.get("/api/models/nope/architecture").status_code == 404
    w = client.get("/api/models/tiny-gpt2/weights", params={"layer": 0}).json()
    assert {"name": "Wq", "shape": [32, 32], "dtype": "float32"} in w["weights"]
    t = client.get("/api/models/tiny-gpt2/weights/tensor", params={"layer": 0, "name": "Wq", "index": "0,0:4"}).json()
    assert t["values"] == models.get("tiny-gpt2").adapter.layer_weights(0)["Wq"][0, :4].tolist()
    act = client.get("/api/models/tiny-gpt2/activation").json()
    assert act["name"] == "gelu_new" and len(act["x"]) == len(act["y"])
    emb = client.get("/api/models/tiny-gpt2/embedding", params={"token_id": 5}).json()
    assert len(emb["values"]) == 32 and len(emb["neighbors"]) == 8


def test_download_requires_confirmation(client):
    r = client.post("/api/models/load", json={"source": "definitely-not-a-real-org/not-a-model"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "download_required"
    assert client.post("/api/models/load", json={"source": "/nonexistent/dir"}).status_code == 404


def test_trace_endpoints(client, rid):
    tr = client.get(f"/api/inference/{rid}/trace", params={"step": 0, "limit": 2000}).json()
    assert tr["total"] == len(tr["events"]) and tr["phase"] == "prefill"
    assert all(e["values"] == [] if "values" in e else True for e in tr["events"])
    page = client.get(f"/api/inference/{rid}/trace", params={"step": 0, "offset": 5, "limit": 3}).json()
    assert [e["event_id"] for e in page["events"]] == [e["event_id"] for e in tr["events"][5:8]]
    only = client.get(f"/api/inference/{rid}/trace", params={"step": 0, "layer": 1, "stage": "attention"}).json()
    assert only["total"] > 5 and all(e["layer"] == 1 and e["stage"] == "attention" for e in only["events"])
    ev = next(e for e in tr["events"] if e["tensor_name"] == "weights" and e["layer"] == 2)
    one = client.get(f"/api/inference/{rid}/trace/{ev['event_id']}", params={"index": "0,1,2"}).json()
    assert one["event"]["operation"] == "softmax" and one["tensor"]["shape"] == [9]
    assert sum(one["tensor"]["values"]) == pytest.approx(1.0, abs=1e-5)
    csv = client.get(f"/api/inference/{rid}/trace-export", params={"step": 0})
    assert csv.headers["content-type"].startswith("text/csv") and len(csv.text.splitlines()) == tr["total"] + 1
    assert client.get(f"/api/inference/{rid}/trace/999999").status_code == 404
    assert client.get(f"/api/inference/{rid}/trace/{ev['event_id']}", params={"index": "0,99"}).status_code == 422


def test_analysis_endpoints(client, rid):
    a = client.get(f"/api/inference/{rid}/attention", params={"layer": 1}).json()
    assert a["verified"] and a["n_heads"] == 4 and a["seq_len"] == 10 and len(a["weights"][0]) == 10
    assert a["weights"][0][0][1] is None  # masked future position
    assert client.get(f"/api/inference/{rid}/attention", params={"layer": 9}).status_code == 422
    g = client.get(f"/api/inference/{rid}/attention/grid").json()
    assert len(g["matrices"]) == 3
    d = client.get(f"/api/inference/{rid}/layers/1", params={"position": 9}).json()
    assert d["phase"] == "decode" and len(d["vectors"]["resid_post"]) == 32 and len(d["vectors"]["mlp_pre"]) == 128
    assert all(c["verified"] for c in d["checks"].values()) and len(d["norm"]["ln1"]["gamma"]) == 32
    x, m, v = d["vectors"]["resid_pre"], d["scalars"]["ln1_mean"], d["scalars"]["ln1_var"]
    assert m == pytest.approx(sum(x) / 32, abs=1e-5)
    assert v == pytest.approx(sum((e - m) ** 2 for e in x) / 32, rel=1e-4)
    assert len(client.get(f"/api/inference/{rid}/layers").json()["layers"]) == 3
    gl = client.get(f"/api/inference/{rid}/global", params={"position": 0}).json()
    assert {"tok_emb", "pos_emb", "emb_out", "final_norm_out"} <= set(gl["vectors"])
    assert client.get(f"/api/inference/{rid}/hidden-states", params={"stage": "mlp_out"}).status_code == 200
    assert client.get(f"/api/inference/{rid}/hidden-states", params={"stage": "bogus"}).status_code == 422
    lg = client.get(f"/api/inference/{rid}/logits", params={"top_n": 5, "temperature": 0.5}).json()
    assert lg["what_if"] and len(lg["top"]) == 5 and lg["vocab_size"] == 101
    assert client.get(f"/api/inference/{rid}/kv-cache").json()["seq_len"] == 10
    assert client.get(f"/api/inference/{rid}/kv-cache/compare").json()["same_argmax"]
    t = client.get(f"/api/inference/{rid}/tensor", params={"name": "resid_post", "layer": 0, "assembled": True}).json()
    assert t["assembled"] and t["shape"] == [1, 10, 32]


def test_control_endpoints(client):
    s = client.post("/api/inference/run", json={"model_id": "tiny-gpt2", "prompt": PROMPT, "max_new_tokens": 3}).json()
    rid = s["run_id"]
    s = client.post(f"/api/inference/{rid}/step", json={"kind": "next_op"}).json()
    assert s["cursor"] == {"step": 0, "order": 1}
    s = client.post(f"/api/inference/{rid}/step", json={"kind": "next_token"}).json()
    s = client.post(f"/api/inference/{rid}/step", json={"kind": "next_token"}).json()
    assert s["state"] == "finished" and len(s["steps"]) == 3 and sum(t["generated"] for t in s["tokens"]) == 3
    assert client.post(f"/api/inference/{rid}/step", json={}).status_code == 409
    s = client.post(f"/api/inference/{rid}/reset").json()
    assert s["state"] == "ready" and len(s["steps"]) == 1
    assert client.post(f"/api/inference/{rid}/pause").status_code == 200
    with client.stream("GET", f"/api/inference/{rid}/events") as r:
        line = next(l for l in r.iter_lines() if l.startswith("data:"))
        assert f'"run_id": "{rid}"' in line
    assert client.delete(f"/api/inference/{rid}").status_code == 200
    assert client.get(f"/api/inference/{rid}").status_code == 404
    assert client.post("/api/inference/run", json={"model_id": "tiny-gpt2", "prompt": "a " * 80}).status_code == 422


def test_tensor_view_slicing_and_pooling():
    t = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    assert tv.view(t, "1,2")["values"] == [20.0, 21.0, 22.0, 23.0]
    assert tv.view(t, "0,:,1:3")["shape"] == [3, 2]
    assert tv.view(t, "0,-1,::2")["values"] == [8.0, 10.0]
    big = torch.randn(600, 500)
    v = tv.view(big, None, max_cells=10_000)
    assert v["pooled"] and v["pooled_shape"][0] * v["pooled_shape"][1] <= 10_000 and "Visualization transform" in v["pool_note"]
    assert torch.tensor(v["values"]).mean().item() == pytest.approx(big.mean().item(), abs=1e-3)
    assert tv.view(big, "0:10,0:10", max_cells=10_000)["pooled"] is False
    with pytest.raises(Exception) as e:
        tv.view(torch.zeros(3, 50, 500), None, max_cells=10_000)
    assert e.value.status_code == 413
    half = tv.view(torch.tensor([1.5, float("-inf"), float("nan")], dtype=torch.bfloat16))
    assert half["values"] == [1.5, None, None] and "upcast" in half["serialization_note"]
    assert tv.stats(torch.tensor([1.0, 3.0, torch.finfo(torch.float32).min])).max == 3.0
