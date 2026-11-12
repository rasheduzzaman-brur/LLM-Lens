"""Captured and re-derived tensors must equal what plain PyTorch computes."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from app.inference.sampling import SamplingConfig
from app.inference.session import create_run
from app.tracing import derive
from app.tracing.schema import CaptureConfig
from tests.conftest import PROMPT

ATOL = 1e-5


def make_run(lm, n=4, capture=None, sampling=None, prompt=PROMPT):
    run = create_run(lm, prompt, capture or CaptureConfig(), sampling or SamplingConfig(), n)
    while run.state != "finished":
        run.next_token()
    return run


@pytest.fixture(scope="module")
def run(lm):
    return make_run(lm)


@pytest.fixture(scope="module")
def ref(lm, run):
    """Plain forward over the whole final sequence, no hooks, no cache."""
    ids = torch.tensor([run.all_ids()[: run.steps[-1].seq_len]])
    with torch.inference_mode():
        return lm.model(input_ids=ids, output_hidden_states=True, output_attentions=True, use_cache=False)


def close(a, b, atol=ATOL):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert torch.allclose(a.float(), b.float(), atol=atol, rtol=1e-4), float((a.float() - b.float()).abs().max())


def test_every_verification_passes(run):
    for rec in run.steps:
        checks = [e for e in rec.events if e.verified is not None]
        assert len(checks) > 10
        assert [(e.layer, e.tensor_name, e.max_abs_err) for e in checks if not e.verified] == []


def test_event_order_and_schema(run, lm):
    rec = run.steps[0]
    assert [e.order for e in rec.events] == list(range(len(rec.events)))
    ids = [e.event_id for r in run.steps for e in r.events]
    assert ids == list(range(len(ids)))
    assert rec.phase == "prefill" and all(r.phase == "decode" for r in run.steps[1:])
    assert rec.events[0].stage == "embedding" and rec.events[-1].stage == "sampling"
    layers = [e.layer for e in rec.events if e.layer is not None]
    assert layers == sorted(layers) and set(layers) == set(range(lm.adapter.n_layers))
    for e in rec.events:
        if e.stored:
            t = rec.tensors[e.event_id]
            assert list(t.shape) == e.shape and str(t.dtype).endswith(e.dtype)
            assert not t.requires_grad


def test_embeddings(run, lm, ref):
    ad = lm.adapter
    ids = torch.tensor(run.prompt_ids)
    tok = run.steps[0].tensor(None, "tok_emb")[0]
    close(tok, ad.embed_tokens.weight[ids])
    emb = run.steps[0].tensor(None, "emb_out")[0]
    close(emb, ref.hidden_states[0][0, : len(ids)])
    if ad.family == "gpt2":
        close(run.steps[0].tensor(None, "pos_emb")[0], lm.model.transformer.wpe.weight[: len(ids)])
        # decode step uses the next absolute position
        close(run.steps[1].tensor(None, "pos_emb")[0], lm.model.transformer.wpe.weight[len(ids) : len(ids) + 1])


def test_hidden_states_match_model(run, lm, ref):
    """Hugging Face reports hidden_states[i + 1] = output of block i, except that the last entry
    has the final normalization applied."""
    n, last = len(run.prompt_ids), lm.adapter.n_layers - 1
    for i in range(lm.adapter.n_layers):
        name, layer = ("resid_post", i) if i < last else ("final_norm_out", None)
        close(run.steps[0].tensor(layer, name)[0], ref.hidden_states[i + 1][0, :n], atol=1e-4)
        # decode steps (computed incrementally with the KV cache) equal the full recompute
        for s, rec in enumerate(run.steps[1:], start=1):
            close(rec.tensor(layer, name)[0, 0], ref.hidden_states[i + 1][0, n + s - 1], atol=1e-4)


def test_qkv_projection(run, lm):
    ad = lm.adapter
    rec = run.steps[0]
    for i in range(ad.n_layers):
        x = rec.tensor(i, "ln1_out")
        w = ad.layer_weights(i)
        for name, wn, bn in (("q", "Wq", "bq"), ("k", "Wk", "bk"), ("v", "Wv", "bv")):
            expect = x @ w[wn]
            if bn in w:
                expect = expect + w[bn]
            close(rec.tensor(i, name), expect)


def test_attention_scores_mask_softmax(run, lm, ref):
    ad = lm.adapter
    rec = run.steps[0]
    n = len(run.prompt_ids)
    for i in range(ad.n_layers):
        q, k, v = (rec.tensor(i, x) for x in ("q_used", "k_used", "v_used"))
        assert list(q.shape) == [1, ad.n_heads, n, ad.head_dim] == list(k.shape) == list(v.shape)
        raw = rec.tensor(i, "scores_raw")
        close(raw, q @ k.transpose(-1, -2))
        scores = rec.tensor(i, "scores")
        close(scores, raw / math.sqrt(ad.head_dim))
        mask = rec.tensor(i, "mask")
        assert torch.equal(mask, torch.tril(torch.ones(n, n, dtype=torch.bool)))
        masked = rec.tensor(i, "scores_masked")
        close(masked[..., mask], scores[..., mask])
        assert (masked[..., ~mask] < -1e30).all()
        weights = rec.tensor(i, "weights")
        close(weights, torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1))
        close(weights.sum(-1), torch.ones(1, ad.n_heads, n))
        assert (weights[..., ~mask] == 0).all()
        # the model's own attention weights
        close(weights, ref.attentions[i][:, :, :n, :n])
        close(rec.tensor(i, "head_out"), weights @ v)


def test_attention_output_and_residuals(run, lm):
    ad = lm.adapter
    rec = run.steps[0]
    for i in range(ad.n_layers):
        w = ad.layer_weights(i)
        ho = rec.tensor(i, "head_out")
        merged = ho.transpose(1, 2).reshape(1, ho.shape[2], -1)
        close(rec.tensor(i, "attn_merged"), merged)
        out = merged @ w["Wo"]
        if "bo" in w:
            out = out + w["bo"]
        close(rec.tensor(i, "attn_out"), out)
        close(rec.tensor(i, "resid_mid"), rec.tensor(i, "resid_pre") + rec.tensor(i, "attn_out"))
        close(rec.tensor(i, "resid_post"), rec.tensor(i, "resid_mid") + rec.tensor(i, "mlp_out"))
        if i:
            close(rec.tensor(i, "resid_pre"), run.steps[0].tensor(i - 1, "resid_post"))


def test_normalization(run, lm):
    ad = lm.adapter
    rec = run.steps[0]
    for i in range(ad.n_layers):
        for prefix, x_name, mod in zip(("ln1", "ln2"), ("resid_pre", "resid_mid"), ad.norm_modules(i)):
            x = rec.tensor(i, x_name)
            out = rec.tensor(i, f"{prefix}_out")
            close(out, mod(x))
            if ad.norm_type == "layernorm":
                assert isinstance(mod, nn.LayerNorm)
                mean, var = rec.tensor(i, f"{prefix}_mean"), rec.tensor(i, f"{prefix}_var")
                close(mean, x.mean(-1, keepdim=True))
                close(var, x.var(-1, unbiased=False, keepdim=True))
                close(rec.tensor(i, f"{prefix}_normalized") * mod.weight + mod.bias, out)
                close(out, F.layer_norm(x, x.shape[-1:], mod.weight, mod.bias, mod.eps))
            else:
                ms, rms = rec.tensor(i, f"{prefix}_meansq"), rec.tensor(i, f"{prefix}_rms")
                close(ms, x.pow(2).mean(-1, keepdim=True))
                close(rms, torch.sqrt(ms + mod.variance_epsilon))
                close(rec.tensor(i, f"{prefix}_normalized"), x / rms)
                close((x / rms) * mod.weight, out)


def test_layernorm_and_rmsnorm_formulas():
    torch.manual_seed(1)
    x, g, b = torch.randn(2, 5, 16) * 3, torch.randn(16), torch.randn(16)
    s = derive.layernorm_steps(x, g, b, 1e-5)
    close(s["out"], F.layer_norm(x, (16,), g, b, 1e-5))
    r = derive.rmsnorm_steps(x, g, 1e-6)
    close(r["out"], F.rms_norm(x, (16,), g, 1e-6))
    close(r["rms"], torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))


def test_mlp(run, lm):
    ad = lm.adapter
    rec = run.steps[0]
    act = ad.activation_module()
    for i in range(ad.n_layers):
        w = ad.layer_weights(i)
        x = rec.tensor(i, "ln2_out")
        if ad.capabilities()["mlp"] == "standard":
            close(rec.tensor(i, "mlp_pre"), x @ w["W1"] + w["b1"])
            close(rec.tensor(i, "mlp_act"), act(rec.tensor(i, "mlp_pre")))
            close(rec.tensor(i, "mlp_out"), rec.tensor(i, "mlp_act") @ w["W2"] + w["b2"])
        else:
            assert ad.capabilities()["mlp"] == "swiglu"
            close(rec.tensor(i, "mlp_gate"), x @ w["W_gate"])
            close(rec.tensor(i, "mlp_gate_act"), F.silu(rec.tensor(i, "mlp_gate")))
            close(rec.tensor(i, "mlp_up"), x @ w["W_up"])
            close(rec.tensor(i, "mlp_hidden"), rec.tensor(i, "mlp_gate_act") * rec.tensor(i, "mlp_up"))
            close(rec.tensor(i, "mlp_out"), rec.tensor(i, "mlp_hidden") @ w["W_down"])
        close(rec.tensor(i, "mlp_out"), ad.blocks[i].mlp(x))


def test_logits_and_probabilities(run, lm, ref):
    n = len(run.prompt_ids)
    rec = run.steps[0]
    close(rec.tensor(None, "logits")[0], ref.logits[0, :n], atol=1e-4)
    close(rec.tensor(None, "final_norm_out")[0], lm.adapter.final_norm(rec.tensor(None, "final_norm_in"))[0])
    for s, r in enumerate(run.steps):
        logits = r.tensor(None, "next_token_logits")
        close(logits, ref.logits[0, n + s - 1], atol=1e-4)
        probs = r.tensor(None, "next_token_probs")
        close(probs, torch.softmax(logits, -1))
        assert abs(float(probs.sum()) - 1) < 1e-5
        assert r.selected_token_id == int(logits.argmax())  # greedy
        assert r.sampling["prob_raw"] == pytest.approx(float(probs[r.selected_token_id]))


def test_rope_and_gqa(llama):
    run = make_run(llama, 2)
    ad = llama.adapter
    assert ad.capabilities()["grouped_query_attention"] and ad.n_rep == 2
    rec = run.steps[0]
    k_cache, k_used = rec.tensor(0, "k_cache"), rec.tensor(0, "k_used")
    assert k_cache.shape[1] == 2 and k_used.shape[1] == 4
    close(k_used, k_cache.repeat_interleave(2, dim=1))
    # rotation preserves the norm of every (q, k) vector
    q_plain = derive.split_heads(rec.tensor(0, "q"), 4)
    close(rec.tensor(0, "q_used").norm(dim=-1), q_plain.norm(dim=-1))
    # position 0 has zero rotation angle
    close(rec.tensor(0, "q_used")[:, :, 0], q_plain[:, :, 0])


def test_capture_modes(gpt2):
    meta = make_run(gpt2, 2, CaptureConfig(mode="metadata"))
    rec = meta.steps[0]
    assert {e.tensor_name for e in rec.events if e.stored} == {"next_token_logits", "next_token_probs", "sampling_probs"}
    assert all(e.shape and e.stats is not None for e in rec.events if e.provenance != "unavailable")
    sel = make_run(gpt2, 2, CaptureConfig(mode="selected", tensors=["weights", "resid_post"], layers=[1]))
    stored = {(e.layer, e.tensor_name) for e in sel.steps[0].events if e.stored and e.layer is not None}
    assert stored == {(1, "weights"), (1, "resid_post")}
    full = make_run(gpt2, 2)
    assert full.stored_bytes() > sel.stored_bytes() > meta.stored_bytes()


def test_no_hooks_left_behind(lm, run):
    for m in lm.model.modules():
        assert not m._forward_hooks and not m._forward_pre_hooks


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_other_dtypes(tiny_factory, dtype):
    lm = tiny_factory("llama", dtype)
    run = make_run(lm, 2)
    for rec in run.steps:
        assert [(e.layer, e.tensor_name, e.max_abs_err) for e in rec.events if e.verified is False] == []
    assert run.steps[0].event(0, "resid_post").dtype == str(dtype).replace("torch.", "")
