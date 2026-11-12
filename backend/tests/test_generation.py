"""Sampling, step-by-step generation, KV cache and retention."""
from __future__ import annotations

import time

import pytest
import torch

from app.core.config import settings
from app.inference.sampling import SamplingConfig, distribution, select
from app.inference.session import create_run
from app.services import analysis
from app.tracing.engine import cache_kv
from app.tracing.schema import CaptureConfig
from tests.conftest import PROMPT
from tests.test_trace import close, make_run


def test_stepwise_generation_matches_generate(lm):
    run = make_run(lm, 8)
    with torch.inference_mode():
        out = lm.model.generate(torch.tensor([run.prompt_ids]), max_new_tokens=8, do_sample=False, pad_token_id=0,
                                eos_token_id=None)
    expected = out[0, len(run.prompt_ids):].tolist()
    assert run.generated[: len(expected)] == expected[: len(run.generated)]
    assert [r.phase for r in run.steps] == ["prefill"] + ["decode"] * (len(run.steps) - 1)
    assert all(r.seq_len - r.past_len == 1 for r in run.steps[1:])
    assert run.steps[0].past_len == 0 and run.steps[0].seq_len == len(run.prompt_ids)


def test_kv_cache_growth_and_reuse(lm):
    run = make_run(lm, 5)
    ad = lm.adapter
    n = len(run.prompt_ids)
    for s, rec in enumerate(run.steps):
        assert len(rec.cache_layers) == ad.n_layers
        for layer in rec.cache_layers:
            assert layer["key_shape"] == [1, ad.n_kv_heads, n + s, ad.head_dim] == layer["value_shape"]
        assert rec.cache_bytes == ad.n_layers * 2 * ad.n_kv_heads * (n + s) * ad.head_dim * 4
    # decode: attention row covers every cached position, Q is a single row
    last = run.steps[-1]
    assert list(last.tensor(0, "q_used").shape) == [1, ad.n_heads, 1, ad.head_dim]
    assert list(last.tensor(0, "weights").shape) == [1, ad.n_heads, 1, last.seq_len]
    # cached prefix is exactly what earlier steps computed (reuse, not recompute)
    k, v = cache_kv(run.cache, 1)
    assert torch.equal(k[:, :, :n], run.steps[0].tensor(1, "k_cache"))
    assert torch.equal(v[:, :, : n + 1], run.steps[1].tensor(1, "v_cache"))
    cmp = analysis.kv_cache_compare(run, len(run.steps) - 1)
    assert cmp["max_abs_logit_diff"] < 1e-4 and cmp["same_argmax"]
    assert cmp["with_cache"]["positions_processed"] == 1 and cmp["without_cache"]["positions_processed"] == last.seq_len
    info = analysis.kv_cache(run, 2)
    assert info["seq_len"] == n + 2 and [h["seq_len"] for h in info["history"]] == [n + i for i in range(len(run.steps))]
    t = analysis.kv_cache_tensor(run, 2, 0, "key", 0, None)
    assert t["shape"] == [n + 2, ad.head_dim] and t["new_from"] == n + 1


def test_assembled_attention_matches_full_forward(lm):
    run = make_run(lm, 4)
    total = run.steps[-1].seq_len
    with torch.inference_mode():
        ref = lm.model(input_ids=torch.tensor([run.all_ids()[:total]]), output_attentions=True, use_cache=False)
    for layer in range(lm.adapter.n_layers):
        m = analysis.attention_matrix(run, layer, len(run.steps) - 1)
        mask = torch.tril(torch.ones(total, total, dtype=torch.bool))
        assert torch.isnan(m[:, ~mask]).all()
        close(m[:, mask], ref.attentions[layer][0][:, mask], atol=1e-4)
    row = analysis.attention_row(run, 1, 2, total - 1, len(run.steps) - 1)
    assert row["phase"] == "decode" and abs(row["weights_sum"] - 1) < 1e-5 and row["recompute_max_err"] < 1e-5
    assert abs(sum(row["exp"]) - row["denominator"]) < 1e-4


def test_sampling_distribution():
    torch.manual_seed(0)
    logits = torch.randn(50) * 3
    d = distribution(logits, SamplingConfig())
    close(d.probs, torch.softmax(logits, -1))
    assert select(d, SamplingConfig(), None) == int(logits.argmax())
    hot, cold = distribution(logits, SamplingConfig(temperature=2.0)), distribution(logits, SamplingConfig(temperature=0.5))
    close(hot.probs, torch.softmax(logits / 2, -1))
    assert cold.probs.max() > d.probs.max() > hot.probs.max()
    k = distribution(logits, SamplingConfig(top_k=5))
    assert int(k.kept.sum()) == 5 and abs(float(k.probs.sum()) - 1) < 1e-6
    assert set(k.kept.nonzero().flatten().tolist()) == set(torch.topk(logits, 5).indices.tolist())
    p = distribution(logits, SamplingConfig(top_p=0.8))
    kept_mass = d.probs[p.kept].sum()
    smallest = d.probs[p.kept].min()
    assert kept_mass >= 0.8 and kept_mass - smallest < 0.8  # minimal nucleus
    assert (p.probs[~p.kept] == 0).all()
    one = distribution(logits, SamplingConfig(top_p=1e-6))
    assert int(one.kept.sum()) == 1 and bool(one.kept[logits.argmax()])
    # sampling: seeded, only kept tokens, roughly follows the distribution
    cfg = SamplingConfig(strategy="sample", top_k=3, seed=7)
    g = torch.Generator().manual_seed(7)
    dk = distribution(logits, cfg)
    draws = [select(dk, cfg, g) for _ in range(300)]
    assert set(draws) <= set(dk.kept.nonzero().flatten().tolist())
    assert max(set(draws), key=draws.count) == int(logits.argmax())


def test_seeded_sampling_is_reproducible(gpt2):
    cfg = SamplingConfig(strategy="sample", temperature=1.3, top_k=20, seed=11)
    a, b = make_run(gpt2, 6, sampling=cfg), make_run(gpt2, 6, sampling=cfg)
    assert a.generated == b.generated
    before = list(a.generated)
    a.reset()
    assert len(a.steps) == 1 and a.state == "ready"
    while a.state != "finished":
        a.next_token()
    assert a.generated == before


def test_logits_view_what_if(gpt2):
    run = make_run(gpt2, 2)
    base = analysis.logits_view(run, 0, None, 10)
    assert [r["rank"] for r in base["top"]] == list(range(10)) and not base["what_if"]
    assert base["top"][0]["token_id"] == run.steps[0].selected_token_id
    assert base["top"][0]["prob_raw"] == pytest.approx(base["top"][0]["prob_final"])
    hot = analysis.logits_view(run, 0, SamplingConfig(temperature=3.0, top_k=4), 10)
    assert hot["kept"] == 4 and hot["entropy_raw"] == pytest.approx(base["entropy_raw"])
    assert hot["top"][0]["prob_temperature"] < base["top"][0]["prob_raw"]
    assert sum(r["prob_final"] for r in hot["top"]) == pytest.approx(1.0, abs=1e-5)
    assert [r["kept"] for r in hot["top"]] == [True] * 4 + [False] * 6


def test_cursor_and_auto_run(gpt2):
    run = create_run(gpt2, PROMPT, CaptureConfig(), SamplingConfig(), 6)
    n_events = len(run.steps[0].events)
    run.move_cursor("next_op")
    assert run.cursor == {"step": 0, "order": 1}
    run.move_cursor("prev_op"); run.move_cursor("prev_op")
    assert run.cursor == {"step": 0, "order": 0}
    run.move_cursor("goto", 0, n_events - 1)
    run.move_cursor("next_op")  # past the end of the trace -> really generates the next token
    assert len(run.steps) == 2 and run.cursor == {"step": 1, "order": 0}
    run.resume()
    deadline = time.time() + 30
    while run.state == "running" and time.time() < deadline:
        time.sleep(0.02)
    assert run.state == "finished" and run.stop_reason == "max_new_tokens" and len(run.generated) == 6
    with pytest.raises(Exception):
        run.next_token()


def test_pause_stops_generation(gpt2):
    run = create_run(gpt2, PROMPT, CaptureConfig(), SamplingConfig(), 40)
    run.delay_ms = 20
    run.resume()
    time.sleep(0.15)
    run.pause()
    n = len(run.steps)
    assert run.state == "paused" and 1 < n < 41
    time.sleep(0.1)
    assert len(run.steps) == n


def test_retention_releases_old_steps(gpt2, monkeypatch):
    monkeypatch.setattr(settings, "retain_steps", 2)
    run = make_run(gpt2, 6)
    kept = [r.index for r in run.steps if len(r.tensors) > 3]
    assert kept == [0, len(run.steps) - 2, len(run.steps) - 1]
    old = run.steps[1]
    assert old.tensor(None, "next_token_logits") is not None  # cheap logits survive
    ev = old.event(0, "weights")
    assert not ev.stored and "retention" in ev.note and ev.shape  # metadata survives
    with pytest.raises(Exception) as e:
        analysis.attention(run, 0, len(run.steps) - 1, None)
    assert e.value.status_code == 409


def test_hidden_state_views(lm):
    run = make_run(lm, 3)
    h = analysis.hidden_states(run, "resid_post", len(run.steps) - 1, -1, -1)
    n_depth, t = lm.adapter.n_layers + 2, run.steps[-1].seq_len
    assert len(h["depth_labels"]) == n_depth and len(h["norms"]) == n_depth and len(h["norms"][0]) == t
    assert len(h["pca"]["points"]) == n_depth and len(h["pca"]["points"][0]) == t
    sim = torch.tensor(h["token_similarity"])
    close(sim.diagonal(), torch.ones(t), atol=1e-4)
    assert sim.shape == (t, t) and torch.allclose(sim, sim.T, atol=1e-5)
    assert len(h["consecutive_similarity"]) == n_depth - 1
    stack, _ = analysis.hidden_stack(run, "resid_post", len(run.steps) - 1)
    pts, ratio = analysis.pca2(stack.reshape(-1, stack.shape[-1]))
    x = stack.reshape(-1, stack.shape[-1]); xc = x - x.mean(0)
    # PCA preserves pairwise structure in the top-2 subspace and reports a valid variance ratio
    assert 0 < ratio[0] <= 1 and ratio[0] >= ratio[1] >= 0
    close(pts.var(0, unbiased=False).sum() / xc.var(0, unbiased=False).sum(), torch.tensor(sum(ratio)), atol=1e-4)
