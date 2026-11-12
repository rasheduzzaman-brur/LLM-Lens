"""GPT-2 adapter (reference implementation).

GPT-2 specifics: a single fused Conv1D (`c_attn`) produces Q, K and V at once,
positions are learned absolute embeddings, normalization is LayerNorm (pre-LN)
and the MLP is Linear → GELU → Linear.
"""
from __future__ import annotations

import torch
from torch import nn

from app.adapters.base import BaseModelAdapter
from app.instrumentation.hooks import Capture, HookManager
from app.tracing import derive
from app.tracing.engine import Emitter, StepContext


class GPT2Adapter(BaseModelAdapter):
    family = "gpt2"
    model_types = ("gpt2",)
    mask_mode = "fill"

    def validate(self) -> None:
        tr = getattr(self.model, "transformer", None)
        if tr is None or not all(hasattr(tr, a) for a in ("wte", "wpe", "h", "ln_f")):
            raise ValueError("GPT2Adapter: model has no GPT-2 style `transformer.{wte,wpe,h,ln_f}`")
        b = tr.h[0]
        if not all(hasattr(b, a) for a in ("ln_1", "attn", "ln_2", "mlp")) or not hasattr(b.attn, "c_attn"):
            raise ValueError("GPT2Adapter: unexpected block structure")
        if getattr(self.cfg, "add_cross_attention", False):
            raise ValueError("GPT2Adapter: cross-attention variants are not supported")
        if getattr(self.cfg, "reorder_and_upcast_attn", False):
            raise ValueError("GPT2Adapter: reorder_and_upcast_attn is not supported")

    # --- structure
    @property
    def blocks(self):
        return self.model.transformer.h

    @property
    def embed_tokens(self):
        return self.model.transformer.wte

    @property
    def final_norm(self):
        return self.model.transformer.ln_f

    @property
    def lm_head(self):
        return self.model.lm_head

    @property
    def intermediate_size(self) -> int:
        return int(self.blocks[0].mlp.c_fc.weight.shape[1])

    def norm_modules(self, layer):
        b = self.blocks[layer]
        return b.ln_1, b.ln_2

    def activation_module(self):
        return self.blocks[0].mlp.act

    def attention_scale(self, layer: int) -> float:
        s = 1.0
        if getattr(self.cfg, "scale_attn_weights", True):
            s /= float(self.head_dim) ** 0.5
        if getattr(self.cfg, "scale_attn_by_inverse_layer_idx", False):
            s /= float(layer + 1)
        return s

    def capabilities(self) -> dict:
        return {
            "norm": "layernorm",
            "norm_placement": "pre",
            "positional": "learned_absolute",
            "mlp": "standard",
            "activation": self.activation_name,
            "fused_qkv": True,
            "qkv_bias": True,
            "grouped_query_attention": False,
            "rope": False,
            "mask_mode": self.mask_mode,
        }

    # --- hooks
    def register_global_hooks(self, hm: HookManager) -> None:
        tr = self.model.transformer
        hm.output(tr.wte, None, "tok_emb")
        hm.output(tr.wpe, None, "pos_emb")
        hm.output(tr.drop, None, "emb_out")
        hm.input(tr.ln_f, None, "final_norm_in")
        hm.output(tr.ln_f, None, "final_norm_out")
        hm.output(self.model.lm_head, None, "logits")

    def register_layer_hooks(self, hm: HookManager, i: int) -> None:
        b = self.blocks[i]
        hm.input(b, i, "resid_pre")
        hm.output(b.ln_1, i, "ln1_out")
        hm.output(b.attn.c_attn, i, "qkv")
        hm.input(b.attn.c_proj, i, "attn_merged")
        hm.output(b.attn.c_proj, i, "attn_out")
        hm.output(b.attn, i, "attn_weights_model", pick=lambda o: o[1] if isinstance(o, tuple) and len(o) > 1 else None)
        hm.input(b.ln_2, i, "resid_mid")
        hm.output(b.ln_2, i, "ln2_out")
        hm.output(b.mlp.c_fc, i, "mlp_pre")
        hm.output(b.mlp.act, i, "mlp_act")
        hm.output(b.mlp.c_proj, i, "mlp_out")
        hm.output(b, i, "resid_post")

    # --- events
    def emit_embeddings(self, cap: Capture, em: Emitter, ctx: StepContext) -> None:
        tok, pos, out = cap.need(None, "tok_emb"), cap.need(None, "pos_emb"), cap.need(None, "emb_out")
        em.emit(stage="embedding", layer=None, op="token_embedding", name="tok_emb", tensor=tok,
                formula="tok_emb = W_E[token_ids]")
        em.emit(stage="embedding", layer=None, op="positional_embedding", name="pos_emb", tensor=pos,
                formula=f"pos_emb = W_P[positions]   (positions {ctx.past_len}…{ctx.total_len - 1})",
                note="Learned absolute position embeddings.")
        em.emit(stage="embedding", layer=None, op="embedding_sum", name="emb_out", tensor=out,
                formula="emb_out = tok_emb + pos_emb", inputs=("tok_emb", "pos_emb"),
                check=derive.compare(tok + pos, out), against="tok_emb + pos_emb")

    def emit_qkv(self, cap, em, ctx, i):
        d, h = self.hidden_size, self.n_heads
        fused = cap.need(i, "qkv")
        em.emit(stage="attention", layer=i, op="fused_qkv_projection", name="qkv", tensor=fused,
                formula="[Q | K | V] = X · W_attn + b_attn", inputs=("ln1_out",),
                note="GPT-2 computes Q, K and V with one fused Conv1D; the three thirds of the output are Q, K, V.")
        q, k, v = fused.split(d, dim=2)
        note = "Slice of the hooked fused projection output."
        em.emit(stage="attention", layer=i, op="query_projection", name="q", tensor=q, raw_key="qkv",
                formula="Q = X · Wq + bq", inputs=("ln1_out",), note=note)
        em.emit(stage="attention", layer=i, op="key_projection", name="k", tensor=k, raw_key="qkv",
                formula="K = X · Wk + bk", inputs=("ln1_out",), note=note)
        em.emit(stage="attention", layer=i, op="value_projection", name="v", tensor=v, raw_key="qkv",
                formula="V = X · Wv + bv", inputs=("ln1_out",), note=note)
        qh, kh, vh = (derive.split_heads(t, h) for t in (q, k, v))
        em.emit(stage="attention", layer=i, op="split_heads", name="q_used", tensor=qh, provenance="derived",
                formula=f"reshape [1, T, {d}] → [1, {h}, T, {d // h}]", inputs=("q",),
                note="Reshape only (no arithmetic). This is the Q that enters Q·Kᵀ.")
        k_all, v_all = self.emit_cache_kv(em, ctx, i, kh, vh, "k", "v")
        em.emit(stage="attention", layer=i, op="keys_for_attention", name="k_used", tensor=k_all, raw_key="k_cache",
                formula="K = K_cache", inputs=("k_cache",), note="Read from the model's KV cache.")
        em.emit(stage="attention", layer=i, op="values_for_attention", name="v_used", tensor=v_all, raw_key="v_cache",
                formula="V = V_cache", inputs=("v_cache",), note="Read from the model's KV cache.")
        return qh, k_all, v_all

    def emit_mlp(self, cap, em, ctx, i):
        act = self.activation_name
        em.emit(stage="mlp", layer=i, op="mlp_fc", name="mlp_pre", tensor=cap.need(i, "mlp_pre"),
                formula="pre = X · W1 + b1", inputs=("ln2_out",))
        em.emit(stage="mlp", layer=i, op="mlp_activation", name="mlp_act", tensor=cap.need(i, "mlp_act"),
                formula=f"hidden = {act}(pre)", inputs=("mlp_pre",))
        em.emit(stage="mlp", layer=i, op="mlp_projection", name="mlp_out", tensor=cap.need(i, "mlp_out"),
                formula="mlp_out = hidden · W2 + b2", inputs=("mlp_act",))

    # --- parameters
    def layer_weights(self, i: int) -> dict[str, torch.Tensor]:
        b, d = self.blocks[i], self.hidden_size
        w, bias = b.attn.c_attn.weight, b.attn.c_attn.bias
        return {
            "Wq": w[:, :d], "Wk": w[:, d : 2 * d], "Wv": w[:, 2 * d :],
            "bq": bias[:d], "bk": bias[d : 2 * d], "bv": bias[2 * d :],
            "Wo": b.attn.c_proj.weight, "bo": b.attn.c_proj.bias,
            "W1": b.mlp.c_fc.weight, "b1": b.mlp.c_fc.bias,
            "W2": b.mlp.c_proj.weight, "b2": b.mlp.c_proj.bias,
            "ln1_gamma": b.ln_1.weight, "ln1_beta": b.ln_1.bias,
            "ln2_gamma": b.ln_2.weight, "ln2_beta": b.ln_2.bias,
        }

    def global_weights(self) -> dict[str, torch.Tensor]:
        tr = self.model.transformer
        return {"W_E": tr.wte.weight, "W_P": tr.wpe.weight, "final_norm_gamma": tr.ln_f.weight,
                "final_norm_beta": tr.ln_f.bias, "W_U": self.model.lm_head.weight}

    # --- architecture
    def embedding_nodes(self) -> list[dict]:
        d, tr = self.hidden_size, self.model.transformer
        return [
            {"id": "tok_emb", "kind": "embedding", "label": "Token Embedding", "shape": f"[1, T, {d}]",
             "tensors": ["tok_emb"], "stage": "embedding", "formula": "W_E[token_ids]",
             "module": "transformer.wte", "params": {"weight": list(tr.wte.weight.shape)}},
            {"id": "pos_emb", "kind": "embedding", "label": "Positional Embedding", "shape": f"[1, T, {d}]",
             "tensors": ["pos_emb", "emb_out"], "stage": "embedding", "formula": "emb_out = tok_emb + W_P[positions]",
             "module": "transformer.wpe", "params": {"weight": list(tr.wpe.weight.shape)}},
        ]

    def layer_template(self) -> list[dict]:
        d, h, dh, m = self.hidden_size, self.n_heads, self.head_dim, self.intermediate_size
        return [
            {"key": "ln1", "label": "LayerNorm 1", "stage": "norm", "module": "ln_1", "in": ["resid_pre"],
             "out": ["ln1_out"], "detail": ["ln1_mean", "ln1_var", "ln1_normalized"], "shape": f"[1, T, {d}]",
             "formula": "(x − mean) / √(var + ε) · γ + β"},
            {"key": "qkv", "label": "Q / K / V projection", "stage": "attention", "module": "attn.c_attn",
             "in": ["ln1_out"], "out": ["q", "k", "v"], "detail": ["qkv", "q_used", "k_cache", "v_cache", "k_used", "v_used"],
             "shape": f"[1, T, {d}] ×3", "formula": "Q = XWq + bq,  K = XWk + bk,  V = XWv + bv  (one fused Conv1D)"},
            {"key": "scores", "label": "Attention scores", "stage": "attention", "module": None, "in": ["q_used", "k_used"],
             "out": ["scores"], "detail": ["scores_raw"], "shape": f"[1, {h}, T, T]", "formula": f"Q·Kᵀ / √{dh}"},
            {"key": "mask", "label": "Causal mask", "stage": "attention", "module": None, "in": ["scores"],
             "out": ["scores_masked"], "detail": ["mask"], "shape": f"[1, {h}, T, T]", "formula": "future positions → finfo.min"},
            {"key": "softmax", "label": "Softmax", "stage": "attention", "module": None, "in": ["scores_masked"],
             "out": ["weights"], "detail": [], "shape": f"[1, {h}, T, T]", "formula": "softmax over each row"},
            {"key": "attn_v", "label": "Weights × V", "stage": "attention", "module": None, "in": ["weights", "v_used"],
             "out": ["head_out"], "detail": ["attn_merged"], "shape": f"[1, {h}, T, {dh}]", "formula": "weights · V, heads concatenated"},
            {"key": "attn_proj", "label": "Attention output projection", "stage": "attention", "module": "attn.c_proj",
             "in": ["attn_merged"], "out": ["attn_out"], "detail": [], "shape": f"[1, T, {d}]", "formula": "merged · Wo + bo"},
            {"key": "res1", "label": "Residual add", "stage": "residual", "module": None, "in": ["resid_pre", "attn_out"],
             "out": ["resid_mid"], "detail": [], "shape": f"[1, T, {d}]", "formula": "resid_pre + attn_out"},
            {"key": "ln2", "label": "LayerNorm 2", "stage": "norm", "module": "ln_2", "in": ["resid_mid"],
             "out": ["ln2_out"], "detail": ["ln2_mean", "ln2_var", "ln2_normalized"], "shape": f"[1, T, {d}]",
             "formula": "(x − mean) / √(var + ε) · γ + β"},
            {"key": "mlp_fc", "label": "MLP expand", "stage": "mlp", "module": "mlp.c_fc", "in": ["ln2_out"],
             "out": ["mlp_pre"], "detail": [], "shape": f"[1, T, {m}]", "formula": "X · W1 + b1"},
            {"key": "mlp_act", "label": f"Activation ({self.activation_name})", "stage": "mlp", "module": "mlp.act",
             "in": ["mlp_pre"], "out": ["mlp_act"], "detail": [], "shape": f"[1, T, {m}]", "formula": f"{self.activation_name}(pre)"},
            {"key": "mlp_proj", "label": "MLP project", "stage": "mlp", "module": "mlp.c_proj", "in": ["mlp_act"],
             "out": ["mlp_out"], "detail": [], "shape": f"[1, T, {d}]", "formula": "hidden · W2 + b2"},
            {"key": "res2", "label": "Residual add", "stage": "residual", "module": None, "in": ["resid_mid", "mlp_out"],
             "out": ["resid_post"], "detail": [], "shape": f"[1, T, {d}]", "formula": "resid_mid + mlp_out"},
        ]
