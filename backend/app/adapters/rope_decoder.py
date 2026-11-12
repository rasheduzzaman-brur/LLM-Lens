"""Shared adapter for Llama-style decoders (Llama, Qwen2, SmolLM2, TinyLlama, …).

Specifics: separate q/k/v projections, rotary position embeddings applied to Q and K
inside attention, optional grouped-query attention, RMSNorm (pre-norm) and a gated
SwiGLU MLP: down(act(gate(x)) * up(x)).
"""
from __future__ import annotations

import torch
from torch import nn

from app.adapters.base import BaseModelAdapter, lin_w
from app.instrumentation.hooks import Capture, HookManager
from app.tracing import derive
from app.tracing.engine import Emitter, StepContext


class RopeDecoderAdapter(BaseModelAdapter):
    family = "rope_decoder"
    mask_mode = "add"
    softmax_dtype = torch.float32

    def validate(self) -> None:
        core = getattr(self.model, "model", None)
        if core is None or not all(hasattr(core, a) for a in ("embed_tokens", "layers", "norm", "rotary_emb")):
            raise ValueError(f"{type(self).__name__}: model has no `model.{{embed_tokens,layers,norm,rotary_emb}}`")
        b = core.layers[0]
        need_block = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
        need_attn = ("q_proj", "k_proj", "v_proj", "o_proj")
        need_mlp = ("gate_proj", "up_proj", "down_proj", "act_fn")
        if not (all(hasattr(b, a) for a in need_block) and all(hasattr(b.self_attn, a) for a in need_attn)
                and all(hasattr(b.mlp, a) for a in need_mlp)):
            raise ValueError(f"{type(self).__name__}: unexpected decoder layer structure")
        if not isinstance(b.mlp.act_fn, nn.Module):
            raise ValueError(f"{type(self).__name__}: MLP activation is not an nn.Module and cannot be hooked")
        if getattr(self.cfg, "sliding_window", None) and getattr(self.cfg, "use_sliding_window", False):
            raise ValueError(f"{type(self).__name__}: sliding-window attention is not supported")

    @property
    def core(self):
        return self.model.model

    @property
    def blocks(self):
        return self.core.layers

    @property
    def embed_tokens(self):
        return self.core.embed_tokens

    @property
    def final_norm(self):
        return self.core.norm

    @property
    def lm_head(self):
        return self.model.lm_head

    @property
    def intermediate_size(self) -> int:
        return int(self.blocks[0].mlp.gate_proj.weight.shape[0])

    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv_heads

    def norm_modules(self, layer):
        b = self.blocks[layer]
        return b.input_layernorm, b.post_attention_layernorm

    def activation_module(self):
        return self.blocks[0].mlp.act_fn

    def attention_scale(self, layer: int) -> float:
        return float(self.blocks[layer].self_attn.scaling)

    def capabilities(self) -> dict:
        a = self.blocks[0].self_attn
        return {
            "norm": self.norm_type,
            "norm_placement": "pre",
            "positional": "rope",
            "mlp": "swiglu" if self.activation_name in ("silu", "swish") else "gated",
            "activation": self.activation_name,
            "fused_qkv": False,
            "qkv_bias": a.q_proj.bias is not None,
            "grouped_query_attention": self.n_rep > 1,
            "rope": True,
            "mask_mode": self.mask_mode,
        }

    # --- hooks
    def register_global_hooks(self, hm: HookManager) -> None:
        c = self.core
        hm.output(c.embed_tokens, None, "tok_emb")
        hm.output(c.rotary_emb, None, "rope_cos", pick=lambda o: o[0])
        hm.output(c.rotary_emb, None, "rope_sin", pick=lambda o: o[1])
        hm.input(c.norm, None, "final_norm_in")
        hm.output(c.norm, None, "final_norm_out")
        hm.output(self.model.lm_head, None, "logits")

    def register_layer_hooks(self, hm: HookManager, i: int) -> None:
        b = self.blocks[i]
        a, m = b.self_attn, b.mlp
        hm.input(b, i, "resid_pre")
        hm.output(b.input_layernorm, i, "ln1_out")
        hm.output(a.q_proj, i, "q")
        hm.output(a.k_proj, i, "k")
        hm.output(a.v_proj, i, "v")
        hm.input(a.o_proj, i, "attn_merged")
        hm.output(a.o_proj, i, "attn_out")
        hm.output(a, i, "attn_weights_model", pick=lambda o: o[1] if isinstance(o, tuple) and len(o) > 1 else None)
        hm.input(b.post_attention_layernorm, i, "resid_mid")
        hm.output(b.post_attention_layernorm, i, "ln2_out")
        hm.output(m.gate_proj, i, "mlp_gate")
        hm.output(m.act_fn, i, "mlp_gate_act")
        hm.output(m.up_proj, i, "mlp_up")
        hm.input(m.down_proj, i, "mlp_hidden")
        hm.output(m.down_proj, i, "mlp_out")
        hm.output(b, i, "resid_post")

    # --- events
    def emit_embeddings(self, cap: Capture, em: Emitter, ctx: StepContext) -> None:
        tok = cap.need(None, "tok_emb")
        em.emit(stage="embedding", layer=None, op="token_embedding", name="tok_emb", tensor=tok,
                formula="tok_emb = W_E[token_ids]")
        em.emit(stage="embedding", layer=None, op="rope_cos", name="rope_cos", tensor=cap.need(None, "rope_cos"),
                formula="cos(position · θ_j),  θ_j = base^(−2j/d_head)",
                note=f"Rotary position table for positions {ctx.past_len}…{ctx.total_len - 1}; applied to Q and K in every layer.")
        em.emit(stage="embedding", layer=None, op="rope_sin", name="rope_sin", tensor=cap.need(None, "rope_sin"),
                formula="sin(position · θ_j)")
        em.emit(stage="embedding", layer=None, op="embedding_output", name="emb_out", tensor=tok, raw_key="tok_emb",
                formula="emb_out = tok_emb", inputs=("tok_emb",),
                note="No additive positional embedding in this architecture; position enters through RoPE inside attention.")

    def emit_qkv(self, cap, em, ctx, i):
        h, hk, dh = self.n_heads, self.n_kv_heads, self.head_dim
        q, k, v = cap.need(i, "q"), cap.need(i, "k"), cap.need(i, "v")
        em.emit(stage="attention", layer=i, op="query_projection", name="q", tensor=q, formula="Q = X · Wq (+ bq)", inputs=("ln1_out",))
        em.emit(stage="attention", layer=i, op="key_projection", name="k", tensor=k, formula="K = X · Wk (+ bk)", inputs=("ln1_out",))
        em.emit(stage="attention", layer=i, op="value_projection", name="v", tensor=v, formula="V = X · Wv (+ bv)", inputs=("ln1_out",))
        qh, kh, vh = derive.split_heads(q, h), derive.split_heads(k, hk), derive.split_heads(v, hk)
        cos, sin = cap.need(None, "rope_cos"), cap.need(None, "rope_sin")
        q_rot, k_rot = derive.apply_rope(qh, cos, sin), derive.apply_rope(kh, cos, sin)
        em.emit(stage="attention", layer=i, op="rope_query", name="q_used", tensor=q_rot, provenance="derived",
                formula="Q_rot = Q ⊙ cos + rotate_half(Q) ⊙ sin", inputs=("q", "rope_cos", "rope_sin"),
                note=f"Heads split to [1, {h}, T, {dh}] then rotated. Correctness is established by the attn_merged check.")
        em.emit(stage="attention", layer=i, op="rope_key", name="k_rot", tensor=k_rot, provenance="derived",
                formula="K_rot = K ⊙ cos + rotate_half(K) ⊙ sin", inputs=("k", "rope_cos", "rope_sin"))
        k_all, v_all = self.emit_cache_kv(em, ctx, i, k_rot, vh, "k_rot", "v")
        if self.n_rep > 1:
            note = f"Grouped-query attention: {hk} key/value heads are each shared by {self.n_rep} query heads."
            k_used, v_used = derive.repeat_kv(k_all, self.n_rep), derive.repeat_kv(v_all, self.n_rep)
            em.emit(stage="attention", layer=i, op="repeat_kv_keys", name="k_used", tensor=k_used, provenance="derived",
                    formula=f"repeat each K head ×{self.n_rep}", inputs=("k_cache",), note=note)
            em.emit(stage="attention", layer=i, op="repeat_kv_values", name="v_used", tensor=v_used, provenance="derived",
                    formula=f"repeat each V head ×{self.n_rep}", inputs=("v_cache",), note=note)
        else:
            k_used, v_used = k_all, v_all
            em.emit(stage="attention", layer=i, op="keys_for_attention", name="k_used", tensor=k_used, raw_key="k_cache",
                    formula="K = K_cache", inputs=("k_cache",), note="Read from the model's KV cache.")
            em.emit(stage="attention", layer=i, op="values_for_attention", name="v_used", tensor=v_used, raw_key="v_cache",
                    formula="V = V_cache", inputs=("v_cache",), note="Read from the model's KV cache.")
        return q_rot, k_used, v_used

    def emit_mlp(self, cap, em, ctx, i):
        act = self.activation_name
        gate, gact, up = cap.need(i, "mlp_gate"), cap.need(i, "mlp_gate_act"), cap.need(i, "mlp_up")
        hidden = cap.need(i, "mlp_hidden")
        em.emit(stage="mlp", layer=i, op="mlp_gate_projection", name="mlp_gate", tensor=gate,
                formula="gate = X · W_gate", inputs=("ln2_out",))
        em.emit(stage="mlp", layer=i, op="mlp_activation", name="mlp_gate_act", tensor=gact,
                formula=f"{act}(gate)", inputs=("mlp_gate",))
        em.emit(stage="mlp", layer=i, op="mlp_up_projection", name="mlp_up", tensor=up,
                formula="up = X · W_up", inputs=("ln2_out",))
        em.emit(stage="mlp", layer=i, op="mlp_gating", name="mlp_hidden", tensor=hidden,
                formula=f"hidden = {act}(gate) ⊙ up", inputs=("mlp_gate_act", "mlp_up"),
                check=derive.compare(gact * up, hidden), against=f"{act}(gate) ⊙ up")
        em.emit(stage="mlp", layer=i, op="mlp_down_projection", name="mlp_out", tensor=cap.need(i, "mlp_out"),
                formula="mlp_out = hidden · W_down", inputs=("mlp_hidden",))

    # --- parameters
    def layer_weights(self, i: int) -> dict[str, torch.Tensor]:
        b = self.blocks[i]
        a, m = b.self_attn, b.mlp
        out = {"Wq": lin_w(a.q_proj), "Wk": lin_w(a.k_proj), "Wv": lin_w(a.v_proj), "Wo": lin_w(a.o_proj),
               "W_gate": lin_w(m.gate_proj), "W_up": lin_w(m.up_proj), "W_down": lin_w(m.down_proj),
               "ln1_weight": b.input_layernorm.weight, "ln2_weight": b.post_attention_layernorm.weight}
        for name, mod in (("bq", a.q_proj), ("bk", a.k_proj), ("bv", a.v_proj), ("bo", a.o_proj)):
            if mod.bias is not None:
                out[name] = mod.bias
        return out

    def global_weights(self) -> dict[str, torch.Tensor]:
        return {"W_E": self.embed_tokens.weight, "final_norm_weight": self.final_norm.weight,
                "W_U": self.lm_head.weight, "rope_inv_freq": self.core.rotary_emb.inv_freq}

    # --- architecture
    def embedding_nodes(self) -> list[dict]:
        d = self.hidden_size
        return [
            {"id": "tok_emb", "kind": "embedding", "label": "Token Embedding", "shape": f"[1, T, {d}]",
             "tensors": ["tok_emb", "emb_out"], "stage": "embedding", "formula": "W_E[token_ids]",
             "module": self.module_names.get(id(self.embed_tokens)), "params": {"weight": list(self.embed_tokens.weight.shape)}},
            {"id": "pos_emb", "kind": "embedding", "label": "Rotary Position (RoPE)", "shape": f"[1, T, {self.head_dim}]",
             "tensors": ["rope_cos", "rope_sin"], "stage": "embedding",
             "formula": "cos/sin(position · θ) — applied to Q and K inside every attention layer, not added to the embedding",
             "module": self.module_names.get(id(self.core.rotary_emb)), "params": {"inv_freq": list(self.core.rotary_emb.inv_freq.shape)}},
        ]

    def layer_template(self) -> list[dict]:
        d, h, hk, dh, m = self.hidden_size, self.n_heads, self.n_kv_heads, self.head_dim, self.intermediate_size
        nt, act = type(self.final_norm).__name__, self.activation_name
        nf = "(x / √(mean(x²) + ε)) · weight"
        gqa = f", {hk} K/V heads shared ×{self.n_rep}" if self.n_rep > 1 else ""
        return [
            {"key": "ln1", "label": f"{nt} 1", "stage": "norm", "module": "input_layernorm", "in": ["resid_pre"],
             "out": ["ln1_out"], "detail": ["ln1_meansq", "ln1_rms", "ln1_normalized"], "shape": f"[1, T, {d}]", "formula": nf},
            {"key": "qkv", "label": "Q / K / V projection", "stage": "attention", "module": "self_attn.{q,k,v}_proj",
             "in": ["ln1_out"], "out": ["q", "k", "v"], "detail": [], "shape": f"Q [1, T, {h * dh}], K/V [1, T, {hk * dh}]",
             "formula": "Q = XWq,  K = XWk,  V = XWv" + gqa},
            {"key": "rope", "label": "Rotary position (RoPE)", "stage": "attention", "module": None, "in": ["q", "k"],
             "out": ["q_used", "k_rot"], "detail": ["k_cache", "v_cache", "k_used", "v_used"], "shape": f"[1, {h}, T, {dh}]",
             "formula": "x ⊙ cos + rotate_half(x) ⊙ sin"},
            {"key": "scores", "label": "Attention scores", "stage": "attention", "module": None, "in": ["q_used", "k_used"],
             "out": ["scores"], "detail": ["scores_raw"], "shape": f"[1, {h}, T, T]", "formula": f"Q·Kᵀ / √{dh}"},
            {"key": "mask", "label": "Causal mask", "stage": "attention", "module": None, "in": ["scores"],
             "out": ["scores_masked"], "detail": ["mask"], "shape": f"[1, {h}, T, T]", "formula": "future positions += finfo.min"},
            {"key": "softmax", "label": "Softmax", "stage": "attention", "module": None, "in": ["scores_masked"],
             "out": ["weights"], "detail": [], "shape": f"[1, {h}, T, T]", "formula": "softmax over each row (float32)"},
            {"key": "attn_v", "label": "Weights × V", "stage": "attention", "module": None, "in": ["weights", "v_used"],
             "out": ["head_out"], "detail": ["attn_merged"], "shape": f"[1, {h}, T, {dh}]", "formula": "weights · V, heads concatenated"},
            {"key": "attn_proj", "label": "Attention output projection", "stage": "attention", "module": "self_attn.o_proj",
             "in": ["attn_merged"], "out": ["attn_out"], "detail": [], "shape": f"[1, T, {d}]", "formula": "merged · Wo"},
            {"key": "res1", "label": "Residual add", "stage": "residual", "module": None, "in": ["resid_pre", "attn_out"],
             "out": ["resid_mid"], "detail": [], "shape": f"[1, T, {d}]", "formula": "resid_pre + attn_out"},
            {"key": "ln2", "label": f"{nt} 2", "stage": "norm", "module": "post_attention_layernorm", "in": ["resid_mid"],
             "out": ["ln2_out"], "detail": ["ln2_meansq", "ln2_rms", "ln2_normalized"], "shape": f"[1, T, {d}]", "formula": nf},
            {"key": "mlp_gate", "label": f"Gate: {act}(X·W_gate)", "stage": "mlp", "module": "mlp.gate_proj", "in": ["ln2_out"],
             "out": ["mlp_gate_act"], "detail": ["mlp_gate"], "shape": f"[1, T, {m}]", "formula": f"{act}(X · W_gate)"},
            {"key": "mlp_up", "label": "Up projection", "stage": "mlp", "module": "mlp.up_proj", "in": ["ln2_out"],
             "out": ["mlp_up"], "detail": [], "shape": f"[1, T, {m}]", "formula": "X · W_up"},
            {"key": "mlp_mul", "label": "Gating product", "stage": "mlp", "module": None, "in": ["mlp_gate_act", "mlp_up"],
             "out": ["mlp_hidden"], "detail": [], "shape": f"[1, T, {m}]", "formula": f"{act}(gate) ⊙ up"},
            {"key": "mlp_proj", "label": "Down projection", "stage": "mlp", "module": "mlp.down_proj", "in": ["mlp_hidden"],
             "out": ["mlp_out"], "detail": [], "shape": f"[1, T, {d}]", "formula": "hidden · W_down"},
            {"key": "res2", "label": "Residual add", "stage": "residual", "module": None, "in": ["resid_mid", "mlp_out"],
             "out": ["resid_post"], "detail": [], "shape": f"[1, T, {d}]", "formula": "resid_mid + mlp_out"},
        ]
