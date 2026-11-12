"""Model adapter interface.

An adapter knows where the real modules of one architecture family live, which
hooks to register, and how to turn the captured tensors into the standardized
trace events the UI understands. The shared event pipeline lives here; subclasses
only supply what genuinely differs between architectures.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from app.instrumentation.hooks import Capture, HookManager
from app.tracing import derive
from app.tracing.engine import MAX_FULL_LOGITS_BYTES, Emitter, StepContext, cache_kv


def lin_w(linear: nn.Module) -> torch.Tensor:
    """Weight in the 'y = x @ W + b' convention ([in, out]) for nn.Linear and HF Conv1D."""
    w = linear.weight
    return w.T if isinstance(linear, nn.Linear) else w


class BaseModelAdapter(ABC):
    family: str = "base"
    model_types: tuple[str, ...] = ()
    mask_mode: str = "fill"
    softmax_dtype: Optional[torch.dtype] = None

    def __init__(self, model: nn.Module, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = model.config
        self.module_names: dict[int, str] = {id(m): n for n, m in model.named_modules()}
        self.validate()

    # ------------------------------------------------------------------ structure
    @property
    @abstractmethod
    def blocks(self) -> nn.ModuleList: ...
    @property
    @abstractmethod
    def embed_tokens(self) -> nn.Embedding: ...
    @property
    @abstractmethod
    def final_norm(self) -> nn.Module: ...
    @property
    @abstractmethod
    def lm_head(self) -> nn.Module: ...
    @abstractmethod
    def validate(self) -> None:
        """Raise if the loaded model does not have the structure this adapter expects."""
    @abstractmethod
    def norm_modules(self, layer: int) -> tuple[nn.Module, nn.Module]: ...
    @abstractmethod
    def activation_module(self) -> nn.Module: ...
    @abstractmethod
    def register_layer_hooks(self, hm: HookManager, i: int) -> None: ...
    @abstractmethod
    def register_global_hooks(self, hm: HookManager) -> None: ...
    @abstractmethod
    def emit_embeddings(self, cap: Capture, em: Emitter, ctx: StepContext) -> None: ...
    @abstractmethod
    def emit_qkv(self, cap: Capture, em: Emitter, ctx: StepContext, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Emit projection events; return (q, k, v) exactly as they enter the attention matmul."""
    @abstractmethod
    def emit_mlp(self, cap: Capture, em: Emitter, ctx: StepContext, i: int) -> None: ...
    @abstractmethod
    def layer_weights(self, i: int) -> dict[str, torch.Tensor]: ...
    @abstractmethod
    def global_weights(self) -> dict[str, torch.Tensor]: ...
    @abstractmethod
    def layer_template(self) -> list[dict]: ...
    @abstractmethod
    def embedding_nodes(self) -> list[dict]: ...

    # ------------------------------------------------------------------ dimensions
    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def hidden_size(self) -> int:
        return int(self.cfg.hidden_size)

    @property
    def n_heads(self) -> int:
        return int(self.cfg.num_attention_heads)

    @property
    def n_kv_heads(self) -> int:
        return int(getattr(self.cfg, "num_key_value_heads", None) or self.n_heads)

    @property
    def head_dim(self) -> int:
        return int(getattr(self.cfg, "head_dim", None) or self.hidden_size // self.n_heads)

    @property
    def vocab_size(self) -> int:
        return int(self.embed_tokens.weight.shape[0])

    @property
    @abstractmethod
    def intermediate_size(self) -> int: ...

    def attention_scale(self, layer: int) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def norm_type(self) -> str:
        return "layernorm" if isinstance(self.final_norm, nn.LayerNorm) else "rmsnorm"

    @property
    def activation_name(self) -> str:
        return str(getattr(self.cfg, "activation_function", None) or getattr(self.cfg, "hidden_act", "unknown"))

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    @abstractmethod
    def capabilities(self) -> dict: ...

    # ------------------------------------------------------------------ hooks
    def register_hooks(self, hm: HookManager) -> None:
        self.register_global_hooks(hm)
        for i in range(self.n_layers):
            self.register_layer_hooks(hm, i)

    # ------------------------------------------------------------------ events
    def emit_step(self, cap: Capture, em: Emitter, ctx: StepContext) -> None:
        self.emit_embeddings(cap, em, ctx)
        for i in range(self.n_layers):
            self.emit_layer(cap, em, ctx, i)
        self.emit_head(cap, em, ctx)

    def emit_norm(self, em: Emitter, layer: Optional[int], prefix: str, op: str, module: nn.Module,
                  x: torch.Tensor, out: torch.Tensor, x_name: str, stage: str = "norm") -> None:
        """Step-by-step normalization with the module's real parameters, verified against its output."""
        eps = getattr(module, "eps", None)
        if eps is None:
            eps = getattr(module, "variance_epsilon", None)
        if isinstance(module, nn.LayerNorm):
            s = derive.layernorm_steps(x, module.weight, module.bias, module.eps)
            em.emit(stage=stage, layer=layer, op=f"{op}.mean", name=f"{prefix}_mean", tensor=s["mean"],
                    provenance="derived", formula="mean = mean(x)", inputs=(x_name,))
            em.emit(stage=stage, layer=layer, op=f"{op}.variance", name=f"{prefix}_var", tensor=s["var"],
                    provenance="derived", formula="variance = mean((x - mean)^2)", inputs=(x_name,))
            em.emit(stage=stage, layer=layer, op=f"{op}.normalize", name=f"{prefix}_normalized", tensor=s["normalized"],
                    provenance="derived", formula=f"normalized = (x - mean) / sqrt(variance + {eps:g})", inputs=(x_name,))
            formula = "output = normalized * gamma + beta"
        elif eps is not None and hasattr(module, "weight") and getattr(module, "bias", None) is None:
            s = derive.rmsnorm_steps(x, module.weight, float(eps))
            em.emit(stage=stage, layer=layer, op=f"{op}.mean_square", name=f"{prefix}_meansq", tensor=s["mean_sq"],
                    provenance="derived", formula="mean_sq = mean(x^2)", inputs=(x_name,))
            em.emit(stage=stage, layer=layer, op=f"{op}.rms", name=f"{prefix}_rms", tensor=s["rms"],
                    provenance="derived", formula=f"rms = sqrt(mean(x^2) + {float(eps):g})", inputs=(x_name,))
            em.emit(stage=stage, layer=layer, op=f"{op}.normalize", name=f"{prefix}_normalized", tensor=s["normalized"],
                    provenance="derived", formula="normalized = x / rms", inputs=(x_name,))
            formula = "output = (x / rms) * weight"
        else:
            em.emit(stage=stage, layer=layer, op=f"{op}.internals", name=f"{prefix}_normalized", tensor=None,
                    note=f"Normalization module {type(module).__name__} is not a LayerNorm/RMSNorm this adapter can decompose.")
            em.emit(stage=stage, layer=layer, op=op, name=f"{prefix}_out", tensor=out, inputs=(x_name,))
            return
        em.emit(stage=stage, layer=layer, op=op, name=f"{prefix}_out", tensor=out, formula=formula,
                inputs=(x_name,), check=derive.compare(s["out"], out),
                against="step-by-step formula recomputed with the module's parameters")

    def emit_layer(self, cap: Capture, em: Emitter, ctx: StepContext, i: int) -> None:
        n1, n2 = self.norm_modules(i)
        resid_pre = cap.need(i, "resid_pre")
        em.emit(stage="residual", layer=i, op="block_input", name="resid_pre", tensor=resid_pre,
                formula="x  (residual stream entering the block)")
        self.emit_norm(em, i, "ln1", "norm_1", n1, resid_pre, cap.need(i, "ln1_out"), "resid_pre")

        q, k, v = self.emit_qkv(cap, em, ctx, i)
        scale = self.attention_scale(i)
        a = derive.attention_core(q, k, v, scale, self.mask_mode, self.softmax_dtype)  # type: ignore[arg-type]
        em.emit(stage="attention", layer=i, op="qk_matmul", name="scores_raw", tensor=a["scores_raw"],
                provenance="derived", formula="Q · Kᵀ", inputs=("q_used", "k_used"))
        em.emit(stage="attention", layer=i, op="scale", name="scores", tensor=a["scores"], provenance="derived",
                formula=f"Q · Kᵀ / √d_k   (d_k = {self.head_dim}, factor = {scale:.6g})", inputs=("scores_raw",))
        em.emit(stage="attention", layer=i, op="causal_mask", name="mask", tensor=a["allowed"], provenance="derived",
                formula="allowed[i, j] = (j ≤ i)", note="True = position may be attended to.")
        em.emit(stage="attention", layer=i, op="apply_mask", name="scores_masked", tensor=a["scores_masked"],
                provenance="derived", inputs=("scores", "mask"),
                formula=("where(allowed, scores, finfo.min)" if self.mask_mode == "fill" else "scores + (0 | finfo.min)"),
                note="Masked entries hold the most negative finite value of the dtype, as in the model.")
        model_w = cap.get(i, "attn_weights_model")
        em.emit(stage="attention", layer=i, op="softmax", name="weights", tensor=a["weights"], provenance="derived",
                formula="softmax(scores_masked, dim=-1)", inputs=("scores_masked",),
                check=derive.compare(a["weights"], model_w) if model_w is not None else None,
                against="attention weights returned by the model's attention module",
                note=None if model_w is not None else "The model's attention module returned no weights to compare against.")
        em.emit(stage="attention", layer=i, op="weighted_values", name="head_out", tensor=a["head_out"],
                provenance="derived", formula="weights · V   (per head)", inputs=("weights", "v_used"))
        merged_model = cap.need(i, "attn_merged")
        em.emit(stage="attention", layer=i, op="merge_heads", name="attn_merged", tensor=merged_model,
                formula="concat(head_0 … head_H-1)", inputs=("head_out",),
                check=derive.compare(a["merged"], merged_model),
                against="re-derived attention (QKᵀ → scale → mask → softmax → ·V) vs the tensor the model fed to its output projection")
        em.emit(stage="attention", layer=i, op="output_projection", name="attn_out", tensor=cap.need(i, "attn_out"),
                formula="attn_merged · Wo + bo", inputs=("attn_merged",))

        resid_mid = cap.need(i, "resid_mid")
        em.emit(stage="residual", layer=i, op="residual_add_attention", name="resid_mid", tensor=resid_mid,
                formula="resid_mid = resid_pre + attn_out", inputs=("resid_pre", "attn_out"),
                check=derive.compare(resid_pre + cap.need(i, "attn_out"), resid_mid), against="resid_pre + attn_out")
        self.emit_norm(em, i, "ln2", "norm_2", n2, resid_mid, cap.need(i, "ln2_out"), "resid_mid")
        self.emit_mlp(cap, em, ctx, i)
        resid_post = cap.need(i, "resid_post")
        em.emit(stage="residual", layer=i, op="residual_add_mlp", name="resid_post", tensor=resid_post,
                formula="resid_post = resid_mid + mlp_out", inputs=("resid_mid", "mlp_out"),
                check=derive.compare(resid_mid + cap.need(i, "mlp_out"), resid_post), against="resid_mid + mlp_out")

    def emit_cache_kv(self, em: Emitter, ctx: StepContext, i: int, k_new: torch.Tensor, v_new: torch.Tensor,
                      k_name: str, v_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Read this layer's K/V from the model's cache (past + new) and check the new tail."""
        k_all, v_all = cache_kv(ctx.cache, i)
        k_all, v_all = k_all[..., : ctx.total_len, :], v_all[..., : ctx.total_len, :]
        reuse = f"{ctx.past_len} cached position(s) reused, {ctx.new_len} appended." if ctx.past_len else \
            "Prefill: the cache is filled with keys/values for every prompt position."
        em.emit(stage="attention", layer=i, op="kv_cache_keys", name="k_cache", tensor=k_all, inputs=(k_name,),
                formula="K_cache = concat(K_past, K_new)", check=derive.compare(k_all[..., ctx.past_len :, :], k_new),
                against=f"tail of the cache vs {k_name} computed this step", note="Read from the model's KV cache. " + reuse)
        em.emit(stage="attention", layer=i, op="kv_cache_values", name="v_cache", tensor=v_all, inputs=(v_name,),
                formula="V_cache = concat(V_past, V_new)", check=derive.compare(v_all[..., ctx.past_len :, :], v_new),
                against=f"tail of the cache vs {v_name} computed this step", note="Read from the model's KV cache. " + reuse)
        return k_all, v_all

    def emit_head(self, cap: Capture, em: Emitter, ctx: StepContext) -> None:
        x = cap.need(None, "final_norm_in")
        out = cap.need(None, "final_norm_out")
        em.emit(stage="final_norm", layer=None, op="final_input", name="final_norm_in", tensor=x,
                formula="residual stream after the last block")
        self.emit_norm(em, None, "final_norm", "final_norm", self.final_norm, x, out, "final_norm_in", stage="final_norm")
        logits = cap.need(None, "logits")
        w = lin_w(self.lm_head)
        last = out[0, -1] @ w
        if getattr(self.lm_head, "bias", None) is not None:
            last = last + self.lm_head.bias
        big = logits.numel() * logits.element_size() > MAX_FULL_LOGITS_BYTES
        em.emit(stage="lm_head", layer=None, op="lm_head", name="logits", tensor=logits,
                formula="logits = final_norm_out · W_Uᵀ", inputs=("final_norm_out",),
                check=derive.compare(last, logits[0, -1]), against="last position recomputed as h · W_U",
                never_store=big,
                note="Too large to retain for every position; the last position is kept as next_token_logits." if big else None)
        em.emit(stage="lm_head", layer=None, op="select_last_position", name="next_token_logits", tensor=logits[0, -1],
                formula="logits[0, -1, :]", inputs=("logits",), force_store=True,
                note="Slice of the hooked logits. Always retained (needed for token selection).")

    # ------------------------------------------------------------------ architecture description
    def info(self) -> dict:
        n_params = sum(p.numel() for p in self.model.parameters())
        return {
            "family": self.family,
            "model_type": self.cfg.model_type,
            "model_class": type(self.model).__name__,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_positions": int(getattr(self.cfg, "max_position_embeddings", 0) or getattr(self.cfg, "n_positions", 0)),
            "parameter_count": n_params,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
            "attn_implementation": getattr(self.cfg, "_attn_implementation", None),
            "capabilities": self.capabilities(),
        }

    def architecture(self) -> dict:
        d, v = self.hidden_size, self.vocab_size
        fn = self.final_norm
        nodes = [
            {"id": "input", "kind": "io", "label": "Input tokens", "shape": "[1, T]", "tensors": [],
             "formula": "token ids from the tokenizer", "module": None, "params": {}},
            *self.embedding_nodes(),
            *[
                {"id": f"layer{i}", "kind": "layer", "layer": i, "label": f"Transformer Layer {i}",
                 "shape": f"[1, T, {d}]", "tensors": ["resid_pre", "resid_post"],
                 "module": self.module_names.get(id(self.blocks[i])), "params": {}}
                for i in range(self.n_layers)
            ],
            {"id": "final_norm", "kind": "norm", "label": f"Final {type(fn).__name__}", "shape": f"[1, T, {d}]",
             "tensors": ["final_norm_in", "final_norm_out"], "stage": "final_norm",
             "formula": "output = normalized * gamma + beta" if self.norm_type == "layernorm" else "output = (x / rms) * weight",
             "module": self.module_names.get(id(fn)), "params": {n: list(p.shape) for n, p in fn.named_parameters()}},
            {"id": "lm_head", "kind": "head", "label": "LM Head", "shape": f"[1, T, {v}]", "tensors": ["logits"],
             "stage": "lm_head", "formula": "logits = h · W_Uᵀ", "module": self.module_names.get(id(self.lm_head)),
             "params": {n: list(p.shape) for n, p in self.lm_head.named_parameters()},
             "note": "Weight tied to the token embedding matrix." if self.lm_head.weight is self.embed_tokens.weight else None},
            {"id": "logits", "kind": "op", "label": "Logits → Softmax", "shape": f"[{v}]",
             "tensors": ["next_token_logits", "next_token_probs"], "stage": "sampling",
             "formula": "p = softmax(logits[-1] / temperature)", "module": None, "params": {}},
            {"id": "output", "kind": "io", "label": "Next token", "shape": "[1]", "tensors": ["sampling_probs"],
             "stage": "sampling", "formula": "greedy argmax or sample from p", "module": None, "params": {}},
        ]
        for order, n in enumerate(nodes):
            n["order"] = order
        return {"nodes": nodes, "layer_template": self.layer_template(), "info": self.info()}
