# Computation trace

## Event schema (`app/tracing/schema.py`)

```json
{
  "event_id": 37, "step": 0, "phase": "prefill", "order": 37,
  "stage": "attention", "layer": 1, "operation": "softmax", "tensor_name": "weights",
  "shape": [1, 12, 11, 11], "dtype": "float32",
  "provenance": "derived", "verified": true, "max_abs_err": 0.0,
  "verified_against": "attention weights returned by the model's attention module",
  "formula": "softmax(scores_masked, dim=-1)", "inputs": ["scores_masked"],
  "module_path": null, "execution_time_ms": null,
  "stats": {"min": 0.0, "max": 1.0, "mean": 0.09, "std": 0.2, "l2": 7.2},
  "stored": true, "nbytes": 5808, "note": null
}
```

`event_id` is unique in a run; `order` is the execution order inside the step. Events are addressed either
by id or by `(step, layer, tensor_name)`.

## Provenance

| value | meaning |
|---|---|
| `hooked` | Input or output of a real module, captured by a forward hook. `execution_time_ms` is the module's time. |
| `derived` | Recomputed from hooked tensors with the model's parameters (`tracing/derive.py`). Carries `verified` / `max_abs_err` when a model-produced tensor exists to compare with. |
| `viz` | Visualization transform (PCA, cosine similarity, pooling, means over heads). Reported by analysis endpoints, never stored as trace events. |
| `unavailable` | The operation exists but cannot be decomposed; `note` explains why. |

## Verification

Tolerances depend on dtype (`derive.tolerance`): float32 `rtol 1e-4, atol 1e-5`; float16/bfloat16
`rtol 3e-2, atol 3e-3`; float64 `rtol 1e-9`.

| check | compared with |
|---|---|
| `weights` | attention weights returned by the model's attention module |
| `attn_merged` | re-derived `QKᵀ → scale → mask → softmax → ·V` vs the hooked input of the output projection. This is the end-to-end check for the whole attention derivation, including RoPE and grouped-query repetition. |
| `k_cache`, `v_cache` | tail of the model's cache vs K/V computed in this step |
| `ln1_out`, `ln2_out`, `final_norm_out` | step-by-step formula vs the module output |
| `resid_mid`, `resid_post` | `resid_pre + attn_out`, `resid_mid + mlp_out` |
| `mlp_hidden` (gated MLP) | `act(gate) ⊙ up` vs the hooked input of the down projection |
| `emb_out` (GPT-2) | `tok_emb + pos_emb` |
| `logits` | last position recomputed as `h · W_U` |

## Tensor names

Global (`layer = null`): `tok_emb`, `pos_emb` (GPT-2), `rope_cos`, `rope_sin` (RoPE models), `emb_out`,
`final_norm_in`, `final_norm_mean|var|normalized` or `final_norm_meansq|rms|normalized`, `final_norm_out`,
`logits`, `next_token_logits`, `next_token_probs`, `sampling_probs`.

Per layer, in execution order:

| stage | names |
|---|---|
| residual | `resid_pre` |
| norm | `ln1_mean`, `ln1_var` (LayerNorm) or `ln1_meansq`, `ln1_rms` (RMSNorm), `ln1_normalized`, `ln1_out` |
| attention | `qkv` (GPT-2 fused output), `q`, `k`, `v`, `q_used`, `k_rot` (RoPE), `k_cache`, `v_cache`, `k_used`, `v_used`, `scores_raw`, `scores`, `mask`, `scores_masked`, `weights`, `head_out`, `attn_merged`, `attn_out` |
| residual | `resid_mid` |
| norm | `ln2_*` |
| mlp | GPT-2: `mlp_pre`, `mlp_act`, `mlp_out`. Gated: `mlp_gate`, `mlp_gate_act`, `mlp_up`, `mlp_hidden`, `mlp_out` |
| residual | `resid_post` |

`q_used`, `k_used`, `v_used` are exactly the tensors that enter `Q·Kᵀ` and `weights·V`: after the head
split, after RoPE, read from the KV cache, after grouped-query repetition.

In a decode step, per-position tensors have sequence length 1 and attention tensors are `[1, H, 1, T]`.
Analysis endpoints stitch rows from different steps by absolute position; that is exact because a
position's activations never change once computed.

## Precision

Values are computed in the model's dtype. float16/bfloat16 tensors are upcast to float32 for transfer
(lossless) and the response says so. Softmax row breakdowns evaluate `exp(score − max)` for stability and
state it. Masked scores hold `finfo.min` as in the model; non-finite values serialize as `null`.
