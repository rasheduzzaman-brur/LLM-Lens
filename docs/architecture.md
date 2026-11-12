# Architecture

```
Browser (Next.js, React, Tailwind, React Flow, D3 canvas heatmaps, Plotly)
   │  REST + Server-Sent Events
FastAPI  app/api/{models,inference,misc}.py
   │
   ├─ services/model_manager.py    load / unload, local-first resolution, download confirmation
   ├─ inference/session.py         Run: prefill + decode steps, sampling, pause/resume/reset, retention
   ├─ tracing/engine.py            trace_forward(): one instrumented forward pass → StepRecord
   │     ├─ instrumentation/hooks.py   Capture + HookManager (forward hooks, timing)
   │     ├─ adapters/*                 where the modules are, what to emit, architecture description
   │     └─ tracing/derive.py          attention / normalization / RoPE math mirrored from Hugging Face
   ├─ services/analysis.py         read-side views: attention, softmax row, layer detail, hidden states,
   │                               logits, KV cache, cached-vs-uncached comparison
   └─ services/tensor_view.py      slicing, pagination, pooling, JSON-safe serialization
```

## Execution model

A **run** is one prompt plus the tokens generated from it.

- **Step 0, prefill**: the whole prompt goes through the model in one pass with an empty `DynamicCache`.
- **Step n, decode**: only the token selected in step n−1 is fed in, with the cache from earlier steps.

Each step is a separate call of `trace_forward`, which registers hooks, calls the real
`model(input_ids, past_key_values=cache, use_cache=True)` under `torch.inference_mode()`, removes the hooks
and asks the adapter to emit ordered trace events. Token selection (greedy, temperature, top-k, top-p,
seeded sampling) is implemented in `inference/sampling.py` instead of `generate()` so every stage of the
distribution is a tensor that can be inspected. Greedy stepping is tested to equal `model.generate`.

Models are loaded with `attn_implementation="eager"`. Fused kernels (SDPA / flash attention) produce the
same output but never materialise scores or weights, so there would be nothing to verify against.

"Next operation / previous operation" in the UI moves a cursor over the recorded events of a completed
forward pass. A forward pass is not paused mid-computation; the UI says so.

## Adapters

```
BaseModelAdapter            shared event pipeline, verification, architecture description
 ├─ GPT2Adapter             fused Conv1D Q/K/V, learned positions, LayerNorm, Linear→GELU→Linear
 └─ RopeDecoderAdapter      q/k/v_proj, RoPE, grouped-query attention, RMSNorm, SwiGLU
     ├─ LlamaAdapter        model_type "llama"
     └─ QwenAdapter         model_type "qwen2"
```

The adapter is chosen from `config.model_type` (`adapters/registry.py`). Each adapter validates the module
structure at load and raises if it differs from what it expects, rather than guessing. `capabilities()`
tells the frontend which operations exist (norm type, MLP structure, RoPE, GQA, biases), so only applicable
operations are drawn.

## Memory

- Captured tensors stay on the model's device, detached, with no autograd graph. They are copied to the
  CPU only when a slice is serialized.
- Capture modes per run: `full`, `selected` (tensor-name and layer filters), `metadata` (no tensors; shapes,
  dtypes, statistics and verification results are still real).
- Retention: the prefill step and the most recent `retain_steps` decode steps keep tensors; older steps keep
  metadata plus their next-token logits. A global byte budget releases older runs first.
- `logits` for all positions is not stored above 48 MB; the last position always is.
- Responses never contain a whole large tensor: numpy-style `index` slicing, a `max_cells` cap, and
  mean-pooling of oversized 2-D slices (flagged as a visualization transform).

## Frontend

`app/` holds one page per sidebar section. Shared state (model, run, step, layer, head, token position,
inspector selection, sampling and capture settings) lives in a Zustand store; server data goes through
TanStack Query; run updates arrive over SSE. `components/tensors/heatmap.tsx` is the canvas heatmap used
everywhere; `components/architecture/flow.tsx` is the React Flow diagram; `components/inspector` is the
right-hand panel that shows any selected tensor, weight or trace event.
