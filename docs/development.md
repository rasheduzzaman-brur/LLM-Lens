# Development

## Layout
```
backend/app/{api,core,adapters,instrumentation,tracing,inference,services}   backend/tests
frontend/{app,components,hooks,services,types,lib}
models/        model directories (git-ignored)
```
`backend/app/schemas` and `backend/app/models` are reserved packages; request and response models currently
live next to the routes that use them.

## Running
```bash
cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000
cd frontend && npm run dev
```
The frontend reads `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

## Adding an architecture
1. Subclass `BaseModelAdapter` (or `RopeDecoderAdapter` for Llama-like decoders).
2. Implement `validate`, the structure accessors, `register_global_hooks`, `register_layer_hooks`,
   `emit_embeddings`, `emit_qkv` (return Q, K, V exactly as they enter the matmul), `emit_mlp`,
   `layer_weights`, `global_weights`, `layer_template`, `embedding_nodes`, `capabilities`.
3. Set `mask_mode` and `softmax_dtype` to mirror the model's attention function.
4. Register it in `adapters/registry.py` and add a tiny config to `tests/conftest.py`. The parametrised tests
   then check every tensor against PyTorch. A failed `attn_merged` check means the derivation does not
   match the model.

## Roadmap status
| phase | status |
|---|---|
| 1 Foundation | done: model loading, GPT-2, tokenization explorer |
| 2 Transformer internals | done: architecture diagram, Q/K/V, attention, softmax, residuals, normalization, MLP |
| 3 Generation | done: logits, sampling, step-by-step execution, timeline, SSE |
| 4 Advanced inspection | done: KV cache, hidden-state comparison, head/layer comparison, tensor inspector, memory monitoring |
| 5 Model expansion | done: Qwen2/2.5 and Llama-compatible adapters |

## Known limitations
- Batch size is 1.
- CUDA code paths (device placement, float16, GPU memory reporting) are written but were developed on a
  CPU-only machine and have not been executed on a GPU.
- Stepping through operations replays a recorded trace; a forward pass is not suspended mid-computation.
- Sliding-window attention, cross-attention and GPT-2 `reorder_and_upcast_attn` are rejected at load.
- Trace retention and settings are in memory and reset when the backend restarts.
- The spec lists Three.js as optional 3-D tensor rendering; it is not implemented.
- Docker images are provided and the compose file is syntax-checked, but the images were not built here.
- `transformers` is pinned to 4.57.6; other versions may move modules the adapters rely on, in which case
  the adapter's `validate` or a verification check fails visibly.
