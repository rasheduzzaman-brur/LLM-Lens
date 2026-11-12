# API

Interactive documentation with all schemas: `http://localhost:8000/docs`.

## Models
| method | path | |
|---|---|---|
| GET | `/api/models` | loaded models, local model directories, supported architectures |
| POST | `/api/models/load` | `{source, device, dtype, allow_download}`. Returns `409 {code: "download_required", size_bytes}` when weights are not local and `allow_download` is false |
| DELETE | `/api/models/{model_id}` | unload |
| GET | `/api/models/{model_id}/architecture` | nodes in execution order + per-layer operation template |
| GET | `/api/models/{model_id}/weights?layer=` | parameter listing (`y = x·W + b` convention) |
| GET | `/api/models/{model_id}/weights/tensor?name=&layer=&index=` | parameter slice |
| GET | `/api/models/{model_id}/embedding?token_id=` | embedding row + nearest tokens |
| POST | `/api/models/{model_id}/embeddings/pca` | 2-D PCA of the embedding rows of `token_ids` |
| GET | `/api/models/{model_id}/activation` | the model's activation module evaluated on a grid |
| POST | `/api/tokenize` | tokens, ids, offsets, vocabulary information |

## Inference
| method | path | |
|---|---|---|
| POST | `/api/inference/run` | `{model_id, prompt, capture, sampling, max_new_tokens}` → run summary after the traced prefill |
| GET | `/api/inference/{run_id}` | run summary |
| POST | `/api/inference/{run_id}/step` | `{kind: next_token \| next_op \| prev_op \| goto, sampling?}` |
| POST | `/api/inference/{run_id}/resume` | start / continue automatic generation `{delay_ms, sampling?}` |
| POST | `/api/inference/{run_id}/pause` | |
| POST | `/api/inference/{run_id}/reset` | discard generated tokens, re-run prefill |
| GET | `/api/inference/{run_id}/events` | SSE stream: `run` event with the summary on every change |
| GET | `/api/inference/{run_id}/trace?step=&layer=&stage=&offset=&limit=` | trace events (metadata) |
| GET | `/api/inference/{run_id}/trace/{event_id}?index=` | one event and a slice of its tensor |
| GET | `/api/inference/{run_id}/trace-export?step=` | events as CSV |
| GET | `/api/inference/{run_id}/tensor?name=&step=&layer=&index=&assembled=` | tensor by name |
| GET | `/api/inference/{run_id}/layers?step=` | per-layer summary |
| GET | `/api/inference/{run_id}/layers/{layer}?position=&step=` | every per-token vector of one layer |
| GET | `/api/inference/{run_id}/global?position=&step=` | embedding and final-norm vectors |
| GET | `/api/inference/{run_id}/attention?layer=&step=&head=` | attention weights, all heads or one |
| GET | `/api/inference/{run_id}/attention/grid?step=&reduce=mean\|max\|head` | one matrix per layer |
| GET | `/api/inference/{run_id}/attention/row?layer=&head=&position=` | scores → exp → denominator → weights |
| GET | `/api/inference/{run_id}/hidden-states?stage=&step=&position=&depth=` | norms, PCA, cosine similarities |
| GET | `/api/inference/{run_id}/logits?step=&top_n=&temperature=&top_k=&top_p=` | next-token distribution; sampling parameters give a what-if view |
| GET | `/api/inference/{run_id}/kv-cache?step=` | cache shapes, bytes, growth history |
| GET | `/api/inference/{run_id}/kv-cache/tensor?layer=&which=&head=` | cached keys or values |
| GET | `/api/inference/{run_id}/kv-cache/compare?step=` | runs the step with and without the cache; timing and logit difference |

`step=-1` (default) means the latest step. `index` is numpy-style: `0,3,:,0:64`; integers drop a dimension.
Oversized 2-D slices are mean-pooled and flagged (`pooled: true`); other oversized slices return `413`.
Tensors released by retention or not captured return `409` with the reason.

## System
`GET /api/system` (RAM, CUDA devices, trace-store usage), `GET|PATCH /api/settings`, `GET /api/health`.
Environment variables use the `LLMLENS_` prefix, for example `LLMLENS_MODELS_DIR`, `LLMLENS_MAX_STORE_BYTES`.
