"""Tiny randomly initialised models built from configs: no downloads, fast, deterministic."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from app.adapters.registry import build_adapter
from app.services.model_manager import LoadedModel

torch.set_grad_enabled(False)
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class ToyTokenizer:
    """Minimal whitespace tokenizer for architecture tests (real tokenizers are tested in the `real` suite)."""
    bos_token_id = None
    eos_token_id = 0
    bos_token = None
    eos_token = "<eos>"
    pad_token = None
    is_fast = False

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: str, **_):
        return {"input_ids": [1 + (sum(map(ord, w)) % (self.vocab_size - 1)) for w in text.split()]}

    def decode(self, ids, **_):
        return " ".join(f"t{i}" for i in ids)


def _tiny(kind: str, dtype=torch.float32):
    torch.manual_seed(0)
    if kind == "gpt2":
        m = GPT2LMHeadModel(GPT2Config(vocab_size=101, n_positions=64, n_embd=32, n_layer=3, n_head=4,
                                       attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0, attn_implementation="eager"))
    elif kind == "llama":  # grouped-query attention: 4 query heads share 2 key/value heads
        m = LlamaForCausalLM(LlamaConfig(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                         attn_implementation="eager"))
    else:
        m = Qwen2ForCausalLM(Qwen2Config(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                         attn_implementation="eager"))
    # random gains/biases so normalization tests are not trivially passing with gamma=1, beta=0
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 1:
                p.add_(torch.randn_like(p) * 0.1)
    m = m.to(dtype).eval()
    return LoadedModel(model_id=f"tiny-{kind}", source="test", path="", adapter=build_adapter(m, ToyTokenizer(101)))


@pytest.fixture(scope="session", params=["gpt2", "llama", "qwen2"])
def lm(request):
    return _tiny(request.param)


@pytest.fixture(scope="session")
def gpt2():
    return _tiny("gpt2")


@pytest.fixture(scope="session")
def llama():
    return _tiny("llama")


@pytest.fixture(scope="session")
def tiny_factory():
    return _tiny


PROMPT = "the quick brown fox jumps over the lazy dog"
