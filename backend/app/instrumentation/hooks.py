"""Forward-hook instrumentation.

`Capture` holds the raw tensors a single forward pass produced. `HookManager`
registers (and always removes) hooks on the real model modules; nothing in the
model is replaced or re-implemented here.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

import torch
from torch import nn

Key = tuple[Optional[int], str]


class Capture:
    def __init__(self) -> None:
        self.raw: dict[Key, torch.Tensor] = {}
        self.time_ms: dict[Key, float] = {}
        self.module_path: dict[Key, str] = {}
        self.order: list[Key] = []

    def put(self, layer: Optional[int], key: str, value: Any) -> None:
        if value is None:
            return
        self.raw[(layer, key)] = value.detach()
        self.order.append((layer, key))

    def get(self, layer: Optional[int], key: str) -> Optional[torch.Tensor]:
        return self.raw.get((layer, key))

    def need(self, layer: Optional[int], key: str) -> torch.Tensor:
        try:
            return self.raw[(layer, key)]
        except KeyError:
            raise RuntimeError(
                f"instrumentation did not capture '{key}' (layer {layer}); the model structure "
                "differs from what this adapter expects"
            ) from None


def _first(x: Any) -> Any:
    return x[0] if isinstance(x, (tuple, list)) else x


class HookManager:
    def __init__(self, cap: Capture, names: dict[int, str]) -> None:
        self.cap = cap
        self.names = names
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def output(
        self,
        module: nn.Module,
        layer: Optional[int],
        key: str,
        pick: Callable[[Any], Any] = _first,
    ) -> None:
        """Capture a module's output and how long the module took."""
        k = (layer, key)
        self.cap.module_path[k] = self.names.get(id(module), type(module).__name__)
        t0: list[float] = []

        def pre(_m, _args, _kwargs=None):
            t0.append(time.perf_counter())

        def post(_m, _args, out):
            if t0:
                self.cap.time_ms[k] = (time.perf_counter() - t0.pop()) * 1e3
            self.cap.put(layer, key, pick(out))

        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def input(self, module: nn.Module, layer: Optional[int], key: str, arg: int = 0) -> None:
        """Capture a module's positional input."""
        k = (layer, key)
        self.cap.module_path[k] = self.names.get(id(module), type(module).__name__) + " (input)"

        def pre(_m, args, kwargs):
            if len(args) > arg:
                self.cap.put(layer, key, args[arg])
            elif kwargs:
                self.cap.put(layer, key, next(iter(kwargs.values())))

        self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
