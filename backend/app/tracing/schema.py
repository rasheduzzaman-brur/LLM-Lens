"""Trace schema: the contract between the instrumentation engine and the UI."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Provenance = Literal["hooked", "derived", "viz", "unavailable"]
Phase = Literal["prefill", "decode"]
CaptureMode = Literal["metadata", "selected", "full"]


class TensorStats(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    l2: Optional[float] = None


class TraceEvent(BaseModel):
    """One computation in the forward pass.

    provenance:
      hooked      captured directly from a module input/output by a forward hook
      derived     recomputed from hooked tensors with the model's own parameters;
                  `verified` says whether it matched what the model produced
      viz         visualization-only transform (never part of the model computation)
      unavailable the operation exists but cannot be inspected (see `note`)
    """

    event_id: int
    step: int = Field(description="0 = prefill, n>0 = n-th decoded token")
    phase: Phase
    order: int = Field(description="Execution order inside the step")
    stage: str
    layer: Optional[int] = None
    operation: str
    tensor_name: str
    shape: list[int]
    dtype: str
    provenance: Provenance = "hooked"
    verified: Optional[bool] = None
    max_abs_err: Optional[float] = None
    verified_against: Optional[str] = None
    formula: Optional[str] = None
    inputs: list[str] = []
    module_path: Optional[str] = None
    execution_time_ms: Optional[float] = None
    stats: Optional[TensorStats] = None
    stored: bool = False
    nbytes: int = 0
    note: Optional[str] = None
    values: list = Field(default_factory=list, description="Empty in listings; use the tensor endpoints")


class CaptureConfig(BaseModel):
    mode: CaptureMode = "full"
    tensors: list[str] = Field(default_factory=list, description="tensor_name filter for mode=selected")
    layers: list[int] = Field(default_factory=list, description="layer filter for mode=selected (empty = all)")

    def wants(self, layer: Optional[int], name: str) -> bool:
        if self.mode == "metadata":
            return False
        if self.mode == "full":
            return True
        if self.tensors and name not in self.tensors:
            return False
        if self.layers and layer is not None and layer not in self.layers:
            return False
        return True
