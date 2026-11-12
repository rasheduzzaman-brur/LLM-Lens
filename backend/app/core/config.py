"""Runtime configuration (env-overridable, also mutable through /api/settings)."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLMLENS_", extra="ignore")

    models_dir: Path = ROOT / "models"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # --- memory / retention -------------------------------------------------
    max_loaded_models: int = Field(1, ge=1, le=4)
    max_runs: int = Field(4, ge=1, le=32, description="Runs kept before the oldest is evicted")
    max_store_bytes: int = Field(1_500_000_000, ge=10_000_000, description="Byte budget for captured tensors")
    retain_steps: int = Field(12, ge=1, le=256, description="Decode steps per run that keep tensors (prefill is always kept)")
    max_prompt_tokens: int = Field(256, ge=1, le=4096)
    max_new_tokens: int = Field(64, ge=1, le=1024)

    # --- serialization caps -------------------------------------------------
    max_cells: int = Field(70_000, ge=1_000, description="Max numbers in one tensor response")
    max_pool_side: int = Field(192, ge=16, le=1024, description="Max side of a pooled 2-D heatmap")


settings = Settings()
