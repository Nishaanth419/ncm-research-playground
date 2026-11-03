import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[3] / "results" / "playground.sqlite"


@dataclass(frozen=True)
class Settings:
    # CPU by default: MPS fails in research-ncm/models/quant.py (non-divisible
    # adaptive pooling), see docs/ASSESSMENT.md.
    device: str = field(default_factory=lambda: os.environ.get("NCM_DEVICE", "cpu"))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        os.environ.get("NCM_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")))
    log_level: str = field(default_factory=lambda: os.environ.get("NCM_LOG_LEVEL", "INFO"))
    db_path: str = field(default_factory=lambda: os.environ.get("NCM_DB_PATH", str(DEFAULT_DB)))
