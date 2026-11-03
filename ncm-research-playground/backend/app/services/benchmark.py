"""Reproducible latency/throughput benchmarks.

For every (model, batch size): `warmup_runs` untimed generations, then
`repeats` timed generations with seeds seed, seed+1, ... (no trace, so the
measurement contains no instrumentation-side decoding). Runs execute on the
same single worker as experiments, so nothing else uses the device meanwhile.
Timings come from the runners (device-synchronised; checkpoint load excluded).
"""
import json
import logging
import math
import os
import platform
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.research import RESEARCH_ROOT
from app.schemas.benchmark import BenchmarkRequest
from app.schemas.generation import GenerateRequest

log = logging.getLogger("ncm.bench")

NCM_CHECKPOINTS_REVISION = "d342ce8aaea58830eef3b8ad473664673e41f692"


def percentile(xs: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    r = (len(s) - 1) * q / 100
    lo = math.floor(r)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (r - lo)


def describe(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0, "median": None, "p90": None, "p95": None, "mean": None, "std": None, "min": None, "max": None}
    m = sum(xs) / len(xs)
    std = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return {"n": len(xs), "median": percentile(xs, 50), "p90": percentile(xs, 90), "p95": percentile(xs, 95),
            "mean": m, "std": std, "min": min(xs), "max": max(xs)}


RUN_FIELDS = ("model", "batch_size", "repeat", "seed", "generate_s", "decode_s", "total_s", "first_output_s",
              "output_tokens", "output_words", "passes", "memory_peak_bytes")


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault((r["model"], r["batch_size"]), []).append(r)
    out = []
    for (model, bs), rs in groups.items():
        out.append({
            "model": model, "batch_size": bs, "n": len(rs),
            "generate_s": describe([r["generate_s"] for r in rs]),
            "total_s": describe([r["total_s"] for r in rs]),
            "first_output_s": describe([r["first_output_s"] for r in rs if r["first_output_s"] is not None]),
            "per_sample_s": describe([r["total_s"] / bs for r in rs]),
            "words_per_s": describe([r["output_words"] / r["total_s"] for r in rs if r["total_s"] > 0]),
            "tokens_per_s": describe([r["output_tokens"] / r["generate_s"] for r in rs if r["generate_s"] > 0]),
            "passes": rs[-1]["passes"],
            "mean_output_tokens": sum(r["output_tokens"] for r in rs) / len(rs) / bs,
            "memory_peak_bytes": max((r["memory_peak_bytes"] for r in rs if r["memory_peak_bytes"] is not None),
                                     default=None),
        })
    return out


def hardware_info(device: str) -> dict[str, Any]:
    import torch
    import transformers
    cpu = platform.processor()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=2).stdout.strip() or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram = None
    try:
        commit = subprocess.run(["git", "-C", str(RESEARCH_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=2).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "platform": platform.platform(), "machine": platform.machine(), "cpu": cpu, "cpu_count": os.cpu_count(),
        "ram_bytes": ram, "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "torch_threads": torch.get_num_threads(), "device": device,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ncm_commit": commit, "ncm_checkpoints_revision": NCM_CHECKPOINTS_REVISION,
    }


SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, error TEXT,
    name TEXT, config TEXT NOT NULL, hardware TEXT, models TEXT, progress TEXT
);
CREATE TABLE IF NOT EXISTS benchmark_runs (
    benchmark_id TEXT NOT NULL REFERENCES benchmarks(id), model TEXT NOT NULL, batch_size INTEGER NOT NULL,
    repeat INTEGER NOT NULL, seed INTEGER NOT NULL, generate_s REAL NOT NULL, decode_s REAL NOT NULL,
    total_s REAL NOT NULL, first_output_s REAL, output_tokens INTEGER NOT NULL, output_words INTEGER NOT NULL,
    passes INTEGER NOT NULL, memory_peak_bytes INTEGER
);
CREATE INDEX IF NOT EXISTS benchmark_runs_bid ON benchmark_runs(benchmark_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class BenchmarkStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def create(self, bid: str, req: BenchmarkRequest) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO benchmarks (id, created_at, status, name, config) VALUES (?, ?, 'queued', ?, ?)",
                      (bid, _now(), req.name, req.model_dump_json()))

    def update(self, bid: str, **f: Any) -> None:
        if f.get("status") in ("completed", "failed"):
            f.setdefault("finished_at", _now())
        f = {k: json.dumps(v) if k in ("hardware", "models", "progress") else v for k, v in f.items()}
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE benchmarks SET {', '.join(f'{k} = ?' for k in f)} WHERE id = ?", (*f.values(), bid))

    def add_run(self, bid: str, run: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute(f"INSERT INTO benchmark_runs (benchmark_id, {', '.join(RUN_FIELDS)}) "
                      f"VALUES (?, {', '.join('?' * len(RUN_FIELDS))})", (bid, *(run[k] for k in RUN_FIELDS)))

    def runs(self, bid: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(f"SELECT {', '.join(RUN_FIELDS)} FROM benchmark_runs WHERE benchmark_id = ? "
                             "ORDER BY rowid", (bid,)).fetchall()
        return [dict(r) for r in rows]

    def _row(self, r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        for k in ("config", "hardware", "models", "progress"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        return d

    def get(self, bid: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM benchmarks WHERE id = ?", (bid,)).fetchone()
        if r is None:
            return None
        d = self._row(r)
        d["runs"] = self.runs(bid)
        d["summary"] = summarize(d["runs"])
        return d

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM benchmarks ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]


class BenchmarkService:
    def __init__(self, runner: Any, store: BenchmarkStore, executor: ThreadPoolExecutor, device: str):
        self.runner, self.store, self.executor, self.device = runner, store, executor, device

    def create(self, req: BenchmarkRequest) -> str:
        bid = uuid.uuid4().hex
        self.store.create(bid, req)
        self.executor.submit(self._run, bid, req)
        return bid

    def _run(self, bid: str, req: BenchmarkRequest) -> None:
        total = len(req.models) * len(req.batch_sizes) * req.repeats
        done = 0
        try:
            infos = {m.id: m.model_dump() for m in self.runner.list_models()}
            self.store.update(bid, status="running", hardware=hardware_info(self.device),
                              models={m: infos.get(m) for m in req.models},
                              progress={"done": 0, "total": total, "phase": "starting"})
            for model in req.models:
                for bs in req.batch_sizes:
                    base = dict(model=model, num_samples=bs, sampling=req.sampling, include_trace=False)
                    for w in range(req.warmup_runs):
                        self.store.update(bid, progress={"done": done, "total": total, "phase": "warmup",
                                                         "model": model, "batch_size": bs, "warmup": w + 1})
                        self.runner.generate(GenerateRequest(**base, seed=req.seed))
                    for i in range(req.repeats):
                        r = self.runner.generate(GenerateRequest(**base, seed=req.seed + i))
                        self.store.add_run(bid, {
                            "model": model, "batch_size": bs, "repeat": i, "seed": req.seed + i,
                            "generate_s": r.timing.generate_s, "decode_s": r.timing.decode_s,
                            "total_s": r.timing.total_s, "first_output_s": r.timing.first_output_s,
                            "output_tokens": sum(r.output_tokens), "output_words": sum(len(o.split()) for o in r.outputs),
                            "passes": r.transformer_passes,
                            "memory_peak_bytes": r.memory.peak_bytes if r.memory else None})
                        done += 1
                        self.store.update(bid, progress={"done": done, "total": total, "phase": "timed",
                                                         "model": model, "batch_size": bs})
            self.store.update(bid, status="completed", progress={"done": done, "total": total, "phase": "done"})
            log.info("benchmark %s completed runs=%d", bid, done)
        except Exception as e:
            log.exception("benchmark %s failed", bid)
            self.store.update(bid, status="failed", error=f"{type(e).__name__}: {e}",
                              progress={"done": done, "total": total, "phase": "failed"})


def runs_csv(runs: list[dict[str, Any]]) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(RUN_FIELDS))
    w.writeheader()
    w.writerows(runs)
    return buf.getvalue()


def summary_csv(summary: list[dict[str, Any]], hardware: Optional[dict[str, Any]]) -> str:
    import csv
    import io
    stats = ("generate_s", "total_s", "first_output_s", "per_sample_s", "words_per_s", "tokens_per_s")
    cols = ["model", "batch_size", "n", "passes", "mean_output_tokens", "memory_peak_bytes"] + [
        f"{s}_{k}" for s in stats for k in ("median", "p90", "p95", "mean", "std", "min", "max")] + ["device", "cpu"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for g in summary:
        row = {k: g[k] for k in cols[:6]}
        for s in stats:
            for k in ("median", "p90", "p95", "mean", "std", "min", "max"):
                row[f"{s}_{k}"] = g[s][k]
        row["device"] = (hardware or {}).get("device")
        row["cpu"] = (hardware or {}).get("cpu")
        w.writerow(row)
    return buf.getvalue()
