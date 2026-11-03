"""Run a generation as a recorded experiment and stream its real events.

Event order for one experiment:
  generation.started
  [model.loading, model.loaded]          (only if the checkpoint was not in memory)
  generation.level_started / level_completed ...   (from TraceCollector hooks)
  generation.completed | generation.failed
Every event carries experiment_id, seq (0-based) and t (seconds since the
experiment started running).
"""
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.schemas.generation import GenerateRequest, GenerateResponse
from app.services.event_hub import EventHub
from app.services.experiment_store import ExperimentStore

log = logging.getLogger("ncm.experiments")


def compute_metrics(resp: GenerateResponse) -> dict[str, Any]:
    per_level = []
    if resp.trace is not None:
        per_level = [{"level": lv.level, "n_codes": lv.n_codes, "source": lv.source, "seconds": lv.seconds,
                      "unique": lv.stats.unique, "entropy_bits": lv.stats.entropy_bits} for lv in resp.trace.levels]
    return {
        "model": resp.model, "task": resp.task, "device": resp.device, "num_samples": len(resp.outputs),
        "timing": resp.timing.model_dump(), "transformer_passes": resp.transformer_passes, "levels": resp.levels,
        "per_level": per_level, "output_words": [len(o.split()) for o in resp.outputs],
        "output_chars": [len(o) for o in resp.outputs], "sampling": resp.sampling.model_dump(),
        "prompt_tokens": resp.prompt_tokens, "prompt_truncated": resp.prompt_truncated,
        "reference": resp.reference,
        "output_tokens": resp.output_tokens, "tokenizer": resp.tokenizer,
        "throughput_tokens_per_s": sum(resp.output_tokens) / resp.timing.generate_s if resp.timing.generate_s > 0 else None,
        "throughput_words_per_s": sum(len(o.split()) for o in resp.outputs) / resp.timing.total_s if resp.timing.total_s > 0 else None,
        "output_entropy_own": resp.output_entropy_own,
        "memory": resp.memory.model_dump() if resp.memory else None,
        "quality": None,
    }


def with_request_facts(metrics: dict[str, Any], req: GenerateRequest) -> dict[str, Any]:
    return {**metrics, "warmup": req.warmup, "tag": req.tag}


class ExperimentService:
    def __init__(self, runner: Any, store: ExperimentStore, hub: EventHub):
        self.runner, self.store, self.hub = runner, store, hub
        # one worker: inference is serialised on the device anyway; extra
        # requests wait with status "queued"
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ncm-gen")

    def create(self, req: GenerateRequest) -> str:
        eid = uuid.uuid4().hex
        self.store.create(eid, model=req.model, request=req.model_dump())
        self.hub.open(eid)
        self.executor.submit(self._run, eid, req)
        return eid

    def _run(self, eid: str, req: GenerateRequest) -> None:
        t0 = time.perf_counter()

        def emit(ev: dict[str, Any]) -> None:
            self.hub.publish(eid, {**ev, "experiment_id": eid, "t": round(time.perf_counter() - t0, 6)})

        self.store.update(eid, status="running")
        emit({"type": "generation.started", "model": req.model, "num_samples": req.num_samples, "seed": req.seed})
        try:
            resp: GenerateResponse = self.runner.generate(req, on_event=emit)
            emit({"type": "generation.completed", "outputs": resp.outputs, "timing": resp.timing.model_dump(),
                  "transformer_passes": resp.transformer_passes})
            self.store.update(
                eid, status="completed", outputs=resp.outputs, timing=resp.timing.model_dump(),
                trace=resp.trace.model_dump() if resp.trace is not None else None,
                metrics=with_request_facts(compute_metrics(resp), req), events=self.hub.history(eid))
            log.info("experiment %s completed model=%s total_s=%.3f", eid, req.model, resp.timing.total_s)
        except Exception as e:
            log.exception("experiment %s failed", eid)
            emit({"type": "generation.failed", "error": str(e), "error_type": type(e).__name__})
            self.store.update(eid, status="failed", error=f"{type(e).__name__}: {e}", events=self.hub.history(eid))
        finally:
            self.hub.close(eid)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
