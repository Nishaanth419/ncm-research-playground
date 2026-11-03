import asyncio
import threading

from app.services.event_hub import EventHub
from app.services.experiment_store import ExperimentStore


def test_store_round_trip(tmp_path):
    s = ExperimentStore(tmp_path / "x.sqlite")
    s.create("e1", model="ncm-tinystories", request={"model": "ncm-tinystories", "seed": 1})
    rec = s.get("e1")
    assert rec["status"] == "queued" and rec["request"]["seed"] == 1 and rec["outputs"] is None
    s.update("e1", status="completed", outputs=["a"], timing={"total_s": 1.0}, trace={"levels": []},
             metrics={"x": 1}, events=[{"type": "generation.completed"}])
    rec = s.get("e1")
    assert rec["status"] == "completed" and rec["outputs"] == ["a"] and rec["events"][0]["type"] == "generation.completed"
    assert s.get_field("e1", "trace") == {"levels": []}
    assert s.get("nope") is None


def test_store_list_newest_first_and_survives_reopen(tmp_path):
    p = tmp_path / "x.sqlite"
    s = ExperimentStore(p)
    s.create("a", model="m", request={})
    s.create("b", model="m", request={})
    assert [r["id"] for r in ExperimentStore(p).list(limit=10)] == ["b", "a"]


def test_hub_replays_then_streams_live_in_order():
    async def main():
        hub = EventHub()
        hub.open("e")
        hub.publish("e", {"type": "generation.started"})
        sub = hub.subscribe("e", asyncio.get_running_loop())
        t = threading.Thread(target=lambda: [hub.publish("e", {"type": "x", "i": i}) for i in range(3)]
                             + [hub.close("e")])
        t.start()
        got = [ev async for ev in sub]
        t.join()
        return got
    got = asyncio.run(main())
    assert [e["type"] for e in got] == ["generation.started", "x", "x", "x"]
    assert [e["seq"] for e in got] == [0, 1, 2, 3]


def test_hub_subscribe_after_close_gets_full_history():
    async def main():
        hub = EventHub()
        hub.open("e")
        hub.publish("e", {"type": "a"})
        hub.close("e")
        return [ev async for ev in hub.subscribe("e", asyncio.get_running_loop())]
    assert [e["type"] for e in asyncio.run(main())] == ["a"]


def test_hub_unknown_id():
    assert not EventHub().known("zzz")
