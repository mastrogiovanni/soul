import asyncio
import contextlib
from dataclasses import replace

import pytest
from conftest import eventually, make_model

from soul.catalog import inspect_model
from soul.controller.engine import Engine
from soul.controller.registry import Peer
from soul.controller.store import Store
from soul.node.runtime import Runtime

ADMIN = {"Authorization": "Bearer admin"}
API = {"Authorization": "Bearer api"}


async def test_delete_model_frees_loaded_shards_and_preserves_other_models(cluster, tmp_path):
    engine, http, model, _, clients, _ = cluster
    second = str(tmp_path / "retained")
    make_model(tmp_path / "retained", layers=2)
    assert (
        await http.post("/admin/models/preload", headers=ADMIN, json={"model": second})
    ).status_code == 200
    assert (await http.post("/admin/models/load", headers=ADMIN, json={"model": model})).status_code == 200
    key = engine.specs[model].key
    for headers in ({}, API):
        assert (
            await http.post("/admin/models/delete", headers=headers, json={"model": model})
        ).status_code == 401
    assert (
        await http.post("/admin/models/delete", headers=ADMIN, json={"model": model, "revision": "main"})
    ).status_code == 400
    response = await http.post("/admin/models/delete", headers=ADMIN, json={"model": model})
    assert response.status_code == 200, response.text
    assert response.json()["pending_nodes"] == []
    assert response.json()["freed_bytes"] > 0
    assert (
        model not in engine.models.records and model not in engine.specs and model not in engine.deployments
    )
    assert all(r["model"] != model for r in engine.store.load("models"))
    for client in clients:
        assert not (client.runtime.cache_dir / "weights" / key).exists()
        assert client.runtime.model is None and not client.runtime.sessions
        assert (client.runtime.cache_dir / "identity.json").exists()
    assert engine.models.executable(second)
    assert (
        await http.post("/v1/completions", headers=API, json={"model": model, "prompt": "Hello"})
    ).status_code == 404
    # Repeated deletion is harmless, and explicit re-addition is allowed after cleanup.
    assert (await http.post("/admin/models/delete", headers=ADMIN, json={"model": model})).status_code == 200
    assert (await http.post("/admin/models/preload", headers=ADMIN, json={"model": model})).json()[
        "executable"
    ]
    assert model not in engine.cache.deleted


async def test_delete_offline_model_cleans_on_reconnect_without_resurrection(cluster):
    engine, http, model, _, clients, tasks = cluster
    key = engine.specs[model].key
    tasks[1].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks[1]
    await eventually(lambda: not engine.nodes["node1"].connected)
    response = await http.post("/admin/models/delete", headers=ADMIN, json={"model": model})
    assert response.json()["pending_nodes"] == ["node1"]
    assert (clients[1].runtime.cache_dir / "weights" / key).exists()
    assert (await http.get("/admin/state", headers=ADMIN)).json()["models"] == []
    assert (await http.post("/admin/models/preload", headers=ADMIN, json={"model": model})).status_code == 409
    tasks[1] = asyncio.create_task(clients[1].run())
    await eventually(lambda: not engine.cache.deleted[model]["pending_nodes"])
    assert not (clients[1].runtime.cache_dir / "weights" / key).exists()
    assert model not in engine.models.records
    assert (await http.get("/v1/models", headers=API)).json()["data"] == []


async def test_clear_node_pauses_refills_and_resume_rebuilds_pipeline(cluster, tmp_path):
    engine, http, model, _, clients, _ = cluster
    second = str(tmp_path / "second")
    make_model(tmp_path / "second", layers=2)
    await http.post("/admin/models/preload", headers=ADMIN, json={"model": second})
    await http.post("/admin/models/load", headers=ADMIN, json={"model": model})
    runtime = clients[0].runtime
    identities = (runtime.cache_dir / "identity.json").read_bytes()
    for headers in ({}, API):
        assert (await http.delete("/admin/nodes/node0/cache", headers=headers)).status_code == 401
    response = await http.delete("/admin/nodes/node0/cache", headers=ADMIN)
    assert response.status_code == 200 and not response.json()["pending"]
    assert response.json()["cache_paused"] and response.json()["freed_bytes"] > 0
    assert runtime.cached == {} and runtime.model is None
    assert not (runtime.cache_dir / "weights").exists()
    assert (runtime.cache_dir / "identity.json").read_bytes() == identities
    assert (runtime.cache_dir / "node.log").exists()
    assert clients[1].runtime.cached
    assert set(engine.models.records) == {model, second}
    assert not engine.models.executable(model)
    await http.post("/admin/models/preload", headers=ADMIN, json={"model": model})
    assert not (runtime.cache_dir / "weights").exists()
    state = (await http.get("/admin/state", headers=ADMIN)).json()
    node = next(n for n in state["nodes"] if n["node_id"] == "node0")
    assert node["cache_paused"] and not node["cache_clear_pending"] and node["cached_models"] == []
    assert (await http.post("/admin/nodes/node0/cache/resume", headers=ADMIN)).status_code == 200
    await eventually(lambda: bool(engine.models.executable(model)))
    assert runtime.cached and not engine.cache.paused("node0")


async def test_offline_clear_waits_for_reconnection_and_stays_paused(cluster):
    engine, http, _, _, clients, tasks = cluster
    tasks[1].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks[1]
    await eventually(lambda: not engine.nodes["node1"].connected)
    response = await http.delete("/admin/nodes/node1/cache", headers=ADMIN)
    assert response.json()["pending"]
    assert (await http.post("/admin/nodes/node1/cache/resume", headers=ADMIN)).status_code == 409
    tasks[1] = asyncio.create_task(clients[1].run())
    await eventually(lambda: not engine.cache.nodes["node1"]["pending"])
    assert engine.cache.paused("node1") and not clients[1].runtime.cached


async def test_legacy_nodes_report_pending_deletion_until_upgraded(cluster):
    engine, http, model, _, _, _ = cluster
    engine.nodes["node0"].capabilities.pop("cache_management")
    response = await http.post("/admin/models/delete", headers=ADMIN, json={"model": model})
    assert response.status_code == 200
    assert response.json()["pending_nodes"] == ["node0"]
    assert "Update and restart" in response.json()["errors"]["node0"]
    engine.nodes["node0"].capabilities["cache_management"] = True
    assert (await http.post("/admin/cache/retry", headers=API)).status_code == 401
    assert (await http.post("/admin/cache/retry", headers=ADMIN)).json()["pending_models"] == []


async def test_delete_removes_unadvertised_cache_with_damaged_manifest(cluster):
    engine, http, model, _, clients, _ = cluster
    runtime = clients[0].runtime
    cached = runtime.cache_dir / "weights" / engine.specs[model].key
    (cached / "manifest.json").write_text("invalid manifest")
    runtime.scan_cache()
    engine.nodes["node0"].update_inventory(runtime.inventory())
    assert not runtime.cached and cached.exists()
    response = await http.post("/admin/models/delete", headers=ADMIN, json={"model": model})
    assert response.json()["pending_nodes"] == []
    assert not cached.exists()


async def test_repeated_deletion_does_not_queue_already_cleaned_offline_nodes(cluster):
    engine, http, model, _, _, tasks = cluster
    await http.post("/admin/models/delete", headers=ADMIN, json={"model": model})
    tasks[1].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks[1]
    await eventually(lambda: not engine.nodes["node1"].connected)
    response = await http.post("/admin/models/delete", headers=ADMIN, json={"model": model})
    assert response.json()["pending_nodes"] == []


async def test_failed_clear_keeps_node_paused_until_retry(cluster, monkeypatch):
    engine, http, _, _, clients, _ = cluster
    original = clients[0].runtime.delete_cache

    def fail(*args):
        raise OSError("Cache directory is not writable")

    monkeypatch.setattr(clients[0].runtime, "delete_cache", fail)
    response = await http.delete("/admin/nodes/node0/cache", headers=ADMIN)
    assert response.json()["pending"] and "not writable" in response.json()["error"]
    assert engine.cache.paused("node0")
    monkeypatch.setattr(clients[0].runtime, "delete_cache", original)
    await http.post("/admin/cache/retry", headers=ADMIN)
    assert not engine.cache.nodes["node0"]["pending"] and not clients[0].runtime.cached


@pytest.mark.parametrize("operation", ["delete_model", "clear_node"])
async def test_cleanup_intentions_survive_controller_restart(tiny_model, tmp_path, operation):
    path, _ = tiny_model
    spec = inspect_model(str(path), allow_local=True, max_context=64)
    runtime = Runtime(tmp_path / "node", "cpu", memory_gb=1)
    runtime.preload(spec, 0, spec.layers)
    peer = Peer(
        None, {"node_id": "node", "capabilities": runtime.capabilities(), "inventory": runtime.inventory()}
    )
    peer.connected = False
    db = tmp_path / "controller.db"
    store = Store(db)
    engine = Engine(store, allow_local=True, max_context=64)
    engine.nodes[peer.id] = peer
    engine.models.remember(spec)
    store.save("nodes", peer.id, peer.snapshot())
    await getattr(engine.cache, operation)(spec.model_id if operation == "delete_model" else peer.id)
    store.close()
    store = Store(db)
    try:
        restored = Engine(store, allow_local=True, max_context=64)
        peer.connected = True
        restored.nodes[peer.id] = peer
        restored.models.adopt_inventory(peer)

        async def call(method, args=None, timeout=None):
            freed = runtime.delete_cache(args["model"] if method == "delete_model" else None)
            peer.update_inventory(runtime.inventory())
            return {"freed_bytes": freed}

        peer.call = call
        async with restored.gate:
            await restored.cache.reconcile()
        assert runtime.cached == {}
        if operation == "delete_model":
            assert spec.model_id not in restored.models.records
            assert restored.cache.deleted[spec.model_id]["pending_nodes"] == []
        else:
            assert spec.model_id in restored.models.records
            assert restored.cache.paused(peer.id) and not restored.cache.nodes[peer.id]["pending"]
    finally:
        store.close()


def test_local_cache_removal_is_scoped_includes_all_revisions_and_handles_partial_files(tiny_model, tmp_path):
    path, _ = tiny_model
    spec = inspect_model(str(path), allow_local=True, max_context=64)
    runtime = Runtime(tmp_path / "node", "cpu", memory_gb=1)
    runtime.preload(spec, 0, 2)
    for invalid_keys in (["../.."], [str(path)], [None], "0" * 24):
        with pytest.raises(ValueError, match="Invalid cache keys"):
            runtime.delete_cache(spec.model_id, invalid_keys)
        assert (runtime.cache_dir / "weights" / spec.key).exists()
    old = replace(spec, revision="old-revision")
    runtime.preload(old, 2, 4)
    (runtime.cache_dir / "weights" / old.key / "unfinished.partial").write_bytes(b"partial")
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep").write_text("untouched")
    (runtime.cache_dir / "weights" / old.key / "linked").symlink_to(external, target_is_directory=True)
    assert runtime.delete_cache(spec.model_id) > 0
    assert not (runtime.cache_dir / "weights" / spec.key).exists()
    assert not (runtime.cache_dir / "weights" / old.key).exists()
    assert (external / "keep").read_text() == "untouched"
    assert (path / "model.safetensors").exists()
    corrupt = runtime.cache_dir / "weights" / "corrupt"
    corrupt.mkdir()
    (corrupt / "bad.partial").write_text("bad")
    assert runtime.delete_cache() == 3
    assert not (runtime.cache_dir / "weights").exists()
    assert runtime.delete_cache() == 0


async def test_cleanup_waits_for_inference_gate_and_rejects_unknown_targets(cluster):
    engine, http, model, _, clients, _ = cluster
    assert (await http.delete("/admin/nodes/unknown/cache", headers=ADMIN)).status_code == 404
    assert (
        await http.post("/admin/models/delete", headers=ADMIN, json={"model": "unknown"})
    ).status_code == 404
    await engine.gate.acquire()
    task = asyncio.create_task(engine.cache.delete_model(model))
    await asyncio.sleep(0.05)
    assert not task.done() and clients[0].runtime.cached
    engine.gate.release()
    await task
    assert not clients[0].runtime.cached
