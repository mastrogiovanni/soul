import asyncio
import contextlib

import pytest
from conftest import eventually, make_model

from soul.catalog import inspect_model
from soul.controller.engine import Engine
from soul.controller.registry import Peer
from soul.controller.store import Store
from soul.node.client import NodeClient
from soul.node.runtime import Runtime
from soul.node.weights import WeightStore
from soul.protocol import SoulError

ADMIN = {"Authorization": "Bearer admin"}
API = {"Authorization": "Bearer api"}


async def test_preloaded_model_is_listed_without_gpu_load_or_download_on_inference(cluster, monkeypatch):
    engine, http, model, _, clients, _ = cluster
    assert all(c.runtime.model is None for c in clients)
    assert not engine.deployments
    assert (await http.get("/models")).status_code == 401
    assert (await http.get("/models", headers=ADMIN)).status_code == 401
    for endpoint in ("/models", "/v1/models"):
        response = await http.get(endpoint, headers=API)
        assert [m["id"] for m in response.json()["data"]] == [model]
    state = (await http.get("/admin/state", headers=ADMIN)).json()
    assert state["models"][0]["status"] == "available"
    assert sorted(n["cached_models"][0]["layers"] for n in state["nodes"]) == [[0, 1], [2, 3]]

    def no_download(*args, **kwargs):
        raise AssertionError("Inference attempted to download weights")

    monkeypatch.setattr(WeightStore, "ensure", no_download)
    response = await http.post(
        "/v1/completions", headers=API, json={"model": model, "prompt": "Hello", "max_tokens": 2}
    )
    assert response.status_code == 200, response.text


async def test_unknown_model_requires_preload(cluster):
    engine, http, _, _, _, _ = cluster
    response = await http.post(
        "/v1/completions", headers=API, json={"model": "unknown/model", "prompt": "Hello", "max_tokens": 2}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_preloaded"
    assert "unknown/model" not in engine.models.records


async def test_multiple_preloaded_models_stay_listed_after_rotation_and_unload(cluster, tmp_path):
    _, http, model, _, clients, _ = cluster
    second = tmp_path / "second"
    make_model(second, layers=2)
    response = await http.post("/admin/models/preload", headers=ADMIN, json={"model": str(second)})
    assert response.status_code == 200, response.text
    assert all(c.runtime.model is None for c in clients)
    for current in (model, str(second)):
        response = await http.post("/admin/models/load", headers=ADMIN, json={"model": current})
        assert response.status_code == 200, response.text
    await http.post("/admin/models/unload", headers=ADMIN, json={"model": str(second)})
    listed = (await http.get("/v1/models", headers=API)).json()["data"]
    assert {m["id"] for m in listed} == {model, str(second)}


async def test_remove_only_offline_nodes_and_repair_missing_layers(cluster):
    engine, http, model, _, clients, tasks = cluster
    assert (await http.delete("/admin/nodes/node1")).status_code == 401
    assert (await http.delete("/admin/nodes/node1", headers=API)).status_code == 401
    assert (await http.delete("/admin/nodes/node1", headers=ADMIN)).status_code == 409
    tasks[1].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks[1]
    await eventually(lambda: not engine.nodes["node1"].connected)
    assert (await http.get("/v1/models", headers=API)).json()["data"] == []
    # Remaining node has ample memory once the deliberate two-layer test cap is removed.
    engine.max_layers_per_node = None
    response = await http.delete("/admin/nodes/node1", headers=ADMIN)
    assert response.status_code == 200
    await eventually(lambda: bool(engine.models.executable(model)))
    assert "node1" not in engine.nodes
    assert all(n["node_id"] != "node1" for n in engine.store.load("nodes"))
    assert len(clients[0].runtime.cached[engine.specs[model].key]["tensors"]) == len(
        engine.specs[model].tensors
    )
    assert clients[0].runtime.model is None
    assert (await http.delete("/admin/nodes/node1", headers=ADMIN)).status_code == 404
    assert len((await http.get("/v1/models", headers=API)).json()["data"]) == 1


async def test_reconnect_discovers_cache_before_first_call_without_redownload(cluster, monkeypatch):
    engine, http, model, url, clients, tasks = cluster
    old_generation = engine.nodes["node1"].generation
    tasks[1].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks[1]
    await eventually(lambda: not engine.nodes["node1"].connected)

    def no_download(*args, **kwargs):
        raise AssertionError("Valid cached weights should not be downloaded again")

    monkeypatch.setattr(WeightStore, "ensure", no_download)
    clients[1] = NodeClient(url, "node", clients[1].runtime.cache_dir, device="cpu", memory_gb=1)
    tasks[1] = asyncio.create_task(clients[1].run())
    await eventually(
        lambda: (
            engine.nodes["node1"].connected
            and engine.nodes["node1"].generation != old_generation
            and engine.models.executable(model)
        )
    )
    assert clients[1].runtime.model is None
    assert engine.nodes["node1"].snapshot()["cached_models"][0]["layers"] == [2, 3]
    assert len((await http.get("/v1/models", headers=API)).json()["data"]) == 1


async def test_preload_intention_survives_missing_capacity_and_repairs_on_join(cluster):
    engine, http, model, url, clients, tasks = cluster
    tasks[1].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks[1]
    await eventually(lambda: not engine.nodes["node1"].connected)
    response = await http.post("/admin/models/preload", headers=ADMIN, json={"model": model})
    assert response.status_code == 503
    assert engine.models.records[model]["status"] == "partial"
    replacement = NodeClient(
        url,
        "node",
        clients[1].runtime.cache_dir.parent / "replacement",
        node_id="replacement",
        device="cpu",
        memory_gb=1,
    )
    clients[1] = replacement
    tasks[1] = asyncio.create_task(replacement.run())
    await eventually(lambda: bool(engine.models.executable(model)))
    assert replacement.runtime.cached[engine.specs[model].key]["tensors"]
    assert replacement.runtime.model is None


def test_cache_inventory_and_catalog_survive_process_restart(tiny_model, tmp_path):
    directory, _ = tiny_model
    spec = inspect_model(str(directory), allow_local=True, max_context=64)
    root = tmp_path / "node"
    runtime = Runtime(root, "cpu", memory_gb=1)
    runtime.preload(spec, 0, spec.layers)
    runtime = Runtime(root, "cpu", memory_gb=1)
    inventory = runtime.scan_cache()
    store = Store(tmp_path / "controller.db")
    engine = Engine(store, allow_local=True, max_context=64)
    peer = Peer(None, {"node_id": "node", "capabilities": runtime.capabilities(), "inventory": inventory})
    engine.nodes[peer.id] = peer
    engine.models.adopt_inventory(peer)
    assert engine.models.executable(spec.model_id)
    store.close()
    store = Store(tmp_path / "controller.db")
    restored = Engine(store, allow_local=True, max_context=64)
    assert spec.model_id in restored.models.records
    assert not restored.models.executable(spec.model_id)
    restored.nodes[peer.id] = peer
    assert restored.models.executable(spec.model_id)
    store.close()


def test_inventory_excludes_corrupt_and_partial_tensors(tiny_model, tmp_path):
    directory, _ = tiny_model
    spec = inspect_model(str(directory), allow_local=True)
    runtime = Runtime(tmp_path / "node", "cpu", memory_gb=1)
    runtime.preload(spec, 0, 2)
    store = WeightStore(runtime.cache_dir, spec)
    name = spec.names_for(0, 2)[0]
    store.path(name).write_bytes(b"corrupt")
    inventory = runtime.scan_cache()
    assert name not in inventory[0]["tensors"]
    assert inventory[0]["offer"]["missing_bytes"][0][1] > 0
    with pytest.raises(SoulError, match="missing or corrupt"):
        runtime.load(spec, 0, 2, "new", download=False)
