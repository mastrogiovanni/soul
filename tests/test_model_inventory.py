import asyncio
import contextlib

import pytest
from conftest import eventually, make_model
from test_quantization import quantize_fixture

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
    assert response.status_code == 200
    assert response.json()["warning"]
    assert response.json()["distribution"]["cached_layers"] == 2
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


@pytest.mark.parametrize("quantization", ["none", "awq", "gptq"])
async def test_partial_download_expands_with_new_nodes_without_redownloading(
    cluster, tmp_path, monkeypatch, quantization
):
    engine, http, _, url, clients, tasks = cluster
    path = tmp_path / "growing-model"
    if quantization == "none":
        make_model(path)
    else:
        quantize_fixture(path, method=quantization)
    model = str(path)
    engine.max_layers_per_node = 1
    downloaded = []
    ensure = WeightStore.ensure

    def track(store, names, progress):
        if store.spec.model_id == model:
            downloaded.append((store.root, set(names) - set(store.available())))
        return ensure(store, names, progress)

    monkeypatch.setattr(WeightStore, "ensure", track)
    response = await http.post("/admin/models/preload", headers=ADMIN, json={"model": model})
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["status"] == "partial" and not view["executable"]
    assert view["error"] is None and "automatically" in view["warning"]
    coverage = view["distribution"]
    assert coverage["cached_layers"] == 2 and coverage["total_layers"] == 4
    assert 0 < coverage["cached_bytes"] < coverage["total_bytes"]
    assert coverage["missing_bytes"] == coverage["total_bytes"] - coverage["cached_bytes"]
    assert coverage["nodes"] == 2
    assert len(downloaded) == 2
    first_names = set.union(*(names for _, names in downloaded))
    for client in clients:
        assert client.runtime.model is None
        assert client.runtime.cached[engine.specs[model].key]["tensors"]

    # Neither a manual retry nor an inference/load attempt redownloads cached stages.
    assert (await http.post("/admin/models/preload", headers=ADMIN, json={"model": model})).status_code == 200
    assert (await http.post("/admin/models/load", headers=ADMIN, json={"model": model})).status_code == 503
    response = await http.post(
        "/v1/completions", headers=API, json={"model": model, "prompt": "Hello", "max_tokens": 2}
    )
    assert response.status_code == 503 and response.json()["error"]["code"] == "model_not_ready"
    assert model not in [m["id"] for m in (await http.get("/v1/models", headers=API)).json()["data"]]
    assert len(downloaded) == 2

    def current():
        return next(v for v in engine.models.views() if v["model"] == model)

    for index in (2, 3):
        client = NodeClient(
            url, "node", tmp_path / f"node{index}", node_id=f"node{index}", device="cpu", memory_gb=1
        )
        clients.append(client)
        tasks.append(asyncio.create_task(client.run()))
        await eventually(lambda expected=index + 1: current()["distribution"]["cached_layers"] == expected)
        assert len(downloaded) == index + 1
        assert client.runtime.model is None
    await eventually(lambda: current()["executable"])
    view = current()
    assert view["status"] == "available" and view["warning"] is None and view["error"] is None
    assert view["distribution"]["missing_bytes"] == 0
    assert view["distribution"]["tokenizer_cached"]
    assert all(not (names & first_names) for _, names in downloaded[2:])
    response = await http.post(
        "/v1/completions", headers=API, json={"model": model, "prompt": "Hello", "max_tokens": 2}
    )
    assert response.status_code == 200, response.text
    assert len(downloaded) == 4

    # Offline cache is not counted as connected coverage; reconnect reuses it.
    tasks[-1].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks[-1]
    await eventually(lambda: not engine.nodes["node3"].connected)
    assert current()["distribution"]["cached_layers"] == 3
    assert current()["warning"] and not current()["executable"]
    tasks[-1] = asyncio.create_task(clients[-1].run())
    await eventually(lambda: current()["executable"])
    assert current()["warning"] is None
    assert len(downloaded) == 4


async def test_zero_capacity_preload_is_durable_and_does_not_download(tiny_model, tmp_path, monkeypatch):
    path, _ = tiny_model
    database = tmp_path / "controller.db"
    store = Store(database)
    engine = Engine(store, allow_local=True)

    def no_download(*args, **kwargs):
        raise AssertionError("No node is available")

    monkeypatch.setattr(WeightStore, "ensure", no_download)
    view = await engine.models.preload(str(path))
    assert view["status"] == "partial" and view["warning"] and not view["executable"]
    assert view["assignments"] == [] and view["error"] is None
    assert view["distribution"]["cached_bytes"] == 0
    store.close()
    store = Store(database)
    try:
        restored = Engine(store, allow_local=True)
        assert restored.specs[str(path)].revision == view["revision"]
        assert restored.models.views()[0]["warning"]
        view = await restored.models.preload(str(path))
        assert view["status"] == "partial"
    finally:
        store.close()


async def test_failed_distribution_keeps_completed_tensors_and_can_retry(cluster, tmp_path, monkeypatch):
    engine, http, _, _, clients, _ = cluster
    path = tmp_path / "interrupted-model"
    make_model(path)
    ensure = WeightStore.ensure
    failed = False

    def interrupt(store, names, progress):
        nonlocal failed
        if (
            store.spec.model_id == str(path)
            and store.root.is_relative_to(clients[0].runtime.cache_dir)
            and not failed
        ):
            failed = True
            ensure(store, names[:1], progress)
            raise SoulError("Download interrupted for test")
        return ensure(store, names, progress)

    monkeypatch.setattr(WeightStore, "ensure", interrupt)
    response = await http.post("/admin/models/preload", headers=ADMIN, json={"model": str(path)})
    assert response.status_code == 503
    view = next(v for v in engine.models.views() if v["model"] == str(path))
    assert "Download interrupted" in view["error"]
    assert "retry" in view["warning"] and not view["executable"]
    assert view["distribution"]["cached_bytes"] > 0
    retained = WeightStore(clients[0].runtime.cache_dir, engine.specs[str(path)])
    first = retained.available()[0]
    modified = retained.path(first).stat().st_mtime_ns
    response = await http.post("/admin/models/preload", headers=ADMIN, json={"model": str(path)})
    assert response.status_code == 200 and response.json()["executable"]
    assert response.json()["warning"] is None and response.json()["error"] is None
    assert retained.path(first).stat().st_mtime_ns == modified


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
