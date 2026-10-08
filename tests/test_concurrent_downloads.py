import asyncio
import contextlib
import threading

import pytest
from conftest import eventually, make_model

from soul.node.weights import WeightStore
from soul.protocol import SoulError

ADMIN = {"Authorization": "Bearer admin"}
API = {"Authorization": "Bearer api"}


def hold_downloads(monkeypatch, model, clients, *, fail=False):
    started = {c.runtime.cache_dir: threading.Event() for c in clients}
    release = threading.Event()
    original = WeightStore.ensure

    def held(store, names, progress):
        if store.spec.model_id == model:
            progress(0, sum(store.spec.tensors[n]["length"] for n in names), names[0])
            started[store.root.parent.parent].set()
            if not release.wait(15):
                raise AssertionError("Test did not release the download")
            if fail:
                raise SoulError("Injected download failure")
        return original(store, names, progress)

    monkeypatch.setattr(WeightStore, "ensure", held)
    return started, release


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("background", [False, True])
async def test_inference_and_state_remain_responsive_during_download(
    cluster, tmp_path, monkeypatch, resident, background
):
    engine, http, model, _, clients, _ = cluster
    if resident:
        response = await http.post("/admin/models/load", headers=ADMIN, json={"model": model})
        assert response.status_code == 200
    second = str(tmp_path / "downloading-model")
    make_model(tmp_path / "downloading-model")
    started, release = hold_downloads(monkeypatch, second, clients)
    if background:
        async with engine.preparation_gate:
            await engine.models.register(second)
        engine.models.wake()
        download = None
    else:
        download = asyncio.create_task(
            http.post("/admin/models/preload", headers=ADMIN, json={"model": second})
        )
    try:
        await eventually(lambda: all(event.is_set() for event in started.values()))
        assert engine.preparation_gate.locked() and not engine.gate.locked()
        async with asyncio.timeout(8):
            state = await http.get("/admin/state", headers=ADMIN)
            assert state.status_code == 200
            assert all(node["download"]["model"] == second for node in state.json()["nodes"])
            listed = await http.get("/v1/models", headers=API)
            assert [entry["id"] for entry in listed.json()["data"]] == [model]
            for stream in (False, True):
                response = await http.post(
                    "/v1/completions",
                    headers=API,
                    json={"model": model, "prompt": "Hello", "max_tokens": 3, "stream": stream},
                )
                assert response.status_code == 200, response.text
                if stream:
                    assert "data: [DONE]" in response.text
                else:
                    assert response.json()["usage"]["completion_tokens"] > 0
            # Missing models fail promptly, without waiting behind the transfer.
            incomplete = await http.post(
                "/v1/completions", headers=API, json={"model": second, "prompt": "Hello", "max_tokens": 1}
            )
            assert incomplete.status_code == 503
        assert not release.is_set()
        assert all(c.status["model"] == model and c.status["status"] == "ready" for c in clients)
        assert all(p.status["epoch"] == engine.deployments[model]["epoch"] for p in engine.nodes.values())
        assert all(not c.runtime.sessions for c in clients)
    finally:
        release.set()
        if download:
            response = await download
            assert response.status_code == 200, response.text
        else:
            await eventually(lambda: bool(engine.models.executable(second)))
    # Finishing a download must not restore an obsolete GPU status/epoch.
    assert all(c.status["model"] == model and c.status["status"] == "ready" for c in clients)
    assert engine.valid(engine.deployments[model])
    assert all(c.download_status is None for c in clients)


async def test_failed_download_preserves_ready_pipeline(cluster, tmp_path, monkeypatch):
    engine, http, model, _, clients, _ = cluster
    await http.post("/admin/models/load", headers=ADMIN, json={"model": model})
    second = str(tmp_path / "failed-download")
    make_model(tmp_path / "failed-download")
    started, release = hold_downloads(monkeypatch, second, clients, fail=True)
    download = asyncio.create_task(http.post("/admin/models/preload", headers=ADMIN, json={"model": second}))
    try:
        await eventually(lambda: all(event.is_set() for event in started.values()))
        async with asyncio.timeout(8):
            response = await http.post(
                "/v1/completions", headers=API, json={"model": model, "prompt": "Hello", "max_tokens": 2}
            )
            assert response.status_code == 200
    finally:
        release.set()
        response = await download
    assert response.status_code == 503 and "Injected download failure" in response.text
    assert engine.valid(engine.deployments[model])
    assert all(c.status["status"] == "ready" and c.download_status is None for c in clients)
    response = await http.post(
        "/v1/completions", headers=API, json={"model": model, "prompt": "Hello", "max_tokens": 2}
    )
    assert response.status_code == 200


async def test_pending_cache_clear_does_not_block_serving_while_download_finishes(
    cluster, tmp_path, monkeypatch
):
    engine, http, model, _, clients, _ = cluster
    second = str(tmp_path / "clear-during-download")
    make_model(tmp_path / "clear-during-download")
    started, release = hold_downloads(monkeypatch, second, clients)
    download = asyncio.create_task(http.post("/admin/models/preload", headers=ADMIN, json={"model": second}))
    clear = None
    try:
        await eventually(lambda: all(event.is_set() for event in started.values()))
        clear = asyncio.create_task(http.delete("/admin/nodes/node0/cache", headers=ADMIN))
        async with asyncio.timeout(8):
            response = await http.post(
                "/v1/completions", headers=API, json={"model": model, "prompt": "Hello", "max_tokens": 2}
            )
            assert response.status_code == 200, response.text
        assert not clear.done()
    finally:
        release.set()
        await download
        if clear:
            response = await clear
            assert response.status_code == 200 and not response.json()["pending"]
    assert engine.cache.paused("node0")
    assert not clients[0].runtime.cached and not (clients[0].runtime.cache_dir / "weights").exists()


async def test_legacy_nodes_keep_ready_stages_until_unloaded(cluster, tmp_path, monkeypatch):
    engine, http, model, _, clients, _ = cluster
    await http.post("/admin/models/load", headers=ADMIN, json={"model": model})
    for peer in engine.nodes.values():
        peer.capabilities.pop("concurrent_preload")
    second = str(tmp_path / "legacy-download")
    make_model(tmp_path / "legacy-download")
    started, release = hold_downloads(monkeypatch, second, clients)
    release.set()
    response = await http.post("/admin/models/preload", headers=ADMIN, json={"model": second})
    assert response.status_code == 200 and not response.json()["executable"]
    assert "update to download while serving" in response.json()["warning"]
    assert not any(event.is_set() for event in started.values())
    response = await http.post(
        "/v1/completions", headers=API, json={"model": model, "prompt": "Hello", "max_tokens": 2}
    )
    assert response.status_code == 200
    await http.post("/admin/models/unload", headers=ADMIN, json={"model": model})
    response = await http.post("/admin/models/preload", headers=ADMIN, json={"model": second})
    assert response.status_code == 200 and response.json()["executable"]


async def test_cancelled_preparation_fences_download_workers(cluster, tmp_path, monkeypatch):
    engine, _, _, _, clients, _ = cluster
    second = str(tmp_path / "cancelled-download")
    make_model(tmp_path / "cancelled-download")
    started, release = hold_downloads(monkeypatch, second, clients)
    download = asyncio.create_task(engine.models.preload(second))
    try:
        await eventually(lambda: all(event.is_set() for event in started.values()))
        peers = list(engine.nodes.values())
        download.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await download
        assert all(not peer.connected for peer in peers)
        assert not engine.preparation_gate.locked() and not engine.gate.locked()
    finally:
        release.set()
    await eventually(lambda: bool(engine.models.executable(second)))
    assert all(engine.nodes[p.id] is not p for p in peers)


async def test_cached_pipeline_does_not_query_unrelated_legacy_downloader(cluster, monkeypatch):
    engine, http, model, _, clients, _ = cluster
    # Give node0 a full cached replica, leaving the other legacy node busy.
    spec = engine.specs[model]
    await asyncio.to_thread(clients[0].runtime.preload, spec, 0, spec.layers)
    engine.nodes["node0"].update_inventory(clients[0].runtime.inventory())
    engine.max_layers_per_node = None
    busy = engine.nodes["node1"]
    busy.capabilities.pop("concurrent_preload")
    busy.downloading = True
    calls = []

    async def no_rpc(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Inference queried a node whose legacy worker is downloading")

    monkeypatch.setattr(busy, "call", no_rpc)
    try:
        response = await http.post(
            "/v1/completions", headers=API, json={"model": model, "prompt": "Hello", "max_tokens": 2}
        )
        assert response.status_code == 200, response.text
        assert [a["node_id"] for a in engine.deployments[model]["assignments"]] == ["node0"]
        assert not calls
    finally:
        busy.downloading = False
