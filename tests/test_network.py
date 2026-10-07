import asyncio
import contextlib
import json

from anthropic import AsyncAnthropic
from conftest import eventually
from openai import AsyncOpenAI


async def test_real_nodes_openai_anthropic_and_admin(cluster):
    engine, http, model, url, clients, _ = cluster
    async with AsyncOpenAI(api_key="api", base_url=url + "/v1", max_retries=0) as client:
        result = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Hello world"}], max_tokens=4, temperature=0
        )
        assert result.object == "chat.completion"
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello world"}],
            max_tokens=4,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [chunk async for chunk in stream]
        text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        assert text == result.choices[0].message.content
        assert chunks[-1].usage.completion_tokens == result.usage.completion_tokens
        assert chunks[-2].choices[0].finish_reason in ("stop", "length")
        completion = await client.completions.create(
            model=model, prompt="Hello world", max_tokens=3, temperature=0
        )
        assert completion.object == "text_completion"
        stream = await client.completions.create(
            model=model, prompt="Hello world", max_tokens=3, temperature=0, stream=True
        )
        assert "".join([c.choices[0].text async for c in stream]) == completion.choices[0].text
        listed = await client.models.list()
        assert listed.data[0].id == model
    async with AsyncAnthropic(api_key="api", base_url=url, max_retries=0) as client:
        normal = await client.messages.create(
            model=model, messages=[{"role": "user", "content": "Hello world"}], max_tokens=4, temperature=0
        )
        assert normal.type == "message"
        async with client.messages.stream(
            model=model, messages=[{"role": "user", "content": "Hello world"}], max_tokens=4, temperature=0
        ) as stream:
            text = "".join([part async for part in stream.text_stream])
            final = await stream.get_final_message()
        assert normal.content[0].text == text == final.content[0].text
        assert normal.usage == final.usage
    state = (await http.get("/admin/state", headers={"Authorization": "Bearer admin"})).json()
    assert state["metrics"]["counts"]["completed"] == 6
    assert [(a["start"], a["end"]) for a in state["models"][0]["assignments"]] == [(0, 2), (2, 4)]
    assert all(not c.runtime.sessions for c in clients)
    assert not engine.gate.locked()
    logs = (await http.get("/admin/nodes/node0/logs", headers={"Authorization": "Bearer admin"})).json()
    assert any("load completed" in entry["message"] for entry in logs["logs"])


async def test_auth_invalid_requests_context_and_missing_node(cluster):
    engine, http, model, _, clients, tasks = cluster
    assert (await http.get("/admin/state")).status_code == 401
    assert (await http.get("/admin/state", headers={"Authorization": "Bearer api"})).status_code == 401
    assert (await http.get("/v1/models", headers={"Authorization": "Bearer admin"})).status_code == 401
    request = {"model": model, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2}
    auth = {"Authorization": "Bearer api"}
    assert (
        await http.post("/v1/chat/completions", headers=auth, json={**request, "tools": []})
    ).status_code == 400
    assert (
        await http.post("/v1/chat/completions", headers=auth, json={**request, "temperature": -1})
    ).status_code == 400
    assert (
        await http.post("/v1/chat/completions", headers=auth, json={**request, "max_tokens": 256})
    ).status_code == 400
    assert not engine.gate.locked()
    tasks[1].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks[1]
    await eventually(lambda: not engine.nodes["node1"].connected)
    failed = await http.post("/v1/chat/completions", headers=auth, json={**request, "stream": True})
    assert failed.status_code == 503
    assert failed.headers["content-type"].startswith("application/json")
    assert not clients[0].runtime.sessions
    assert not engine.gate.locked()


async def test_reconnect_fences_old_assignment_and_reloads(cluster):
    engine, http, model, _, clients, tasks = cluster
    auth = {"Authorization": "Bearer api"}
    request = {"model": model, "prompt": "Hello", "max_tokens": 2, "temperature": 0}
    first = await http.post("/v1/completions", headers=auth, json=request)
    assert first.status_code == 200
    epoch = engine.deployments[model]["epoch"]
    old_generation = engine.nodes["node1"].generation
    tasks[1].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks[1]
    await eventually(lambda: not engine.nodes["node1"].connected)
    assert engine.deployments[model]["status"] == "stale"
    tasks[1] = asyncio.create_task(clients[1].run())
    await eventually(
        lambda: engine.nodes["node1"].connected and engine.nodes["node1"].generation != old_generation
    )
    second = await http.post("/v1/completions", headers=auth, json=request)
    assert second.status_code == 200, second.text
    assert second.json()["choices"] == first.json()["choices"]
    assert engine.deployments[model]["epoch"] != epoch
    assert all(not c.runtime.sessions for c in clients)


async def test_chunked_prefill_through_network_and_queue(cluster):
    engine, http, model, _, _, _ = cluster
    request = {"model": model, "prompt": " ".join(["Hello"] * 140), "max_tokens": 2, "temperature": 0}
    responses = await asyncio.gather(
        *(
            http.post("/v1/completions", headers={"Authorization": "Bearer api"}, json=request)
            for _ in range(3)
        )
    )
    assert all(r.status_code == 200 for r in responses), [r.text for r in responses]
    assert responses[0].json()["usage"]["prompt_tokens"] == 140
    assert len({json.dumps(r.json()["choices"]) for r in responses}) == 1
    assert engine.waiting == 0
    assert not engine.gate.locked()


async def test_stream_disconnect_releases_request(cluster):
    engine, http, model, _, clients, _ = cluster
    async with http.stream(
        "POST",
        "/v1/completions",
        headers={"Authorization": "Bearer api"},
        json={"model": model, "prompt": "Hello", "max_tokens": 150, "stream": True},
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                break
    await eventually(lambda: not engine.gate.locked())
    assert all(not c.runtime.sessions for c in clients)


async def test_model_rotation_unloads_orphan_stages(cluster, tmp_path):
    from conftest import make_model

    engine, http, model, _, clients, _ = cluster
    auth = {"Authorization": "Bearer admin"}
    response = await http.post("/admin/models/load", headers=auth, json={"model": model})
    assert response.status_code == 200
    smaller = tmp_path / "smaller"
    make_model(smaller, layers=2)
    response = await http.post("/admin/models/load", headers=auth, json={"model": str(smaller)})
    assert response.status_code == 200, response.text
    assert engine.deployments[model]["status"] == "stale"
    assert clients[1].runtime.model is None
    # Unloading the stale record must not unload the replacement on its former first node.
    response = await http.post("/admin/models/unload", headers=auth, json={"model": model})
    assert response.status_code == 200
    assert clients[0].runtime.model is not None
    response = await http.post("/admin/models/unload", headers=auth, json={"model": str(smaller)})
    assert response.status_code == 200
    assert all(c.runtime.model is None for c in clients)


async def test_fault_after_stream_headers_is_error_event(cluster, monkeypatch):
    from soul.protocol import SoulError

    engine, http, model, _, clients, _ = cluster
    response = await http.post(
        "/admin/models/load", headers={"Authorization": "Bearer admin"}, json={"model": model}
    )
    assert response.status_code == 200

    def broken(request):
        raise SoulError("Injected device failure")

    monkeypatch.setattr(clients[1].runtime, "forward", broken)
    response = await http.post(
        "/v1/messages",
        headers={"x-api-key": "api"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4,
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "Injected device failure" in response.text
    assert "event: message_stop" not in response.text
    assert not engine.gate.locked()
    assert all(not c.runtime.sessions for c in clients)
    assert engine.deployments[model]["status"] == "stale"


async def test_bounded_queue_and_queue_timeout(cluster):
    engine, http, model, _, _, _ = cluster
    engine.max_queue = 1
    engine.queue_timeout = 0.2
    await engine.gate.acquire()
    request = {"model": model, "prompt": "Hello", "max_tokens": 2}
    queued = asyncio.create_task(
        http.post("/v1/completions", headers={"Authorization": "Bearer api"}, json=request)
    )
    try:
        await eventually(lambda: engine.waiting == 1)
        overflow = await http.post("/v1/completions", headers={"Authorization": "Bearer api"}, json=request)
        assert overflow.status_code == 503
        assert "queue is full" in overflow.text
        expired = await queued
        assert expired.status_code == 503
        assert "timed out" in expired.text
        assert engine.waiting == 0
    finally:
        engine.gate.release()


async def test_disconnect_while_loading_fences_and_cleans_nodes(cluster, monkeypatch):
    import time

    engine, http, model, _, clients, _ = cluster
    original = clients[0].runtime.load

    def slow_load(*args, **kwargs):
        time.sleep(0.4)
        return original(*args, **kwargs)

    monkeypatch.setattr(clients[0].runtime, "load", slow_load)
    request = asyncio.create_task(
        http.post(
            "/v1/completions",
            headers={"Authorization": "Bearer api"},
            json={"model": model, "prompt": "Hello", "max_tokens": 2},
        )
    )
    await eventually(lambda: clients[0].status["status"] == "loading")
    request.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await request
    await eventually(lambda: not engine.gate.locked())
    await eventually(lambda: all(c.runtime.model is None for c in clients))
    assert engine.deployments[model]["status"] != "ready"
    assert all(not c.runtime.sessions for c in clients)
