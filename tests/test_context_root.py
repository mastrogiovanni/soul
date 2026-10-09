import pytest
from anthropic import AsyncAnthropic
from conftest import BasicIngress
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from soul.controller.app import Settings, create_app


@pytest.mark.parametrize("prefix", ["/soul", "/team/soul/"])
def test_mounted_ui_docs_auth_and_health(prefix, tmp_path):
    app = create_app(Settings("api", "admin", "node", state_dir=tmp_path, context_root=prefix))
    base = prefix.rstrip("/")
    with TestClient(app, base_url="https://example.test") as client:
        for path in ("/", base):
            redirect = client.get(path, follow_redirects=False)
            assert redirect.status_code == 307
            assert redirect.headers["location"] == base + "/"
        assert client.get("/healthz").status_code == 200
        assert client.get(base + "/healthz").status_code == 200
        page = client.get(base + "/")
        assert page.status_code == 200
        assert 'href="static/style.css?v=9"' in page.text
        assert 'href="./"' in page.text and 'href="docs"' in page.text
        assert client.get(base + "/static/app.js?v=10").status_code == 200
        docs = client.get(base + "/docs")
        assert f"{base}/openapi.json" in docs.text
        schema = client.get(base + "/openapi.json").json()
        assert schema["servers"] == [{"url": base}]
        assert "/v1/models" in schema["paths"]
        for path in ("/admin/state", "/v1/models", "/models"):
            assert client.get(base + path).status_code == 401
            assert client.get(path).status_code == 404
        assert client.get(base + "/admin/state", headers={"X-Soul-API-Key": "api"}).status_code == 401
        assert client.get(base + "/admin/state", headers={"X-Soul-Admin-Key": "admin"}).status_code == 200
        assert client.get(base + "/models", headers={"Authorization": "Bearer api"}).status_code == 200
        assert client.get("/soul-other/admin/state").status_code == 404
        error = client.post(base + "/v1/messages", json={}).json()
        assert error["type"] == "error" and error["error"]["type"] == "authentication_error"
        invalid = client.post(base + "/v1/messages", headers={"x-api-key": "api"}, json={})
        assert invalid.status_code == 400 and invalid.json()["type"] == "error"


@pytest.mark.parametrize("prefix", ["soul", "//soul", "/soul/../admin", "/soul?query", "/soul#hash"])
def test_invalid_context_root_rejected(prefix, tmp_path):
    with pytest.raises(ValueError, match="Context root"):
        create_app(Settings("api", "admin", "node", state_dir=tmp_path, context_root=prefix))


@pytest.mark.parametrize("cluster", [{"context_root": "/soul", "ingress_auth": True}], indirect=True)
async def test_prefixed_nodes_and_sdk_streams_behind_basic_ingress(cluster):
    engine, http, model, url, nodes, _ = cluster
    assert len(engine.nodes) == 2
    assert all(n.url.endswith("/soul/internal/nodes") for n in nodes)
    assert (await http.get("/admin/state")).status_code == 401
    assert (await http.get("/admin/state", headers={"X-Soul-Admin-Key": "admin"})).status_code == 200
    # A valid Soul key alone cannot bypass the outer Basic Auth gate.
    assert (await http.get("/models", headers={"Authorization": "Bearer api"})).status_code == 401
    headers = {"Authorization": BasicIngress.authorization.decode(), "X-Soul-API-Key": "api"}
    async with AsyncOpenAI(
        api_key="api", base_url=url + "/v1", default_headers=headers, max_retries=0
    ) as sdk:
        assert (await sdk.models.list()).data[0].id == model
        stream = await sdk.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=2,
            stream=True,
        )
        chunks = [c async for c in stream]
        assert chunks[-1].choices[0].finish_reason in ("stop", "length")
    async with AsyncAnthropic(api_key="api", base_url=url, default_headers=headers, max_retries=0) as sdk:
        result = await sdk.messages.create(
            model=model, messages=[{"role": "user", "content": "Hello"}], max_tokens=2
        )
        assert result.type == "message"
    assert all(not node.runtime.sessions for node in nodes)
