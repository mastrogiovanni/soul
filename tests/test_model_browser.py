import httpx
import pytest
from fastapi.testclient import TestClient
from test_quantization import quantize_fixture

from soul.controller.app import Settings, create_app
from soul.controller.browser import ModelBrowser
from soul.protocol import SoulError


def item(model="owner/model", files=None):
    return {
        "id": model,
        "sha": "abc123",
        "downloads": 50,
        "gated": False,
        "siblings": [
            {"rfilename": f} for f in (files if files is not None else ["config.json", "model.safetensors"])
        ],
    }


@pytest.mark.parametrize(
    "config,status,format",
    [
        ({"model_type": "qwen3"}, "candidate", "none"),
        ({"model_type": "qwen3_5"}, "unsupported", "none"),
        ({"model_type": "qwen3_moe"}, "unsupported", "none"),
        (
            {
                "model_type": "qwen2",
                "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128},
            },
            "candidate",
            "awq-gemm-4",
        ),
        (
            {
                "model_type": "llama",
                "quantization_config": {"quant_method": "gptq", "bits": 8, "group_size": 128},
            },
            "unsupported",
            "gptq",
        ),
        ({"model_type": "qwen3", "quantization_config": {"quant_method": "fp8"}}, "unsupported", "fp8"),
    ],
)
async def test_discovery_uses_config_not_repository_name(config, status, format):
    browser = ModelBrowser()
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=config)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        for _ in range(2):
            row = await browser.describe(client, item("owner/Qwen3-claimed-compatible"))
            assert row["status"] == status
            assert row["quantization"] == format
    assert len(requests) == 1  # Metadata is cached by immutable commit, never by model name alone.
    assert "/abc123/config.json" in str(requests[0].url)


async def test_gguf_gated_and_missing_metadata_are_not_claimed_supported():
    browser = ModelBrowser()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403))) as client:
        gguf = await browser.describe(client, item(files=["model.gguf"]))
        assert gguf["status"] == "unsupported" and "GGUF" in gguf["reason"]
        gated = await browser.describe(client, {**item(), "gated": "auto"})
        assert gated["status"] == "unknown" and "HF_TOKEN" in gated["reason"]
        assert not browser.configs  # Transient/access errors are retryable.


async def test_search_paginates_and_never_follows_returned_urls(monkeypatch):
    requests = []
    original_client = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        if request.url.path == "/api/models":
            return httpx.Response(
                200,
                json=[item()],
                headers={"link": '<https://untrusted.invalid/path?cursor=next-page>; rel="next"'},
            )
        return httpx.Response(200, json={"model_type": "llama"})

    monkeypatch.setattr("soul.controller.browser.get_token", lambda: None)
    monkeypatch.setattr(
        "soul.controller.browser.httpx.AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    browser = ModelBrowser()
    result = await browser.search("test", "llama")
    assert result["next_cursor"] == "next-page"
    assert result["models"][0]["status"] == "candidate"
    await browser.search("test", "llama", result["next_cursor"])
    assert all(r.url.host == "huggingface.co" for r in requests)
    assert requests[-1].url.params["cursor"] == "next-page"
    assert requests[-1].url.params["filter"] == "llama"


@pytest.mark.parametrize("prefix", ["", "/soul", "/team/soul"])
def test_catalog_routes_auth_and_read_only_check(tmp_path, prefix, monkeypatch):
    path = tmp_path / "model"
    quantize_fixture(path, "qwen2", "gptq")
    app = create_app(
        Settings("api", "admin", "node", state_dir=tmp_path / "state", context_root=prefix, allow_local=True)
    )
    with TestClient(app) as client:
        page = client.get(prefix + "/model-browser")
        assert page.status_code == 200
        assert 'src="static/model-browser.js?v=3"' in page.text
        assert 'href="model-browser"' in client.get(prefix + "/").text
        assert client.get(prefix + "/static/model-browser.js").status_code == 200
        for route in ("support", "search"):
            assert client.get(prefix + f"/admin/catalog/{route}").status_code == 401
        assert client.post(prefix + "/admin/catalog/check", json={"model": str(path)}).status_code == 401
        headers = {"X-Soul-Admin-Key": "admin"}
        response = client.post(prefix + "/admin/catalog/check", headers=headers, json={"model": str(path)})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "supported" and data["quantization"] == "gptq-4"
        assert data["compatible_nodes"] == 0
        assert data["weight_bytes_fp16"] < data["weight_bytes_fp32"]
        assert client.get(prefix + "/admin/state", headers=headers).json()["models"] == []
        matrix = client.get(prefix + "/admin/catalog/support", headers=headers).json()
        assert not next(f for f in matrix["families"] if f["id"] == "qwen3_5")["supported"]


async def test_allowlist_checked_before_inspection(tmp_path):
    from soul.controller.engine import Engine
    from soul.controller.store import Store

    store = Store(tmp_path / "state.sqlite3")
    engine = Engine(store, allowed_models=["allowed/model"])
    try:
        with pytest.raises(SoulError, match="allowlist"):
            await ModelBrowser().check(engine, "other/model")
    finally:
        store.close()


@pytest.mark.parametrize("method", ["awq", "gptq"])
async def test_quantized_model_preload_load_and_inference_over_network(cluster, tmp_path, method):
    engine, http, _, _, nodes, _ = cluster
    path = tmp_path / method
    quantize_fixture(path, "qwen2", method)
    admin = {"X-Soul-Admin-Key": "admin"}
    check = await http.post("/admin/catalog/check", headers=admin, json={"model": str(path)})
    assert check.status_code == 200, check.text
    data = check.json()
    assert data["compatible_nodes"] == 2
    prepared = await http.post(
        "/admin/models/load", headers=admin, json={"model": str(path), "revision": data["revision"]}
    )
    assert prepared.status_code == 200, prepared.text
    result = await http.post(
        "/v1/chat/completions",
        headers={"X-Soul-API-Key": "api"},
        json={"model": str(path), "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2},
    )
    assert result.status_code == 200, result.text
    assert result.json()["usage"]["completion_tokens"] > 0
    wrong = await http.post(
        "/admin/models/preload", headers=admin, json={"model": str(path), "revision": "different"}
    )
    assert wrong.status_code == 409
    # Legacy nodes must never receive a quantized placement even if they report its cache.
    for peer in engine.nodes.values():
        peer.capabilities.pop("quantizations")
    assert engine.models.executable(str(path)) is None
