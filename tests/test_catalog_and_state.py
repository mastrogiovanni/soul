import json
import os
import subprocess
import sys

import httpx
import pytest

from soul.catalog import RangeReader
from soul.controller.store import Store
from soul.protocol import SoulError, pack, unpack


def test_controller_import_does_not_load_torch():
    script = "from soul.controller.app import create_app; import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_http_range_requires_exact_partial_response():
    reader = RangeReader("owner/repo", "commit")
    reader.client.close()
    reader.client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"abcdefgh"))
    )
    with pytest.raises(SoulError, match="byte ranges"):
        reader.read("model.safetensors", 0, 8)
    reader.client.close()
    reader.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                206, headers={"content-range": "bytes 0-7/999"}, content=b"abcdefgh"
            )
        )
    )
    assert reader.read("model.safetensors", 0, 8) == b"abcdefgh"
    with pytest.raises(ValueError, match="filename"):
        reader.read("../secret", 0, 8)
    reader.close()


def test_range_response_length_is_validated():
    reader = RangeReader("owner/repo", "commit")
    reader.client.close()
    reader.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(206, headers={"content-range": "bytes 0-7/999"}, content=b"abc")
        )
    )
    with pytest.raises(ValueError, match="Truncated"):
        reader.read("model.safetensors", 0, 8)
    reader.close()


def test_restart_preserves_counters_but_fails_unfinished_requests(tmp_path):
    path = tmp_path / "controller.sqlite3"
    store = Store(path)
    store.request("done", "model")
    store.update_request("done", "completed", input_tokens=2, output_tokens=3)
    store.request("active", "model")
    store.update_request("active", "running")
    store.request("queued", "model")
    store.close()
    store = Store(path)
    assert store.stats()["counts"] == {"completed": 1, "failed": 2}
    assert store.stats()["output_tokens"] == 3
    assert all(
        row["error"] == "Controller restarted" for row in store.stats()["recent"] if row["status"] == "failed"
    )
    store.close()


def test_protocol_uses_binary_and_rejects_untyped_maps():
    data = {"type": "rpc", "args": {"data": b"\x00\xff"}}
    assert unpack(pack(data)) == data
    with pytest.raises(ValueError):
        unpack(pack({"method": "forward"}))


def test_generation_config_eos_takes_precedence(tiny_model, tmp_path):
    from soul.catalog import inspect_model
    from soul.node.runtime import Runtime

    path, _ = tiny_model
    generation = path / "generation_config.json"
    content = json.loads(generation.read_text())
    content["eos_token_id"] = [2, 15]
    generation.write_text(json.dumps(content))
    spec = inspect_model(str(path), allow_local=True)
    node = Runtime(tmp_path / "node", "cpu", memory_gb=1)
    node.load(spec, 0, 2, "epoch")
    assert node.tokenize({"prompt": "Hello"})["eos_ids"] == [2, 15]
    node.unload()
