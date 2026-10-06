import json

import pytest
import torch
from conftest import make_model
from transformers import DynamicCache

from soul.catalog import inspect_model
from soul.node.runtime import Runtime, decode_tensor, encode_tensor
from soul.node.weights import WeightStore
from soul.protocol import SoulError


@pytest.mark.parametrize("architecture", ["llama", "mistral", "qwen2", "qwen3"])
@pytest.mark.parametrize("mode", ["resident", "stream"])
def test_shards_match_full_model_prefill_and_cached_decode(tmp_path, architecture, mode):
    directory = tmp_path / "model"
    reference = make_model(directory, architecture)
    spec = inspect_model(str(directory), allow_local=True, max_context=64)
    stages = [Runtime(tmp_path / f"node{i}", "cpu", memory_gb=1, weight_mode=mode) for i in range(3)]
    for stage, (start, end) in zip(stages, [(0, 1), (1, 3), (3, 4)], strict=True):
        stage.load(spec, start, end, "epoch")
        assert len(stage.model.model.layers) == end - start
        names = {stage.store.path(n).name for n in spec.names_for(start, end)}
        assert {p.name for p in stage.store.root.glob("*.safetensors")} == names
    cache = DynamicCache(config=reference.config)
    position = 0
    # Multiple prefill chunks cross Mistral's sliding window, followed by decode tokens.
    for tokens in ([4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14], [15], [16], [17]):
        with torch.inference_mode():
            expected = reference(torch.tensor([tokens]), past_key_values=cache, use_cache=True).logits[0, -1]
        data = {
            "epoch": "epoch",
            "session_id": "req",
            "position": position,
            "input_ids": tokens,
            "return_logits": True,
        }
        for stage in stages:
            output = stage.forward(data)
            data.pop("input_ids", None)
            data.pop("hidden", None)
            data.update(output)
        torch.testing.assert_close(torch.tensor(output["logits"]), expected, atol=2e-6, rtol=2e-5)
        position += len(tokens)
    for stage in stages:
        assert stage.sessions["req"][1] == position
        stage.unload()
        assert not stage.sessions


def test_tied_weights_and_single_stage(tmp_path):
    reference = make_model(tmp_path / "model", tied=True)
    spec = inspect_model(str(tmp_path / "model"), allow_local=True)
    node = Runtime(tmp_path / "node", "cpu", memory_gb=1)
    node.load(spec, 0, 4, "epoch")
    result = node.forward(
        {"epoch": "epoch", "session_id": "r", "position": 0, "input_ids": [4, 5], "return_logits": True}
    )
    with torch.inference_mode():
        expected = reference(torch.tensor([[4, 5]])).logits[0, -1]
    torch.testing.assert_close(torch.tensor(result["logits"]), expected)
    with pytest.raises(SoulError, match="Stale"):
        node.forward({"epoch": "old"})
    with pytest.raises(SoulError, match="sequence"):
        node.forward({"epoch": "epoch", "session_id": "r", "position": 12})


def test_rejects_unsupported_architecture(tmp_path):
    reference = make_model(tmp_path / "model")
    del reference
    path = tmp_path / "model/config.json"
    data = json.loads(path.read_text())
    data["model_type"] = "mamba"
    path.write_text(json.dumps(data))
    with pytest.raises(SoulError, match="adapter"):
        inspect_model(str(path.parent), allow_local=True)


def test_tensor_wire_format_validates_shape_and_length():
    value = torch.randn(1, 3, 4).to(torch.bfloat16)
    wire = encode_tensor(value)
    torch.testing.assert_close(decode_tensor(wire, "cpu", torch.bfloat16, 4, 128), value)
    with pytest.raises(ValueError):
        decode_tensor({**wire, "shape": [1, 3, 8]}, "cpu", torch.float32, 4, 128)
    with pytest.raises(ValueError):
        decode_tensor({**wire, "data": b""}, "cpu", torch.float32, 4, 128)


def test_cache_repairs_corrupted_tensor(tiny_model, tmp_path):
    path, _ = tiny_model
    spec = inspect_model(str(path), allow_local=True)
    store = WeightStore(tmp_path / "weights", spec)
    names = spec.names_for(1, 2)
    store.ensure(names)
    target = store.path(names[0])
    expected = target.read_bytes()
    target.write_bytes(b"corrupted")
    store.ensure(names)
    assert target.read_bytes() == expected
