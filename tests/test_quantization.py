import json

import numpy as np
import pytest
import torch
from conftest import make_model
from safetensors.torch import save_file
from transformers import DynamicCache

from soul.catalog import inspect_model
from soul.node.quantized import PackedLinear
from soul.node.runtime import Runtime
from soul.protocol import SoulError
from soul.quantization import quantization_format, supports


def quantize_fixture(path, architecture="llama", method="awq", group_size=8, act_order=False, hidden_size=32):
    """Independent NumPy checkpoint writer + dense dequantized reference."""
    model = make_model(path, architecture, hidden_size=hidden_size)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = np.random.default_rng(24)
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.startswith("model.layers."):
            continue
        inputs, outputs = module.in_features, module.out_features
        group = inputs if group_size == -1 else group_size
        indices = np.arange(inputs, dtype=np.int32) // group
        if act_order:
            rng.shuffle(indices)
        groups = int(indices.max()) + 1
        scale = rng.uniform(0.001, 0.01, size=(groups, outputs)).astype(np.float32)
        zero = rng.integers(1, 15, size=(groups, outputs), dtype=np.int32)
        values = rng.integers(0, 16, size=(inputs, outputs), dtype=np.int32)
        dense = (values - zero[indices]) * scale[indices]
        module.weight.data.copy_(torch.from_numpy(dense.T.copy()))
        del state[name + ".weight"]
        qweight = np.zeros(
            (inputs, outputs // 8) if method == "awq" else (inputs // 8, outputs), dtype=np.uint32
        )
        qzeros = np.zeros((groups, outputs // 8), dtype=np.uint32)
        for col in range(outputs):
            slot = [0, 4, 1, 5, 2, 6, 3, 7][col % 8] if method == "awq" else col % 8
            qzeros[:, col // 8] |= (zero[:, col] - (method == "gptq")).astype(np.uint32) << (slot * 4)
            for row in range(inputs):
                if method == "awq":
                    qweight[row, col // 8] |= np.uint32(values[row, col]) << (slot * 4)
                else:
                    qweight[row // 8, col] |= np.uint32(values[row, col]) << ((row % 8) * 4)
        state[name + ".qweight"] = torch.from_numpy(qweight.view(np.int32))
        state[name + ".qzeros"] = torch.from_numpy(qzeros.view(np.int32))
        state[name + ".scales"] = torch.from_numpy(scale)
        if method == "gptq":
            state[name + ".g_idx"] = torch.from_numpy(indices)
    save_file(state, path / "model.safetensors", metadata={"format": "pt"})
    config = json.loads((path / "config.json").read_text())
    config["quantization_config"] = {"quant_method": method, "bits": 4, "group_size": group_size}
    if method == "awq":
        config["quantization_config"].update(version="gemm", zero_point=True)
    else:
        config["quantization_config"].update(checkpoint_format="gptq", desc_act=act_order)
    (path / "config.json").write_text(json.dumps(config))
    return model


@pytest.mark.parametrize("architecture", ["llama", "mistral", "qwen2", "qwen3"])
@pytest.mark.parametrize("method", ["awq", "gptq"])
@pytest.mark.parametrize("mode", ["resident", "stream"])
def test_packed_shards_match_dense_reference(tmp_path, architecture, method, mode):
    reference = quantize_fixture(tmp_path / "model", architecture, method, act_order=method == "gptq")
    spec = inspect_model(str(tmp_path / "model"), allow_local=True, max_context=64)
    stages = [Runtime(tmp_path / f"node{i}", "cpu", memory_gb=1, weight_mode=mode) for i in range(3)]
    for stage, (start, end) in zip(stages, [(0, 1), (1, 3), (3, 4)], strict=True):
        stage.load(spec, start, end, "epoch")
        assert sum(isinstance(m, PackedLinear) for m in stage.model.modules()) == 7 * (end - start)
        assert all(".weight" not in n for n, _ in stage.model.named_parameters() if "_proj." in n)
    cache = DynamicCache(config=reference.config)
    position = 0
    for tokens in ([4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14], [15], [16]):
        with torch.inference_mode():
            expected = reference(torch.tensor([tokens]), past_key_values=cache, use_cache=True).logits[0, -1]
        request = {
            "epoch": "epoch",
            "session_id": "req",
            "position": position,
            "input_ids": tokens,
            "return_logits": True,
        }
        for stage in stages:
            output = stage.forward(request)
            request.pop("input_ids", None)
            request.pop("hidden", None)
            request.update(output)
        torch.testing.assert_close(torch.tensor(output["logits"]), expected, atol=2e-6, rtol=2e-5)
        position += len(tokens)
    for stage in stages:
        if mode == "stream":
            assert all(
                b.device.type == "meta" for n, b in stage.model.named_buffers() if n.endswith(".qweight")
            )
        stage.unload()
    # Disk cache survives restart and preserves packed integer tensors.
    restarted = Runtime(tmp_path / "node0", "cpu", memory_gb=1)
    assert restarted.scan_cache()[0]["spec"]["config"]["quantization_config"]["bits"] == 4
    restarted.load(spec, 0, 1, "new", download=False)
    assert restarted.model.model.layers[0].self_attn.q_proj.qweight.dtype == torch.int32


@pytest.mark.parametrize("method", ["awq", "gptq"])
def test_group_minus_one_and_memory_accounting(tmp_path, method):
    quantize_fixture(tmp_path / "model", method=method, group_size=-1)
    spec = inspect_model(str(tmp_path / "model"), allow_local=True)
    dense_parameters = sum(p.numel() for p in make_model(tmp_path / "dense").parameters())
    assert spec.weight_bytes(0, spec.layers, 4) < dense_parameters * 4
    assert spec.quantization_workspace() > 0
    node = Runtime(tmp_path / "node", "cpu", memory_gb=1)
    node.load(spec, 0, 4, "epoch")
    assert "token_id" in node.forward({"epoch": "epoch", "session_id": "r", "position": 0, "input_ids": [4]})
    assert supports(node.capabilities(), spec)
    assert not supports({"architectures": ["llama"]}, spec)


@pytest.mark.parametrize(
    "options",
    [
        {"bits": 8},
        {"version": "gemv"},
        {"zero_point": False},
        {"quant_method": "bitsandbytes"},
        {"quant_method": "gptq", "checkpoint_format": "gptq_v2"},
        {"quant_method": "gptq", "format": "gptq_v2"},
        {"quant_method": "gptq", "format": "marlin"},
        {"quant_method": "awq", "format": "gemv"},
        {"method": "gptq"},
        {"quant_method": "gptq", "is_marlin_format": True},
        {"group_size": 0},
        {"dynamic": {".*": {"bits": 8}}},
        {"lm_head": True},
    ],
)
def test_rejects_unsupported_quantization_variants(options):
    with pytest.raises(SoulError):
        quantization_format(
            {"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128, **options}}
        )


def test_rejects_malformed_packed_checkpoint(tmp_path):
    from safetensors.torch import load_file

    quantize_fixture(tmp_path / "model")
    filename = tmp_path / "model/model.safetensors"
    tensors = load_file(filename)
    tensors.pop("model.layers.0.self_attn.q_proj.qzeros")
    save_file(tensors, filename)
    with pytest.raises(SoulError, match="qzeros"):
        inspect_model(str(filename.parent), allow_local=True)


def test_rejects_missing_projection_even_when_quantized_metadata_is_valid(tmp_path):
    from safetensors.torch import load_file

    quantize_fixture(tmp_path / "model")
    filename = tmp_path / "model/model.safetensors"
    tensors = {
        name: value
        for name, value in load_file(filename).items()
        if not name.startswith("model.layers.2.self_attn.q_proj.")
    }
    save_file(tensors, filename)
    with pytest.raises(SoulError, match="q_proj.weight"):
        inspect_model(str(filename.parent), allow_local=True)


@pytest.mark.parametrize("method", ["awq", "gptq"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_multiple_dequantization_tiles_and_floating_dtypes(tmp_path, method, dtype):
    # Hidden/output sizes cross the 128-column tile boundary and group_size=48
    # exercises a partially filled final group. Compare with independently decoded weights.
    reference = quantize_fixture(tmp_path / "model", "qwen2", method, group_size=48, hidden_size=160)
    spec = inspect_model(str(tmp_path / "model"), allow_local=True)
    runtime = Runtime(tmp_path / "node", "cpu", memory_gb=1, dtype=dtype)
    runtime.load(spec, 0, 4, "epoch")
    packed = runtime.model.model.layers[0].mlp.up_proj
    dense = reference.model.layers[0].mlp.up_proj
    x = torch.randn(1, 5, dense.in_features).to(getattr(torch, dtype))
    # Rounding follows scales -> dequantized weights in the node's float dtype.
    source = "model.layers.0.mlp.up_proj"
    scales = runtime.store.tensor(source + ".scales", "cpu", torch.float32)
    rounded = scales.to(x.dtype).float()
    groups = packed.g_idx.long() if method == "gptq" else torch.arange(dense.in_features) // 48
    weights = (dense.weight.detach().T / scales[groups]).round().to(x.dtype) * rounded[groups].to(x.dtype)
    with torch.inference_mode():
        expected = torch.matmul(x, weights)
        actual = packed(x)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)


def test_rejects_invalid_gptq_group_values_before_inference(tmp_path):
    from safetensors.torch import load_file

    quantize_fixture(tmp_path / "model", method="gptq")
    filename = tmp_path / "model/model.safetensors"
    tensors = load_file(filename)
    tensors["model.layers.0.self_attn.q_proj.g_idx"][0] = 999
    save_file(tensors, filename)
    spec = inspect_model(str(filename.parent), allow_local=True)
    node = Runtime(tmp_path / "node", "cpu", memory_gb=1)
    with pytest.raises(SoulError, match="group indices"):
        node.load(spec, 0, 4, "epoch")


@pytest.mark.parametrize("method", ["awq", "gptq"])
@pytest.mark.parametrize("mode", ["resident", "stream"])
@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_quantized_accelerator_when_available(tmp_path, method, mode, device):
    available = torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()
    if not available:
        pytest.skip(f"{device} is not available")
    quantize_fixture(tmp_path / "model", "qwen2", method, act_order=method == "gptq")
    spec = inspect_model(str(tmp_path / "model"), allow_local=True, max_context=64)
    reference = Runtime(tmp_path / "cpu", "cpu", memory_gb=1, dtype="float16")
    node = Runtime(tmp_path / device, device, memory_gb=1, dtype="float16", weight_mode=mode)
    for runtime in (reference, node):
        runtime.load(spec, 0, 4, "epoch")
    for position, tokens in [(0, [4, 5, 6]), (3, [7])]:
        request = {
            "epoch": "epoch",
            "session_id": "r",
            "position": position,
            "input_ids": tokens,
            "return_logits": True,
        }
        expected = reference.forward(request)["logits"]
        result = node.forward(request)["logits"]
        torch.testing.assert_close(torch.tensor(result), torch.tensor(expected), atol=0.003, rtol=0.02)
    reference.unload()
    node.unload()
