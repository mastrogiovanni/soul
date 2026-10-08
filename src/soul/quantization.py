"""Checkpoint format contracts shared by the controller and nodes (no torch)."""

import math
import re

from soul.protocol import ARCHITECTURES, SoulError

QUANTIZATIONS = ("awq-gemm-4", "gptq-4")
DEQUANT_BLOCK = 128
FLOAT_DTYPES = {"F16", "BF16", "F32"}


def unsupported(message):
    raise SoulError(message, 400, "unsupported_model")


def quantization_format(config):
    q = config.get("quantization_config")
    if q is None:
        return "none"
    if not isinstance(q, dict):
        unsupported("Invalid quantization_config")
    method = q.get("quant_method", q.get("method"))
    if "method" in q and q["method"] != method:
        unsupported("Conflicting quantization method fields.")
    if method not in ("awq", "gptq"):
        unsupported(f"Unsupported quantization {method!r}. Use AWQ GEMM 4-bit or GPTQ v1 4-bit safetensors.")
    group = q.get("group_size")
    if q.get("bits") != 4 or type(group) is not int or not (group == -1 or 1 <= group <= 65536):
        unsupported("Quantized checkpoints require 4 bits and a positive group_size (or -1).")
    if q.get("dynamic") or q.get("lm_head"):
        unsupported("Mixed quantization settings and quantized output heads are not supported.")
    if method == "awq":
        if (
            any(
                str(q.get(field, "gemm")).lower() != "gemm"
                for field in ("version", "format", "checkpoint_format")
            )
            or q.get("zero_point", True) is not True
        ):
            unsupported(
                "Only AWQ GEMM with 4-bit weights and zero_point=true is supported; GEMV/Marlin are not."
            )
        if q.get("backend", "autoawq") != "autoawq":
            unsupported("Only the AutoAWQ GEMM checkpoint layout is supported.")
        return "awq-gemm-4"
    if any(q.get(field, "gptq") != "gptq" for field in ("format", "checkpoint_format")) or q.get(
        "is_marlin_format", False
    ):
        unsupported("Only GPTQ v1 checkpoints are supported; GPTQ v2 and Marlin require a different adapter.")
    if q.get("pack_dtype", "int32") != "int32":
        unsupported("GPTQ requires int32 packed weights.")
    return "gptq-4"


def validate_config(config):
    architecture = config.get("model_type")
    if architecture not in ARCHITECTURES:
        unsupported(
            f"Architecture {architecture or 'unknown'} is not supported. Supported architectures: "
            f"{', '.join(ARCHITECTURES)}. An architecture adapter is required; quantization does not change this."
        )
    return quantization_format(config)


def projection_shapes(config):
    hidden = config["hidden_size"]
    head = config.get("head_dim", hidden // config["num_attention_heads"])
    query = config["num_attention_heads"] * head
    kv = config.get("num_key_value_heads", config["num_attention_heads"]) * head
    intermediate = config["intermediate_size"]
    return {
        "self_attn.q_proj": (hidden, query),
        "self_attn.k_proj": (hidden, kv),
        "self_attn.v_proj": (hidden, kv),
        "self_attn.o_proj": (query, hidden),
        "mlp.gate_proj": (hidden, intermediate),
        "mlp.up_proj": (hidden, intermediate),
        "mlp.down_proj": (intermediate, hidden),
    }


def quantized_modules(config, tensors):
    """Validate packed tensor metadata before downloading any weight payloads."""
    format = validate_config(config)
    modules = {}
    packed = {
        name.rsplit(".", 1)[0]
        for name in tensors
        if name.endswith((".qweight", ".qzeros", ".scales", ".g_idx"))
    }
    if format == "none":
        if packed:
            unsupported("Packed tensors require a supported quantization_config.")
        return modules
    shapes = projection_shapes(config)
    for prefix in sorted(packed):
        match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", prefix)
        if not match or int(match[1]) >= config["num_hidden_layers"] or match[2] not in shapes:
            unsupported(f"Unsupported quantized module: {prefix}")
        inputs, outputs = shapes[match[2]]
        group = config["quantization_config"]["group_size"]
        group = inputs if group == -1 else group
        groups = math.ceil(inputs / group)
        if inputs % 8 or outputs % 8 or prefix + ".weight" in tensors:
            unsupported(f"Invalid packed dimensions or duplicate float weight: {prefix}")
        expected = {
            "qweight": (
                [inputs, outputs // 8] if format == "awq-gemm-4" else [inputs // 8, outputs],
                {"I32"},
            ),
            "qzeros": ([groups, outputs // 8], {"I32"}),
            "scales": ([groups, outputs], FLOAT_DTYPES),
        }
        if format == "gptq-4":
            expected["g_idx"] = ([inputs], {"I32"})
        elif prefix + ".g_idx" in tensors:
            unsupported(f"Unexpected AWQ g_idx tensor: {prefix}")
        for suffix, (shape, dtypes) in expected.items():
            tensor = tensors.get(prefix + "." + suffix, {})
            if tensor.get("shape") != shape or tensor.get("dtype") not in dtypes:
                unsupported(f"Missing or invalid {prefix}.{suffix}; expected shape {shape}.")
        modules[prefix] = (inputs, outputs)
    if not modules:
        unsupported("Quantization is declared but no supported packed linear weights were found.")
    for name, tensor in tensors.items():
        if not name.endswith((".qweight", ".qzeros", ".g_idx")) and tensor["dtype"] not in FLOAT_DTYPES:
            unsupported(f"Unsupported tensor dtype for {name}: {tensor['dtype']}")
    return modules


def supports(capabilities, spec):
    return spec.config["model_type"] in capabilities.get("architectures", []) and (
        spec.quantization == "none" or spec.quantization in capabilities.get("quantizations", [])
    )


def validate_tensor_layout(config, tensors):
    """Verify the required dense topology as well as converted projections."""
    modules = quantized_modules(config, tensors)
    hidden = config["hidden_size"]
    shapes = projection_shapes(config)
    required = {
        "model.embed_tokens.weight": [config["vocab_size"], hidden],
        "model.norm.weight": [hidden],
    }
    if "lm_head.weight" in tensors or not config.get("tie_word_embeddings", False):
        required["lm_head.weight"] = [config["vocab_size"], hidden]
    for layer in range(config["num_hidden_layers"]):
        root = f"model.layers.{layer}"
        for name, (inputs, outputs) in shapes.items():
            prefix = root + "." + name
            if prefix not in modules:
                required[prefix + ".weight"] = [outputs, inputs]
        required[root + ".input_layernorm.weight"] = [hidden]
        required[root + ".post_attention_layernorm.weight"] = [hidden]
        if config["model_type"] == "qwen3":
            head = config.get("head_dim", hidden // config["num_attention_heads"])
            required[root + ".self_attn.q_norm.weight"] = [head]
            required[root + ".self_attn.k_norm.weight"] = [head]
    for name, shape in required.items():
        tensor = tensors.get(name, {})
        if tensor.get("shape") != shape or tensor.get("dtype") not in FLOAT_DTYPES | {"F64"}:
            unsupported(f"Missing or invalid {name}; expected floating-point shape {shape}.")
