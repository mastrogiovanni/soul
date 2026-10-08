"""Model metadata without importing torch or downloading model weights."""

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from huggingface_hub import HfApi, get_token, hf_hub_download, hf_hub_url

from soul.protocol import SoulError
from soul.quantization import (
    DEQUANT_BLOCK,
    quantization_format,
    quantized_modules,
    validate_config,
    validate_tensor_layout,
)

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


@dataclass
class ModelSpec:
    model_id: str
    revision: str
    config: dict
    tensors: dict
    max_context: int
    generation_config: dict = field(default_factory=dict)

    @property
    def layers(self):
        return self.config["num_hidden_layers"]

    @property
    def quantization(self):
        return quantization_format(self.config)

    def tensor_bytes(self, name, element_size):
        tensor = self.tensors[name]
        size = 4 if name.endswith((".qweight", ".qzeros", ".g_idx")) else element_size
        return math.prod(tensor["shape"]) * size

    def quantization_workspace(self):
        modules = quantized_modules(self.config, self.tensors)
        # Integer unpacking, grouped zeros/scales, float tiles and matmul temporaries.
        return max(
            (inputs * min(outputs, DEQUANT_BLOCK) * 32 for inputs, outputs in modules.values()), default=0
        )

    @property
    def key(self):
        return hashlib.sha256(f"{self.model_id}@{self.revision}".encode()).hexdigest()[:24]

    def to_dict(self):
        return asdict(self)

    def names_for(self, start: int, end: int) -> list[str]:
        names = []
        for name in self.tensors:
            match = re.match(r"model\.layers\.(\d+)\.", name)
            if match and start <= int(match[1]) < end:
                names.append(name)
            elif start == 0 and name.startswith("model.embed_tokens."):
                names.append(name)
            elif end == self.layers and name.startswith(("model.norm.", "lm_head.")):
                names.append(name)
        # Tied embeddings must exist on both edge nodes, even if the checkpoint omits lm_head.
        if end == self.layers and "lm_head.weight" not in self.tensors:
            if "model.embed_tokens.weight" not in names:
                names.append("model.embed_tokens.weight")
        return names

    def weight_bytes(self, start, end, element_size):
        return sum(self.tensor_bytes(n, element_size) for n in self.names_for(start, end))

    def kv_per_layer(self, element_size):
        c = self.config
        return (
            2
            * c.get("num_key_value_heads", c["num_attention_heads"])
            * c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])
            * self.max_context
            * element_size
        )


class RangeReader:
    """Strict HTTP ranges. Never silently fetch a multi-GB shard when ranges are ignored."""

    def __init__(self, model_id, revision):
        self.model_id, self.revision = model_id, revision
        self.local = Path(model_id) if Path(model_id).is_dir() else None
        self.client = httpx.Client(timeout=120, follow_redirects=True)

    def close(self):
        self.client.close()

    def read(self, filename: str, start: int, length: int) -> bytes:
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("Invalid checkpoint filename")
        if start < 0 or length < 1:
            raise ValueError("Invalid byte range")
        if self.local:
            root = self.local.resolve()
            target = (root / filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Checkpoint path escapes model directory")
            with target.open("rb") as f:
                f.seek(start)
                data = f.read(length)
        else:
            token = get_token()
            headers = {"Range": f"bytes={start}-{start + length - 1}", "Accept-Encoding": "identity"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            url = hf_hub_url(self.model_id, filename, revision=self.revision)
            # Include the range in the cache key; some CDN/proxy caches otherwise reuse a signed range URL.
            url += f"?soul_range={start}-{start + length - 1}"
            with self.client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{start + length - 1}/"
                if response.status_code != 206 or not response.headers.get("content-range", "").startswith(
                    expected
                ):
                    raise SoulError(
                        "Model host does not honor byte ranges; refusing a full-shard download", 502
                    )
                chunks, count = [], 0
                for chunk in response.iter_bytes():
                    count += len(chunk)
                    if count > length:
                        raise ValueError("Range response exceeds requested size")
                    chunks.append(chunk)
                data = b"".join(chunks)
        if len(data) != length:
            raise ValueError("Truncated tensor range")
        return data

    def header(self, filename):
        size = struct.unpack("<Q", self.read(filename, 0, 8))[0]
        if size > 32 * 1024 * 1024 or size < 2:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(self.read(filename, 8, size))
        result = {}
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            shape, dtype = tensor["shape"], tensor["dtype"]
            a, b = tensor["data_offsets"]
            if dtype not in DTYPE_BYTES or not all(isinstance(x, int) and x > 0 for x in shape):
                raise SoulError(f"Unsupported checkpoint tensor: {name} ({dtype})", 400, "unsupported_model")
            if a < 0 or b - a != math.prod(shape) * DTYPE_BYTES[dtype]:
                raise ValueError(f"Invalid offsets for {name}")
            result[name] = {
                "file": filename,
                "offset": 8 + size + a,
                "length": b - a,
                "dtype": dtype,
                "shape": shape,
            }
        return result


def inspect_model(
    model_id: str, revision: str = "main", max_context: int = 2048, allow_local: bool = False
) -> ModelSpec:
    local = Path(model_id).is_dir()
    if local and not allow_local:
        raise SoulError("Local model paths are disabled", 400, "invalid_model")
    if local:
        directory = Path(model_id).resolve()
        config = json.loads((directory / "config.json").read_text())
        generation_path = directory / "generation_config.json"
        generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}
        index = directory / "model.safetensors.index.json"
        files = (
            sorted(set(json.loads(index.read_text())["weight_map"].values()))
            if index.exists()
            else ["model.safetensors"]
        )
        # Local fixtures are immutable for the lifetime of a controller; metadata pins their identity.
        identity = [
            (name, (directory / name).stat().st_size, (directory / name).stat().st_mtime_ns) for name in files
        ]
        revision = hashlib.sha256(json.dumps([config, identity], sort_keys=True).encode()).hexdigest()
        model_id = str(directory)
    else:
        info = HfApi().model_info(model_id, revision=revision)
        revision = info.sha
        config_path = hf_hub_download(model_id, "config.json", revision=revision)
        config = json.loads(Path(config_path).read_text())
        available = {s.rfilename for s in info.siblings}
        generation_config = {}
        if "generation_config.json" in available:
            generation_path = hf_hub_download(model_id, "generation_config.json", revision=revision)
            generation_config = json.loads(Path(generation_path).read_text())
        if "model.safetensors.index.json" in available:
            index = hf_hub_download(model_id, "model.safetensors.index.json", revision=revision)
            files = sorted(set(json.loads(Path(index).read_text())["weight_map"].values()))
        elif "model.safetensors" in available:
            files = ["model.safetensors"]
        else:
            raise SoulError("A standard safetensors checkpoint is required", 400, "unsupported_model")
    validate_config(config)
    if not 1 <= config.get("num_hidden_layers", 0) <= 512:
        raise SoulError("Invalid layer count", 400, "unsupported_model")
    reader = RangeReader(model_id, revision)
    try:
        tensors = {}
        for filename in files:
            tensors.update(reader.header(filename))
    finally:
        reader.close()
    if "model.embed_tokens.weight" not in tensors:
        raise SoulError(
            "Checkpoint does not match the architecture's tensor layout", 400, "unsupported_model"
        )
    if "lm_head.weight" not in tensors and not config.get("tie_word_embeddings", False):
        raise SoulError("Checkpoint is missing its output head", 400, "unsupported_model")
    validate_tensor_layout(config, tensors)
    context = min(max_context, config.get("max_position_embeddings", max_context))
    return ModelSpec(model_id, revision, config, tensors, context, generation_config)
