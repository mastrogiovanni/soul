"""Atomic per-tensor disk cache, populated using bounded HTTP byte ranges."""

import hashlib
import json
import os
import struct
from pathlib import Path

from safetensors import safe_open

from soul.catalog import ModelSpec, RangeReader
from soul.protocol import SoulError


class WeightStore:
    def __init__(self, root: Path, spec: ModelSpec):
        self.spec = spec
        self.root = root / "weights" / spec.key
        self.root.mkdir(parents=True, exist_ok=True)

    def remember(self):
        temporary = self.root / "manifest.partial"
        temporary.write_text(json.dumps(self.spec.to_dict()))
        temporary.replace(self.root / "manifest.json")

    def available(self):
        result = []
        for name in self.spec.tensors:
            path = self.path(name)
            checksum = path.with_suffix(".sha256")
            if path.exists() and checksum.exists() and self.digest(path) == checksum.read_text():
                result.append(name)
        return result

    def require(self, names):
        if not set(names).issubset(self.available()):
            raise SoulError("Assigned tensors are missing or corrupt; preload the model first", 503)

    @classmethod
    def manifests(cls, root):
        for path in sorted((root / "weights").glob("*/manifest.json")):
            try:
                spec = ModelSpec(**json.loads(path.read_text()))
                if path.parent.name == spec.key:
                    yield spec
            except (OSError, ValueError, TypeError, KeyError):
                continue

    def path(self, name):
        return self.root / (hashlib.sha256(name.encode()).hexdigest() + ".safetensors")

    def ensure(self, names: list[str], progress=lambda *_: None):
        # Persist metadata before downloading so interrupted shards can be rediscovered.
        self.remember()
        reader = RangeReader(self.spec.model_id, self.spec.revision)
        total = sum(self.spec.tensors[n]["length"] for n in names)
        completed = 0
        try:
            for name in names:
                tensor = self.spec.tensors[name]
                path = self.path(name)
                checksum = path.with_suffix(".sha256")
                if path.exists() and checksum.exists() and self.digest(path) == checksum.read_text():
                    completed += tensor["length"]
                    progress(completed, total, name)
                    continue
                header = json.dumps(
                    {
                        "weight": {
                            "dtype": tensor["dtype"],
                            "shape": tensor["shape"],
                            "data_offsets": [0, tensor["length"]],
                        }
                    },
                    separators=(",", ":"),
                ).encode()
                header += b" " * (-len(header) % 8)
                temporary = path.with_suffix(".partial")
                try:
                    with temporary.open("wb") as f:
                        f.write(struct.pack("<Q", len(header)))
                        f.write(header)
                        for offset in range(0, tensor["length"], 8 * 1024 * 1024):
                            length = min(8 * 1024 * 1024, tensor["length"] - offset)
                            f.write(reader.read(tensor["file"], tensor["offset"] + offset, length))
                            completed += length
                            progress(completed, total, name)
                        f.flush()
                        os.fsync(f.fileno())
                    temporary.replace(path)
                    checksum.write_text(self.digest(path))
                finally:
                    temporary.unlink(missing_ok=True)
        finally:
            reader.close()

    @staticmethod
    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    def tensor(self, name, device, dtype):
        with safe_open(self.path(name), framework="pt", device="cpu") as f:
            # A copy detaches CPU tensors from mmap too; no disk-backed writable parameters.
            return f.get_tensor("weight").to(device=device, dtype=dtype, copy=True)
