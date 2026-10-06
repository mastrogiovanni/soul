"""Versioned, language-neutral MessagePack RPC; tensors are opaque to the controller."""

import msgpack

VERSION = 1
MAX_FRAME = 64 * 1024 * 1024
ARCHITECTURES = ("llama", "mistral", "qwen2", "qwen3")


class SoulError(Exception):
    def __init__(self, message: str, status: int = 503, code: str = "service_unavailable"):
        super().__init__(message)
        self.status = status
        self.code = code


def pack(message: dict) -> bytes:
    payload = msgpack.packb(message, use_bin_type=True)
    if len(payload) > MAX_FRAME:
        raise SoulError("RPC frame exceeds the 64 MiB limit", 413, "payload_too_large")
    return payload


def unpack(payload: bytes) -> dict:
    if len(payload) > MAX_FRAME:
        raise ValueError("RPC frame exceeds limit")
    message = msgpack.unpackb(payload, raw=False, strict_map_key=True)
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ValueError("Expected a typed RPC map")
    return message
