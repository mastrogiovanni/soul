"""Live Hub discovery and explicit checkpoint verification, without registering models."""

import asyncio
import json
from collections import OrderedDict
from urllib.parse import parse_qs, quote, urlsplit

import httpx
from huggingface_hub import get_token
from huggingface_hub.utils import validate_repo_id

from soul.catalog import inspect_model
from soul.protocol import ARCHITECTURES, SoulError
from soul.quantization import QUANTIZATIONS, supports, validate_config

FAMILIES = (
    ("llama", "Llama / SmolLM2"),
    ("mistral", "Mistral"),
    ("qwen2", "Qwen2 / Qwen2.5"),
    ("qwen3", "Qwen3 dense"),
    ("qwen3_5", "Qwen3.5"),
    ("qwen3_moe", "Qwen3 MoE"),
    ("qwen3_5_moe", "Qwen3.5 MoE"),
    ("qwen3_next", "Qwen3 Next"),
    ("qwen3_vl", "Qwen3 VL"),
    ("gemma", "Gemma"),
    ("gemma2", "Gemma 2"),
    ("gemma3", "Gemma 3"),
    ("mixtral", "Mixtral"),
    ("deepseek_v3", "DeepSeek V3"),
    ("gpt_oss", "GPT OSS"),
    ("phi3", "Phi 3 / 4"),
    ("mamba", "Mamba"),
)


def support_matrix():
    return {
        "families": [{"id": id, "name": name, "supported": id in ARCHITECTURES} for id, name in FAMILIES],
        "formats": ["none", *QUANTIZATIONS],
    }


class ModelBrowser:
    def __init__(self):
        self.configs = OrderedDict()

    async def search(self, query="", family="", cursor="", sort="downloads"):
        params = {"search": query, "limit": 24, "sort": sort, "direction": -1, "full": "true"}
        if family:
            params["filter"] = family
        if cursor:
            params["cursor"] = cursor
        token = get_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
                response = await client.get("https://huggingface.co/api/models", params=params)
                response.raise_for_status()
                items = response.json()
                if not isinstance(items, list):
                    raise ValueError("Invalid Hub response")
                semaphore = asyncio.Semaphore(4)

                async def describe(item):
                    async with semaphore:
                        return await self.describe(client, item)

                rows = await asyncio.gather(
                    *(describe(item) for item in items[:24] if not item.get("private"))
                )
                # Only pass the opaque cursor back. Never follow an upstream/user-provided URL.
                next_url = response.links.get("next", {}).get("url", "")
                next_cursor = parse_qs(urlsplit(next_url).query).get("cursor", [None])[0]
                return {"models": rows, "next_cursor": next_cursor, **support_matrix()}
        except (httpx.HTTPError, ValueError) as exc:
            raise SoulError(
                "Hugging Face search is unavailable. Try again shortly; cached models are unaffected.",
                502,
                "hub_unavailable",
            ) from exc

    async def describe(self, client, item):
        model = item["id"]
        validate_repo_id(model)
        revision = item.get("sha")
        files = {f["rfilename"] for f in item.get("siblings", [])}
        row = {
            "model": model,
            "revision": revision,
            "downloads": item.get("downloads", 0),
            "gated": bool(item.get("gated")),
            "architecture": "unknown",
            "quantization": "unknown",
            "status": "unknown",
            "reason": "Check this checkpoint to determine compatibility.",
            "url": "https://huggingface.co/" + quote(model, safe="/"),
        }
        if "model.safetensors" not in files and "model.safetensors.index.json" not in files:
            row.update(
                status="unsupported",
                reason="A standard safetensors checkpoint is required. GGUF and adapter-only repositories are not supported.",
            )
            return row
        if "config.json" not in files or not revision:
            return row
        cache_key = (model, revision)
        config = self.configs.get(cache_key)
        if config is None:
            try:
                url = f"https://huggingface.co/{quote(model, safe='/')}/resolve/{quote(revision, safe='')}/config.json"
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > 512 * 1024:
                            raise ValueError("Config too large")
                config = json.loads(data)
                if not isinstance(config, dict):
                    raise ValueError("Invalid config")
                self.configs[cache_key] = config
                while len(self.configs) > 256:
                    self.configs.popitem(last=False)
            except (httpx.HTTPError, ValueError):
                row["reason"] = (
                    "Cannot read config.json. For gated/private models, grant access and set HF_TOKEN on the controller and nodes."
                )
                return row
        row["architecture"] = config.get("model_type", "unknown")
        q = config.get("quantization_config")
        row["quantization"] = str(q.get("quant_method", "unknown")) if isinstance(q, dict) else "none"
        try:
            row["quantization"] = validate_config(config)
            row.update(
                status="candidate",
                reason="Architecture and format are supported. Check tensor layout before preparing; node capacity is checked during preparation.",
            )
        except SoulError as exc:
            row.update(status="unsupported", reason=str(exc))
        return row

    async def check(self, engine, model, revision="main"):
        if not engine.models.allowed(model):
            raise SoulError("Model is not in the controller allowlist", 403, "model_not_allowed")
        try:
            spec = await asyncio.to_thread(
                inspect_model, model, revision, engine.max_context, engine.allow_local
            )
        except SoulError:
            raise
        except Exception as exc:
            raise SoulError(f"Cannot inspect checkpoint: {str(exc)[:300]}", 400, "invalid_model") from exc
        return {
            "model": model,
            "revision": spec.revision,
            "architecture": spec.config["model_type"],
            "quantization": spec.quantization,
            "status": "supported",
            "layers": spec.layers,
            "context": spec.max_context,
            "weight_bytes_fp16": spec.weight_bytes(0, spec.layers, 2),
            "weight_bytes_fp32": spec.weight_bytes(0, spec.layers, 4),
            "download_bytes": sum(t["length"] for t in spec.tensors.values()),
            "compatible_nodes": sum(
                p.connected and supports(p.capabilities, spec) for p in engine.nodes.values()
            ),
            "reason": "Checkpoint layout verified. Weight estimates exclude KV cache, workspace and placement overhead. Prepare to check capacity and download assigned shards.",
        }
