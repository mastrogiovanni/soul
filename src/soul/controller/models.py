"""Durable model catalog and proactive disk-shard reconciliation."""

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

from soul.catalog import ModelSpec, inspect_model
from soul.planner import plan
from soul.protocol import ARCHITECTURES, SoulError
from soul.quantization import supports

log = logging.getLogger("soul.models")


class Models:
    def __init__(self, engine):
        self.engine = engine
        self.records = {r["model"]: r for r in engine.store.load("models")}
        # Upgrade existing controllers: deployments become durable preload intentions.
        for deployment in engine.deployments.values():
            self.records.setdefault(
                deployment["model"],
                {
                    "model": deployment["model"],
                    "revision": deployment["revision"],
                    "created": int(time.time()),
                    "status": "pending",
                    "assignments": [],
                },
            )
        for record in self.records.values():
            record["status"] = "pending"
            if "spec" in record:
                spec = self.contextual(ModelSpec(**record["spec"]))
                engine.specs[spec.model_id] = spec
            self.save(record)
        self.changed = asyncio.Event()
        self.task = None

    def contextual(self, spec):
        return replace(
            spec,
            max_context=min(
                self.engine.max_context, spec.config.get("max_position_embeddings", self.engine.max_context)
            ),
        )

    def save(self, record):
        self.engine.store.save("models", record["model"], record)

    def allowed(self, model):
        if Path(model).is_absolute() and not self.engine.allow_local:
            return False
        return not self.engine.allowed_models or model in self.engine.allowed_models

    def adopt_inventory(self, peer, *, wake=True):
        for entry in peer.inventory.values():
            spec = self.contextual(ModelSpec(**entry["spec"]))
            if not self.allowed(spec.model_id) or spec.config.get("model_type") not in ARCHITECTURES:
                continue
            if spec.model_id not in self.records:
                self.remember(spec)
        if wake:
            self.wake()

    def remember(self, spec):
        self.engine.specs[spec.model_id] = spec
        record = self.records.setdefault(
            spec.model_id,
            {
                "model": spec.model_id,
                "created": int(time.time()),
                "status": "pending",
                "assignments": [],
            },
        )
        record.update(spec=spec.to_dict(), revision=spec.revision)
        self.save(record)
        return record

    async def register(self, model, revision=None):
        if not self.allowed(model):
            raise SoulError("Model is not in the controller allowlist", 403, "model_not_allowed")
        existing = self.engine.specs.get(model)
        if revision and existing and revision != existing.revision:
            raise SoulError(
                "This model is already pinned to a different revision on this controller.",
                409,
                "revision_conflict",
            )
        if model not in self.engine.specs:
            revision = revision or self.records.get(model, {}).get("revision", "main")
            try:
                spec = await asyncio.to_thread(
                    inspect_model, model, revision, self.engine.max_context, self.engine.allow_local
                )
            except SoulError:
                raise
            except Exception as exc:
                raise SoulError(f"Cannot inspect model: {str(exc)[:300]}", 400, "invalid_model") from exc
            return self.remember(spec)
        return self.records[model]

    def placements(self, model, *, cached_only):
        spec = self.engine.specs.get(model)
        if not spec or not self.allowed(model):
            raise SoulError("Model has not been preloaded", 404, "model_not_preloaded")
        offers = []
        for peer in self.engine.nodes.values():
            entry = peer.inventory.get(spec.key)
            if not peer.connected or not entry or not supports(peer.capabilities, spec):
                continue
            offer = entry.get("offer", {})
            if offer.get("context") != spec.max_context:
                continue
            costs = offer["costs"]
            if cached_only:
                costs = [
                    [
                        cost if offer["missing_bytes"][s][i] == 0 else float("inf")
                        for i, cost in enumerate(row)
                    ]
                    for s, row in enumerate(costs)
                ]
            offers.append({**offer, "costs": costs, "node_id": peer.id})
        # Cache order also handles shards returning in a different capacity order.
        ordered = []
        for offer in offers:
            cached_starts = [s for s, row in enumerate(offer["missing_bytes"]) if any(n == 0 for n in row)]
            ordered.append({**offer, "order": min(cached_starts, default=0)})
        candidates = []
        for options in (offers, ordered):
            with contextlib.suppress(SoulError):
                candidates.append(plan(spec.layers, options, self.engine.max_layers_per_node))
        if not candidates:
            quantization_hint = (
                " Quantized models require updated nodes with the matching format capability."
                if spec.quantization != "none"
                else ""
            )
            raise SoulError(
                "Model is not executable: insufficient connected memory, disk space, or cached layers. "
                "Preload the model or connect more nodes." + quantization_hint,
                503,
                "model_not_ready",
            )
        by_id = {o["node_id"]: o for o in offers}
        return min(
            candidates,
            key=lambda assignments: (
                sum(by_id[a.node_id]["missing_bytes"][a.start][a.end - a.start - 1] for a in assignments),
                len(assignments),
            ),
        )

    def executable(self, model):
        try:
            return self.placements(model, cached_only=True)
        except SoulError:
            return None

    def views(self):
        result = []
        for model, record in self.records.items():
            spec = self.engine.specs.get(model)
            placement = self.executable(model)
            deployment = self.engine.deployments.get(model)
            ready = bool(deployment and self.engine.valid(deployment))
            status = "ready" if ready else "available" if placement else record["status"]
            if not placement and status in ("ready", "available"):
                status = "partial"
            result.append(
                {
                    "model": model,
                    "revision": record.get("revision", ""),
                    "created": record["created"],
                    "architecture": spec.config["model_type"] if spec else "unknown",
                    "quantization": spec.quantization if spec else "unknown",
                    "layers": spec.layers if spec else 0,
                    "context": spec.max_context if spec else 0,
                    "status": status,
                    "executable": bool(placement),
                    "error": record.get("error"),
                    "assignments": deployment["assignments"]
                    if ready
                    else [asdict(a) for a in placement]
                    if placement
                    else record.get("assignments", []),
                }
            )
        return result

    async def refresh_offers(self, spec):
        peers = [p for p in self.engine.nodes.values() if p.connected and supports(p.capabilities, spec)]
        await asyncio.gather(
            *(p.call("offer", {"spec": spec.to_dict()}, timeout=self.engine.rpc_timeout) for p in peers),
            return_exceptions=True,
        )

    async def prepare(self, record):
        model = record["model"]
        spec = self.engine.specs[model]
        try:
            await self.refresh_offers(spec)
            assignments = self.placements(model, cached_only=False)
            record.update(status="downloading", assignments=[asdict(a) for a in assignments], error=None)
            self.save(record)
            pending = []
            for assignment in assignments:
                peer = self.engine.nodes[assignment.node_id]
                offer = peer.inventory[spec.key]["offer"]
                if offer["missing_bytes"][assignment.start][assignment.end - assignment.start - 1]:
                    pending.append(
                        peer.call(
                            "preload",
                            {
                                "spec": spec.to_dict(),
                                "start": assignment.start,
                                "end": assignment.end,
                            },
                            timeout=self.engine.load_timeout,
                        )
                    )
            results = await asyncio.gather(*pending, return_exceptions=True)
            error = next((r for r in results if isinstance(r, BaseException)), None)
            if error:
                raise error
            self.placements(model, cached_only=True)
            record["status"] = "available"
        except Exception as exc:
            record.update(status="partial", error=str(exc)[:500])
            raise
        finally:
            self.save(record)
        return next(v for v in self.views() if v["model"] == model)

    async def preload(self, model, revision=None):
        async with self.engine.gate:
            record = await self.register(model, revision)
            return await self.prepare(record)

    def wake(self):
        self.changed.set()

    def start(self):
        self.task = asyncio.create_task(self.reconcile())
        self.wake()

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    async def reconcile(self):
        while True:
            await self.changed.wait()
            self.changed.clear()
            # Use the same gate as inference: no shard replacement during a request.
            async with self.engine.gate:
                for model in list(self.records):
                    if not self.allowed(model):
                        continue
                    try:
                        record = await self.register(model)
                        await self.prepare(record)
                    except Exception as exc:
                        record = self.records[model]
                        record.update(status="partial", error=str(exc)[:500])
                        self.save(record)
                        log.info("Model %s awaits capacity or preload retry: %s", model, exc)
