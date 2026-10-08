"""Durable cache removal intentions, including disconnected nodes."""

import time

from soul.catalog import ModelSpec
from soul.protocol import SoulError


class CacheManager:
    def __init__(self, engine):
        self.engine = engine
        self.deleted = {r["model"]: r for r in engine.store.load("deleted_models")}
        self.nodes = {r["node_id"]: r for r in engine.store.load("node_caches")}
        # A crash after writing a tombstone must never restore the old catalog.
        for model in self.deleted:
            engine.store.forget_model(model)
            engine.deployments.pop(model, None)

    def paused(self, node_id):
        return node_id in self.nodes

    def save_model(self, record):
        self.engine.store.save("deleted_models", record["model"], record)

    def save_node(self, record):
        self.engine.store.save("node_caches", record["node_id"], record)

    def observe(self, peer):
        for entry in peer.inventory.values():
            record = self.deleted.get(entry["spec"]["model_id"])
            if record and peer.id not in record["pending_nodes"]:
                record["pending_nodes"].append(peer.id)
                self.save_model(record)

    def check_registration(self, model):
        if self.deleted.get(model, {}).get("pending_nodes"):
            raise SoulError(
                "Model deletion is still pending on nodes. Reconnect/update them and retry cleanup before adding it again.",
                409,
                "deletion_pending",
            )

    def restore(self, model):
        self.engine.store.restore_model(model)
        self.deleted.pop(model, None)

    def snapshot(self):
        return [dict(r) for r in self.deleted.values() if r["pending_nodes"]]

    def node_view(self, node_id):
        record = self.nodes.get(node_id, {})
        return {
            "cache_paused": bool(record),
            "cache_clear_pending": record.get("pending", False),
            "cache_error": record.get("error"),
        }

    async def delete_model(self, model):
        async with self.engine.gate:
            if model not in self.engine.models.records and model not in self.deleted:
                raise SoulError("Unknown model", 404, "not_found")
            first_deletion = model not in self.deleted
            record = self.deleted.setdefault(
                model,
                {
                    "model": model,
                    "created": int(time.time()),
                    "pending_nodes": [],
                    "errors": {},
                    "freed_bytes": 0,
                    "keys": [],
                },
            )
            # Ask every known node, including caches whose manifest is damaged
            # and therefore absent from the advertised inventory.
            targets = set(record["pending_nodes"])
            if first_deletion:
                targets |= self.engine.nodes.keys() | self.engine.old_nodes.keys()
            keys = set(record.get("keys", []))
            spec = self.engine.specs.get(model)
            if spec:
                keys.add(spec.key)
            for peer in self.engine.nodes.values():
                cached_keys = {
                    ModelSpec(**entry["spec"]).key
                    for entry in peer.inventory.values()
                    if entry["spec"]["model_id"] == model
                }
                if cached_keys:
                    targets.add(peer.id)
                    keys.update(cached_keys)
            record["pending_nodes"] = sorted(targets)
            record["keys"] = sorted(keys)
            self.save_model(record)  # Persist the tombstone before sending destructive RPCs.
            self.engine.store.forget_model(model)
            self.engine.models.records.pop(model, None)
            self.engine.specs.pop(model, None)
            self.engine.deployments.pop(model, None)
            await self.reconcile()
            self.engine.models.wake()
            return {"deleted": model, **record}

    async def clear_node(self, node_id):
        async with self.engine.gate:
            if node_id not in self.engine.nodes and node_id not in self.engine.old_nodes:
                raise SoulError("Unknown node", 404, "not_found")
            record = {"node_id": node_id, "pending": True, "error": None, "freed_bytes": 0}
            self.nodes[node_id] = record
            self.save_node(record)
            self.engine.invalidate(node_id)
            await self.reconcile()
            self.engine.models.wake()
            return {**record, "cache_paused": True}

    async def resume_node(self, node_id):
        async with self.engine.gate:
            if node_id not in self.engine.nodes and node_id not in self.engine.old_nodes:
                raise SoulError("Unknown node", 404, "not_found")
            if self.nodes.get(node_id, {}).get("pending"):
                raise SoulError(
                    "Cache clearing must finish before downloads can resume", 409, "cleanup_pending"
                )
            self.engine.store.resume_node(node_id)
            self.nodes.pop(node_id, None)
            self.engine.models.wake()
            return {"node_id": node_id, "cache_paused": False}

    async def retry(self):
        async with self.engine.gate:
            await self.reconcile()
            self.engine.models.wake()
            return {"pending_models": self.snapshot(), "nodes": list(self.nodes.values())}

    async def call(self, peer, method, args=None):
        if not peer.capabilities.get("cache_management"):
            raise SoulError(
                "Update and restart this node to enable cache deletion", 409, "node_upgrade_required"
            )
        result = await peer.call(method, args, timeout=self.engine.load_timeout)
        if not peer.connected or self.engine.nodes.get(peer.id) is not peer:
            raise SoulError("Node reconnected during cache deletion; cleanup will retry")
        return result

    async def reconcile(self):
        # Callers hold the same gate as inference and preparation. Intentions stay
        # pending until the current connection acknowledges deletion; cancellation
        # or a restart cannot allow downloads to race an unfinished removal.
        for node_id, record in list(self.nodes.items()):
            peer = self.engine.nodes.get(node_id)
            if not record["pending"] or not peer or not peer.connected:
                continue
            try:
                result = await self.call(peer, "clear_cache")
                record.update(pending=False, error=None, freed_bytes=result["freed_bytes"])
                for deletion in self.deleted.values():
                    if node_id in deletion["pending_nodes"]:
                        deletion["pending_nodes"].remove(node_id)
                        deletion["errors"].pop(node_id, None)
                        self.save_model(deletion)
            except Exception as exc:
                record["error"] = str(exc)[:500]
            finally:
                self.save_node(record)
        for model, record in list(self.deleted.items()):
            for node_id in list(record["pending_nodes"]):
                peer = self.engine.nodes.get(node_id)
                if not peer or not peer.connected:
                    continue
                try:
                    result = await self.call(
                        peer, "delete_model", {"model": model, "keys": record.get("keys", [])}
                    )
                    record["pending_nodes"].remove(node_id)
                    record["errors"].pop(node_id, None)
                    record["freed_bytes"] += result["freed_bytes"]
                except Exception as exc:
                    record["errors"][node_id] = str(exc)[:500]
                finally:
                    self.save_model(record)
