import asyncio
import contextlib
import time
import uuid
from dataclasses import asdict

import anyio

from soul.controller.models import Models
from soul.protocol import SoulError


class TextBuffer:
    """Hold incomplete UTF-8 and stop-string prefixes until they can be emitted safely."""

    def __init__(self, stops):
        self.stops = stops
        self.emitted = ""
        self.matched = None

    def update(self, text, final=False):
        found = [(text.find(s), s) for s in self.stops if s in text]
        if found:
            end, self.matched = min(found, key=lambda x: x[0])
            text = text[:end]
            final = True
        if not final:
            text = text[: text.find("\ufffd")] if "\ufffd" in text else text
            hold = max((n for s in self.stops for n in range(1, len(s)) if text.endswith(s[:n])), default=0)
            if hold:
                text = text[:-hold]
        if not text.startswith(self.emitted):
            raise SoulError("Tokenizer rewrote already streamed text", 500, "decode_error")
        delta = text[len(self.emitted) :]
        self.emitted = text
        return delta


class Engine:
    def __init__(
        self,
        store,
        *,
        max_context=2048,
        max_queue=32,
        queue_timeout=300,
        rpc_timeout=120,
        load_timeout=1800,
        allow_local=False,
        max_layers_per_node=None,
        allowed_models=None,
    ):
        self.store = store
        self.nodes = {}
        self.old_nodes = {
            n["node_id"]: {**n, "connected": False, "status": "offline"} for n in store.load("nodes")
        }
        self.deployments = {d["model"]: {**d, "status": "stale"} for d in store.load("deployments")}
        self.specs = {}
        self.gate = asyncio.Lock()
        self.waiting = 0
        self.active = None
        self.max_context, self.max_queue, self.queue_timeout = max_context, max_queue, queue_timeout
        self.rpc_timeout, self.load_timeout = rpc_timeout, load_timeout
        self.allow_local, self.max_layers_per_node = allow_local, max_layers_per_node
        self.allowed_models = set(allowed_models or [])
        self.models = Models(self)

    async def remove_node(self, node_id):
        peer = self.nodes.get(node_id)
        if peer and peer.connected:
            raise SoulError("Connected nodes cannot be removed", 409, "node_connected")
        if peer is None and node_id not in self.old_nodes:
            raise SoulError("Unknown node", 404, "not_found")
        self.invalidate(node_id)
        self.nodes.pop(node_id, None)
        self.old_nodes.pop(node_id, None)
        self.store.delete_node(node_id)
        for record in self.models.records.values():
            record["assignments"] = [a for a in record.get("assignments", []) if a["node_id"] != node_id]
            self.models.save(record)
        self.models.wake()
        return {"removed": node_id}

    def invalidate(self, node_id):
        for deployment in self.deployments.values():
            if any(a["node_id"] == node_id for a in deployment["assignments"]):
                deployment["status"] = "stale"
                self.store.save("deployments", deployment["model"], deployment)

    def snapshot(self):
        nodes = {**self.old_nodes, **{id: n.snapshot() for id, n in self.nodes.items()}}
        return {
            "nodes": list(nodes.values()),
            "models": self.models.views(),
            "metrics": self.store.stats(),
            "queue_depth": self.waiting,
            "active_request": self.active,
            "max_context": self.max_context,
        }

    def valid(self, deployment):
        return deployment["status"] == "ready" and all(
            (peer := self.nodes.get(a["node_id"])) and peer.connected and peer.generation == a["generation"]
            for a in deployment["assignments"]
        )

    async def ensure_model(self, model_id):
        if self.allowed_models and model_id not in self.allowed_models:
            raise SoulError("Model is not in the controller allowlist", 403, "model_not_allowed")
        existing = self.deployments.get(model_id)
        if existing and self.valid(existing):
            return existing
        if model_id not in self.specs:
            raise SoulError(
                "Preload this model through /admin/models/preload before inference",
                404,
                "model_not_preloaded",
            )
        spec = self.specs[model_id]
        await self.models.refresh_offers(spec)
        assignments = self.models.placements(model_id, cached_only=True)
        selected_ids = {a.node_id for a in assignments}
        retired = [
            d
            for d in self.deployments.values()
            if any(a["node_id"] in selected_ids for a in d["assignments"])
        ]
        # Free orphan stages of a pipeline being replaced, including stages on nodes
        # not needed by the new model. The generation lock excludes active sessions.
        orphan_peers = {}
        for old in retired:
            for assignment in old["assignments"]:
                peer = self.nodes.get(assignment["node_id"])
                if (
                    peer
                    and peer.connected
                    and peer.id not in selected_ids
                    and peer.generation == assignment["generation"]
                    and peer.status.get("epoch") == old["epoch"]
                ):
                    orphan_peers[peer.id] = peer
        epoch = uuid.uuid4().hex
        deployment = {
            "model": model_id,
            "revision": spec.revision,
            "architecture": spec.config["model_type"],
            "layers": spec.layers,
            "context": spec.max_context,
            "status": "loading",
            "epoch": epoch,
            "assignments": [
                {**asdict(a), "generation": self.nodes[a.node_id].generation} for a in assignments
            ],
        }
        # Every affected old pipeline is invalid before the first node is repurposed.
        for assignment in assignments:
            self.invalidate(assignment.node_id)
        self.deployments[model_id] = deployment
        self.store.save("deployments", model_id, deployment)
        selected = [self.nodes[a.node_id] for a in assignments]
        try:
            await asyncio.gather(
                *(p.call("unload", timeout=10) for p in orphan_peers.values()), return_exceptions=True
            )
            results = await asyncio.gather(
                *(
                    peer.call(
                        "load",
                        {"spec": spec.to_dict(), "start": a.start, "end": a.end, "epoch": epoch},
                        timeout=self.load_timeout,
                    )
                    for peer, a in zip(selected, assignments, strict=True)
                ),
                return_exceptions=True,
            )
            error = next((r for r in results if isinstance(r, BaseException)), None)
            if error:
                raise error
            if any(not p.connected or self.nodes.get(p.id) is not p for p in selected):
                raise SoulError("Node reconnected while the model was loading")
            deployment["status"] = "ready"
        except BaseException:
            deployment["status"] = "error"
            # A load may outlive a cancelled RPC while executing in a worker thread.
            # Fence every selected connection; reconnect unloads before new work.
            with anyio.CancelScope(shield=True):
                for peer in selected:
                    peer.disconnect("Deployment interrupted")
                    with contextlib.suppress(Exception):
                        await peer.ws.close(code=1011)
            raise
        finally:
            self.store.save("deployments", model_id, deployment)
        return deployment

    async def load_model(self, model):
        await self.models.preload(model)
        async with self.gate:
            return await self.ensure_model(model)

    async def unload_model(self, model):
        async with self.gate:
            deployment = self.deployments.get(model)
            if not deployment:
                raise SoulError("Unknown model", 404, "not_found")
            # Only unload still-owned stages; stale records may refer to repurposed nodes.
            owned = [
                p
                for a in deployment["assignments"]
                if (p := self.nodes.get(a["node_id"]))
                and p.connected
                and p.generation == a["generation"]
                and p.status.get("epoch") == deployment["epoch"]
            ]
            results = await asyncio.gather(
                *(p.call("unload", timeout=10) for p in owned), return_exceptions=True
            )
            failure = next((r for r in results if isinstance(r, BaseException)), None)
            if failure:
                deployment["status"] = "stale"
                self.store.save("deployments", model, deployment)
                raise SoulError("Some nodes could not acknowledge unload; their deployment was invalidated")
            deployment["status"] = "unloaded"
            self.store.save("deployments", model, deployment)
            return deployment

    async def generate(self, request, request_id):
        if self.waiting >= self.max_queue:
            raise SoulError("Inference queue is full", 503, "overloaded")
        self.store.request(request_id, request["model"])
        self.waiting += 1
        started = time.monotonic()
        acquired = False
        deployment = None
        peers = []
        input_tokens = output_tokens = 0
        status, error = "failed", None
        try:
            try:
                await asyncio.wait_for(self.gate.acquire(), self.queue_timeout)
                acquired = True
            except TimeoutError as exc:
                raise SoulError("Inference queue wait timed out", 503, "overloaded") from exc
            finally:
                self.waiting -= 1
            self.active = request_id
            self.store.update_request(request_id, "running")
            deployment = await self.ensure_model(request["model"])
            peers = [self.nodes[a["node_id"]] for a in deployment["assignments"]]
            epoch = deployment["epoch"]
            common = {"epoch": epoch, "session_id": request_id}
            tokens = await peers[0].call("tokenize", {**request, "epoch": epoch}, timeout=self.rpc_timeout)
            ids, eos = tokens["input_ids"], tokens["eos_ids"]
            input_tokens = len(ids)
            if input_tokens + request["max_tokens"] > deployment["context"]:
                raise SoulError(
                    f"Prompt ({input_tokens}) + output ({request['max_tokens']}) exceeds context ({deployment['context']})",
                    400,
                    "context_length_exceeded",
                )
            yield {"kind": "start", "input_tokens": input_tokens}

            async def step(input_ids, position, sample=True):
                if not self.valid(deployment):
                    raise SoulError("Pipeline changed during inference; retry the request")
                data = {
                    **common,
                    "input_ids": input_ids,
                    "position": position,
                    "sample": sample,
                    "temperature": request["temperature"],
                    "top_p": request["top_p"],
                    "seed": request.get("seed"),
                }
                for peer in peers:
                    result = await peer.call("forward", data, timeout=self.rpc_timeout)
                    data.pop("input_ids", None)
                    data.pop("hidden", None)
                    data.update(result)
                return result

            # Chunked prefill keeps activation frames and GPU scratch space bounded.
            for offset in range(0, input_tokens, 128):
                chunk = ids[offset : offset + 128]
                result = await step(chunk, offset, sample=offset + len(chunk) == input_tokens)
            generated = []
            buffer = TextBuffer(request.get("stop", []))
            finish, full_text = "length", ""
            for index in range(request["max_tokens"]):
                token = result["token_id"]
                generated.append(token)
                output_tokens += 1
                final = token in eos or index == request["max_tokens"] - 1
                if token in eos:
                    finish = "stop"
                full_text = (
                    await peers[0].call("decode", {**common, "ids": generated}, timeout=self.rpc_timeout)
                )["text"]
                delta = buffer.update(full_text, final=final)
                if delta:
                    yield {"kind": "text", "text": delta}
                if buffer.matched or token in eos:
                    finish = "stop"
                    break
                if index < request["max_tokens"] - 1:
                    result = await step([token], input_tokens + index)
            status = "completed"
            yield {
                "kind": "done",
                "finish_reason": finish,
                "stop_sequence": buffer.matched,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        except (asyncio.CancelledError, GeneratorExit):
            if status != "completed":
                status, error = "cancelled", "Client disconnected"
            raise
        except Exception as exc:
            error = str(exc)[:500]
            if deployment and (not isinstance(exc, SoulError) or exc.status >= 500):
                deployment["status"] = "stale"
                self.store.save("deployments", deployment["model"], deployment)
            raise
        finally:

            async def cleanup():
                try:
                    if peers and deployment:
                        results = await asyncio.gather(
                            *(
                                p.call(
                                    "release",
                                    {"epoch": deployment["epoch"], "session_id": request_id},
                                    timeout=10,
                                )
                                for p in peers
                            ),
                            return_exceptions=True,
                        )
                        if any(isinstance(r, BaseException) for r in results):
                            deployment["status"] = "stale"
                            self.store.save("deployments", deployment["model"], deployment)
                finally:
                    if acquired:
                        self.active = None
                        self.gate.release()
                    self.store.update_request(
                        request_id,
                        status,
                        duration=time.monotonic() - started,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        error=error,
                    )

            # Starlette uses AnyIO level cancellation; asyncio.shield alone does not
            # protect subsequent awaits inside an already-cancelled AnyIO scope.
            with anyio.CancelScope(shield=True):
                task = asyncio.create_task(cleanup())
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    await asyncio.shield(task)
                    raise
