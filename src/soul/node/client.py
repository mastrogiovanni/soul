import asyncio
import base64
import contextlib
import json
import logging
import random
import socket
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import websockets

from soul.catalog import ModelSpec
from soul.node.runtime import Runtime
from soul.protocol import MAX_FRAME, VERSION, SoulError, pack, unpack

log = logging.getLogger("soul.node")


async def in_thread(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await task
        raise


def identity(cache_dir: Path, node_id=None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "identity.json"
    if path.exists():
        saved = json.loads(path.read_text())["node_id"]
        if node_id and saved != node_id:
            raise ValueError("This cache directory belongs to another node ID; use a separate --cache-dir")
        return saved
    node_id = node_id or str(uuid.uuid4())
    path.write_text(json.dumps({"node_id": node_id}))
    return node_id


class NodeClient:
    def __init__(
        self,
        controller,
        token,
        cache_dir,
        node_id=None,
        name=None,
        ingress_username=None,
        ingress_password=None,
        **runtime_options,
    ):
        parsed = urlparse(controller)
        if parsed.scheme not in ("http", "https", "ws", "wss") or not parsed.netloc:
            raise ValueError("Controller must be an http(s) or ws(s) URL")
        self.url = urlunparse(
            (
                "wss" if parsed.scheme in ("https", "wss") else "ws",
                parsed.netloc,
                parsed.path.rstrip("/") + "/internal/nodes",
                "",
                "",
                "",
            )
        )
        self.node_id = identity(cache_dir, node_id)
        self.name = name or socket.gethostname()
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}", "X-Soul-Node-Token": token}
        if ingress_username is not None or ingress_password is not None:
            if not ingress_username or not ingress_password or ":" in ingress_username:
                raise ValueError(
                    "Set both SOUL_INGRESS_USERNAME and SOUL_INGRESS_PASSWORD; username cannot contain ':'"
                )
            credential = base64.b64encode(f"{ingress_username}:{ingress_password}".encode()).decode()
            self.headers["Authorization"] = f"Basic {credential}"
        self.runtime = Runtime(cache_dir, **runtime_options)
        self.status = {"status": "idle"}
        self.initial_inventory = []
        handler = RotatingFileHandler(cache_dir / "node.log", maxBytes=5 * 1024**2, backupCount=2)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)

    async def serve_connection(self, ws):
        send_lock = asyncio.Lock()

        async def send(message):
            async with send_lock:
                await ws.send(pack(message))

        await send(
            {
                "type": "hello",
                "version": VERSION,
                "node_id": self.node_id,
                "name": self.name,
                "capabilities": self.runtime.capabilities(),
                "inventory": self.initial_inventory,
            }
        )
        reply = unpack(await ws.recv())
        if reply["type"] != "welcome":
            raise ValueError("Controller refused the handshake")
        # Reconnection always starts without KV state or an authoritative assignment.
        await asyncio.to_thread(self.runtime.unload)
        self.status = {"status": "idle"}
        loop = asyncio.get_running_loop()

        def progress(done, total, tensor):
            self.status = {
                **self.status,
                "status": "downloading",
                "downloaded_bytes": done,
                "total_bytes": total,
                "tensor": tensor,
            }
            future = asyncio.run_coroutine_threadsafe(send({"type": "status", **self.status}), loop)
            try:
                future.result(timeout=5)
            except Exception:
                # Stop an abandoned download at its next bounded chunk, rather than
                # finishing a multi-GB assignment after the controller has fenced it.
                future.cancel()
                raise

        async def heartbeat():
            while True:
                await send({"type": "heartbeat", **self.status, "time": time.time()})
                await asyncio.sleep(5)

        task = asyncio.create_task(heartbeat())
        try:
            async for data in ws:
                request = unpack(data)
                if request["type"] != "rpc":
                    raise ValueError("Expected an RPC request")
                method, args = request["method"], request.get("args", {})
                try:
                    work = asyncio.create_task(asyncio.to_thread(self.dispatch, method, args, progress))
                    try:
                        result = await asyncio.shield(work)
                    except asyncio.CancelledError:
                        # Do not let an old worker mutate the cache/GPU after a new handshake.
                        with contextlib.suppress(Exception):
                            await work
                        raise
                    if method in ("preload", "load", "unload", "release"):
                        await send({"type": "status", **self.status})
                    await send({"type": "result", "id": request["id"], "result": result})
                    if method in ("preload", "load", "unload", "release"):
                        await send({"type": "log", "level": "INFO", "message": f"{method} completed"})
                except Exception as exc:
                    log.exception("RPC %s failed", method)
                    if method in ("preload", "load"):
                        await send({"type": "inventory", "inventory": self.runtime.inventory()})
                    if method in ("load", "forward"):
                        await asyncio.to_thread(self.runtime.unload)
                        self.status = {"status": "error", "error": str(exc)[:500]}
                    await send(
                        {
                            "type": "result",
                            "id": request["id"],
                            "error": str(exc)[:500],
                            "status": exc.status if isinstance(exc, SoulError) else 503,
                        }
                    )
                    await send({"type": "log", "level": "ERROR", "message": f"{method}: {str(exc)[:500]}"})
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def dispatch(self, method, args, progress):
        if method == "offer":
            offer = self.runtime.offer(ModelSpec(**args["spec"]))
            return {**offer, "inventory": self.runtime.inventory()}
        if method == "preload":
            previous = self.status
            self.status = {
                "status": "downloading",
                "model": args["spec"]["model_id"],
                "start": args["start"],
                "end": args["end"],
            }
            try:
                self.runtime.preload(ModelSpec(**args["spec"]), args["start"], args["end"], progress)
            finally:
                self.status = previous
            return {"cached": True, "inventory": self.runtime.inventory()}
        if method == "load":
            self.status = {"status": "loading"}
            self.runtime.load(
                ModelSpec(**args["spec"]), args["start"], args["end"], args["epoch"], progress, download=False
            )
            self.status = {
                "status": "ready",
                "model": args["spec"]["model_id"],
                "start": args["start"],
                "end": args["end"],
                "epoch": args["epoch"],
            }
            log.info("Loaded %s layers [%d, %d)", args["spec"]["model_id"], args["start"], args["end"])
            return {"ready": True}
        if method == "unload":
            self.runtime.unload()
            self.status = {"status": "idle"}
            return {"unloaded": True}
        self.runtime.check(args["epoch"])
        if method == "tokenize":
            return self.runtime.tokenize(args)
        if method == "decode":
            return {"text": self.runtime.decode(args["ids"])}
        if method == "forward":
            return self.runtime.forward(args)
        if method == "release":
            self.runtime.sessions.pop(args["session_id"], None)
            return {"released": True}
        raise ValueError(f"Unknown RPC method: {method}")

    async def run(self):
        delay = 1
        while True:
            try:
                await in_thread(self.runtime.unload)
                self.initial_inventory = await in_thread(self.runtime.scan_cache)
                async with websockets.connect(
                    self.url,
                    extra_headers=self.headers,
                    max_size=MAX_FRAME,
                    compression=None,
                    ping_interval=20,
                    open_timeout=15,
                ) as ws:
                    log.info("Connected as %s (%s)", self.node_id, self.runtime.device)
                    delay = 1
                    await self.serve_connection(ws)
            except (OSError, ValueError, websockets.WebSocketException) as exc:
                log.warning("Connection lost: %s; retrying", exc)
            await asyncio.to_thread(self.runtime.unload)
            await asyncio.sleep(delay + random.random())
            delay = min(30, delay * 2)
