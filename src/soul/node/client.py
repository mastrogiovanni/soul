import asyncio
import base64
import contextlib
import getpass
import json
import logging
import random
import re
import socket
import threading
import time
import unicodedata
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


def default_node_name():
    """Combine the hostname and login username into a lowercase ASCII slug."""
    parts = []
    for value, fallback in ((socket.gethostname(), "host"), (getpass.getuser(), "user")):
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        parts.append(re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or fallback)
    return "-".join(parts)


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
        self.name = name or default_node_name()
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
        self.download_status = None
        self.initial_inventory = []
        handler = RotatingFileHandler(cache_dir / "node.log", maxBytes=5 * 1024**2, backupCount=2)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)

    def status_snapshot(self):
        status = dict(self.status)
        download = self.download_status
        if download is not None:
            status["download"] = download
            if status["status"] == "idle":
                status.update(status="downloading", model=download["model"])
            status.update(
                {k: v for k, v in download.items() if k in ("downloaded_bytes", "total_bytes", "tensor")}
            )
        return status

    async def serve_connection(self, ws):
        send_lock = asyncio.Lock()

        async def send(message):
            async with send_lock:
                await ws.send(pack(message))

        async def send_status(kind="status"):
            async with send_lock:
                # Capture at publication time, not in the download thread: an
                # older progress callback must not restore a stale GPU epoch.
                await ws.send(pack({"type": kind, **self.status_snapshot(), "time": time.time()}))

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
        self.download_status = None
        loop = asyncio.get_running_loop()
        stopping = threading.Event()

        def progress(done, total, tensor):
            if stopping.is_set():
                raise SoulError("Download connection closed")
            self.download_status = {
                **(self.download_status or {}),
                "downloaded_bytes": done,
                "total_bytes": total,
                "tensor": tensor,
            }
            future = asyncio.run_coroutine_threadsafe(send_status(), loop)
            try:
                future.result(timeout=5)
            except Exception:
                # Stop an abandoned download at its next bounded chunk, rather than
                # finishing a multi-GB assignment after the controller has fenced it.
                future.cancel()
                raise

        async def heartbeat():
            while True:
                await send_status("heartbeat")
                await asyncio.sleep(5)

        # Downloads have a separate worker; all GPU/KV operations retain FIFO
        # ordering. A deletion drains earlier downloads and takes their lock.
        execution_queue, download_queue = asyncio.Queue(32), asyncio.Queue(32)
        cache_lock = asyncio.Lock()
        inventory_lock = asyncio.Lock()

        async def report_inventory(message):
            # Serialize snapshot + publication so a slow offer cannot overwrite
            # the inventory reported by a newly completed concurrent preload.
            async with inventory_lock:
                inventory = await in_thread(self.runtime.inventory)
                if message["type"] == "result":
                    message["result"]["inventory"] = inventory
                else:
                    message["inventory"] = inventory
                await send(message)

        async def execute(request):
            method, args = request["method"], request.get("args", {})
            try:
                result = await in_thread(self.dispatch, method, args, progress)
                if method in ("preload", "load", "unload", "release", "delete_model", "clear_cache"):
                    await send_status()
                message = {"type": "result", "id": request["id"], "result": result}
                if method in ("offer", "preload", "delete_model", "clear_cache"):
                    await report_inventory(message)
                else:
                    await send(message)
                if method in ("preload", "load", "unload", "release", "delete_model", "clear_cache"):
                    await send({"type": "log", "level": "INFO", "message": f"{method} completed"})
            except Exception as exc:
                log.exception("RPC %s failed", method)
                if method in ("preload", "load", "delete_model", "clear_cache"):
                    await report_inventory({"type": "inventory"})
                if method in ("delete_model", "clear_cache") and self.runtime.model is None:
                    self.status = {"status": "idle"}
                if method in ("load", "forward"):
                    await in_thread(self.runtime.unload)
                    self.status = {"status": "error", "error": str(exc)[:500]}
                await send_status()
                await send(
                    {
                        "type": "result",
                        "id": request["id"],
                        "error": str(exc)[:500],
                        "status": exc.status if isinstance(exc, SoulError) else 503,
                    }
                )
                await send({"type": "log", "level": "ERROR", "message": f"{method}: {str(exc)[:500]}"})

        async def worker(queue, downloads=False):
            while True:
                request = await queue.get()
                try:
                    destructive = request["method"] in ("delete_model", "clear_cache")
                    if destructive:
                        await download_queue.join()
                    if downloads or destructive:
                        async with cache_lock:
                            await execute(request)
                    else:
                        await execute(request)
                finally:
                    queue.task_done()

        async def receive():
            async for data in ws:
                request = unpack(data)
                if request["type"] != "rpc":
                    raise ValueError("Expected an RPC request")
                queue = download_queue if request["method"] == "preload" else execution_queue
                await queue.put(request)

        tasks = [
            asyncio.create_task(receive()),
            asyncio.create_task(heartbeat()),
            asyncio.create_task(worker(execution_queue)),
            asyncio.create_task(worker(download_queue, downloads=True)),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            stopping.set()
            for task in tasks:
                task.cancel()
            # in_thread drains active work before reconnecting or unloading.
            await asyncio.gather(*tasks, return_exceptions=True)

    def dispatch(self, method, args, progress):
        if method in ("delete_model", "clear_cache"):
            model = args["model"] if method == "delete_model" else None
            if method == "delete_model" and (not isinstance(model, str) or not model):
                raise ValueError("A model ID is required for deletion")
            freed = self.runtime.delete_cache(model, args.get("keys", []))
            if self.runtime.model is None:
                self.status = {"status": "idle"}
            return {"freed_bytes": freed}
        if method == "offer":
            offer = self.runtime.offer(ModelSpec(**args["spec"]))
            return offer
        if method == "preload":
            self.download_status = {
                "model": args["spec"]["model_id"],
                "start": args["start"],
                "end": args["end"],
            }
            try:
                self.runtime.preload(ModelSpec(**args["spec"]), args["start"], args["end"], progress)
            finally:
                self.download_status = None
            return {"cached": True}
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
