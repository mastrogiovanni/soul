import asyncio
import contextlib
import hmac
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from soul.controller.api import error_body, respond
from soul.controller.engine import Engine
from soul.controller.registry import Peer
from soul.controller.schemas import AnthropicRequest, ChatRequest, CompletionRequest, LoadRequest
from soul.controller.store import Store
from soul.protocol import VERSION, SoulError, pack, unpack

log = logging.getLogger("soul.controller")


@dataclass
class Settings:
    api_key: str
    admin_key: str
    node_token: str
    state_dir: Path = field(default_factory=lambda: Path(".soul/controller"))
    max_context: int = 2048
    max_queue: int = 32
    queue_timeout: float = 300
    rpc_timeout: float = 120
    load_timeout: float = 1800
    allow_local: bool = False
    max_layers_per_node: int | None = None
    allowed_models: list[str] = field(default_factory=list)
    context_root: str = ""


def route_path(scope):
    """The application path, excluding a mounted ASGI context root."""
    path, root = scope["path"], scope.get("root_path", "")
    if root and (path == root or path.startswith(root + "/")):
        return path[len(root) :] or "/"
    return path


class GuardMiddleware:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = route_path(scope)
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        if path.startswith(("/v1/", "/admin/")) or path == "/models":
            key = self.settings.admin_key if path.startswith("/admin/") else self.settings.api_key
            supplied = headers.get("authorization", "").removeprefix("Bearer ")
            if path == "/v1/messages":
                supplied = headers.get("x-api-key", supplied)
            header = "x-soul-admin-key" if path.startswith("/admin/") else "x-soul-api-key"
            supplied = headers.get(header, supplied)
            if not hmac.compare_digest(supplied, key):
                exc = SoulError("Invalid API key", 401, "authentication_error")
                return await JSONResponse(error_body(exc, path == "/v1/messages"), 401)(scope, receive, send)
        if scope["method"] in ("POST", "PUT", "PATCH"):
            data = bytearray()
            while True:
                try:
                    message = await asyncio.wait_for(receive(), 30)
                except TimeoutError:
                    return await JSONResponse({"error": {"message": "Request body timed out"}}, 408)(
                        scope, receive, send
                    )
                if message["type"] == "http.disconnect":
                    return
                data.extend(message.get("body", b""))
                if len(data) > 1024 * 1024:
                    return await JSONResponse({"error": {"message": "Request body exceeds 1 MiB"}}, 413)(
                        scope, receive, send
                    )
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(data), "more_body": False}
                return await receive()

            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)


def create_app(settings: Settings):
    if not all((settings.api_key, settings.admin_key, settings.node_token)):
        raise ValueError("API, admin and node credentials are required")
    context_root = settings.context_root.rstrip("/")
    if context_root and not re.fullmatch(r"(?:/[A-Za-z0-9_-]+)+", context_root):
        raise ValueError("Context root must be an absolute path such as /soul (or empty for /)")

    @asynccontextmanager
    async def lifespan(app):
        store = Store(settings.state_dir / "controller.sqlite3")
        engine = Engine(
            store,
            max_context=settings.max_context,
            max_queue=settings.max_queue,
            queue_timeout=settings.queue_timeout,
            rpc_timeout=settings.rpc_timeout,
            load_timeout=settings.load_timeout,
            allow_local=settings.allow_local,
            max_layers_per_node=settings.max_layers_per_node,
            allowed_models=settings.allowed_models,
        )
        app.state.engine = engine
        engine.models.start()

        async def reap():
            while True:
                await asyncio.sleep(5)
                for peer in list(engine.nodes.values()):
                    if peer.connected and time.time() - peer.last_seen > 30:
                        peer.disconnect("Heartbeat expired")
                        engine.invalidate(peer.id)
                        engine.models.wake()
                        with contextlib.suppress(Exception):
                            await peer.ws.close(code=1011)

        task = asyncio.create_task(reap())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await engine.models.stop()
        for peer in engine.nodes.values():
            if peer.connected:
                peer.disconnect("Controller shutting down")
                with contextlib.suppress(Exception):
                    await peer.ws.close(code=1001)
        store.close()

    app = FastAPI(title="Soul distributed serving", version="0.1.0", lifespan=lifespan)
    app.add_middleware(GuardMiddleware, settings=settings)
    static = Path(__file__).resolve().parents[1] / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.exception_handler(SoulError)
    async def soul_error(request, exc):
        return JSONResponse(error_body(exc, route_path(request.scope) == "/v1/messages"), exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        error = SoulError(details, 400, "invalid_request")
        return JSONResponse(error_body(error, route_path(request.scope) == "/v1/messages"), 400)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        log.exception("Request failed", exc_info=exc)
        error = SoulError("Internal server error; inspect controller logs", 500, "internal_error")
        return JSONResponse(error_body(error, route_path(request.scope) == "/v1/messages"), 500)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(static / "index.html")

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/v1/models")
    @app.get("/models", include_in_schema=False)
    async def models(request: Request):
        engine = request.app.state.engine
        return {
            "object": "list",
            "data": [
                {"id": d["model"], "object": "model", "created": d["created"], "owned_by": "soul"}
                for d in engine.models.views()
                if d["executable"]
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request):
        return await respond(request.app.state.engine, body, "chat", request)

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        return await respond(request.app.state.engine, body, "completion", request)

    @app.post("/v1/messages")
    async def messages(body: AnthropicRequest, request: Request):
        return await respond(request.app.state.engine, body, "anthropic", request)

    @app.get("/admin/state")
    async def state(request: Request):
        return request.app.state.engine.snapshot()

    @app.get("/admin/nodes/{node_id}/logs")
    async def logs(node_id: str, request: Request):
        node = request.app.state.engine.nodes.get(node_id)
        if not node:
            raise SoulError("Node has not connected in this controller session", 404, "not_found")
        return {"logs": list(node.logs)}

    @app.delete("/admin/nodes/{node_id}")
    async def remove_node(node_id: str, request: Request):
        return await request.app.state.engine.remove_node(node_id)

    @app.post("/admin/models/preload")
    async def preload(body: LoadRequest, request: Request):
        return await request.app.state.engine.models.preload(body.model)

    @app.post("/admin/models/load")
    async def load(body: LoadRequest, request: Request):
        return await request.app.state.engine.load_model(body.model)

    @app.post("/admin/models/unload")
    async def unload(body: LoadRequest, request: Request):
        return await request.app.state.engine.unload_model(body.model)

    @app.websocket("/internal/nodes")
    async def nodes(ws: WebSocket):
        # Reject browser cross-origin websocket attempts, even if a node token was exposed.
        supplied = ws.headers.get(
            "x-soul-node-token", ws.headers.get("authorization", "").removeprefix("Bearer ")
        )
        if ws.headers.get("origin") or not hmac.compare_digest(supplied, settings.node_token):
            await ws.close(code=1008)
            return
        await ws.accept()
        engine, peer = ws.app.state.engine, None
        try:
            hello = unpack(await asyncio.wait_for(ws.receive_bytes(), 10))
            if (
                hello.get("type") != "hello"
                or hello.get("version") != VERSION
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", str(hello.get("node_id", "")))
                or not isinstance(hello.get("capabilities"), dict)
                or not isinstance(hello.get("inventory"), list)
            ):
                raise ValueError("Invalid node handshake")
            peer = Peer(ws, hello)
            old = engine.nodes.get(peer.id)
            if old and old.connected:
                old.disconnect("Replaced by a new connection")
                await old.ws.close(code=1012)
            engine.invalidate(peer.id)
            engine.nodes[peer.id] = peer
            engine.old_nodes.pop(peer.id, None)
            engine.models.adopt_inventory(peer)
            engine.store.save("nodes", peer.id, peer.snapshot())
            await ws.send_bytes(pack({"type": "welcome", "version": VERSION, "generation": peer.generation}))
            while True:
                message = unpack(await ws.receive_bytes())
                peer.receive(message)
                if message["type"] == "inventory" and engine.nodes.get(peer.id) is peer:
                    engine.models.adopt_inventory(peer, wake=False)
                if message["type"] in ("inventory", "result") and engine.nodes.get(peer.id) is peer:
                    engine.store.save("nodes", peer.id, peer.snapshot())
        except (WebSocketDisconnect, OSError, RuntimeError):
            pass
        except (ValueError, KeyError, TypeError, TimeoutError) as exc:
            log.warning("Invalid node connection: %s", exc)
            with contextlib.suppress(Exception):
                await ws.close(code=1008)
        finally:
            if peer:
                peer.disconnect()
                if engine.nodes.get(peer.id) is peer:
                    engine.invalidate(peer.id)
                    engine.store.save("nodes", peer.id, peer.snapshot())
                    engine.models.wake()

    if not context_root:
        return app

    # Mount without stripping paths in the proxy, so docs, HTTP and WebSockets
    # share one context. The parent owns the lifespan and both apps share state.
    root = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    root.state = app.state

    @root.get("/", include_in_schema=False)
    @root.get(context_root, include_in_schema=False)
    async def redirect_index():
        # Relative Location keeps the public HTTPS scheme behind TLS termination.
        return RedirectResponse(context_root + "/")

    root.add_api_route("/healthz", health, include_in_schema=False)
    root.mount(context_root, app)
    return root
