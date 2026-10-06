import asyncio
import time
import uuid
from collections import deque

from fastapi import WebSocket

from soul.protocol import SoulError, pack


class Peer:
    def __init__(self, ws: WebSocket, hello: dict):
        self.ws = ws
        self.id = hello["node_id"]
        self.name = hello.get("name", self.id)
        self.capabilities = hello["capabilities"]
        self.generation = str(uuid.uuid4())
        self.last_seen = time.time()
        self.status = {"status": "idle"}
        self.pending = {}
        self.logs = deque(maxlen=300)
        self.connected = True
        self.send_lock = asyncio.Lock()

    async def call(self, method, args=None, timeout=120):
        if not self.connected:
            raise SoulError(f"Node {self.name} is disconnected")
        id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[id] = future
        try:
            async with self.send_lock:
                await self.ws.send_bytes(
                    pack({"type": "rpc", "id": id, "method": method, "args": args or {}})
                )
            return await asyncio.wait_for(future, timeout)
        except TimeoutError as exc:
            # The node might still be executing; revoke this connection before any subsequent work.
            self.disconnect("RPC timeout")
            await self.ws.close(code=1011)
            raise SoulError(f"Node {self.name} timed out during {method}", 504, "timeout") from exc
        except (OSError, RuntimeError) as exc:
            self.disconnect("Transport failed")
            raise SoulError(f"Node {self.name} transport failed") from exc
        finally:
            self.pending.pop(id, None)
            if future.done() and not future.cancelled():
                future.exception()  # Retrieve an error if a concurrent send/disconnect failed.

    def receive(self, message):
        self.last_seen = time.time()
        kind = message["type"]
        if kind == "result":
            future = self.pending.get(message.get("id"))
            if future and not future.done():
                if "error" in message:
                    future.set_exception(SoulError(message["error"], message.get("status", 503)))
                else:
                    future.set_result(message.get("result", {}))
        elif kind in ("heartbeat", "status"):
            self.status = {k: v for k, v in message.items() if k not in ("type", "time")}
        elif kind == "log":
            self.logs.append(
                {
                    "time": time.time(),
                    "level": str(message.get("level", "INFO"))[:10],
                    "message": str(message.get("message", ""))[:2000],
                }
            )
        else:
            raise ValueError("Unknown node event")

    def disconnect(self, reason="Disconnected"):
        self.connected = False
        self.status = {"status": "offline", "error": reason}
        for future in self.pending.values():
            if not future.done():
                future.set_exception(SoulError(f"Node {self.name}: {reason}"))

    def snapshot(self):
        return {
            "node_id": self.id,
            "name": self.name,
            "capabilities": self.capabilities,
            "connected": self.connected,
            "generation": self.generation,
            "last_seen": self.last_seen,
            **self.status,
        }
