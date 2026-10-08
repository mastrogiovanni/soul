import asyncio
import contextlib
import socket

import httpx
import pytest
import torch
import uvicorn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    MistralConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen3Config,
)

from soul.controller.app import Settings, create_app
from soul.node.client import NodeClient

CONFIGS = {"llama": LlamaConfig, "mistral": MistralConfig, "qwen2": Qwen2Config, "qwen3": Qwen3Config}
torch.set_num_threads(1)


class BasicIngress:
    """Test the shared ingress gate without replacing Soul's own authentication."""

    authorization = b"Basic dGVzdDpwYXNz"  # test:pass

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            if dict(scope["headers"]).get(b"authorization") != self.authorization:
                if scope["type"] == "websocket":
                    return await send({"type": "websocket.close", "code": 1008})
                await send({"type": "http.response.start", "status": 401, "headers": []})
                return await send({"type": "http.response.body", "body": b"Ingress authentication required"})
        await self.app(scope, receive, send)


def make_model(directory, architecture="llama", layers=4, tied=False, hidden_size=32):
    torch.manual_seed(42)
    config = CONFIGS[architecture](
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        sliding_window=8,
        tie_word_embeddings=tied,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        head_dim=hidden_size // 4,
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config).eval()
    model.save_pretrained(directory)
    vocab = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "Hello": 4,
        "world": 5,
        "user": 6,
        "assistant": 7,
        ":": 8,
        **{f"word{i}": i for i in range(9, 32)},
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]"
    )
    fast.chat_template = (
        "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}{{ 'assistant: ' }}"
    )
    fast.save_pretrained(directory)
    return model


@pytest.fixture
def tiny_model(tmp_path):
    path = tmp_path / "model"
    model = make_model(path)
    return path, model


async def eventually(predicate, timeout=15):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.03)


@pytest.fixture
async def cluster(tmp_path, tiny_model, request):
    path, _ = tiny_model
    options = getattr(request, "param", {})
    settings = Settings(
        api_key="api",
        admin_key="admin",
        node_token="node",
        state_dir=tmp_path / "controller",
        max_layers_per_node=2,
        allow_local=True,
        max_context=256,
        rpc_timeout=5,
        load_timeout=20,
        context_root=options.get("context_root", ""),
    )
    app = create_app(settings)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            BasicIngress(app) if options.get("ingress_auth") else app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            ws_per_message_deflate=False,
        )
    )
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    await eventually(lambda: server.started)
    url = f"http://127.0.0.1:{port}" + settings.context_root.rstrip("/")
    clients = [
        NodeClient(
            url,
            "node",
            tmp_path / f"node{i}",
            node_id=f"node{i}",
            name=f"Test node {i}",
            device="cpu",
            memory_gb=1,
            ingress_username="test" if options.get("ingress_auth") else None,
            ingress_password="pass" if options.get("ingress_auth") else None,
        )
        for i in range(2)
    ]
    tasks = [asyncio.create_task(c.run()) for c in clients]
    await eventually(lambda: len(app.state.engine.nodes) == 2)
    headers = {"Authorization": BasicIngress.authorization.decode()} if options.get("ingress_auth") else {}
    async with httpx.AsyncClient(base_url=url, timeout=30, headers=headers) as http:
        preloaded = await http.post(
            "/admin/models/preload", headers={"X-Soul-Admin-Key": "admin"}, json={"model": str(path)}
        )
        assert preloaded.status_code == 200, preloaded.text
        yield app.state.engine, http, str(path), url, clients, tasks
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task
    server.should_exit = True
    await server_task
