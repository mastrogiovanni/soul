# Soul — distributed model serving

One model, many machines. Soul splits an LLM into contiguous decoder stages on a star network: nodes connect outward to a small controller, which accepts API requests and routes activations through the stages.

**This is an executable phase-1 inference implementation, not a training system.** It runs real transformer layers and node-local KV caches. The controller never loads model weights or imports PyTorch. The admin console is included, with no frontend build step.

![Soul admin console](docs/admin-console.jpg)

## What works

- CPU-only FastAPI controller with OpenAI Chat Completions, text Completions, and Anthropic Messages endpoints; streaming and ordinary JSON responses.
- PyTorch nodes for NVIDIA CUDA, Apple Metal (`mps`), and CPU, with automatic device selection and persistent node identities.
- Automatic contiguous layer placement based on each node's estimates for weights, endpoint embeddings/head, KV cache, working memory, and its memory budget.
- Pinned Hugging Face revisions; HTTP range downloads of **only assigned tensors**, atomic local safetensors files, checksum validation, and reuse after restarts.
- Resident weights, or optional per-layer streaming from local disk to the GPU.
- Node-local rotary-position calculations, causal/sliding-window attention, and per-request KV caches. Chunked prefill bounds activation size.
- Bounded FIFO inference admission, demand-driven model loading and replacement, generation fencing after reconnects, timeouts, and explicit failures.
- An admin UI for model placement, connected/offline nodes, download progress, logs, queue state, request outcomes, and cumulative counters persisted in SQLite.

**Model support is explicit:** unquantized safetensors checkpoints with Transformers `model_type` **`llama`, `mistral`, `qwen2`, or `qwen3`**, using their standard tensor layout. Examples include Llama-family models, SmolLM2, Qwen2.5, and Qwen3 dense models. A repository's `model_type` is authoritative, not its marketing name. MoE variants, Gemma, GPT-2, multimodal models, Mamba/hybrid models, custom remote code, GGUF, and prequantized checkpoints require additional adapters. Unsupported models fail clearly; Soul does **not** claim arbitrary Hugging Face compatibility.

## Quick start

Use Python 3.11 or 3.12. From this checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[node,dev]'
soul demo
```

Open **http://127.0.0.1:8000** and enter `soul-dev` as the admin key. The demo launches **three separate processes**: one controller and two CPU nodes. It builds a tiny four-layer Llama locally, distributes two layers to each node, and executes a request to populate the dashboard. No model download is required. The demo model has random weights, so its output is deliberately meaningless. Computation, communication, placement, and counters are real. `Ctrl+C` stops all three processes.

If port 8000 is occupied, use `soul demo --port 8766`. Logs and reusable weights go in `.soul/demo/`.

## Run your own network

### Controller

The controller needs only the base package, not the node extra:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

export SOUL_API_KEY='your-inference-key'
export SOUL_ADMIN_KEY='your-admin-key'
export SOUL_NODE_TOKEN='your-node-enrollment-token'
soul controller --host 0.0.0.0 --port 8000 --max-context 2048
```

Use independent, strong values for these three credentials. The API key can run inference; the admin key can inspect logs and load/unload models; the node token admits trusted compute nodes. `--dev` instead uses the public `soul-dev` credential and is restricted to loopback.

The dashboard is at `/`, API documentation at `/docs`, and basic health at `/healthz`. Deployment state and counters live in `.soul/controller/controller.sqlite3`. Run **one controller process / one Uvicorn worker** against this state. Multi-controller consensus and shared scheduling are outside this release.

For a remote network, use HTTPS/WSS through a reverse proxy or a private encrypted network. Nodes see prompts, activations, and portions of the model; enroll trusted machines only. Use `--allow-model owner/repository` repeatedly to restrict which models inference callers can load. Without an allowlist, authenticated callers may initiate downloads of any supported model.

### Mac node

On each machine, copy or clone this project and run:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[node]'
export SOUL_NODE_TOKEN='your-node-enrollment-token'
soul node --controller https://controller.example.com --name mac-studio
```

The node uses CUDA if available, otherwise Metal, otherwise CPU. Metal uses the Mac's unified memory. `--device mps` requires a Metal-capable PyTorch installation; use `--device cpu` for a portable smoke test. No inbound node ports are required.

### NVIDIA node

Install the **PyTorch 2.8 CUDA wheel matching your operating system and driver** from the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v280), then install the node extra in the same environment:

```sh
pip install -e '.[node]'
export SOUL_NODE_TOKEN='your-node-enrollment-token'
soul node --controller https://controller.example.com --device cuda:0 --name gpu-01
```

One process uses one device. For multiple GPUs on one host, run a node per GPU with a different `--cache-dir` and `--device cuda:N`; set explicit memory caps if resources are shared. Do not run Metal nodes inside Docker: use native macOS Python.

### Memory and disk controls

```sh
soul node --controller https://controller.example.com \
  --memory-gb 10 --weight-mode resident --cache-dir /fast-ssd/soul

# Keep only the executing decoder layer's weights on the GPU:
soul node --controller https://controller.example.com \
  --memory-gb 6 --weight-mode stream --cache-dir /fast-ssd/soul
```

A node initially budgets 75% of reported available device memory, capped by `--memory-gb`. Estimates include KV storage for one full context and allocator/workspace headroom. They are estimates, not hard memory isolation. External GPU use can still cause OOM; such a request fails and invalidates its deployment.

`resident` is the fast path. `stream` hydrates each local decoder layer from cached safetensors before executing it, then releases its parameters; edge embeddings/output head and KV caches remain resident. This supports shards larger than VRAM at the cost of substantial I/O and transfer latency. It is synchronous, without prefetch or overlap. KV offloading, paged attention, and quantization are not implemented. Downloaded tensors are retained after unload for reuse; there is no automatic disk quota/eviction yet.

Nodes persist identity in `<cache-dir>/identity.json`; reuse that directory after a restart. Never run two processes against the same node cache directory. Node logs rotate at 5 MiB with two backups. The dashboard shows the latest 300 controller-received lifecycle/error events per node in the current controller session.

## Load a model and generate text

A request automatically inspects and places its requested model. You can also choose **Load a model** in the admin UI. For gated/private models, authenticate **both controller and nodes** with `HF_TOKEN` or `hf auth login`; tokens are not sent through the node protocol. Have sufficient local disk space and permission to use the model.

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $SOUL_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Explain pipeline parallelism in two sentences."}],
    "max_tokens": 80,
    "temperature": 0,
    "stream": true
  }'
```

OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-inference-key")
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64,
    stream=True,
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Anthropic Python SDK:

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="your-inference-key")
with client.messages.stream(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_tokens=64,
    messages=[{"role": "user", "content": "Hello!"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

These are protocol-compatible **local model** endpoints. They do not call OpenAI or Anthropic or provide proprietary models. See [API coverage](docs/api.md) for accepted fields and unsupported features.

## Scheduling and failure behavior

Phase 1 runs **one active generation across the cluster**, with up to 32 queued inference requests by default. This is deliberately conservative: it makes model replacement and exact KV cleanup tractable before adding continuous batching, replicas, and concurrent pipelines. It aggregates memory capacity; it does not promise lower single-request latency than one GPU.

The controller prefers fewer stages, using larger-memory nodes first. It accounts for the larger first/last stage weights and allocates contiguous ranges. `--max-layers-per-node N` can force a smaller shard for experiments. Node compute-speed calibration and network-aware load balancing are future work.

Node reconnects invalidate every affected deployment and all old KV state. Later requests can rebuild the pipeline from disk-cached weights. In-flight requests fail; there is no transparent replay after output has been streamed. Before stream headers, failures use ordinary 4xx/5xx responses. During streaming, errors appear as protocol error events. A request must fit `prompt_tokens + max_tokens <= max_context`; the default is 2048, capped by the model's configured context.

Controller restarts preserve counters and history but mark loaded placement records stale; queued/running requests become failed. The request queue itself is in memory and is not replayed after restart. No prompt or generated text is saved in the controller's ledger.

## Development and verification

```sh
pip install -e '.[node,dev]'
pytest -q
ruff check src tests examples
ruff format --check src tests examples

# Optional live download test: a small public checkpoint, no full-shard download
python examples/verify_hub.py
# Run the same exercise on an available accelerator
python examples/verify_hub.py --device mps
python examples/verify_hub.py --device cuda
```

Tests instantiate small random models locally. They compare sharded vs. unsplit logits for four architectures, chunked prefill, cached decode, sliding attention and disk streaming; run actual HTTP/WebSocket nodes with both official SDKs; and cover placement, auth, validation, reconnect fencing, cancellation cleanup and queueing. The Hub smoke test separately exercises actual remote tensor ranges and tokenizer loading.

The optional node runtime is intentionally pinned to Transformers **4.56.0** and PyTorch **2.8.x**, because model/cache internals are part of the stage adapter contract. Upgrade only together with equivalence tests. See [architecture](docs/architecture.md), [wire protocol](docs/protocol.md), and [phase 2](docs/phase-2.md).

## Repository

```text
src/soul/controller/   API adapters, scheduler, registry, SQLite ledger
src/soul/node/         device runtime, tensor cache, reconnecting client
src/soul/static/       admin console (HTML/CSS/JS; no CDN dependencies)
src/soul/catalog.py    pinned model metadata and strict range reader
src/soul/planner.py    contiguous placement planner
src/soul/protocol.py   MessagePack framing and protocol version
src/soul/demo.py       three-process, zero-download demo
examples/             optional real Hugging Face smoke test
tests/               numerical, protocol, and network integration tests
```
