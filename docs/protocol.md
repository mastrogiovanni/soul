# Node protocol v2

All node traffic goes through an outward WebSocket connection to `<controller-base>/internal/nodes`. Container/Kubernetes deployments use `wss://<host>/soul/internal/nodes`. The handshake accepts `Authorization: Bearer <SOUL_NODE_TOKEN>` or `X-Soul-Node-Token`. When the shared ingress requires Basic Auth, set `SOUL_INGRESS_USERNAME` and `SOUL_INGRESS_PASSWORD` on the node: the client sends those credentials in `Authorization` and its Soul token separately, preserving both authentication gates. Browser-origin connections are rejected; a browser/WebGPU node will need a scoped browser enrollment flow in phase 2. WebSocket compression is disabled and each binary MessagePack map is limited to 64 MiB. Version-1 handshakes are rejected: upgrade controller and nodes together.

## Handshake

Node → controller:

```json
{"type":"hello","version":2,"node_id":"persistent-uuid","name":"mac-studio","capabilities":{"device":"mps","device_name":"Apple Metal (unified memory)","budget_bytes":12884901888,"dtype":"float16","architectures":["llama","mistral","qwen2","qwen3"],"weight_mode":"resident","ram_bytes":34359738368},"inventory":[]}
```

Controller → node: `{"type":"welcome","version":2,"generation":"connection-uuid"}`.

A new connection always discards GPU assignments and KV state. The disk tensor cache survives. Node IDs match `[A-Za-z0-9_.-]{1,128}`. A duplicate identity replaces the old connection and invalidates its placements.

Before opening the connection, nodes scan cache manifests and verify tensor checksums. `inventory` is an array of `{spec: ModelSpec, tensors: [verified tensor names], tokenizer: boolean, offer: MemoryOffer}`. Empty nodes send an empty array. Offers describe the context in their accompanying spec; the controller refreshes offers when its configured context differs. Model metadata can be recovered from returning nodes even by a new controller. A pinned catalog revision takes precedence over a different revision in another node's cache.

## RPC

Controller → node: `{"type":"rpc","id":"uuid","method":"...","args":{...}}`.

Node → controller: `{"type":"result","id":"uuid","result":{...}}` or `{"type":"result","id":"uuid","error":"reason","status":503}`.

| Method | Arguments / response |
| --- | --- |
| `offer` | ModelSpec → memory offer and refreshed full inventory |
| `preload` | ModelSpec, inclusive `start`, exclusive `end` → download assigned tensors/tokenizer to disk; return `cached: true` and full inventory |
| `load` | ModelSpec, inclusive `start`, exclusive `end`, deployment `epoch` → ready; requires locally cached tensors/tokenizer, never downloads |
| `unload` | No arguments → releases weights and all request caches |
| `tokenize` | epoch and `messages` or `prompt` → `input_ids`, `eos_ids`; first stage only |
| `forward` | epoch, session_id, position, IDs or hidden tensor, sample flag, temperature/top_p/seed → hidden tensor or token_id |
| `decode` | epoch and accumulated generated `ids` → text; first stage only |
| `release` | epoch and session_id → deletes local KV state |

`ModelSpec` contains model_id, resolved revision SHA, raw Transformers config, generation_config, max_context, and a map of tensor name → shard filename, byte offset/length, dtype and shape. No Hugging Face access token crosses this protocol. Tensors use the standard checkpoint layout and are reindexed locally inside the runtime.

Nodes send asynchronous events: `heartbeat` every 5 seconds, `status` for progress/state, and `log` with level/message. The controller disconnects a node after 30 seconds without events. Node RPCs run in order; heartbeat/status transport remains live while GPU/download work runs in a worker thread.

A memory offer contains `budget_bytes`, triangular `costs[start][end-start-1]`, triangular `missing_bytes`, `disk_free_bytes`, `context`, and `weight_mode`. Missing bytes include a one-byte sentinel when first-stage tokenizer preparation is needed. Zero means the range is fully prepared. Offers account for the node's dtype, context, endpoints, KV cache, and workspace. Inventory updates accompany successful offer/preload results; an `inventory` event reports the remaining verified cache after a failed preload/load. These reports describe disk contents independently of GPU status. The controller does not send inference to a partial pipeline, and only preload/reconciliation downloads missing tensors.

## Tensor envelope

```text
{"dtype": "float16" | "bfloat16" | "float32",
 "shape": [1, sequence_length, hidden_size],
 "data": <MessagePack binary containing contiguous little-endian tensor bytes>}
```

Current prefill length is at most 128; decode length is one. The receiver validates rank, batch size, hidden size, sequence length and byte count before materialization. Standard supported hosts are little-endian; big-endian machines are not supported. There is no NumPy pickle, torch serialization, base64, or executable payload on the wire. The controller does not interpret tensor contents. Different node precisions are converted by each receiving runtime, so heterogeneous precision affects numerical equivalence.

The epoch fences stale deployments; session IDs isolate requests; the absolute position detects cache gaps/replays. Lost connections cannot recover a session: retry through a newly established pipeline. Error completion invalidates unsafe placements rather than silently continuing.

## Extensibility

The wire protocol does not require Python. Future WebGPU, Swift/Metal, Vulkan, or native CUDA clients can implement it. They must also implement the exact architecture's math, tensor layout, precision, cache semantics and position handling. A browser implementation must address MessagePack binary handling, cross-origin Hub ranges, persistent storage, scoped authentication, and background-tab throttling; simply opening the dashboard does not make a phone a compute node.
