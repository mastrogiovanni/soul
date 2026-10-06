# Node protocol v1

All node traffic goes through an outward WebSocket connection to `/internal/nodes`. The handshake uses `Authorization: Bearer <SOUL_NODE_TOKEN>`. Browser-origin connections are rejected in v1; a browser/WebGPU node will need a scoped browser enrollment flow in phase 2. WebSocket compression is disabled and each binary MessagePack map is limited to 64 MiB.

## Handshake

Node → controller:

```json
{"type":"hello","version":1,"node_id":"persistent-uuid","name":"mac-studio","capabilities":{"device":"mps","device_name":"Apple Metal (unified memory)","budget_bytes":12884901888,"dtype":"float16","architectures":["llama","mistral","qwen2","qwen3"],"weight_mode":"resident","ram_bytes":34359738368}}
```

Controller → node: `{"type":"welcome","version":1,"generation":"connection-uuid"}`.

A new connection always discards GPU assignments and KV state. The disk tensor cache survives. Node IDs match `[A-Za-z0-9_.-]{1,128}`. A duplicate identity replaces the old connection and invalidates its placements.

## RPC

Controller → node: `{"type":"rpc","id":"uuid","method":"...","args":{...}}`.

Node → controller: `{"type":"result","id":"uuid","result":{...}}` or `{"type":"result","id":"uuid","error":"reason","status":503}`.

| Method | Arguments / response |
| --- | --- |
| `offer` | ModelSpec → budget and triangular `costs[start][end-start-1]` matrix |
| `load` | ModelSpec, inclusive `start`, exclusive `end`, deployment `epoch` → ready |
| `unload` | No arguments → releases weights and all request caches |
| `tokenize` | epoch and `messages` or `prompt` → `input_ids`, `eos_ids`; first stage only |
| `forward` | epoch, session_id, position, IDs or hidden tensor, sample flag, temperature/top_p/seed → hidden tensor or token_id |
| `decode` | epoch and accumulated generated `ids` → text; first stage only |
| `release` | epoch and session_id → deletes local KV state |

`ModelSpec` contains model_id, resolved revision SHA, raw Transformers config, generation_config, max_context, and a map of tensor name → shard filename, byte offset/length, dtype and shape. No Hugging Face access token crosses this protocol. Tensors use the standard checkpoint layout and are reindexed locally inside the runtime.

Nodes send asynchronous events: `heartbeat` every 5 seconds, `status` for progress/state, and `log` with level/message. The controller disconnects a node after 30 seconds without events. Node RPCs run in order; heartbeat/status transport remains live while GPU/download work runs in a worker thread.

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
