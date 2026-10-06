# Mobile nodes and the next phase

Phase 2 is not implemented by this release. The controller/node split keeps the GPU runtime local and the wire format language-neutral so it can be added without changing the API clients.

## Suggested sequence

1. **Benchmark the working pipeline first.** Measure first-token latency, decode token/s, boundary payload bytes, controller relay bandwidth, per-stage timing, peak accelerator memory, and streaming-mode disk throughput on a Mac + NVIDIA pair. Establish the network/compute tradeoff before distributing to phones.
2. **Add capability negotiation beyond architecture names.** Advertise adapter version, activation formats, operators, quantization, maximum buffer size/context, thermal/battery constraints and storage quota. Require controller acceptance of an exact compatible adapter contract.
3. **Implement one mobile model family end to end.** Use a small dense Llama/Qwen-family model and numerical reference fixtures from phase 1. Stage boundaries must preserve residuals, rotary position, attention mask, and cache state exactly.
4. **Choose a phone runtime from measured constraints.** A native Swift/Metal or Android GPU runtime offers background/lifecycle and memory controls. A WebGPU page lowers installation friction but is limited by browser GPU features, storage quotas, Hub CORS, tab suspension and mobile thermal throttling. Support one path well before attempting both.
5. **Change browser admission explicitly.** Current nodes authenticate through a WebSocket header and reject browser Origin. A browser cannot set that header directly. Add short-lived, scoped enrollment credentials and an origin-checked session flow; do not put permanent credentials in URLs or expose the shared node token to a webpage.
6. **Add per-node public-key identities and stronger workload isolation** before using an untrusted volunteer network. Phones need explicit controls for charging-only operation, bandwidth, battery, temperature and user consent to store model data.

## Serving improvements before scale

- Continuous batching and pipeline microbatches, with per-session KV admission and fair scheduling.
- Multiple independent pipelines, replica selection and measured speed/network-aware placement.
- Better node permutation search for asymmetric embedding/output-head sizes.
- Broader architecture adapters, each gated by full-vs-sharded tests, rather than unsafe universal model loading.
- Quantized weight formats, paged/offloaded KV, sliding-window-aware memory estimates and prefix caching.
- SSD weight prefetch, double buffering, async transfer overlap, and disk quotas/eviction.
- Request/stream deadlines, richer node-level throughput/temperature metrics, durable log aggregation, metrics export and operational dashboards.
- Controller high availability, deployment reconciler and durable jobs only if the intended operating environment needs them.

Training is a separate future objective: backward activations/gradients, optimizer states, training microbatches, synchronization and checkpoint semantics are absent from the inference protocol.
