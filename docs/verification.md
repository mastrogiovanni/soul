# Verification record

Verified during implementation on 2026-10-06:

- **32 automated tests passed** (31 in the full run, followed by all 9 network tests after adding the cold-load cancellation case). Numerical equivalence for Llama, Mistral, Qwen2 and Qwen3; resident and disk-streamed weights; sharded cached decode and chunked prefill; Mistral sliding-window behavior; tied embeddings; tensor validation; corrupt local cache repair; strict HTTP ranges; restart persistence; no PyTorch import in the controller.
- **Live HTTP/WebSocket integration:** two actual node clients; OpenAI and Anthropic SDK parsing for streaming and non-streaming output; completion usage, authentication, invalid input/context errors, reconnect fencing, queue saturation/timeouts, model rotation, orphan-stage unload, injected node execution failure, and cancellation cleanup.
- **Live Hugging Face model:** `hf-internal-testing/tiny-random-LlamaForCausalLM`, revision `9fb191250dd56d0ba7ec9785a025ed29c03d5998`, 4,129,088 tensor bytes. Assigned tensors downloaded with real HTTP ranges; tokenizer fetched from the same revision. Prefill plus two cached decode steps succeeded through two separate runtime stages on CPU, then on Apple Metal (`mps`). The Metal smoke test used one physical Mac GPU; it was not a cross-machine or NVIDIA benchmark.
- **Three-process demo:** controller + two independent CPU processes, each hosting two layers of a locally generated four-layer Llama. A real inference populated the dashboard's placements and counters.
- **Admin UI:** browser inspection verified connection, real model placement and metrics, node inventory, and the node log dialog. No browser console errors or warnings were observed. The screenshot in this directory is from the running demo.
- **Packaging:** editable installation and wheel build validated; static UI assets are included. Ruff lint and format checks passed.

The automated suite is offline except for loopback networking. The optional `examples/verify_hub.py` test downloads a small public model. The existing machine's unrelated pytest installation required an isolated project environment and disabled third-party plugin autoload for the verification run.

Not verified here: NVIDIA CUDA execution, physical cross-machine mixed-precision pipelines, production-scale checkpoints/long contexts, Raspberry Pi deployment, Docker image runtime, sustained throughput/load or memory benchmarks, and mobile GPUs. The CUDA selection and execution path is implemented using PyTorch but needs hardware validation before being called production-ready. Transformers is pinned to 4.56.0 and PyTorch to 2.8.x.

The WebSocket dependency emits a deprecation warning under the tested Uvicorn stack. All tests pass; replacing that transport dependency should retain the integration/failure tests.
