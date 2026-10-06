# Verification record

## uv migration — 2026-10-06

- **Locked environment:** `uv lock` and `uv sync --locked --extra node --group dev` succeeded with uv 0.9.0 and Python 3.12 on macOS. The lock covers the controller, CPU/Metal, three CUDA profiles, and the development group.
- **Full suite:** `uv run --no-sync pytest -q` passed all **32 tests** in one run, without disabling plugin autoload. Ruff lint and formatting checks passed for all 23 Python files.
- **Platform selection:** Linux x86_64 dry runs selected `torch==2.8.0+cpu`, `2.8.0+cu126`, `2.8.0+cu128`, and `2.8.0+cu129` for their respective extras. All six pairs of incompatible profiles were rejected. Selecting a CUDA profile on macOS failed as expected. These checks resolve packages; they do not validate NVIDIA execution.
- **Controller isolation:** a separate non-editable base install contained no PyTorch. Imports, `/healthz`, the dashboard, and authenticated `/admin/state` succeeded.
- **Demo and UI:** a uv-launched demo started the controller and both CPU nodes and completed an inference. Browser checks verified the CPU/Metal command and all three CUDA command selections in the join dialog, with no console warnings or errors. The temporary demo was stopped after verification.
- **Packaging and Docker:** `uv build` produced a source distribution and wheel. The controller Docker image built and ran on Linux arm64 under Docker Desktop. Its default entrypoint, health endpoint, dashboard, and authenticated admin state worked as a non-root user; PyTorch was absent. The temporary container was stopped and removed.

Reproduction commands and profile selection are in [environment setup](environment.md). GitHub Actions now uses the same locked CPU/development profile; the remote workflow itself was not run during this local verification.

## Initial implementation

Verified during implementation on 2026-10-06:

- **32 automated tests passed** (31 in the full run, followed by all 9 network tests after adding the cold-load cancellation case). Numerical equivalence for Llama, Mistral, Qwen2 and Qwen3; resident and disk-streamed weights; sharded cached decode and chunked prefill; Mistral sliding-window behavior; tied embeddings; tensor validation; corrupt local cache repair; strict HTTP ranges; restart persistence; no PyTorch import in the controller.
- **Live HTTP/WebSocket integration:** two actual node clients; OpenAI and Anthropic SDK parsing for streaming and non-streaming output; completion usage, authentication, invalid input/context errors, reconnect fencing, queue saturation/timeouts, model rotation, orphan-stage unload, injected node execution failure, and cancellation cleanup.
- **Live Hugging Face model:** `hf-internal-testing/tiny-random-LlamaForCausalLM`, revision `9fb191250dd56d0ba7ec9785a025ed29c03d5998`, 4,129,088 tensor bytes. Assigned tensors downloaded with real HTTP ranges; tokenizer fetched from the same revision. Prefill plus two cached decode steps succeeded through two separate runtime stages on CPU, then on Apple Metal (`mps`). The Metal smoke test used one physical Mac GPU; it was not a cross-machine or NVIDIA benchmark.
- **Three-process demo:** controller + two independent CPU processes, each hosting two layers of a locally generated four-layer Llama. A real inference populated the dashboard's placements and counters.
- **Admin UI:** browser inspection verified connection, real model placement and metrics, node inventory, and the node log dialog. No browser console errors or warnings were observed. The screenshot in this directory is from the running demo.
- **Packaging:** editable installation and wheel build validated; static UI assets are included. Ruff lint and format checks passed.

The automated suite is offline except for loopback networking. The optional `examples/verify_hub.py` test downloads a small public model. The initial verification needed disabled third-party pytest plugin autoload due to the machine's unrelated installation; the subsequent uv environment passes without that workaround.

Not verified here: NVIDIA CUDA execution, physical cross-machine mixed-precision pipelines, production-scale checkpoints/long contexts, Raspberry Pi deployment, sustained throughput/load or memory benchmarks, and mobile GPUs. The CUDA selection and execution path is implemented using PyTorch but needs hardware validation before being called production-ready. Transformers is pinned to 4.56.0 and PyTorch to 2.8.0.

The WebSocket dependency emits a deprecation warning under the tested Uvicorn stack. All tests pass; replacing that transport dependency should retain the integration/failure tests.
