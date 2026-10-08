# Verification record

## Serving during model downloads — 2026-10-08

- **Automated suite:** **168 tests passed, 4 CUDA tests skipped** because CUDA was unavailable. Nine new integration cases hold real node downloads open while checking controller state, model listings, normal and streaming inference, both GPU-resident and disk-cached models, explicit and background preparation, prompt rejection of incomplete models, download failures, cancellation/reconnect fencing, queued cache clearing, and legacy-node compatibility. Inference finishes before the test releases downloads.
- **Browser:** an isolated controller under `/soul` and two CPU nodes completed an inference while both nodes' other-model downloads were held. The Nodes page showed **ready** alongside the separate download model/progress; the downloading model's topology showed **downloading**. No browser console warnings/errors appeared. The temporary controller, nodes, and browser tab were stopped afterward.
- **Static checks:** Ruff lint/format, JavaScript syntax, and whitespace checks passed. Existing dependency deprecation warnings remain. This verifies concurrent disk preparation and serving, not simultaneous generations or production-scale I/O throughput. Controller and node agents must both be updated for same-node download/inference concurrency; the production controller was not redeployed here.

## Model deletion and node cache cleanup — 2026-10-08

- **Automated suite:** **159 tests passed, 4 CUDA tests skipped** because CUDA was unavailable. New tests exercise admin authentication, loaded-model removal, retained models, offline cleanup and reconnects, restart persistence, older node capability handling, filesystem errors/retries, per-node download pause/resume, damaged manifests, invalid cache keys, symlink boundaries, repeated deletion, and serialization with inference. An initial run encountered an inference timeout; its isolated rerun and the full rerun passed.
- **Browser:** a disposable controller under `/soul` verified both confirmation dialogs and cancellation. Local API operations removed real fixture caches; the UI showed a cleared node with **Resume downloads**, pending cleanup for offline nodes, and an empty model catalog/topology after deletion. No browser console errors appeared. Temporary nodes, controller, and browser tab were stopped afterward.
- **Static checks:** Ruff lint/format, JavaScript syntax, and whitespace checks passed. Existing dependency deprecation warnings remain. Production caches were not touched or redeployed; controller and node agents must both be updated to enable the new cache-management RPCs.

## Partial model distribution — 2026-10-08

- **Automated suite:** **147 tests passed, 4 CUDA tests skipped** because CUDA was unavailable. New real HTTP/WebSocket tests cover unquantized, AWQ and GPTQ models downloaded across an initially undersized cluster, expansion one node at a time, reuse without redownloading cached stages, inference only after full coverage, offline/reconnecting nodes, no-node persistence, and retained tensors after an interrupted download.
- **Placement:** regression tests exercise memory and disk limits, maximum layers per node, missing edge-stage capacity, gaps, and reuse of a cached suffix. Strict inference placement still rejects incomplete coverage.
- **Browser:** an isolated controller under `/soul` and four small CPU nodes verified the preload form, exclamation mark, byte/tensor/layer progress, manual **Distribute missing** retry, and automatic progression from 2/4 to 3/4 to 4/4 cached layers as nodes joined. The warning and pending notice cleared when the model became available. No browser console warnings or errors appeared.
- **Static checks:** Ruff lint/format, JavaScript syntax and whitespace checks passed. Tests retain existing transport/dependency deprecation warnings. This validates local test models; production-scale checkpoints and physical multi-machine capacity were not benchmarked, and the running production controller was not redeployed.

## Kubernetes and `/soul` context — 2026-10-07

- **Application tests:** all **48 tests** passed, including mounted-path routing, relative redirects/assets, Swagger/OpenAPI paths, role-specific authentication, and real node WebSockets plus OpenAI streaming/Anthropic requests through a simulated Basic Auth ingress. Three additional deployment-workflow tests passed, covering preflight before build, private/idempotent credential creation, and data-preserving removal.
- **Manifests and tooling:** Kustomize rendered the default resources and namespace/image overrides successfully. Bash syntax, JavaScript syntax, Ruff lint/format, and whitespace checks passed.
- **Container:** `make build` produced the controller-only Linux arm64 image. A disposable container ran as UID 1000 with a read-only root filesystem, dropped capabilities, no privilege escalation, writable temporary storage and a data volume. Health, `/soul` redirects, assets, docs, authorization and API paths worked. SQLite request history survived restart; PyTorch and Transformers were absent.
- **Browser:** the container's `/soul/` panel connected successfully after restart, displayed persisted history, generated a node command ending in `/soul`, and linked to `/soul/docs`. No console warnings/errors appeared. The temporary container, volume and browser tab were removed afterward.
- **Deployment limitation:** `make up` stopped in preflight because `kind-k8s` is not configured on this machine. No shared cluster resources or ingress settings were changed. Live Contour routing, public HTTPS/certificate validation and the hostPath mount have not been exercised here; the shared platform must exist before deployment.

## Model inventories and proactive preloading — 2026-10-07

- **Automated suite:** all **40 tests** passed using `uv run --no-sync pytest -q`. New integration coverage exercises disk-only preloading, cache inventories on reconnect, inference without tensor downloads, model listing across GPU rotation/unload, offline-only removal, automatic shard recovery, persisted catalog discovery, and corrupt-cache exclusion.
- **Admin UI:** an isolated uv demo on port 8787 ran a controller and two CPU nodes. Browser checks verified cached layer ranges, GPU unload retaining availability, the disk-only preload form, and removal of a third disposable node after disconnection. Connected nodes had no removal button. No browser console warnings or errors appeared.
- **Static checks:** Ruff lint/format checks, JavaScript syntax checking, and `git diff --check` passed. The existing WebSocket deprecation warnings remain.
- **Scope:** the existing controller on port 8000 and the user's real nodes were not restarted. This change uses protocol version 2; upgrade/restart both controller and nodes together. Validation used CPU processes, not physical mixed Mac/NVIDIA machines or large checkpoints.

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
