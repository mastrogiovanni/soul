# Finding and adding models

The admin console's **Model browser** opens at `<controller-base>/model-browser`, for example `http://127.0.0.1:8000/model-browser` locally or `https://your-host/soul/model-browser` behind the shared ingress. It uses the same admin key as the control room.

Search the live Hugging Face catalog by repository name or publisher. Choose a family shortcut, or enter any architecture tag in **Architecture / family**. Use **Next / Previous** to navigate the Hub's pages. The compatibility dropdown filters the current result page; it does not claim to enumerate every compatible checkpoint. Hub architecture tags can be incomplete, so use an unfiltered name search when a repository is missing. Private repositories are omitted from discovery; add one by exact ID through the control room with an authorized `HF_TOKEN`.

Search results have four states:

| State | Meaning |
| --- | --- |
| Supported format · check layout | The pinned repository configuration matches a supported architecture and quantization format; tensors have not been verified yet. |
| Supported · layout verified | Configuration and safetensors tensor headers have passed the controller's checks. This is not a full-size hardware benchmark or a guarantee of available memory. |
| Unsupported | The architecture, format, or tensor layout needs another adapter. The card explains why. |
| Not checked | Metadata could not be read or is incomplete. Gated models may require accepted license terms and `HF_TOKEN`. Retry **Check compatibility** after resolving access. |

## Current compatibility

| Architecture (`model_type`) | Unquantized safetensors | AWQ GEMM 4-bit | GPTQ v1 4-bit |
| --- | --- | --- | --- |
| `llama` (including SmolLM2) | Yes | Yes | Yes |
| `mistral` | Yes | Yes | Yes |
| `qwen2` (Qwen2 / Qwen2.5) | Yes | Yes | Yes |
| `qwen3` (dense Qwen3) | Yes | Yes | Yes |
| `qwen3_5` / `qwen3_5_text` (Qwen3.5) | No | No | No |
| MoE, vision-language, hybrid, and other architectures | No | No | No |

Quantization compresses weights; it does not make an unsupported architecture compatible. In particular, quantizing `Qwen/Qwen3.5-27B` will not let the current Soul runtime execute it.

AWQ requires AutoAWQ **GEMM**, `bits: 4`, `zero_point: true`, and a positive `group_size` (or `-1` for a whole input channel dimension). GPTQ requires the **v1 `gptq` checkpoint format**, `bits: 4`, int32 packing, and `qweight`, `qzeros`, `scales`, and `g_idx` tensors for converted linear layers. GPTQ activation ordering (`desc_act`) is supported through the saved group indices. Unconverted projections may remain floating point. Embeddings and output heads must remain floating point.

Not supported: GGUF, AWQ GEMV/Marlin, GPTQ v2/Marlin, FP8, bitsandbytes/NF4, mixed-bit settings, quantized output heads, or custom remote model code. Soul does not quantize an unquantized repository on the fly; select a publisher's compatible quantized checkpoint.

## Add a quantized checkpoint

1. **Update the controller and participating nodes.** Deploy this source version to your controller. For this repository's Kind deployment, run `make up`. For nodes running from a checkout, update to the same source version, run `uv sync --locked --extra node` (or the appropriate CUDA extra), and restart the node. For installer-managed nodes, deploy the controller first, stop the old agent, then rerun the installation command with your saved options as described in [installation](install.md#restart-update-and-remove). Existing model caches are retained. Old nodes can still run floating-point models but will not be assigned quantized checkpoints.
2. **Find a compatible repository.** Examples whose publisher configurations use the supported layouts are [`Qwen/Qwen2.5-7B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ) and [`Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4). Choose the full repository ID, including the quantization suffix. No additional AWQ/GPTQ package is needed on nodes.
3. **Click Check compatibility.** Soul reads the configuration and safetensors headers, not weight payloads. The dialog shows the pinned revision, layers, configured context, weight memory estimates for FP16/FP32 nodes, download size, and count of compatible connected nodes.
4. **Click Prepare model.** Choose **Preload to disk** to download assigned shards into node caches, or **Preload and load GPU** to also materialize the pipeline. The browser passes the checked commit to preparation. If the controller already knows a different pinned revision of the same model, it rejects the change rather than silently replacing cached weights.
5. **Wait for available / ready in the control room.** `available` means inference can load the cached model; `ready` means it is already resident. `partial` includes an error describing missing capacity, nodes, or download failures. The weight figures are not total runtime memory: KV cache, workspace, embeddings/head placement, allocator overhead and per-node budgets also matter.
6. **Choose Use model** in the inventory for API/client instructions. Send the exact quantized repository ID as the request's `model`. There is no separate adapter upload or quantization switch.

You can also paste the ID directly into **Preload a model** in the control room. The same compatibility checks run automatically. For gated repositories, accept the publisher's terms in Hugging Face and configure `HF_TOKEN` separately on the controller and each node; the controller does not forward its token to nodes.

## API example

The controller base URL includes its path prefix, if any. Export your existing admin key locally; do not put it in model IDs or URLs.

```sh
export SOUL_BASE_URL='http://127.0.0.1:8000'
export SOUL_ADMIN_KEY='<your-admin-key>'

# Read-only compatibility check; no model weights downloaded or model registered.
curl --fail-with-body "$SOUL_BASE_URL/admin/catalog/check" \
  -H "X-Soul-Admin-Key: $SOUL_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct-AWQ"}'

# Replace the revision with the commit returned by the check.
curl --fail-with-body "$SOUL_BASE_URL/admin/models/preload" \
  -H "X-Soul-Admin-Key: $SOUL_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct-AWQ","revision":"<checked-commit-sha>"}'
```

Omitting `revision` on the first preparation resolves and pins `main`; future preparations reuse the pinned revision. Use `/admin/models/load` instead of `/preload` to load the pipeline immediately. Shared ingress Basic Auth remains separate; retain its normal credentials alongside `X-Soul-Admin-Key`.

## Memory, devices, and verification

Packed int32 weight/zero tensors and group indices retain their integer representation. Scales and unquantized weights use each node's floating dtype. During a matrix multiplication the adapter dequantizes at most 128 output columns at a time. Memory offers include packed storage plus temporary dequantization workspace; disk-streaming hooks load and release packed buffers along with ordinary layer parameters. KV cache and activations remain floating point.

The implementation uses ordinary PyTorch operations with the existing CPU, CUDA and Metal runtime; it does not require CUDA-only extensions. The numerical regression suite compares packed distributed execution with independently reconstructed dense reference weights across all four architectures, resident/stream modes, chunked prefill, cached decode, GPTQ activation ordering and cache restart. CPU tests always run; optional Metal and CUDA numerical tests run when those devices are available. This implementation was also verified on Apple Metal. Real-device speed and memory capacity need measurement on your hardware; this capacity-oriented adapter can be slower than specialized AWQ/GPTQ GPU kernels and unquantized inference. Quantized outputs can differ from the original unquantized checkpoint.

Format references: [AutoAWQ packing](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/utils/packing_utils.py), [AutoGPTQ packed linear layout](https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_cuda.py).
