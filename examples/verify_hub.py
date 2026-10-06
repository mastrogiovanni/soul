"""Optional small, live Hub / accelerator smoke test. Does not download whole shards."""

import argparse
from pathlib import Path

from soul.catalog import inspect_model
from soul.node.runtime import Runtime

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
spec = inspect_model(args.model, max_context=64)
size = sum(t["length"] for t in spec.tensors.values())
if size > 50 * 1024**2:
    raise SystemExit("This smoke test only accepts checkpoints under 50 MiB.")
print(f"Pinned {args.model}@{spec.revision}: {spec.layers} layers, {size:,} bytes", flush=True)
boundary = spec.layers // 2
nodes = [Runtime(Path(".soul/hub-smoke") / f"node{i}", args.device, memory_gb=1) for i in range(2)]
for node, (start, end) in zip(nodes, [(0, boundary), (boundary, spec.layers)], strict=True):
    node.load(spec, start, end, "smoke")
    print(f"Loaded layers [{start}, {end}) on {node.device}", flush=True)
ids = nodes[0].tokenize({"prompt": "Hello world"})["input_ids"]
position = 0
for _ in range(3):
    request = {
        "epoch": "smoke",
        "session_id": "smoke",
        "position": position,
        "input_ids": ids,
        "temperature": 0,
    }
    for node in nodes:
        result = node.forward(request)
        request.pop("input_ids", None)
        request.pop("hidden", None)
        request.update(result)
    position += len(ids)
    ids = [result["token_id"]]
print(f"OK: prefill and two cached decode steps across two {args.device} stages.", flush=True)
for node in nodes:
    node.unload()
