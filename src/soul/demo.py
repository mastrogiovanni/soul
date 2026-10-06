"""Zero-download demonstration with real, randomly initialized transformer layers."""

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def create_model(path: Path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Hello", "world", "user", "assistant", ":"]
    words += [f"word{i}" for i in range(9, 32)]
    tokenizer = Tokenizer(WordLevel(dict(zip(words, range(32), strict=True)), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]"
    )
    fast.chat_template = (
        "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}{{ 'assistant: ' }}"
    )
    fast.save_pretrained(path)


def run(port=8000, state_dir=Path(".soul/demo")):
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    model = state_dir / "tiny-random-llama"
    if not (model / "config.json").exists():
        create_model(model)
    prefix = [sys.executable, "-m", "soul.cli"]
    base = f"http://127.0.0.1:{port}"
    # Never attach to an unrelated service that already owns the requested port.
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", port))
    processes, handles = [], []

    def spawn(name, args):
        handle = (state_dir / f"{name}.log").open("a")
        handles.append(handle)
        process = subprocess.Popen(
            prefix + args,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"},
        )
        processes.append(process)

    try:
        spawn(
            "controller",
            [
                "controller",
                "--dev",
                "--port",
                str(port),
                "--state-dir",
                str(state_dir / "controller"),
                "--max-layers-per-node",
                "2",
                "--allow-local-models",
                "--max-context",
                "512",
            ],
        )
        for i in range(2):
            spawn(
                f"node{i}",
                [
                    "node",
                    "--dev",
                    "--controller",
                    base,
                    "--device",
                    "cpu",
                    "--memory-gb",
                    "1",
                    "--name",
                    f"Demo CPU {i + 1}",
                    "--node-id",
                    f"demo-{i + 1}",
                    "--cache-dir",
                    str(state_dir / f"node{i}"),
                ],
            )
        print("Starting a controller and two CPU node processes…", flush=True)
        with httpx.Client(timeout=120, headers={"Authorization": "Bearer soul-dev"}) as http:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError(f"A demo process exited. See logs in {state_dir}")
                try:
                    state = http.get(base + "/admin/state").json()
                    if len([n for n in state["nodes"] if n["connected"]]) == 2:
                        break
                except (httpx.HTTPError, KeyError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("Nodes did not connect within 60 seconds")
            response = http.post(
                base + "/v1/chat/completions",
                json={
                    "model": str(model),
                    "messages": [{"role": "user", "content": "Hello world"}],
                    "max_tokens": 8,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2), flush=True)
        print(
            f"\nAdmin UI: {base}\nAdmin/API key: soul-dev\nModel: {model}\n"
            "This tiny model is untrained: its text is intentionally meaningless.\n"
            "The layer computation, WebSocket transport and counters are real.\nCtrl+C stops all three processes.",
            flush=True,
        )
        while all(p.poll() is None for p in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in processes:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=15)
            if process.poll() is None:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()
