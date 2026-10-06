# Python environments with uv

Soul is a packaged uv project. Run commands from the repository root so uv can find `pyproject.toml`, `uv.lock`, and `.python-version`. The project supports Python 3.11–3.13 and selects Python 3.12 by default. uv creates `.venv` and downloads Python if a suitable interpreter is unavailable. No manual activation is necessary.

## Install uv

On Linux or macOS, follow the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/). The standalone installer is:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell if necessary, then check `uv --version`. Soul requires uv 0.9.0 or newer; CI and Docker currently use 0.9.0.

## Select an environment

| Purpose | Installation command | PyTorch build |
| --- | --- | --- |
| Controller only | `uv sync --locked` | None |
| CPU node or local demo | `uv sync --locked --extra node` | 2.8.0 CPU on Linux |
| Apple Silicon node | `uv sync --locked --extra node` | 2.8.0 with Metal support on macOS |
| NVIDIA CUDA 12.6 | `uv sync --locked --extra node-cu126` | 2.8.0+cu126 |
| NVIDIA CUDA 12.8 | `uv sync --locked --extra node-cu128` | 2.8.0+cu128 |
| NVIDIA CUDA 12.9 | `uv sync --locked --extra node-cu129` | 2.8.0+cu129 |
| Development on CPU/Mac | `uv sync --locked --extra node --group dev` | CPU/Metal plus tests, lint, and API SDKs |

CUDA profiles include all node dependencies. Select only one node extra per environment; uv rejects combinations and `--all-extras`. CUDA profiles require a platform with an official CUDA wheel and do not install on macOS. These instructions target Linux NVIDIA hosts and native Apple Silicon Macs; a CPU-only controller does not need any node extra.

`uv.lock` records exact versions, hashes, and indexes for all profiles. CPU and CUDA wheels come from [PyTorch's official indexes](https://pytorch.org/get-started/previous-versions/#v280); macOS uses PyPI. The PyTorch indexes are explicit, so unrelated packages continue to come from PyPI. See [uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/) for the source-selection mechanism.

Repeat your selected extra when running a command:

```sh
uv run --locked --extra node soul demo
uv run --locked --extra node-cu128 soul node --controller https://controller.example.com --device cuda:0
```

Alternatively, after a successful sync, `uv run --no-sync ...` uses that environment as-is. This is useful for tests, authentication, or several node processes sharing one Python environment. Each node process must still have its own model cache directory. Resync after pulling dependency changes. A plain `uv sync --locked` selects the controller environment and removes node and development dependencies; do not run it in an environment used by running nodes.

## Linux with NVIDIA CUDA

Check the GPU and installed driver first:

```sh
nvidia-smi
```

The reported CUDA version is the driver's supported CUDA level, not proof that a toolkit is installed. Choose a PyTorch CUDA build compatible with both your GPU and driver. A current driver supporting the selected CUDA runtime avoids reliance on CUDA minor-version compatibility, which has feature limitations; see [NVIDIA's compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). The Python packages provide the runtime libraries used by these wheels; you still need a working system NVIDIA driver. Soul does not require a separate CUDA toolkit for its prebuilt PyTorch runtime.

For a host compatible with CUDA 12.8:

```sh
uv sync --locked --extra node-cu128
uv run --locked --extra node-cu128 python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; x = torch.ones(4, device='cuda'); print('GPU:', torch.cuda.get_device_name(0)); print('GPU result:', (x + x).tolist())"

export SOUL_NODE_TOKEN='your-node-enrollment-token'
uv run --locked --extra node-cu128 soul node \
  --controller https://controller.example.com --device cuda:0 --name gpu-01
```

Replace `node-cu128` with `node-cu126` or `node-cu129` throughout when selecting another build. If the driver cannot support an available build, update the driver before starting the node. If the GPU is unsupported by that PyTorch build, installing a newer driver alone will not add kernel support.

- `torch.version.cuda` is `None`: the environment contains CPU/Metal PyTorch. Sync the intended CUDA profile and use that same profile in subsequent commands.
- `nvidia-smi` fails: fix the host driver or GPU visibility first. uv installs Python packages, not drivers.
- CUDA is unavailable despite the correct wheel: inspect driver compatibility, `CUDA_VISIBLE_DEVICES`, and GPU access if running in a container.
- No matching wheel: check the interpreter, OS, and CPU architecture. The default is Python 3.12; older Linux distributions may need a newer glibc for official wheels. Do not substitute an untested PyTorch version just to satisfy installation.

## Migrate an existing checkout

Stop processes that use the environment before syncing it. Exit any manually activated virtual environment with `deactivate` (or leave your Conda environment), then run the sync command for your chosen profile. uv can reuse a compatible `.venv`; its exact sync removes packages that are not in the selected profile. Other environments, such as `.venv-client`, are not changed.

If you need to preserve the old environment, use a different project environment directory:

```sh
export UV_PROJECT_ENVIRONMENT=.venv-uv
uv sync --locked --extra node
uv run --locked --extra node soul demo
```

Keep `UV_PROJECT_ENVIRONMENT` set for all commands using that environment. Stop relying on manually installed CUDA wheels: the extra and lockfile now select the wheel. Development dependencies moved from the old `dev` extra into `--group dev`.

## Development and dependency updates

```sh
uv sync --locked --extra node --group dev
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests examples
uv run --no-sync ruff format --check src tests examples
uv build
```

For CUDA development, substitute the matching CUDA extra in the sync command. The demo launches child processes with the parent Python interpreter, so they inherit the environment selected by uv.

`--locked` fails if the dependency declarations and lockfile disagree. When intentionally changing dependencies, edit `pyproject.toml`, run `uv lock`, sync the chosen profile with `--group dev`, and rerun the checks. Use `uv lock --upgrade-package PACKAGE` for a deliberate update within the declared constraints. Include `pyproject.toml` and `uv.lock` together in dependency changes. Update all node profiles together when changing their shared runtime dependencies; PyTorch and Transformers upgrades need the numerical equivalence tests.

## Controller container

The Dockerfile installs only locked controller dependencies into `/app/.venv` using uv. Its entrypoint uses the installed `soul` executable directly, so starting the container does not resolve or download packages. Set the three controller credentials from the [README](../README.md#controller), then run:

```sh
docker compose up --build -d
```

This container does not install PyTorch or run GPU nodes. Launch Mac nodes natively and configure Linux nodes with the selected CUDA environment.
