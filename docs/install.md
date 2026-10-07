# Join the Soul community

Run this in a terminal on the machine you want to contribute:

```sh
curl -fsSL https://mastrogiovanni.ddns.net/soul/install.sh | bash
```

The script installs the agent and starts it in the foreground. The controller is always `https://mastrogiovanni.ddns.net/soul`, and the enrollment token is always `your-node-enrollment-token`; both are built into the installer and its generated launcher. No controller or admin setup is needed to join.

The controller serves both `/soul/install.sh` and `/soul/install/agent.tar.gz`. The archive contains the deployed package's source and lockfile, so agent installations follow the controller version and do not need access to the GitHub repository.

Keep the terminal open while contributing. `Ctrl+C` stops the agent; this installer does not create a boot or login service. The node connects outward over HTTPS/WSS and needs no inbound ports. It reconnects automatically after temporary network failures. Its name defaults to your hostname, and its logs report connection errors if it cannot reach or enroll with the controller.

## Supported systems and automatic selection

| Machine | Automatic installation |
| --- | --- |
| Native Apple Silicon macOS | `node` profile; Metal when available, otherwise CPU |
| Linux x86_64 without `nvidia-smi` | `node` CPU profile |
| Linux ARM64 without `nvidia-smi` | `node` CPU profile |
| Linux x86_64 with NVIDIA | `node-cu126` for pre-Blackwell GPUs; `node-cu128` when Blackwell is detected |
| Linux ARM64 with NVIDIA | `node-cu129`, the ARM64 CUDA wheel included in this lockfile |

Linux needs **glibc 2.28 or newer**. Alpine/musl, Intel macOS, 32-bit systems, and native Windows are unsupported. On Windows, run the installer inside a supported WSL2 Linux environment. On Apple Silicon, use a native terminal, with Rosetta disabled.

The script requires Bash, `curl`, `tar`, and standard Unix utilities. It installs its own **uv 0.9.0**, obtains **Python 3.12** as needed, and syncs the selected extra from `uv.lock`. It does not require Git, a preinstalled Python, or administrator privileges. Shell startup files are unchanged. See [uv's installer options](https://docs.astral.sh/uv/reference/installer/) for the isolated uv installation mechanism.

For NVIDIA, `nvidia-smi` must work and report a supported CUDA level: at least 12.6, 12.8, or 12.9 for the corresponding profile. This is a conservative driver check; the script does not install system drivers or rely on CUDA minor-version compatibility. GPU architecture also matters: [PyTorch 2.8's CUDA builds](https://pytorch.org/get-started/previous-versions/#v280) differ in [support for older NVIDIA GPUs](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128). After installing, the script runs a small computation on the selected device before starting the node.

The initial dependency download can take several minutes and several GB for CUDA. Assigned model tensors are downloaded separately when the controller prepares models. A node normally budgets 75% of available device memory; downloaded tensors stay on disk for reuse, with no automatic disk quota.

## Options and resource limits

Pass installer options through `bash -s --`. Put node options after a second `--`; they are saved in the restart command:

```sh
# Set a recognizable node name and cap its memory budget at 8 GiB.
curl -fsSL https://mastrogiovanni.ddns.net/soul/install.sh \
  | bash -s -- -- --name my-community-node --memory-gb 8

# Use CPU even if an NVIDIA GPU is present.
curl -fsSL https://mastrogiovanni.ddns.net/soul/install.sh \
  | bash -s -- --profile cpu

# Choose a CUDA profile explicitly and install without starting yet.
curl -fsSL https://mastrogiovanni.ddns.net/soul/install.sh \
  | bash -s -- --profile cu128 --no-start
```

Available profiles: `auto` (default), `cpu`, `cu126`, `cu128`, and `cu129`. Additional node options include `--device cuda:1`, `--weight-mode stream`, and `--cache-dir /path/to/storage`. Use one cache directory per node process. On machines with different GPU generations, choose a profile compatible with the device you intend to use, and pass `--device cuda:N` explicitly. `CUDA_VISIBLE_DEVICES` can also restrict GPU visibility to PyTorch.

To inspect the script or see its help before installation:

```sh
curl -fsSL https://mastrogiovanni.ddns.net/soul/install.sh -o install.sh
less install.sh
bash install.sh --help
bash install.sh
```

From an existing checkout, `bash install.sh` runs the same workflow and downloads the source served by the community controller; it does not install uncommitted checkout changes.

## Restart, update, and remove

Restart from any directory:

```sh
~/.local/share/soul/bin/soul-node

# Override saved node options for this run.
~/.local/share/soul/bin/soul-node --memory-gb 4
```

| Location | Contents |
| --- | --- |
| `~/.local/share/soul/bin/soul-node` | Launcher with the community configuration and saved node options |
| `~/.local/share/soul/releases/agent.*` | Downloaded source and isolated Python environment for each installation |
| `~/.local/share/soul/tools` | Installer-managed uv executable |
| `~/.cache/soul` | Persistent node identity, downloaded model tensors, and `node.log` (unless `--cache-dir` is supplied) |

To update, stop the agent, then rerun the installation command with the options you want to retain. Each run downloads the agent source served by the community controller into a new directory and replaces the launcher only after dependency installation and device checks pass. A failed installation leaves the previous launcher and release available. Node identity and cached models remain in the same cache directory. Old release directories are retained; after stopping the agent, you can remove unused releases, keeping the one referenced in `bin/soul-node`.

For a custom location, set `SOUL_INSTALL_DIR` (absolute path) on the **bash side** of the pipe:

```sh
curl -fsSL https://mastrogiovanni.ddns.net/soul/install.sh \
  | SOUL_INSTALL_DIR="$HOME/soul-agent" bash -s -- --no-start
```

The installer prints the resulting restart command. To remove the installation, stop the agent and delete its installation directory. Delete the node cache separately only if you also want to discard its identity and model downloads. uv's shared download cache and managed Python installations use uv's usual storage locations.

## Troubleshooting

- **NVIDIA detection or device check fails:** verify `nvidia-smi`, update the driver if necessary, select a compatible `--profile`, or use `--profile cpu`. See the [CUDA environment guide](environment.md#linux-with-nvidia-cuda).
- **No compatible wheel:** check the OS/CPU architecture and glibc version. Do not replace the pinned PyTorch version with an arbitrary release.
- **Controller connection or authorization fails:** check internet access and `~/.cache/soul/node.log`. The community must accept the built-in enrollment token. If its reverse proxy requires Basic Auth, export `SOUL_INGRESS_USERNAME` and `SOUL_INGRESS_PASSWORD` before starting the launcher.
- **Model download needs access:** for gated/private models, export `HF_TOKEN` in the shell that runs the agent. The controller also needs access to that model.
- **Want to run your own controller?** Use the [manual network setup](../README.md#run-your-own-network). This community installer always targets the community controller.

## Publishing the installer

Deploy a controller build containing these changes to make the command available. The container and Python wheel include the installer and its build inputs; no separate download server or public GitHub repository is needed. The installer endpoints require no Soul API/admin key. Any reverse proxy in front of the community controller must allow community access to these endpoints and node enrollment. The deployed controller must use `SOUL_NODE_TOKEN=your-node-enrollment-token` to accept the installer's agents.
