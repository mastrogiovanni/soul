#!/usr/bin/env bash
# Keep execution inside main so curl | bash reads the complete function first.

fail() { printf 'Soul: %s\n' "$*" >&2; exit 1; }

main() (
    set -euo pipefail
    umask 077

    readonly controller='https://mastrogiovanni.ddns.net/soul'
    readonly enrollment_token='your-node-enrollment-token'
    readonly uv_version='0.9.0'
    local profile=auto start=1 os arch extra device cuda_info cuda_major cuda_minor capabilities
    local install_dir=${SOUL_INSTALL_DIR:-$HOME/.local/share/soul}
    local release='' temporary='' installed=0 uv_bin uv_installed=''

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --profile)
                [ "$#" -ge 2 ] || fail '--profile requires auto, cpu, cu126, cu128 or cu129.'
                profile=$2; shift 2 ;;
            --no-start) start=0; shift ;;
            --) shift; break ;;
            --help|-h)
                cat <<'HELP'
Usage: bash install.sh [--profile auto|cpu|cu126|cu128|cu129] [--no-start] [-- NODE_OPTIONS]

Installs Soul and joins https://mastrogiovanni.ddns.net/soul in the foreground.
Linux x86_64/ARM64 and native Apple Silicon macOS are supported. No sudo needed.
The default node name is the normalized hostname-username (for example mac-studio-michele).
Node options (for example --name or --memory-gb) are saved in the restart command.
SOUL_INSTALL_DIR: absolute install path (default: ~/.local/share/soul).
HELP
                exit 0 ;;
            *) fail "Unknown installer option: $1. Use --help, or -- before node options." ;;
        esac
    done
    case "$profile" in auto|cpu|cu126|cu128|cu129) ;; *) fail "Unknown profile: $profile" ;; esac
    # Community enrollment is fixed even when node options are supplied.
    for argument in "$@"; do
        case "$argument" in --controller|--controller=*|--dev) fail 'The community controller and enrollment token are fixed.' ;; esac
    done
    [[ "$install_dir" = /* && "$install_dir" != / ]] || fail 'SOUL_INSTALL_DIR must be an absolute directory other than /.'
    for tool in curl tar uname mktemp; do
        command -v "$tool" >/dev/null || fail "Install $tool with your system package manager, then rerun this installer."
    done

    os=$(uname -s)
    arch=$(uname -m)
    case "$os/$arch" in
        Darwin/arm64) ;;
        Darwin/x86_64)
            fail 'PyTorch 2.8 requires Apple Silicon on macOS. On Apple Silicon, reopen Terminal without Rosetta.' ;;
        Linux/x86_64|Linux/aarch64|Linux/arm64)
            local libc
            libc=$(getconf GNU_LIBC_VERSION 2>/dev/null) || fail 'Linux requires glibc 2.28 or newer; musl/Alpine is unsupported.'
            [[ "$libc" =~ ^glibc\ ([0-9]+)\.([0-9]+)$ ]] || fail "Unrecognized Linux C library: $libc"
            (( BASH_REMATCH[1] > 2 || (BASH_REMATCH[1] == 2 && BASH_REMATCH[2] >= 28) )) ||
                fail 'PyTorch 2.8 requires glibc 2.28 or newer. Upgrade the Linux distribution.'
            ;;
        *) fail "Unsupported system: $os/$arch. Use Linux x86_64/ARM64 or native Apple Silicon macOS (Windows: use WSL2)." ;;
    esac

    extra=node
    device=cpu
    if [ "$os" = Darwin ]; then
        case "$profile" in auto) device=auto ;; cpu) ;; *) fail 'CUDA profiles require Linux and an NVIDIA GPU.' ;; esac
    elif [ "$profile" != cpu ]; then
        if command -v nvidia-smi >/dev/null; then
            cuda_info=$(nvidia-smi 2>/dev/null) || fail 'nvidia-smi failed. Fix the NVIDIA driver or rerun with --profile cpu.'
            [[ "$cuda_info" =~ CUDA\ Version:\ *([0-9]+)\.([0-9]+) ]] ||
                fail 'Cannot detect the NVIDIA driver CUDA level. Fix the driver or use --profile cpu.'
            cuda_major=${BASH_REMATCH[1]}
            cuda_minor=${BASH_REMATCH[2]}
            if [ "$profile" = auto ]; then
                if [ "$arch" != x86_64 ]; then
                    # The lockfile includes Linux ARM64 CUDA wheels only for cu129.
                    profile=cu129
                else
                    # 12.6 retains Maxwell/Pascal support; Blackwell needs 12.8+.
                    profile=cu126
                    capabilities=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null) ||
                        fail 'Cannot detect NVIDIA GPU architecture. Choose --profile cu126, cu128 or cu129 explicitly.'
                    [ -n "$capabilities" ] || fail 'No NVIDIA GPU found. Use --profile cpu.'
                    while IFS= read -r capability; do
                        [[ "$capability" =~ ^([0-9]+)\.([0-9]+)$ ]] ||
                            fail 'Cannot detect NVIDIA GPU architecture. Choose a CUDA profile explicitly.'
                        if (( BASH_REMATCH[1] >= 10 )); then profile=cu128; fi
                    done <<< "$capabilities"
                fi
            fi
            if [ "$arch" != x86_64 ] && [ "$profile" != cu129 ]; then
                fail 'This lockfile supports Linux ARM64 CUDA only with --profile cu129 (driver CUDA 12.9+).'
            fi
            local required_minor=${profile#cu12}
            (( cuda_major > 12 || (cuda_major == 12 && cuda_minor >= required_minor) )) ||
                fail "Profile $profile needs a driver reporting CUDA 12.$required_minor or newer. Update the driver or use --profile cpu."
            extra=node-$profile
            device=cuda:0
        elif [ "$profile" != auto ]; then
            fail 'nvidia-smi is missing. Install a working NVIDIA driver first, or use --profile cpu.'
        fi
    fi
    printf 'Soul: detected %s/%s; installing %s.\n' "$os" "$arch" "$extra"

    mkdir -p "$install_dir/releases" "$install_dir/bin" "$install_dir/tools"
    temporary=$(mktemp -d "$install_dir/.install.XXXXXX")
    trap 'rm -rf "$temporary"; if [ "$installed" = 0 ] && [ -n "$release" ]; then rm -rf "$release"; fi' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    # Download completely before executing/extracting; curl failures stop installation.
    uv_bin=$install_dir/tools/uv
    if [ -x "$uv_bin" ]; then uv_installed=$("$uv_bin" --version); fi
    if [[ ! "$uv_installed" =~ ^uv\ 0\.9\.0($|\ ) ]]; then
        curl --proto '=https' --tlsv1.2 -fsSL --retry 3 \
            "https://astral.sh/uv/$uv_version/install.sh" -o "$temporary/uv-install.sh"
        UV_UNMANAGED_INSTALL="$install_dir/tools" sh "$temporary/uv-install.sh" </dev/null
    fi
    [ -x "$uv_bin" ] || fail 'uv installation did not produce an executable.'
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 \
        "$controller/install/agent.tar.gz" -o "$temporary/source.tar.gz"
    release=$(mktemp -d "$install_dir/releases/agent.XXXXXX")
    tar -xzf "$temporary/source.tar.gz" -C "$release" --strip-components=1
    [ -f "$release/uv.lock" ] && [ -f "$release/pyproject.toml" ] || fail 'The source archive is missing the uv project or lockfile.'
    printf 'Soul: installing Python 3.12 and locked dependencies (the first download can take several minutes).\n'
    UV_PROJECT_ENVIRONMENT="$release/.venv" "$uv_bin" sync --project "$release" \
        --locked --no-dev --extra "$extra" --python 3.12 </dev/null

    # Exercise the selected device, including any --device override, before joining.
    "$release/.venv/bin/python" - "$device" "$@" <<'PY'
import argparse
import sys

import torch

from soul.node.runtime import select_device

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--device", default=sys.argv[1])
options, _ = parser.parse_known_args(sys.argv[2:])
device = select_device(options.device)
try:
    result = (torch.ones(4, device=device) + 1).sum().item()
    assert result == 8
except Exception as exc:
    raise SystemExit(f"Soul: device check failed on {device}: {exc}. Check your driver/profile or use --profile cpu.") from exc
print(f"Soul: PyTorch {torch.__version__} verified on {device}.")
PY

    # Bash %q preserves spaces and shell metacharacters in paths and saved options.
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n'
        printf 'export SOUL_NODE_TOKEN=%q\n' "$enrollment_token"
        printf 'exec %q node --cache-dir %q --device %q' \
            "$release/.venv/bin/soul" "$HOME/.cache/soul" "$device"
        if [ "$#" -gt 0 ]; then printf ' %q' "$@"; fi
        printf ' "$@" --controller %q\n' "$controller"
    } > "$temporary/soul-node"
    chmod 0755 "$temporary/soul-node"
    mv -f "$temporary/soul-node" "$install_dir/bin/soul-node"
    installed=1
    printf 'Soul: installed. Restart with:\n  %q\n' "$install_dir/bin/soul-node"
    if [ "$start" = 1 ]; then
        printf 'Soul: connecting to %s. Keep this terminal open; Ctrl+C stops the agent.\n' "$controller"
        "$install_dir/bin/soul-node" </dev/null
    fi
)

main "$@"
