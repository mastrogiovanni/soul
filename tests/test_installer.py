"""Run the real piped Bash installer with offline downloads and simulated hosts."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = (ROOT / "install.sh").read_text()
CONTROLLER = "https://mastrogiovanni.ddns.net/soul"
TOKEN = "your-node-enrollment-token"


def executable(path, source):
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o755)


@pytest.fixture
def host(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # Spaces and quotes catch accidental shell interpolation in the saved launcher.
    install = tmp_path / "Soul's installation"
    log = tmp_path / "commands.jsonl"
    env = {
        **os.environ,
        "PATH": f"{binaries}:/usr/bin:/bin",
        "SOUL_INSTALL_DIR": str(install),
        "MOCK_LOG": str(log),
        "MOCK_OS": "Linux",
        "MOCK_ARCH": "x86_64",
        "MOCK_LIBC": "glibc 2.36",
    }
    record = """import json, os, sys
from pathlib import Path
def record(command, **data):
    with Path(os.environ['MOCK_LOG']).open('a') as log:
        log.write(json.dumps(dict(command=command, args=sys.argv[1:], **data)) + '\\n')
"""
    executable(
        binaries / "uname",
        "import os, sys\nprint(os.environ['MOCK_OS' if sys.argv[1] == '-s' else 'MOCK_ARCH'])\n",
    )
    executable(binaries / "getconf", "import os\nprint(os.environ['MOCK_LIBC'])\n")
    soul_source = record + "record('soul', token=os.environ.get('SOUL_NODE_TOKEN'))\n"
    python_source = (
        record
        + """record('device-check')
assert 'torch.ones' in sys.stdin.read()
raise SystemExit(int(os.environ.get('MOCK_DEVICE_FAIL', '0')))
"""
    )
    uv_source = (
        record
        + f"""if sys.argv[1] == '--version':
    print('uv 0.9.0 (test build)')
    raise SystemExit(0)
record('uv', environment=os.environ.get('UV_PROJECT_ENVIRONMENT'))
if os.environ.get('MOCK_SYNC_FAIL'):
    raise SystemExit(1)
project = Path(sys.argv[sys.argv.index('--project') + 1])
assert os.environ['UV_PROJECT_ENVIRONMENT'] == str(project / '.venv')
assert (project / 'uv.lock').is_file()
destination = project / '.venv/bin'
destination.mkdir(parents=True)
for name, source in [('soul', {soul_source!r}), ('python', {python_source!r})]:
    command = destination / name
    command.write_text({f"#!{sys.executable}"!r} + '\\n' + source)
    command.chmod(0o755)
"""
    )
    uv = tmp_path / "mock-uv"
    executable(uv, uv_source)
    env["MOCK_UV"] = str(uv)
    project = tmp_path / "source"
    project.mkdir()
    for name in ("uv.lock", "pyproject.toml"):
        (project / name).write_text("fixture\n")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(project, arcname="distributed-system-main")
    env["MOCK_ARCHIVE"] = str(archive)
    executable(
        binaries / "curl",
        record
        + """import shutil
record('curl')
url = next(arg for arg in sys.argv if arg.startswith('https://'))
destination = Path(sys.argv[sys.argv.index('-o') + 1])
if os.environ.get('MOCK_DOWNLOAD_FAIL', 'never-match') in url:
    raise SystemExit(22)
if 'astral.sh' in url:
    destination.write_text('set -eu\\ncp "$MOCK_UV" "$UV_UNMANAGED_INSTALL/uv"\\n')
else:
    shutil.copyfile(os.environ['MOCK_ARCHIVE'], destination)
""",
    )

    class Host:
        def run(self, *args):
            # Feeding stdin reproduces curl | bash, including '--' option handling.
            return subprocess.run(
                ["/bin/bash", "-s", "--", *args],
                input=INSTALLER,
                text=True,
                capture_output=True,
                env=env,
                cwd=tmp_path,
                timeout=20,
            )

        def commands(self, name):
            if not log.exists():
                return []
            return [c for line in log.read_text().splitlines() if (c := json.loads(line))["command"] == name]

        def nvidia(self, version="12.8", capabilities="8.6", broken=False):
            executable(
                binaries / "nvidia-smi",
                f"import sys\n"
                f"if {broken!r}: raise SystemExit(1)\n"
                f"print({capabilities!r} if len(sys.argv) > 1 else 'CUDA Version: {version}')\n",
            )

    machine = Host()
    machine.env = env
    machine.install = install
    machine.launcher = install / "bin/soul-node"
    return machine


@pytest.mark.parametrize(
    ("os_name", "arch", "cuda", "capabilities", "extra", "device"),
    [
        ("Linux", "x86_64", None, None, "node", "cpu"),
        ("Linux", "aarch64", None, None, "node", "cpu"),
        ("Darwin", "arm64", None, None, "node", "auto"),
        ("Linux", "x86_64", "12.6", "6.1", "node-cu126", "cuda:0"),
        ("Linux", "x86_64", "13.0", "6.1", "node-cu126", "cuda:0"),
        ("Linux", "x86_64", "12.8", "8.6", "node-cu126", "cuda:0"),
        ("Linux", "x86_64", "12.8", "12.0", "node-cu128", "cuda:0"),
        ("Linux", "x86_64", "13.0", "8.6\n10.0", "node-cu128", "cuda:0"),
        ("Linux", "aarch64", "12.9", "9.0", "node-cu129", "cuda:0"),
    ],
)
def test_piped_install_selects_profile_and_starts_community_agent(
    host, os_name, arch, cuda, capabilities, extra, device
):
    host.env.update(MOCK_OS=os_name, MOCK_ARCH=arch, SOUL_NODE_TOKEN="wrong-inherited-token")
    if cuda:
        host.nvidia(cuda, capabilities)
    result = host.run()
    assert result.returncode == 0, result.stderr
    sync = host.commands("uv")[0]["args"]
    assert sync[sync.index("--extra") + 1] == extra
    assert sync[sync.index("--python") + 1] == "3.12"
    assert "--locked" in sync and "--no-dev" in sync
    node = host.commands("soul")[0]
    assert node["token"] == TOKEN
    assert node["args"][-2:] == ["--controller", CONTROLLER]
    assert node["args"][node["args"].index("--device") + 1] == device
    assert len(host.commands("device-check")) == 1
    assert host.launcher.is_file()
    assert not list(host.install.glob(".install.*"))


def test_options_are_saved_and_launcher_runs_from_any_directory(host, tmp_path):
    name = "my node's $(touch UNEXPECTED); name"
    result = host.run("--no-start", "--", "--name", name, "--memory-gb", "8")
    assert result.returncode == 0, result.stderr
    assert not host.commands("soul")
    result = subprocess.run(
        [str(host.launcher), "--memory-gb", "4"], env=host.env, cwd="/", capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    node = host.commands("soul")[0]
    assert node["args"][node["args"].index("--name") + 1] == name
    assert node["args"][-6:] == ["--memory-gb", "8", "--memory-gb", "4", "--controller", CONTROLLER]
    assert not (tmp_path / "UNEXPECTED").exists()


def test_explicit_cpu_bypasses_broken_nvidia_driver(host):
    host.nvidia(broken=True)
    result = host.run("--profile", "cpu")
    assert result.returncode == 0, result.stderr
    assert host.commands("uv")[0]["args"][-4:-2] == ["--extra", "node"]


def test_explicit_cuda_and_device_override(host):
    host.nvidia("12.9")
    result = host.run("--profile", "cu129", "--", "--device", "cuda:1")
    assert result.returncode == 0, result.stderr
    assert "node-cu129" in host.commands("uv")[0]["args"]
    assert host.commands("device-check")[0]["args"] == ["-", "cuda:0", "--device", "cuda:1"]


@pytest.mark.parametrize(
    ("settings", "cuda", "args", "message"),
    [
        ({"MOCK_OS": "Darwin", "MOCK_ARCH": "x86_64"}, None, [], "Apple Silicon"),
        ({"MOCK_OS": "MINGW64_NT"}, None, [], "WSL2"),
        ({"MOCK_ARCH": "armv7l"}, None, [], "Unsupported system"),
        ({"MOCK_LIBC": "glibc 2.27"}, None, [], "glibc 2.28"),
        ({"MOCK_LIBC": "musl"}, None, [], "Unrecognized Linux C library"),
        ({}, "12.5", [], "Update the driver"),
        ({"MOCK_ARCH": "aarch64"}, "12.8", [], "CUDA 12.9"),
        ({}, None, ["--profile", "cu128"], "nvidia-smi is missing"),
        ({}, None, ["--profile", "invalid"], "Unknown profile"),
        ({}, None, ["--", "--controller", "https://elsewhere"], "fixed"),
        ({"SOUL_INSTALL_DIR": "relative"}, None, [], "absolute directory"),
    ],
)
def test_unsupported_hosts_and_options_fail_before_download(host, settings, cuda, args, message):
    host.env.update(settings)
    if cuda:
        host.nvidia(cuda)
    result = host.run(*args)
    assert result.returncode != 0
    assert message in result.stderr
    assert not host.commands("curl")
    assert not host.launcher.exists()


@pytest.mark.parametrize(
    "failure",
    [
        {"MOCK_DOWNLOAD_FAIL": "astral.sh"},
        {"MOCK_DOWNLOAD_FAIL": "/install/agent.tar.gz"},
        {"MOCK_SYNC_FAIL": "1"},
        {"MOCK_DEVICE_FAIL": "1"},
    ],
)
def test_failed_install_does_not_start_or_publish_launcher(host, failure):
    host.env.update(failure)
    result = host.run()
    assert result.returncode != 0
    assert not host.commands("soul")
    assert not host.launcher.exists()
    assert not list(host.install.glob(".install.*"))
    assert not list(host.install.glob("releases/agent.*"))


def test_failed_update_preserves_working_launcher_and_successful_update_keeps_cache(host):
    result = host.run("--no-start")
    assert result.returncode == 0, result.stderr
    original = host.launcher.read_bytes()
    host.env["MOCK_SYNC_FAIL"] = "1"
    assert host.run("--no-start").returncode != 0
    assert host.launcher.read_bytes() == original
    assert len(list(host.install.glob("releases/agent.*"))) == 1
    del host.env["MOCK_SYNC_FAIL"]
    assert host.run("--no-start").returncode == 0
    assert host.launcher.read_bytes() != original
    assert len(list(host.install.glob("releases/agent.*"))) == 2
    assert len([c for c in host.commands("curl") if any("astral.sh" in a for a in c["args"])]) == 1
    # Every launcher uses the same persistent cache, regardless of release directory.
    assert f"{os.environ['HOME']}/.cache/soul" in original.decode()
    assert f"{os.environ['HOME']}/.cache/soul" in host.launcher.read_text()
