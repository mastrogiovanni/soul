import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def executable(path, source):
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o755)


def test_make_up_stops_before_build_when_target_cluster_is_missing(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "commands"
    executable(binaries / "kubectl", 'print("unrelated-cluster")\n')
    for command in ("docker", "kind", "openssl"):
        executable(
            binaries / command,
            f"from pathlib import Path\nPath({str(log)!r}).write_text('unexpected mutation')\n",
        )
    result = subprocess.run(
        ["make", "-j4", "up", f"K8S_DIR={tmp_path}", "KUBE_CONTEXT=kind-missing"],
        cwd=ROOT,
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Context kind-missing is missing" in result.stderr
    assert not log.exists()


def test_credentials_bootstrap_is_private_and_does_not_rotate_on_redeploy(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    marker = tmp_path / "secret"
    operations = tmp_path / "operations"
    # Model just Secret get/create/apply. Never contact a cluster or print key material.
    executable(
        binaries / "kubectl",
        f"""import json, sys
from pathlib import Path
marker, operations = Path({str(marker)!r}), Path({str(operations)!r})
args = sys.argv[1:]
with operations.open('a') as log:
    log.write(json.dumps(args) + '\\n')
if 'get' in args:
    raise SystemExit(0 if marker.exists() else 1)
if 'create' in args:
    filename = next(a.split('=', 1)[1] for a in args if a.startswith('--from-env-file='))
    data = dict(line.split('=', 1) for line in Path(filename).read_text().splitlines())
    assert all(data[k] for k in ('SOUL_API_KEY', 'SOUL_ADMIN_KEY', 'SOUL_NODE_TOKEN'))
    print(json.dumps({{'stringData': data}}))
elif 'apply' in args:
    data = json.load(sys.stdin)
    assert len(set(data['stringData'].values())) == 3
    marker.write_text('created')
else:
    raise SystemExit('Unexpected command')
""",
    )
    credentials = tmp_path / "private" / "credentials.env"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("SOUL_API_KEY", "SOUL_ADMIN_KEY", "SOUL_NODE_TOKEN", "HF_TOKEN")
    }
    env.update(PATH=f"{binaries}:{os.environ['PATH']}", CREDENTIALS_FILE=str(credentials))
    command = ["bash", str(ROOT / "scripts/kubernetes.sh"), "secrets"]
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    initial = credentials.read_bytes()
    assert stat.S_IMODE(credentials.stat().st_mode) == 0o600
    assert len(initial.splitlines()) == 3
    for line in initial.decode().splitlines():
        key = line.split("=", 1)[1]
        assert len(key) == 64 and key not in first.stdout
    second = subprocess.run(command, env={**env, "SOUL_ADMIN_KEY": "changed"}, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert credentials.read_bytes() == initial
    commands = [json.loads(line) for line in operations.read_text().splitlines()]
    assert len([c for c in commands if "create" in c]) == 1
    assert len([c for c in commands if "apply" in c]) == 1


def test_down_preserves_data_credentials_and_other_routes(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    operations = tmp_path / "operations"
    executable(
        binaries / "kubectl",
        f"""import json, sys
from pathlib import Path
args = sys.argv[1:]
with Path({str(operations)!r}).open('a') as log:
    log.write(json.dumps(args) + '\\n')
""",
    )
    platform = tmp_path / "platform"
    (platform / "scripts").mkdir(parents=True)
    (platform / "scripts/httpproxy-include.sh").write_text(
        'test "$1" = remove && test "$5" = soul && test "$7" = /soul\n'
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/kubernetes.sh"), "down"],
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "K8S_DIR": str(platform)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in operations.read_text().splitlines()]
    deletion = next(c for c in commands if "delete" in c)
    assert deletion[deletion.index("delete") + 1 : deletion.index("-n")] == [
        "deployment/soul-controller",
        "service/soul-service",
        "httpproxy/soul",
    ]
