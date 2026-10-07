import io
import shutil
import tarfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from soul import installation
from soul.controller.app import Settings, create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_archive_cache():
    installation.agent_source_archive.cache_clear()
    yield
    installation.agent_source_archive.cache_clear()


@pytest.mark.parametrize("prefix", ["", "/soul", "/team/soul"])
def test_installer_and_complete_agent_source_are_public_under_context_root(tmp_path, prefix):
    app = create_app(Settings("api", "admin", "node", state_dir=tmp_path, context_root=prefix))
    with TestClient(app) as client:
        script = client.get(prefix + "/install.sh")
        assert script.status_code == 200
        assert script.text == (ROOT / "install.sh").read_text()
        assert script.headers["content-type"].startswith("text/plain")
        response = client.get(prefix + "/install/agent.tar.gz")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/gzip"
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
            for name in installation.INSTALL_FILES:
                assert archive.extractfile(f"soul-agent/{name}").read() == (ROOT / name).read_bytes()
            assert (
                archive.extractfile("soul-agent/src/soul/cli.py").read()
                == (ROOT / "src/soul/cli.py").read_bytes()
            )
            assert "soul-agent/src/soul/node/runtime.py" in archive.getnames()
            assert "soul-agent/src/soul/static/index.html" in archive.getnames()
            assert not any("__pycache__" in name or "_installer" in name for name in archive.getnames())
        assert client.get(prefix + "/admin/state").status_code == 401
        if prefix:
            assert client.get("/install.sh").status_code == 404


def test_installed_wheel_resources_work_without_checkout_and_exclude_local_state(tmp_path, monkeypatch):
    package = tmp_path / "site-packages/soul"
    shutil.copytree(ROOT / "src/soul", package)
    resources = package / "_installer"
    resources.mkdir()
    for name in installation.INSTALL_FILES:
        shutil.copyfile(ROOT / name, resources / name)
    (package / "credentials.env").write_text("private")
    (package / "static/local-secret.txt").write_text("private")
    (package / "static/linked.js").symlink_to(package / "credentials.env")
    (package / ".soul").mkdir()
    (package / ".soul/controller.sqlite3").write_text("private")
    monkeypatch.setattr(installation, "PACKAGE", package)
    assert installation.installer_file("install.sh") == resources / "install.sh"
    with tarfile.open(fileobj=io.BytesIO(installation.agent_source_archive()), mode="r:gz") as archive:
        names = archive.getnames()
        assert "soul-agent/src/soul/cli.py" in names
        assert "soul-agent/uv.lock" in names
        assert not any(
            part in name
            for name in names
            for part in ("credentials", "local-secret", "linked.js", "controller.sqlite3", "_installer")
        )
