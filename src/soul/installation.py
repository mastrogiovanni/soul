"""Public installer assets for both source checkouts and installed controller wheels."""

import io
import tarfile
from functools import lru_cache
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
INSTALL_FILES = ("install.sh", "pyproject.toml", "uv.lock", "README.md", ".python-version")


def installer_file(name):
    packaged = PACKAGE / "_installer" / name
    if packaged.is_file():
        return packaged
    # Editable installs use the repository's canonical files.
    return PACKAGE.parents[1] / name


@lru_cache(maxsize=1)
def agent_source_archive():
    """Ship only package code/static assets and build inputs, never controller state."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name in INSTALL_FILES:
            archive.add(installer_file(name), arcname=f"soul-agent/{name}", recursive=False)
        for path in sorted(PACKAGE.rglob("*")):
            relative = path.relative_to(PACKAGE)
            if "_installer" in relative.parts or "__pycache__" in relative.parts:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix == ".py" or (
                relative.parts[0] == "static"
                and path.suffix in (".html", ".css", ".js", ".svg", ".png", ".jpg")
            ):
                archive.add(path, arcname=f"soul-agent/src/soul/{relative.as_posix()}", recursive=False)
    return output.getvalue()
