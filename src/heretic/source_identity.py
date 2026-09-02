# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    source_sha256: str
    pyproject_sha256: str
    lock_sha256: str
    python_executable: str
    python_version: str
    package_version: str

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"identity input must be a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_source_identity(
    workdir: str | Path,
    *,
    python_executable: str,
    python_version: str,
    package_version: str,
) -> SourceIdentity:
    root = Path(workdir)
    source_root = root / "src" / "heretic"
    source_files = sorted(source_root.rglob("*.py"))
    if not source_files:
        raise RuntimeError(f"no Heretic Python source found under {source_root}")

    digest = hashlib.sha256()
    for path in source_files:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"source identity input must be a regular file: {path}")
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    return SourceIdentity(
        source_sha256=digest.hexdigest(),
        pyproject_sha256=_file_sha256(root / "pyproject.toml"),
        lock_sha256=_file_sha256(root / "uv.lock"),
        python_executable=python_executable,
        python_version=python_version,
        package_version=package_version,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args(argv)
    identity = build_source_identity(
        args.workdir,
        python_executable=str(Path(sys.executable).resolve()),
        python_version=sys.version.split()[0],
        package_version=version("heretic-llm"),
    )
    print(identity.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
