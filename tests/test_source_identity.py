# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pytest

from heretic.source_identity import build_source_identity


def _tree(root: Path) -> None:
    (root / "src/heretic").mkdir(parents=True)
    (root / "src/heretic/a.py").write_text("A = 1\n", encoding="utf-8")
    (root / "src/heretic/z.py").write_text("Z = 2\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname='heretic-llm'\nversion='2.0.0'\n", encoding="utf-8"
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")


def _identity(root: Path):
    return build_source_identity(
        root,
        python_executable="/opt/heretic/bin/python",
        python_version="3.12.12",
        package_version="2.0.0",
    )


def test_source_identity_changes_when_executable_source_changes(tmp_path: Path) -> None:
    _tree(tmp_path)
    before = _identity(tmp_path)

    (tmp_path / "src/heretic/z.py").write_text("Z = 3\n", encoding="utf-8")
    after = _identity(tmp_path)

    assert before.source_sha256 != after.source_sha256
    assert before.pyproject_sha256 == after.pyproject_sha256
    assert before.lock_sha256 == after.lock_sha256


def test_source_identity_changes_when_locked_environment_changes(tmp_path: Path) -> None:
    _tree(tmp_path)
    before = _identity(tmp_path)

    (tmp_path / "uv.lock").write_text("version = 2\n", encoding="utf-8")

    assert before.lock_sha256 != _identity(tmp_path).lock_sha256


def test_source_identity_is_canonical_and_includes_runtime(tmp_path: Path) -> None:
    _tree(tmp_path)
    identity = _identity(tmp_path)

    assert identity.python_executable == "/opt/heretic/bin/python"
    assert identity.python_version == "3.12.12"
    assert identity.package_version == "2.0.0"
    assert identity.canonical_json() == identity.canonical_json()
    assert len(identity.source_sha256) == 64
    assert len(identity.pyproject_sha256) == 64
    assert len(identity.lock_sha256) == 64


def test_source_identity_rejects_symlinked_source(tmp_path: Path) -> None:
    _tree(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 1\n", encoding="utf-8")
    (tmp_path / "src/heretic/link.py").symlink_to(outside)

    with pytest.raises(RuntimeError, match="regular file"):
        _identity(tmp_path)
