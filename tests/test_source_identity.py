# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

from heretic.source_identity import build_source_identity


def _tree(root: Path) -> None:
    (root / "src/heretic").mkdir(parents=True)
    (root / "src/heretic/a.py").write_text('A = 1\n')
    (root / "src/heretic/z.py").write_text('Z = 2\n')
    (root / "pyproject.toml").write_text("[project]\nname='heretic-llm'\nversion='2.0.0'\n")
    (root / "uv.lock").write_text('version = 1\n')


def test_source_identity_changes_when_executable_source_changes(tmp_path: Path) -> None:
    _tree(tmp_path)
    before = build_source_identity(
        tmp_path, python_executable="/opt/heretic/bin/python",
        python_version="3.12.12", package_version="2.0.0",
    )

    (tmp_path / "src/heretic/z.py").write_text('Z = 3\n')
    after = build_source_identity(
        tmp_path, python_executable="/opt/heretic/bin/python",
        python_version="3.12.12", package_version="2.0.0",
    )

    assert before.source_sha256 != after.source_sha256
    assert before.pyproject_sha256 == after.pyproject_sha256
    assert before.lock_sha256 == after.lock_sha256


def test_source_identity_is_canonical_and_includes_runtime(tmp_path: Path) -> None:
    _tree(tmp_path)
    identity = build_source_identity(
        tmp_path, python_executable="/opt/heretic/bin/python",
        python_version="3.12.12", package_version="2.0.0",
    )

    assert identity.python_executable == "/opt/heretic/bin/python"
    assert identity.python_version == "3.12.12"
    assert identity.package_version == "2.0.0"
    assert len(identity.source_sha256) == 64
    assert len(identity.pyproject_sha256) == 64
    assert len(identity.lock_sha256) == 64
