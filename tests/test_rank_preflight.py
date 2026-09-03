# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import replace
from pathlib import Path

import pytest

from heretic.checkpoint_identity import build_checkpoint_payload_identity
from heretic.rank_preflight import (
    RankPreflightIdentity,
    require_matching_rank_preflights,
)
from heretic.source_identity import build_source_identity


def _source(root: Path):
    (root / "src/heretic").mkdir(parents=True)
    (root / "src/heretic/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return build_source_identity(
        root,
        python_executable="/srv/heretic/.venv/bin/python",
        python_version="3.12.12",
        package_version="2.0.0",
    )


def _checkpoint(root: Path):
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return build_checkpoint_payload_identity(root)


def _preflights(tmp_path: Path):
    source = _source(tmp_path / "source")
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    return (
        RankPreflightIdentity(rank=0, source=source, checkpoint=checkpoint),
        RankPreflightIdentity(rank=1, source=source, checkpoint=checkpoint),
    )


def test_accepts_exact_source_runtime_and_checkpoint_agreement(tmp_path: Path) -> None:
    rank_zero, rank_one = _preflights(tmp_path)

    shared = require_matching_rank_preflights(rank_zero, rank_one)

    assert shared is rank_zero


def test_rejects_source_runtime_mismatch(tmp_path: Path) -> None:
    rank_zero, rank_one = _preflights(tmp_path)
    mismatched = replace(
        rank_one,
        source=replace(rank_one.source, python_version="3.12.13"),
    )

    with pytest.raises(RuntimeError, match="source/runtime"):
        require_matching_rank_preflights(rank_zero, mismatched)


def test_rejects_checkpoint_payload_mismatch(tmp_path: Path) -> None:
    rank_zero, rank_one = _preflights(tmp_path)
    mismatched = replace(
        rank_one,
        checkpoint=replace(rank_one.checkpoint, digest="f" * 64),
    )

    with pytest.raises(RuntimeError, match="checkpoint-payload"):
        require_matching_rank_preflights(rank_zero, mismatched)


def test_rejects_missing_duplicate_or_reversed_ranks(tmp_path: Path) -> None:
    rank_zero, rank_one = _preflights(tmp_path)

    for first, second in (
        (rank_one, rank_zero),
        (rank_zero, replace(rank_one, rank=0)),
        (rank_one, replace(rank_zero, rank=1)),
    ):
        with pytest.raises(RuntimeError, match="ordered"):
            require_matching_rank_preflights(first, second)


@pytest.mark.parametrize("rank", [True, -1, 2, 1.5])
def test_rejects_invalid_rank_identity(rank: object, tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    checkpoint = _checkpoint(tmp_path / "checkpoint")

    with pytest.raises(ValueError, match="rank"):
        RankPreflightIdentity(rank=rank, source=source, checkpoint=checkpoint)
