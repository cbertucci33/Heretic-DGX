# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, cast
from types import SimpleNamespace

import pytest

from heretic.config import ExportStrategy
from heretic.main import save_upstream_export


class _RecordingRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | str]] = []

    def save_adapter(self, directory: str, *, max_shard_size: int | str) -> None:
        self.calls.append(("adapter", directory, max_shard_size))

    def save_merged(self, directory: str, *, max_shard_size: int | str) -> None:
        self.calls.append(("merge", directory, max_shard_size))


class _RecordingArtifact:
    def __init__(self) -> None:
        self.directories: list[object] = []

    def save_pretrained(self, directory) -> None:
        self.directories.append(directory)


def test_adapter_export_preserves_upstream_artifact_type(tmp_path) -> None:
    runtime = _RecordingRuntime()
    tokenizer = _RecordingArtifact()
    processor = _RecordingArtifact()
    model = SimpleNamespace(tokenizer=tokenizer, processor=processor)

    save_upstream_export(
        cast(Any, runtime),
        cast(Any, model),
        ExportStrategy.ADAPTER,
        tmp_path,
        max_shard_size="5GB",
    )

    assert runtime.calls == [("adapter", str(tmp_path), "5GB")]
    assert tokenizer.directories == []
    assert processor.directories == []


def test_merged_export_preserves_upstream_full_model_artifacts(tmp_path) -> None:
    runtime = _RecordingRuntime()
    tokenizer = _RecordingArtifact()
    processor = _RecordingArtifact()
    model = SimpleNamespace(tokenizer=tokenizer, processor=processor)

    save_upstream_export(
        cast(Any, runtime),
        cast(Any, model),
        ExportStrategy.MERGE,
        tmp_path,
        max_shard_size="2GB",
    )

    assert runtime.calls == [("merge", str(tmp_path), "2GB")]
    assert tokenizer.directories == [tmp_path]
    assert processor.directories == [tmp_path]


def test_upstream_export_rejects_model_specific_strategy(tmp_path) -> None:
    runtime = _RecordingRuntime()
    model = SimpleNamespace(tokenizer=_RecordingArtifact(), processor=None)

    with pytest.raises(ValueError, match="adapter or merge"):
        save_upstream_export(
            runtime,
            model,
            ExportStrategy.STANDALONE,
            tmp_path,
            max_shard_size="5GB",
        )
