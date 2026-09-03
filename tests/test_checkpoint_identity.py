# SPDX-License-Identifier: AGPL-3.0-or-later

import hashlib
import json
from pathlib import Path

import pytest

from heretic.checkpoint_identity import build_checkpoint_payload_identity


def _indexed_checkpoint(root: Path) -> None:
    (root / "config.json").write_text(
        json.dumps({"model_type": "example"}), encoding="utf-8"
    )
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first shard")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"second shard")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 23},
                "weight_map": {
                    "z.weight": "model-00002-of-00002.safetensors",
                    "a.weight": "model-00001-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )


def test_identity_binds_exact_checkpoint_payload_bytes(tmp_path: Path) -> None:
    _indexed_checkpoint(tmp_path)
    before = build_checkpoint_payload_identity(tmp_path)

    changed = tmp_path / "model-00002-of-00002.safetensors"
    changed.write_bytes(b"changed shard")
    after = build_checkpoint_payload_identity(tmp_path)

    assert before.digest != after.digest
    assert [file.name for file in before.files] == [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert after.files[-1].sha256 == hashlib.sha256(changed.read_bytes()).hexdigest()


def test_identity_is_deterministic_for_index_key_order(tmp_path: Path) -> None:
    _indexed_checkpoint(tmp_path)
    first = build_checkpoint_payload_identity(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    second = build_checkpoint_payload_identity(tmp_path)

    assert first.files[2:] == second.files[2:]
    assert first.digest != second.digest
    assert first.canonical_json() == first.canonical_json()


def test_supports_single_file_safetensors_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    identity = build_checkpoint_payload_identity(tmp_path)

    assert [file.name for file in identity.files] == [
        "config.json",
        "model.safetensors",
    ]
    assert len(identity.digest) == 64


@pytest.mark.parametrize(
    "index",
    [
        {"weight_map": {"layer.weight": "../outside.safetensors"}},
        {"weight_map": {"": "model-00001-of-00002.safetensors"}},
        {"weight_map": {}},
        {"weight_map": []},
    ],
)
def test_rejects_incomplete_or_unsafe_indexes(
    tmp_path: Path, index: object
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    with pytest.raises((TypeError, ValueError, FileNotFoundError)):
        build_checkpoint_payload_identity(tmp_path)


def test_rejects_missing_empty_and_symlinked_payloads(tmp_path: Path) -> None:
    _indexed_checkpoint(tmp_path)
    first = tmp_path / "model-00001-of-00002.safetensors"
    first.unlink()
    first.symlink_to(tmp_path / "model-00002-of-00002.safetensors")
    with pytest.raises(ValueError, match="regular file"):
        build_checkpoint_payload_identity(tmp_path)

    first.unlink()
    first.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        build_checkpoint_payload_identity(tmp_path)


def test_rejects_duplicate_index_keys(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model.safetensors","a":"other.safetensors"}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_checkpoint_payload_identity(tmp_path)


def test_rejects_symlinked_checkpoint_metadata(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").symlink_to(outside)
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="regular file"):
        build_checkpoint_payload_identity(tmp_path)
