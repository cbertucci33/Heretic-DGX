# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

_INDEX_FILENAME = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class ShardAssignment:
    """One immutable checkpoint shard assigned to exactly one rank."""

    name: str
    size_bytes: int
    sha256: str
    rank: int


@dataclass(frozen=True, slots=True)
class TensorAssignment:
    """One indexed tensor following its containing shard to one rank."""

    name: str
    shard_name: str
    rank: int


@dataclass(frozen=True, slots=True)
class CheckpointPartitionPlan:
    """Deterministic two-rank assignment of whole checkpoint shards."""

    shards: tuple[ShardAssignment, ...]
    tensors: tuple[TensorAssignment, ...]
    rank_bytes: tuple[int, int]
    digest: str


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_checkpoint_index(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("checkpoint index must be UTF-8 JSON") from error
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("checkpoint index must be JSON") from error
    if type(value) is not dict:
        raise TypeError("checkpoint index must be a JSON object")
    if "weight_map" not in value:
        raise ValueError("checkpoint index is missing weight_map")
    if not frozenset(value).issubset({"metadata", "weight_map"}):
        raise ValueError("checkpoint index has unknown top-level fields")
    if "metadata" in value and type(value["metadata"]) is not dict:
        raise TypeError("checkpoint index metadata must be a JSON object")
    return value


def _read_shard_identity(root: Path, shard_name: str) -> tuple[int, str]:
    if type(shard_name) is not str or not shard_name.strip():
        raise ValueError("checkpoint shard name must be a nonempty string")
    if Path(shard_name).name != shard_name or "/" in shard_name or "\\" in shard_name:
        raise ValueError("checkpoint shard name must not contain a path")
    if not shard_name.endswith(".safetensors"):
        raise ValueError("checkpoint shard must be a safetensors file")
    shard_path = root / shard_name
    try:
        before = shard_path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"checkpoint shard does not exist: {shard_name}")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("checkpoint shard must be a regular file inside the checkpoint")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(shard_path, flags)
    except OSError as error:
        raise ValueError("checkpoint shard could not be opened as a regular file") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            before.st_dev,
            before.st_ino,
        ) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ValueError("checkpoint shard changed during validation")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("checkpoint shard changed while being hashed")
        return opened.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _plan_digest(
    shards: tuple[ShardAssignment, ...],
    tensors: tuple[TensorAssignment, ...],
    rank_bytes: tuple[int, int],
) -> str:
    canonical = json.dumps(
        {
            "rank_bytes": rank_bytes,
            "shards": [
                {
                    "name": shard.name,
                    "rank": shard.rank,
                    "sha256": shard.sha256,
                    "size_bytes": shard.size_bytes,
                }
                for shard in shards
            ],
            "tensors": [
                {
                    "name": tensor.name,
                    "rank": tensor.rank,
                    "shard_name": tensor.shard_name,
                }
                for tensor in tensors
            ],
            "version": 1,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def plan_checkpoint_shards(
    checkpoint_directory: str | Path,
) -> CheckpointPartitionPlan:
    """Plan a deterministic, byte-balanced, read-only two-rank shard assignment."""
    root = Path(checkpoint_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    index = _read_checkpoint_index(root / _INDEX_FILENAME)
    weight_map = index["weight_map"]
    if type(weight_map) is not dict:
        raise TypeError("checkpoint index weight_map must be a JSON object")
    if not weight_map:
        raise ValueError("checkpoint index weight_map must not be empty")

    normalized_weight_map: dict[str, str] = {}
    for tensor_name, shard_name in weight_map.items():
        if type(tensor_name) is not str or not tensor_name.strip():
            raise ValueError("tensor name must be a nonempty string")
        if type(shard_name) is not str or not shard_name.strip():
            raise ValueError("checkpoint shard name must be a nonempty string")
        normalized_weight_map[tensor_name] = shard_name

    shard_identities: dict[str, tuple[int, str]] = {}
    for shard_name in sorted(set(normalized_weight_map.values())):
        size_bytes, sha256 = _read_shard_identity(root, shard_name)
        if size_bytes <= 0:
            raise ValueError(f"checkpoint shard is empty: {shard_name}")
        shard_identities[shard_name] = (size_bytes, sha256)
    if len(shard_identities) < 2:
        raise ValueError("checkpoint cannot populate two nonempty ranks")

    rank_totals = [0, 0]
    shard_rank: dict[str, int] = {}
    for shard_name, size_bytes in sorted(
        shard_identities.items(), key=lambda item: (-item[1][0], item[0])
    ):
        size_bytes = size_bytes[0]
        rank = min(range(2), key=lambda candidate: (rank_totals[candidate], candidate))
        shard_rank[shard_name] = rank
        rank_totals[rank] += size_bytes
    if any(total == 0 for total in rank_totals):
        raise ValueError("checkpoint cannot populate two nonempty ranks")

    shards = tuple(
        ShardAssignment(
            name=shard_name,
            size_bytes=shard_identities[shard_name][0],
            sha256=shard_identities[shard_name][1],
            rank=shard_rank[shard_name],
        )
        for shard_name in sorted(shard_identities)
    )
    tensors = tuple(
        TensorAssignment(
            name=tensor_name,
            shard_name=normalized_weight_map[tensor_name],
            rank=shard_rank[normalized_weight_map[tensor_name]],
        )
        for tensor_name in sorted(normalized_weight_map)
    )
    rank_bytes = (rank_totals[0], rank_totals[1])
    return CheckpointPartitionPlan(
        shards=shards,
        tensors=tensors,
        rank_bytes=rank_bytes,
        digest=_plan_digest(shards, tensors, rank_bytes),
    )
