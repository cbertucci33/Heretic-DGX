# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

_CONFIG_FILENAME = "config.json"
_INDEX_FILENAME = "model.safetensors.index.json"
_SINGLE_WEIGHTS_FILENAME = "model.safetensors"


@dataclass(frozen=True, slots=True)
class CheckpointFileIdentity:
    """Identity of one immutable file used to load a checkpoint."""

    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointPayloadIdentity:
    """Canonical identity of the exact local checkpoint payload."""

    files: tuple[CheckpointFileIdentity, ...]
    digest: str

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"checkpoint metadata does not exist: {path.name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"checkpoint metadata must be a regular file: {path.name}")
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"checkpoint metadata must be UTF-8 JSON: {path.name}") from error
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"checkpoint metadata must be JSON: {path.name}") from error
    if type(value) is not dict:
        raise TypeError(f"checkpoint metadata must be a JSON object: {path.name}")
    return value


def _validate_filename(name: object) -> str:
    if type(name) is not str or not name.strip():
        raise ValueError("checkpoint weight filename must be a nonempty string")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("checkpoint weight filename must not contain a path")
    if not name.endswith(".safetensors"):
        raise ValueError("checkpoint weight file must be a safetensors file")
    return name


def _checkpoint_filenames(root: Path) -> tuple[str, ...]:
    config_path = root / _CONFIG_FILENAME
    config = _read_json_object(config_path)

    code_files: tuple[str, ...] = ()
    if config.get("model_type") == "laguna":
        auto_map = config.get("auto_map")
        expected_auto_map = {
            "AutoConfig": "configuration_laguna.LagunaConfig",
            "AutoModelForCausalLM": "modeling_laguna.LagunaForCausalLM",
        }
        if auto_map != expected_auto_map:
            raise ValueError("Laguna checkpoint has unexpected custom-code mapping")
        code_files = ("configuration_laguna.py", "modeling_laguna.py")

    index_path = root / _INDEX_FILENAME
    if not index_path.exists():
        return (_CONFIG_FILENAME, *code_files, _SINGLE_WEIGHTS_FILENAME)

    index = _read_json_object(index_path)
    weight_map = index.get("weight_map")
    if type(weight_map) is not dict or not weight_map:
        raise ValueError("checkpoint index weight_map must be a nonempty JSON object")

    weights: set[str] = set()
    for tensor_name, filename in weight_map.items():
        if type(tensor_name) is not str or not tensor_name.strip():
            raise ValueError("checkpoint tensor name must be a nonempty string")
        weights.add(_validate_filename(filename))
    return (_CONFIG_FILENAME, _INDEX_FILENAME, *code_files, *sorted(weights))


def _hash_regular_file(path: Path) -> CheckpointFileIdentity:
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"checkpoint payload file does not exist: {path.name}")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"checkpoint payload must be a regular file: {path.name}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"checkpoint payload could not be opened as a regular file: {path.name}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            before.st_dev,
            before.st_ino,
        ) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ValueError(f"checkpoint payload changed before hashing: {path.name}")

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
            raise ValueError(f"checkpoint payload changed while hashing: {path.name}")
        if opened.st_size <= 0:
            raise ValueError(f"checkpoint payload file is empty: {path.name}")
        return CheckpointFileIdentity(
            name=path.name,
            size_bytes=opened.st_size,
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(descriptor)


def build_checkpoint_payload_identity(
    checkpoint_directory: str | Path,
) -> CheckpointPayloadIdentity:
    """Hash the exact config, index, and weight bytes used by a checkpoint."""

    root = Path(checkpoint_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    files = tuple(_hash_regular_file(root / name) for name in _checkpoint_filenames(root))
    canonical = json.dumps(
        {"files": [asdict(file) for file in files], "version": 1},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return CheckpointPayloadIdentity(
        files=files,
        digest=hashlib.sha256(canonical).hexdigest(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_directory")
    args = parser.parse_args(argv)
    identity = build_checkpoint_payload_identity(args.checkpoint_directory)
    print(identity.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
