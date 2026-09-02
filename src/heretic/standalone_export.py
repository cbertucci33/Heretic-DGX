# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


@dataclass(frozen=True, slots=True)
class StandaloneVerification:
    changed_targets: tuple[str, ...]
    rewritten_shards: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoraDelta:
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    scaling: float


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    model_id: str
    revision: str
    config_sha256: str
    index_sha256: str


LAGUNA_S_2_1_FP8_IDENTITY = CheckpointIdentity(
    model_id="poolside/Laguna-S-2.1-FP8",
    revision="06d71e91db70a11b08ee6a09c3c4818c85a61953",
    config_sha256="876de1e4a6c8baa234e414c4129a197d2b3dfa34476447ceafb266bebd236376",
    index_sha256="aec4ef10244640b4a60b4c74cddc3c08399acef547e1f6f973f6381b4745ebb7",
)


def checkpoint_identity(
    source_directory: str | Path,
    *,
    model_id: str,
    revision: str,
) -> CheckpointIdentity:
    root = Path(source_directory)
    return CheckpointIdentity(
        model_id=model_id,
        revision=revision,
        config_sha256=_sha256(root / "config.json"),
        index_sha256=_sha256(root / "model.safetensors.index.json"),
    )


def verify_checkpoint_identity(
    source_directory: str | Path,
    expected: CheckpointIdentity,
) -> None:
    actual = checkpoint_identity(
        source_directory,
        model_id=expected.model_id,
        revision=expected.revision,
    )
    if actual != expected:
        raise ValueError(
            "source checkpoint does not match pinned Laguna "
            f"{expected.model_id}@{expected.revision}"
        )


def load_bf16_lora_deltas(adapter_directory: str | Path) -> dict[str, LoraDelta]:
    """Load a strict BF16 LoRA adapter and map it to base checkpoint keys."""
    root = Path(adapter_directory)
    config = json.loads((root / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA":
        raise ValueError("standalone export requires a LoRA adapter")
    if config.get("bias") != "none" or config.get("modules_to_save") is not None:
        raise ValueError("standalone export does not support auxiliary adapter tensors")
    if (
        config.get("fan_in_fan_out")
        or config.get("use_dora")
        or config.get("use_rslora")
    ):
        raise ValueError("standalone export supports ordinary linear LoRA only")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("standalone export requires uniform LoRA rank and alpha")
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if not isinstance(rank, int) or rank <= 0 or not isinstance(alpha, (int, float)):
        raise ValueError("standalone export requires positive LoRA rank and alpha")

    state = load_file(root / "adapter_model.safetensors", device="cpu")
    prefix = "base_model.model."
    suffixes = (".lora_A.weight", ".lora_B.weight")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        if not key.startswith(prefix) or not key.endswith(suffixes):
            raise ValueError(f"unexpected adapter tensor: {key}")
        suffix = suffixes[0] if key.endswith(suffixes[0]) else suffixes[1]
        module = key[len(prefix) : -len(suffix)]
        base_key = f"{module}.weight"
        component = "A" if suffix == suffixes[0] else "B"
        if component in pairs.setdefault(base_key, {}):
            raise ValueError(f"duplicate LoRA tensor for {base_key}")
        pairs[base_key][component] = tensor

    deltas: dict[str, LoraDelta] = {}
    for base_key, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"incomplete LoRA tensor pair for {base_key}")
        if pair["A"].shape[0] != rank or pair["B"].shape[1] != rank:
            raise ValueError(f"LoRA rank mismatch for {base_key}")
        if pair["A"].dtype is not torch.bfloat16 or pair["B"].dtype is not torch.bfloat16:
            raise ValueError(f"LoRA tensors must be BF16 for {base_key}")
        if not torch.isfinite(pair["A"]).all() or not torch.isfinite(pair["B"]).all():
            raise ValueError(f"LoRA tensors must be finite for {base_key}")
        if not base_key.endswith(".self_attn.o_proj.weight"):
            if torch.count_nonzero(pair["B"]).item() != 0:
                raise ValueError(f"non-target adapter is not exactly zero: {base_key}")
            continue
        deltas[base_key] = LoraDelta(pair["A"], pair["B"], float(alpha / rank))
    if not deltas:
        raise ValueError("adapter contains no self_attn.o_proj tensors")
    return deltas


def validate_laguna_checkpoint(
    source_directory: str | Path,
    deltas: dict[str, LoraDelta],
    *,
    expected_layer_count: int = 48,
) -> dict[str, tuple[str, ...]]:
    """Validate Laguna's FP8 metadata and protected BF16 LoRA targets."""
    root = Path(source_directory)
    config = json.loads((root / "config.json").read_text())
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError("Laguna checkpoint is missing quantization_config")
    if (
        quantization.get("quant_method") != "fp8"
        or quantization.get("weight_block_size") != [128, 128]
        or quantization.get("activation_scheme") != "dynamic"
    ):
        raise ValueError("Laguna checkpoint does not use the expected FP8 contract")
    if (
        config.get("model_type") != "laguna"
        or config.get("architectures") != ["LagunaForCausalLM"]
    ):
        raise ValueError("checkpoint does not use the exact Laguna architecture")
    layer_count = config.get("num_hidden_layers")
    if layer_count != expected_layer_count:
        raise ValueError(f"Laguna checkpoint must have exactly {expected_layer_count} layers")
    expected = {
        f"model.layers.{layer}.self_attn.o_proj.weight"
        for layer in range(layer_count)
    }
    if set(deltas) != expected:
        missing = sorted(expected - set(deltas))
        extra = sorted(set(deltas) - expected)
        raise ValueError(f"adapter target inventory mismatch; missing={missing}, extra={extra}")
    ignored = set(quantization.get("ignored_layers") or ())
    for key in expected:
        if key.removesuffix(".weight") not in ignored:
            raise ValueError(f"LoRA target is not protected BF16 in base config: {key}")

    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Laguna checkpoint has invalid safetensors index")
    grouped: dict[str, list[str]] = {}
    for key in sorted(expected):
        shard = weight_map.get(key)
        if not isinstance(shard, str):
            raise ValueError(f"Laguna index is missing LoRA target: {key}")
        grouped.setdefault(shard, []).append(key)

    for shard, keys in grouped.items():
        shard_path = root / shard
        if not shard_path.is_file():
            raise ValueError(f"Laguna shard is missing: {shard}")
        with safe_open(shard_path, framework="pt", device="cpu") as tensors:
            available = set(tensors.keys())
            for key in keys:
                if key not in available:
                    raise ValueError(f"Laguna shard {shard} is missing {key}")
                tensor = tensors.get_slice(key)
                delta = deltas[key]
                if tensor.get_dtype() != "BF16":
                    raise ValueError(f"LoRA target is not stored as BF16: {key}")
                expected_shape = [delta.lora_b.shape[0], delta.lora_a.shape[1]]
                if tensor.get_shape() != expected_shape:
                    raise ValueError(f"LoRA target shape mismatch: {key}")
    return {shard: tuple(keys) for shard, keys in sorted(grouped.items())}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.equal(left.float(), right.float())
    return torch.equal(left, right)


def _range_equal(left: Path, right: Path, start: int, end: int) -> bool:
    remaining = end - start
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        left_stream.seek(start)
        right_stream.seek(start)
        while remaining:
            chunk_size = min(8 * 1024 * 1024, remaining)
            if left_stream.read(chunk_size) != right_stream.read(chunk_size):
                return False
            remaining -= chunk_size
    return True


def _only_target_intervals_may_differ(
    source: Path,
    output: Path,
    affected_keys: tuple[str, ...],
) -> bool:
    if source.stat().st_size != output.stat().st_size:
        return False
    header_end, header = _read_safetensors_header(source)
    intervals: list[tuple[int, int]] = []
    for key in affected_keys:
        entry = header.get(key)
        if not isinstance(entry, dict):
            return False
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and value >= 0 for value in offsets)
        ):
            return False
        intervals.append((header_end + offsets[0], header_end + offsets[1]))
    cursor = 0
    for start, end in sorted(intervals):
        if start < cursor or end < start or end > source.stat().st_size:
            return False
        if not _range_equal(source, output, cursor, start):
            return False
        cursor = end
    return _range_equal(source, output, cursor, source.stat().st_size)


def verify_standalone_laguna(
    source_directory: str | Path,
    adapter_directory: str | Path,
    output_directory: str | Path,
    *,
    expected_layer_count: int = 48,
) -> StandaloneVerification:
    """Prove the standalone tree changed only the intended BF16 targets."""
    source = Path(source_directory).resolve()
    output = Path(output_directory).resolve()
    if not output.is_dir():
        raise ValueError("standalone output directory does not exist")
    deltas = load_bf16_lora_deltas(adapter_directory)
    plan = validate_laguna_checkpoint(
        source,
        deltas,
        expected_layer_count=expected_layer_count,
    )

    source_files = {
        path.relative_to(source).as_posix(): path
        for path in source.rglob("*")
        if path.is_file() and path.relative_to(source).parts[0] != ".cache"
    }
    output_files = {
        path.relative_to(output).as_posix(): path
        for path in output.rglob("*")
        if path.is_file()
    }
    if set(source_files) != set(output_files):
        raise ValueError("standalone file inventory differs from the source checkpoint")
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("standalone output contains a symlink")

    changed_targets: list[str] = []
    for relative, source_path in sorted(source_files.items()):
        output_path = output_files[relative]
        affected_keys = plan.get(relative)
        if affected_keys is None:
            if _sha256(source_path) != _sha256(output_path):
                raise ValueError(f"unexpected standalone file change: {relative}")
            continue
        if not _only_target_intervals_may_differ(
            source_path,
            output_path,
            affected_keys,
        ):
            raise ValueError(
                f"rewritten shard changed outside intended target intervals: {relative}"
            )

        with safe_open(source_path, framework="pt", device="cpu") as source_handle:
            source_keys = set(source_handle.keys())
            source_metadata = source_handle.metadata()
        with safe_open(output_path, framework="pt", device="cpu") as output_handle:
            output_keys = set(output_handle.keys())
            output_metadata = output_handle.metadata()
        if source_keys != output_keys or source_metadata != output_metadata:
            raise ValueError(f"rewritten shard inventory/metadata changed: {relative}")
        source_state = load_file(source_path, device="cpu")
        output_state = load_file(output_path, device="cpu")
        affected = set(affected_keys)
        for key in sorted(source_keys):
            if key in affected:
                delta = deltas[key]
                expected = merge_bf16_lora_weight(
                    source_state[key],
                    delta.lora_a,
                    delta.lora_b,
                    scaling=delta.scaling,
                )
                if not _tensor_equal(output_state[key], expected):
                    raise ValueError(f"standalone target does not match merge oracle: {key}")
                if _tensor_equal(source_state[key], output_state[key]):
                    raise ValueError(f"standalone target did not change: {key}")
                changed_targets.append(key)
            elif not _tensor_equal(source_state[key], output_state[key]):
                raise ValueError(f"unexpected tensor change in {relative}: {key}")

    if set(changed_targets) != set(deltas):
        raise ValueError("not every adapter target was materialized")
    return StandaloneVerification(
        changed_targets=tuple(sorted(changed_targets)),
        rewritten_shards=tuple(sorted(plan)),
    )


def write_sha256_manifest(directory: str | Path) -> Path:
    root = Path(directory).resolve()
    manifest = root / "SHA256SUMS"
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest:
            continue
        relative = path.relative_to(root).as_posix()
        records.append(f"{_sha256(path)}  {relative}")
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(chr(10).join(records) + chr(10))
    os.replace(temporary, manifest)
    return manifest


def verify_sha256_manifest(directory: str | Path) -> None:
    root = Path(directory).resolve()
    manifest = root / "SHA256SUMS"
    expected_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    records: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, separator, relative = line.partition("  ")
        candidate = Path(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in records
        ):
            raise ValueError("invalid SHA256SUMS entry")
        records[relative] = digest
    if set(records) != expected_files:
        raise ValueError("SHA256SUMS file inventory mismatch")
    for relative, digest in sorted(records.items()):
        if _sha256(root / relative) != digest:
            raise ValueError(f"SHA256SUMS mismatch: {relative}")


def save_runtime_as_standalone(
    runtime: object,
    *,
    source_directory: str | Path,
    destination_directory: str | Path,
    max_shard_size: int | str,
    expected_identity: CheckpointIdentity = LAGUNA_S_2_1_FP8_IDENTITY,
    expected_layer_count: int = 48,
) -> StandaloneVerification:
    """Consume a coordinated runtime adapter export into a verified standalone model."""
    destination = Path(destination_directory).resolve()
    verify_checkpoint_identity(source_directory, expected_identity)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".heretic-adapter-", dir=destination.parent
    ) as adapter_directory:
        runtime.save_adapter(  # type: ignore[attr-defined]
            adapter_directory,
            max_shard_size=max_shard_size,
        )
        runtime.shutdown()  # type: ignore[attr-defined]
        export_standalone_laguna(
            source_directory,
            adapter_directory,
            destination,
            expected_layer_count=expected_layer_count,
        )
        report = verify_standalone_laguna(
            source_directory,
            adapter_directory,
            destination,
            expected_layer_count=expected_layer_count,
        )
        write_sha256_manifest(destination)
        verify_sha256_manifest(destination)
        return report


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"invalid safetensors header: {path.name}")
        header_length = int.from_bytes(length_bytes, "little")
        if header_length <= 0 or header_length > 100 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length: {path.name}")
        encoded = stream.read(header_length)
        if len(encoded) != header_length:
            raise ValueError(f"truncated safetensors header: {path.name}")
    header = json.loads(encoded)
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header object: {path.name}")
    return 8 + header_length, header


def _pwrite_all(descriptor: int, data: bytes, offset: int) -> None:
    written = 0
    while written < len(data):
        count = os.pwrite(descriptor, data[written:], offset + written)
        if count <= 0:
            raise OSError("pwrite did not make progress")
        written += count


def _materialize_shard_intervals(
    source_path: Path,
    destination_path: Path,
    keys: tuple[str, ...],
    deltas: dict[str, LoraDelta],
    *,
    rows_per_chunk: int = 128,
) -> None:
    header_end, header = _read_safetensors_header(source_path)
    source_descriptor = os.open(source_path, os.O_RDONLY)
    destination_descriptor = os.open(destination_path, os.O_RDWR)
    try:
        for key in keys:
            entry = header.get(key)
            if not isinstance(entry, dict):
                raise ValueError(f"missing safetensors entry: {key}")
            if entry.get("dtype") != "BF16":
                raise ValueError(f"LoRA target is not BF16: {key}")
            shape = entry.get("shape")
            offsets = entry.get("data_offsets")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or not all(isinstance(value, int) and value > 0 for value in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) and value >= 0 for value in offsets)
            ):
                raise ValueError(f"invalid safetensors target metadata: {key}")
            output_features, input_features = shape
            start, end = offsets
            if end - start != output_features * input_features * 2:
                raise ValueError(f"invalid BF16 payload length: {key}")
            delta = deltas[key]
            for row_start in range(0, output_features, rows_per_chunk):
                row_end = min(row_start + rows_per_chunk, output_features)
                row_count = row_end - row_start
                byte_count = row_count * input_features * 2
                absolute_offset = header_end + start + row_start * input_features * 2
                encoded = os.pread(source_descriptor, byte_count, absolute_offset)
                if len(encoded) != byte_count:
                    raise ValueError(f"truncated BF16 payload: {key}")
                base = torch.frombuffer(
                    bytearray(encoded), dtype=torch.bfloat16
                ).reshape(row_count, input_features)
                merged = merge_bf16_lora_weight(
                    base,
                    delta.lora_a,
                    delta.lora_b[row_start:row_end],
                    scaling=delta.scaling,
                )
                _pwrite_all(
                    destination_descriptor,
                    merged.contiguous().view(torch.uint8).numpy().tobytes(),
                    absolute_offset,
                )
        os.fsync(destination_descriptor)
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)


def export_standalone_laguna(
    source_directory: str | Path,
    adapter_directory: str | Path,
    destination_directory: str | Path,
    *,
    expected_layer_count: int = 48,
) -> None:
    """Materialize a BF16 LoRA delta into Laguna's protected BF16 weights."""
    source = Path(source_directory).resolve()
    adapter = Path(adapter_directory).resolve()
    destination = Path(destination_directory).resolve()
    if not source.is_dir() or not adapter.is_dir():
        raise ValueError("source checkpoint and adapter directories must exist")
    if destination.exists():
        raise ValueError("standalone destination must not already exist")
    if source == destination or source in destination.parents:
        raise ValueError("standalone destination must be outside the source checkpoint")

    deltas = load_bf16_lora_deltas(adapter)
    plan = validate_laguna_checkpoint(
        source,
        deltas,
        expected_layer_count=expected_layer_count,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        for source_path in sorted(source.rglob("*")):
            relative = source_path.relative_to(source)
            if relative.parts[0] == ".cache":
                continue
            destination_path = temporary / relative
            if source_path.is_symlink():
                raise ValueError(f"standalone source contains a symlink: {relative}")
            if source_path.is_dir():
                destination_path.mkdir(parents=True, exist_ok=True)
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            affected_keys = plan.get(relative.as_posix())
            if affected_keys is not None:
                _materialize_shard_intervals(
                    source_path,
                    destination_path,
                    affected_keys,
                    deltas,
                )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def merge_bf16_lora_weight(
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    """Merge one LoRA delta into an existing protected BF16 weight."""
    if base_weight.dtype is not torch.bfloat16:
        raise ValueError("standalone merge requires a BF16 base weight")
    if lora_a.dtype is not torch.bfloat16 or lora_b.dtype is not torch.bfloat16:
        raise ValueError("standalone merge requires BF16 LoRA tensors")
    if lora_a.ndim != 2 or lora_b.ndim != 2 or base_weight.ndim != 2:
        raise ValueError("standalone merge requires rank-2 tensors")
    if lora_b.shape[1] != lora_a.shape[0]:
        raise ValueError("LoRA rank dimensions do not match")
    if base_weight.shape != (lora_b.shape[0], lora_a.shape[1]):
        raise ValueError("LoRA delta shape does not match the base weight")
    if not math.isfinite(scaling):
        raise ValueError("LoRA scaling must be finite")

    delta = (lora_b.float() @ lora_a.float()) * scaling
    return base_weight + delta.to(dtype=torch.bfloat16)
