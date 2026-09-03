# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import timedelta


@dataclass(frozen=True, slots=True)
class CollectiveProbeResult:
    """Evidence returned by one rank after a CPU-only Gloo collective."""

    rank: int
    world_size: int
    backend: str
    reduced_value: int

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def parse_collective_probe_result(payload: str) -> CollectiveProbeResult:
    try:
        raw = json.loads(payload)
        result = CollectiveProbeResult(**raw)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid collective probe output") from error
    if result.canonical_json() != payload.strip():
        raise ValueError("collective probe output must be canonical JSON")
    return result


def _required_integer(name: str) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)

    rank = _required_integer("RANK")
    world_size = _required_integer("WORLD_SIZE")
    timeout_seconds = _required_integer("HERETIC_DGX_TIMEOUT_SECONDS")
    if rank not in (0, 1) or world_size != 2:
        raise ValueError("collective probe requires ranks 0/1 and WORLD_SIZE=2")
    if timeout_seconds <= 0:
        raise ValueError("HERETIC_DGX_TIMEOUT_SECONDS must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("collective probe requires CUDA_VISIBLE_DEVICES to be empty")

    import torch
    import torch.distributed as distributed

    initialized = False
    try:
        distributed.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
        initialized = True
        value = torch.tensor([rank + 1], dtype=torch.int64, device="cpu")
        distributed.all_reduce(value)
        distributed.barrier()
        result = CollectiveProbeResult(
            rank=rank,
            world_size=distributed.get_world_size(),
            backend=str(distributed.get_backend()),
            reduced_value=int(value.item()),
        )
    finally:
        if initialized:
            distributed.destroy_process_group()

    print(result.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
