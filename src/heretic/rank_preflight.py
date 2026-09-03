# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from .checkpoint_identity import (
    CheckpointPayloadIdentity,
    build_checkpoint_payload_identity,
)
from .rank_environment import read_rank_environment
from .source_identity import SourceIdentity, build_source_identity


@dataclass(frozen=True, slots=True)
class RankPreflightIdentity:
    """Source/runtime and checkpoint evidence reported by one rank."""

    rank: int
    source: SourceIdentity
    checkpoint: CheckpointPayloadIdentity

    def __post_init__(self) -> None:
        if self.rank not in (0, 1) or type(self.rank) is not int:
            raise ValueError("preflight rank must be 0 or 1")
        if type(self.source) is not SourceIdentity:
            raise TypeError("preflight source must be exactly SourceIdentity")
        if type(self.checkpoint) is not CheckpointPayloadIdentity:
            raise TypeError(
                "preflight checkpoint must be exactly CheckpointPayloadIdentity"
            )

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def require_matching_rank_preflights(
    first: RankPreflightIdentity,
    second: RankPreflightIdentity,
) -> RankPreflightIdentity:
    """Require exact two-rank identity agreement before process launch."""

    if (
        type(first) is not RankPreflightIdentity
        or type(second) is not RankPreflightIdentity
    ):
        raise TypeError("rank preflights must be exactly RankPreflightIdentity")
    if (first.rank, second.rank) != (0, 1):
        raise RuntimeError("rank preflights must be ordered as rank 0 then rank 1")
    if first.source != second.source:
        raise RuntimeError("rank source/runtime identities do not match")
    if first.checkpoint != second.checkpoint:
        raise RuntimeError("rank checkpoint-payload identities do not match")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("checkpoint_directory")
    args = parser.parse_args(argv)
    environment = read_rank_environment(os.environ)
    identity = RankPreflightIdentity(
        rank=environment.rank,
        source=build_source_identity(
            args.workdir,
            python_executable=str(Path(sys.executable).resolve()),
            python_version=sys.version.split()[0],
            package_version=version("heretic-llm"),
        ),
        checkpoint=build_checkpoint_payload_identity(args.checkpoint_directory),
    )
    print(identity.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
