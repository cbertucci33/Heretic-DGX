# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING

from .rank_environment import RankEnvironment, read_rank_environment

if TYPE_CHECKING:
    from .config import Settings


def _validate_application_argv(argv: Sequence[str]) -> None:
    seed_count = 0
    for index, argument in enumerate(argv):
        if argument == "--cluster" or argument.startswith("--cluster="):
            raise ValueError("rank application argv must not contain --cluster")
        if argument == "--seed":
            if index + 1 >= len(argv) or argv[index + 1].startswith("-"):
                raise ValueError("rank application --seed requires a value")
            seed_count += 1
        elif argument.startswith("--seed="):
            if argument == "--seed=":
                raise ValueError("rank application --seed requires a value")
            seed_count += 1
    if seed_count != 1:
        raise ValueError("rank application argv must contain exactly one seed")


def run_rank_application(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    application_main: Callable[[], int | None],
) -> tuple[RankEnvironment, int]:
    """Validate one rank before invoking the upstream application entry."""

    rank_environment = read_rank_environment(environment)
    application_argv = tuple(argv)
    _validate_application_argv(application_argv)
    sys.argv = [sys.argv[0], *application_argv]
    result = application_main()
    if result is None:
        result = 0
    if type(result) is not int:
        raise TypeError("rank application main must return an integer or None")
    return rank_environment, result


def synchronize_rank_settings(
    settings: Settings | None,
    *,
    rank: int,
) -> Settings | None:
    """Broadcast rank 0's finalized settings before either model is constructed."""

    if rank not in (0, 1):
        raise ValueError("rank settings synchronization requires rank 0 or 1")
    if rank == 1 and settings is not None:
        raise ValueError("worker settings must come from rank 0")

    import torch.distributed as dist

    payload: list[object] = [
        settings.model_dump_json() if rank == 0 and settings is not None else None
    ]
    dist.broadcast_object_list(payload, src=0)
    serialized = payload[0]
    if serialized is None:
        return None
    if type(serialized) is not str:
        raise TypeError("rank settings payload must be serialized JSON or null")

    from .config import Settings

    return Settings.model_validate_json(serialized)


def main(argv: list[str] | None = None) -> int:
    application_argv = sys.argv[1:] if argv is None else argv
    rank_environment = read_rank_environment(os.environ)
    _validate_application_argv(application_argv)
    sys.argv = [sys.argv[0], *application_argv]

    # Keep heavyweight imports after the complete non-recursive rank contract.
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("DGX NCCL execution requires CUDA")
    torch.cuda.set_device(rank_environment.local_rank)
    dist.init_process_group(
        backend=rank_environment.backend,
        init_method="env://",
        rank=rank_environment.rank,
        world_size=rank_environment.world_size,
        timeout=timedelta(seconds=rank_environment.timeout_seconds),
    )

    coordinator_runtime = None
    coordinator_settings_sent = False
    try:
        from .dgx_runtime import (
            DgxCoordinatorRuntime,
            TorchDistributedCommandChannel,
            run_dgx_worker,
        )
        from .main import run
        from .runtime import LocalModelRuntime

        channel = TorchDistributedCommandChannel()

        def synchronize(settings):
            nonlocal coordinator_settings_sent
            if rank_environment.role == "coordinator":
                coordinator_settings_sent = True
            return synchronize_rank_settings(
                settings,
                rank=rank_environment.rank,
            )

        if rank_environment.role == "worker":
            run(
                worker_runner=lambda model: run_dgx_worker(
                    LocalModelRuntime(model),
                    channel,
                ),
                settings_synchronizer=synchronize,
            )
        else:

            def runtime_factory(model):
                nonlocal coordinator_runtime
                coordinator_runtime = DgxCoordinatorRuntime(
                    LocalModelRuntime(model),
                    channel,
                )
                return coordinator_runtime

            try:
                run(
                    runtime_factory=runtime_factory,
                    settings_synchronizer=synchronize,
                )
            finally:
                if coordinator_runtime is not None:
                    coordinator_runtime.shutdown()
                elif not coordinator_settings_sent:
                    synchronize_rank_settings(None, rank=rank_environment.rank)
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
