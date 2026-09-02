# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from io import BufferedReader, BufferedWriter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from .cluster import ClusterConfig
from .source_identity import SourceIdentity


_SAFE_FORWARDED_ENVIRONMENT = frozenset(
    {
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
    }
)
_HEARTBEAT = b"HERETIC_DGX_HEARTBEAT\n"
_LEASE_SECONDS = 10
_TERM_GRACE_SECONDS = 5


@dataclass(frozen=True, slots=True)
class RankLaunch:
    """Complete process contract for one DGX rank."""

    rank: int
    role: Literal["coordinator", "worker"]
    host: str
    argv: tuple[str, ...]
    workdir: str
    environment: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class DgxLaunchResult:
    exit_codes: tuple[int, int]
    log_paths: tuple[Path, Path]


class DgxLaunchError(RuntimeError):
    pass


def verify_source_identities(
    launches: tuple[RankLaunch, RankLaunch],
    *,
    identity_reader: Callable[[RankLaunch], SourceIdentity] | None = None,
) -> SourceIdentity:
    """Require exact source/runtime identity before starting either rank."""
    reader = read_source_identity if identity_reader is None else identity_reader
    identities = tuple(reader(launch) for launch in launches)
    if identities[0] != identities[1]:
        raise DgxLaunchError(
            "DGX source identity mismatch between rank 0 and rank 1"
        )
    return identities[0]


def remove_seed_arguments(argv: tuple[str, ...]) -> tuple[str, ...]:
    """Remove user seed syntax before appending one resolved shared seed."""

    result: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--seed":
            if index + 1 >= len(argv):
                raise ValueError("--seed requires a value")
            index += 2
            continue
        if argument.startswith("--seed="):
            index += 1
            continue
        result.append(argument)
        index += 1
    return tuple(result)


def build_ssh_argv(
    launch: RankLaunch,
    *,
    ssh_program: str = "ssh",
) -> tuple[str, ...]:
    """Build a local coordinator command or safely quoted worker SSH command."""

    if launch.role == "coordinator":
        return launch.argv

    assignments = tuple(
        f"{name}={value}" for name, value in launch.environment.items()
    )
    worker_command = shlex.join(
        (
            "env",
            *assignments,
            launch.argv[0],
            "-m",
            "heretic.remote_rank_supervisor",
            "--lease-seconds",
            str(_LEASE_SECONDS),
            "--term-grace-seconds",
            str(_TERM_GRACE_SECONDS),
            "--",
            *launch.argv,
        )
    )
    remote_command = f"cd {shlex.quote(launch.workdir)} && exec {worker_command}"
    return ssh_program, "-T", "--", launch.host, remote_command


def read_source_identity(
    launch: RankLaunch,
    *,
    ssh_program: str = "ssh",
    timeout_seconds: float = 30,
) -> SourceIdentity:
    """Read one rank's source identity before starting the distributed runtime."""
    module_argv = (
        launch.argv[0],
        "-m",
        "heretic.source_identity",
        "--workdir",
        launch.workdir,
    )
    environment: Mapping[str, str] | None
    workdir: str | None
    if launch.role == "coordinator":
        argv = module_argv
        environment = {**os.environ, **launch.environment}
        workdir = launch.workdir
    else:
        pythonpath = launch.environment.get("PYTHONPATH")
        if pythonpath is None:
            raise DgxLaunchError("DGX worker identity preflight requires PYTHONPATH")
        remote_command = (
            f"cd {shlex.quote(launch.workdir)} && exec "
            + shlex.join(("env", f"PYTHONPATH={pythonpath}", *module_argv))
        )
        argv = (ssh_program, "-o", "BatchMode=yes", "-T", "--", launch.host, remote_command)
        environment = None
        workdir = None

    try:
        result = subprocess.run(
            argv,
            cwd=workdir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DgxLaunchError(
            f"DGX rank {launch.rank} source identity preflight failed: {error}"
        ) from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise DgxLaunchError(
            f"DGX rank {launch.rank} source identity preflight exited "
            f"{result.returncode}: {detail}"
        )
    lines = result.stdout.splitlines()
    if len(lines) != 1:
        raise DgxLaunchError(
            f"DGX rank {launch.rank} source identity preflight returned malformed output"
        )
    try:
        payload = json.loads(lines[0])
        if not isinstance(payload, dict):
            raise TypeError("identity payload must be an object")
        identity = SourceIdentity(**payload)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise DgxLaunchError(
            f"DGX rank {launch.rank} source identity preflight returned invalid JSON"
        ) from error
    return identity


def _stream_coordinator_output(
    stream: BufferedReader,
    log_file: BufferedWriter,
) -> None:
    """Tee rank 0 output to its durable log and the invoking terminal."""

    console = getattr(sys.stdout, "buffer", None)
    while chunk := stream.read(8192):
        log_file.write(chunk)
        log_file.flush()
        if console is not None:
            console.write(chunk)
            console.flush()
        else:
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()


def _send_worker_heartbeats(
    stream: BufferedWriter,
    stop: threading.Event,
) -> None:
    try:
        while not stop.is_set():
            stream.write(_HEARTBEAT)
            stream.flush()
            if stop.wait(_LEASE_SECONDS / 3):
                break
    except (BrokenPipeError, OSError):
        pass
    finally:
        stream.close()


def _stop_processes(
    processes: list[subprocess.Popen[bytes]],
    heartbeat_stop: threading.Event,
    heartbeat_thread: threading.Thread | None,
) -> None:
    heartbeat_stop.set()
    if heartbeat_thread is not None:
        heartbeat_thread.join(timeout=2)

    for process in processes[:1]:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + _LEASE_SECONDS + _TERM_GRACE_SECONDS + 2
    for process in reversed(processes):
        if process.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_rank_launches(
    launches: tuple[RankLaunch, RankLaunch],
    *,
    log_dir: str | Path,
    command_builder: Callable[[RankLaunch], tuple[str, ...]] = build_ssh_argv,
    identity_reader: Callable[[RankLaunch], SourceIdentity] = read_source_identity,
    peer_exit_grace_seconds: float = 30,
    poll_interval_seconds: float = 0.1,
    local_hostname: str | None = None,
) -> DgxLaunchResult:
    """Run both DGX ranks, preserve logs, and fail both ranks together."""

    if peer_exit_grace_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("DGX launch timing values must be positive")
    actual_hostname = local_hostname or socket.gethostname()
    if launches[0].host not in {actual_hostname, socket.getfqdn()}:
        raise DgxLaunchError(
            "DGX launch must run on configured coordinator "
            f"{launches[0].host!r}; current host is {actual_hostname!r}"
        )
    verify_source_identities(launches, identity_reader=identity_reader)

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_paths = directory / "rank-0.log", directory / "rank-1.log"
    log_files = [path.open("wb") for path in log_paths]
    processes: list[subprocess.Popen[bytes]] = []
    output_threads: list[threading.Thread] = []
    first_exit_at: float | None = None
    failure_rank: int | None = None
    peer_timeout = False
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None

    try:
        for launch, log_file in zip(launches, log_files, strict=True):
            coordinator = launch.role == "coordinator"
            process = subprocess.Popen(
                command_builder(launch),
                stdin=None if coordinator else subprocess.PIPE,
                stdout=subprocess.PIPE if coordinator else log_file,
                stderr=subprocess.STDOUT,
                cwd=launch.workdir if coordinator else None,
                env=(os.environ | dict(launch.environment)) if coordinator else None,
            )
            processes.append(process)
            if coordinator:
                if process.stdout is None:
                    raise RuntimeError("DGX coordinator output pipe was not created")
                output_thread = threading.Thread(
                    target=_stream_coordinator_output,
                    args=(process.stdout, log_file),
                    name="heretic-dgx-rank-0-output",
                    daemon=True,
                )
                output_thread.start()
                output_threads.append(output_thread)
            else:
                if process.stdin is None:
                    raise RuntimeError("DGX worker control pipe was not created")
                heartbeat_thread = threading.Thread(
                    target=_send_worker_heartbeats,
                    args=(process.stdin, heartbeat_stop),
                    name="heretic-dgx-rank-1-heartbeat",
                    daemon=True,
                )
                heartbeat_thread.start()

        while True:
            exit_codes = [process.poll() for process in processes]
            for rank, exit_code in enumerate(exit_codes):
                if exit_code is not None and exit_code != 0:
                    failure_rank = rank
                    break
            if failure_rank is not None:
                _stop_processes(processes, heartbeat_stop, heartbeat_thread)
                break
            if all(exit_code is not None for exit_code in exit_codes):
                break
            if any(exit_code is not None for exit_code in exit_codes):
                if first_exit_at is None:
                    first_exit_at = time.monotonic()
                elif time.monotonic() - first_exit_at > peer_exit_grace_seconds:
                    peer_timeout = True
                    _stop_processes(processes, heartbeat_stop, heartbeat_thread)
                    break
            time.sleep(poll_interval_seconds)

        final_codes = processes[0].wait(), processes[1].wait()
        for output_thread in output_threads:
            output_thread.join(timeout=5)
            if output_thread.is_alive():
                raise DgxLaunchError("DGX coordinator output stream did not close")
    except BaseException:
        _stop_processes(processes, heartbeat_stop, heartbeat_thread)
        raise
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
        for log_file in log_files:
            log_file.close()

    if failure_rank is not None:
        raise DgxLaunchError(
            f"DGX rank {failure_rank} exited with status {final_codes[failure_rank]}; "
            f"logs: {log_paths[0]}, {log_paths[1]}"
        )
    if peer_timeout:
        raise DgxLaunchError(
            "DGX peer did not exit after its rank completed; "
            f"logs: {log_paths[0]}, {log_paths[1]}"
        )
    if final_codes != (0, 0):
        raise DgxLaunchError(
            f"DGX ranks exited with statuses {final_codes}; "
            f"logs: {log_paths[0]}, {log_paths[1]}"
        )
    return DgxLaunchResult(exit_codes=final_codes, log_paths=log_paths)


def build_rank_launches(
    config: ClusterConfig,
    heretic_argv: tuple[str, ...],
    *,
    seed: int,
    host_environment: Mapping[str, str] | None = None,
) -> tuple[RankLaunch, RankLaunch]:
    """Build the exact coordinator and worker process contracts."""

    if "--seed" in heretic_argv:
        raise ValueError("rank argv must not provide its own seed")

    workdir = str(Path(config.workdir).absolute())
    command = (
        config.python,
        "-m",
        "heretic.cluster_entry",
        *heretic_argv,
        "--seed",
        str(seed),
    )
    forwarded_environment = {
        name: value
        for name, value in (host_environment or {}).items()
        if name in _SAFE_FORWARDED_ENVIRONMENT
    }
    launches: list[RankLaunch] = []

    for rank, node in enumerate(config.nodes):
        role: Literal["coordinator", "worker"] = (
            "coordinator" if rank == 0 else "worker"
        )
        environment = {
            **forwarded_environment,
            "HERETIC_DGX_ACTIVE": "1",
            "HERETIC_DGX_ROLE": role,
            "MASTER_ADDR": config.master_address,
            "MASTER_PORT": str(config.master_port),
            "WORLD_SIZE": str(config.world_size),
            "RANK": str(rank),
            "LOCAL_RANK": "0",
            "HERETIC_DGX_BACKEND": config.backend,
            "HERETIC_DGX_TIMEOUT_SECONDS": str(config.timeout_seconds),
            "HF_DEACTIVATE_ASYNC_LOAD": "1",
            "PYTHONPATH": str(Path(workdir) / "src"),
        }
        if config.nccl_socket_ifname is not None:
            environment["NCCL_SOCKET_IFNAME"] = config.nccl_socket_ifname

        launches.append(
            RankLaunch(
                rank=rank,
                role=role,
                host=node.host,
                argv=command,
                workdir=workdir,
                environment=environment,
            )
        )

    return launches[0], launches[1]
