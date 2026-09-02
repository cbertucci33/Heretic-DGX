# SPDX-License-Identifier: AGPL-3.0-or-later

import shlex
import sys
import time
from pathlib import Path

import pytest

from heretic.cluster import ClusterConfig, ClusterNode
from heretic.source_identity import SourceIdentity
from heretic.cluster_launcher import (
    DgxLaunchError,
    RankLaunch,
    build_rank_launches,
    build_ssh_argv,
    remove_seed_arguments,
    read_source_identity,
    run_rank_launches,
    verify_source_identities,
)

_TEST_SOURCE_IDENTITY = SourceIdentity(
    "a" * 64, "b" * 64, "c" * 64, sys.executable, "3.12.12", "2.0.0"
)


def test_build_rank_launches_creates_one_coordinator_and_one_worker() -> None:
    config = ClusterConfig(
        nodes=(
            ClusterNode(host="dgx-01", rank_address="10.10.10.1"),
            ClusterNode(host="dgx-02", rank_address="10.10.10.2"),
        ),
        python="/opt/heretic/bin/python",
        workdir="/srv/heretic",
        master_port=29517,
        nccl_socket_ifname="rocep1s0f0,roceP2p1s0f0",
    )

    launches = build_rank_launches(
        config,
        ("--cluster", "dgx-cluster.toml", "/models/model-under-test"),
        seed=123456789,
    )

    assert len(launches) == 2
    assert [launch.rank for launch in launches] == [0, 1]
    assert [launch.role for launch in launches] == ["coordinator", "worker"]
    assert [launch.host for launch in launches] == ["dgx-01", "dgx-02"]
    assert all(
        launch.argv
        == (
            "/opt/heretic/bin/python",
            "-m",
            "heretic.cluster_entry",
            "--cluster",
            "dgx-cluster.toml",
            "/models/model-under-test",
            "--seed",
            "123456789",
        )
        for launch in launches
    )
    assert all(launch.workdir == "/srv/heretic" for launch in launches)

    for rank, launch in enumerate(launches):
        assert launch.environment == {
            "HERETIC_DGX_ACTIVE": "1",
            "HERETIC_DGX_ROLE": "coordinator" if rank == 0 else "worker",
            "MASTER_ADDR": "10.10.10.1",
            "MASTER_PORT": "29517",
            "WORLD_SIZE": "2",
            "RANK": str(rank),
            "LOCAL_RANK": "0",
            "HERETIC_DGX_BACKEND": "nccl",
            "HERETIC_DGX_TIMEOUT_SECONDS": "900",
            "HF_DEACTIVATE_ASYNC_LOAD": "1",
            "PYTHONPATH": "/srv/heretic/src",
            "NCCL_SOCKET_IFNAME": "rocep1s0f0,roceP2p1s0f0",
        }


def test_build_rank_launches_normalizes_relative_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = ClusterConfig(
        nodes=(
            ClusterNode(host="dgx-01", rank_address="10.10.10.1"),
            ClusterNode(host="dgx-02", rank_address="10.10.10.2"),
        ),
        python="/opt/heretic/bin/python",
        workdir="candidate tree",
    )

    launches = build_rank_launches(config, ("--model", "org/model"), seed=1)

    expected_workdir = str((tmp_path / "candidate tree").absolute())
    assert all(launch.workdir == expected_workdir for launch in launches)
    assert all(
        launch.environment["PYTHONPATH"] == str(Path(expected_workdir) / "src")
        for launch in launches
    )


def test_build_rank_launches_forwards_cache_paths_but_not_secrets() -> None:
    config = ClusterConfig(
        nodes=(
            ClusterNode(host="dgx-01", rank_address="10.10.10.1"),
            ClusterNode(host="dgx-02", rank_address="10.10.10.2"),
        ),
        python="/opt/heretic/bin/python",
        workdir="/srv/heretic",
    )

    launches = build_rank_launches(
        config,
        ("--model", "org/model"),
        seed=1,
        host_environment={
            "HF_HOME": "/srv/heretic/.cache/huggingface",
            "HF_HUB_CACHE": "/srv/heretic/.cache/hub",
            "TRITON_CACHE_DIR": "/srv/heretic/.cache/triton",
            "HF_TOKEN": "must-not-cross-rank-boundary",
            "AUTHORIZATION": "must-not-cross-rank-boundary",
            "UNRELATED": "must-not-cross-rank-boundary",
        },
    )

    for launch in launches:
        assert launch.environment["HF_HOME"] == "/srv/heretic/.cache/huggingface"
        assert launch.environment["HF_HUB_CACHE"] == "/srv/heretic/.cache/hub"
        assert launch.environment["TRITON_CACHE_DIR"] == "/srv/heretic/.cache/triton"
        assert "HF_TOKEN" not in launch.environment
        assert "AUTHORIZATION" not in launch.environment
        assert "UNRELATED" not in launch.environment


def test_remove_seed_arguments_supports_both_cli_forms() -> None:
    assert remove_seed_arguments(
        ("--model", "org/model", "--seed", "7", "--batch-size", "2")
    ) == ("--model", "org/model", "--batch-size", "2")
    assert remove_seed_arguments(("--seed=7", "--model", "org/model")) == (
        "--model",
        "org/model",
    )


def test_build_ssh_argv_runs_coordinator_locally_and_quotes_worker_values() -> None:
    config = ClusterConfig(
        nodes=(
            ClusterNode(host="dgx-01", rank_address="10.10.10.1"),
            ClusterNode(host="dgx-02", rank_address="10.10.10.2"),
        ),
        python="/opt/heretic env/bin/python",
        workdir="/srv/heretic candidate",
    )
    launches = build_rank_launches(
        config,
        ("--model", "/models/model with spaces"),
        seed=17,
    )

    coordinator_argv = build_ssh_argv(launches[0])
    assert coordinator_argv == launches[0].argv

    ssh_argv = build_ssh_argv(launches[1])
    assert ssh_argv[:4] == ("ssh", "-T", "--", "dgx-02")
    remote = ssh_argv[4]
    assert remote.startswith(f"cd {shlex.quote('/srv/heretic candidate')} && ")
    expected_rank_command = shlex.join(
        (
            *launches[1].argv,
        )
    )
    assert "-m heretic.remote_rank_supervisor" in remote
    assert "--lease-seconds 10 --term-grace-seconds 5 --" in remote
    assert expected_rank_command in remote
    assert "HERETIC_DGX_RUN_ID" not in remote


def test_run_rank_launches_streams_coordinator_and_preserves_separate_logs(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    config = ClusterConfig(
        nodes=(
            ClusterNode(host="dgx-01", rank_address="127.0.0.1"),
            ClusterNode(host="dgx-02", rank_address="127.0.0.2"),
        ),
        python=sys.executable,
        workdir=str(tmp_path),
        backend="gloo",
    )
    launches = build_rank_launches(config, ("--model", "unused"), seed=1)

    result = run_rank_launches(
        launches,
        log_dir=tmp_path / "logs",
        identity_reader=lambda _launch: _TEST_SOURCE_IDENTITY,
        command_builder=lambda launch: (
            sys.executable,
            "-c",
            f"print('rank-{launch.rank}-complete')",
        ),
        peer_exit_grace_seconds=5,
        poll_interval_seconds=0.01,
        local_hostname="dgx-01",
    )

    assert result.exit_codes == (0, 0)
    assert result.log_paths[0].read_text().strip() == "rank-0-complete"
    assert result.log_paths[1].read_text().strip() == "rank-1-complete"
    assert "rank-0-complete" in capfd.readouterr().out


def test_run_rank_launches_terminates_peer_after_failure(tmp_path: Path) -> None:
    config = ClusterConfig(
        nodes=(
            ClusterNode(host="dgx-01", rank_address="127.0.0.1"),
            ClusterNode(host="dgx-02", rank_address="127.0.0.2"),
        ),
        python=sys.executable,
        workdir=str(tmp_path),
        backend="gloo",
    )
    launches = build_rank_launches(config, ("--model", "unused"), seed=1)
    started = time.monotonic()

    with pytest.raises(DgxLaunchError, match="rank 0"):
        run_rank_launches(
            launches,
            log_dir=tmp_path / "logs",
            identity_reader=lambda _launch: _TEST_SOURCE_IDENTITY,
            command_builder=lambda launch: (
                sys.executable,
                "-c",
                "raise SystemExit(7)"
                if launch.rank == 0
                else "import sys; sys.stdin.buffer.read()",
            ),
            peer_exit_grace_seconds=5,
            poll_interval_seconds=0.01,
            local_hostname="dgx-01",
        )

    assert time.monotonic() - started < 10


@pytest.mark.skipif(sys.platform != "linux", reason="DGX runtime is Linux-only")
def test_run_rank_launches_revokes_worker_lease_after_rank_failure(
    tmp_path: Path,
) -> None:
    config = ClusterConfig(
        nodes=(
            ClusterNode(host="dgx-01", rank_address="127.0.0.1"),
            ClusterNode(host="dgx-02", rank_address="127.0.0.2"),
        ),
        python=sys.executable,
        workdir=str(tmp_path),
        backend="gloo",
    )
    launches = build_rank_launches(config, ("--model", "unused"), seed=1)
    marker = tmp_path / "remote-child.pid"
    child_script = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )

    def command(launch: RankLaunch) -> tuple[str, ...]:
        if launch.rank == 0:
            return sys.executable, "-c", "import time; time.sleep(.3); raise SystemExit(7)"
        return (
            sys.executable,
            "-m",
            "heretic.remote_rank_supervisor",
            "--lease-seconds",
            "0.5",
            "--term-grace-seconds",
            "0.2",
            "--",
            sys.executable,
            "-c",
            child_script,
        )

    with pytest.raises(DgxLaunchError, match="rank 0"):
        run_rank_launches(
            launches,
            log_dir=tmp_path / "logs",
            identity_reader=lambda _launch: _TEST_SOURCE_IDENTITY,
            command_builder=command,
                poll_interval_seconds=0.01,
            local_hostname="dgx-01",
        )

    assert marker.exists()
    child_pid = int(marker.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()


def test_run_rank_launches_rejects_wrong_coordinator_host(tmp_path: Path) -> None:
    config = ClusterConfig(
        nodes=(
            ClusterNode(host="dgx-01", rank_address="127.0.0.1"),
            ClusterNode(host="dgx-02", rank_address="127.0.0.2"),
        ),
        python=sys.executable,
        workdir=str(tmp_path),
        backend="gloo",
    )

    with pytest.raises(DgxLaunchError, match="must run on configured coordinator"):
        run_rank_launches(
            build_rank_launches(config, ("--model", "unused"), seed=1),
            log_dir=tmp_path / "logs",
            local_hostname="some-other-host",
        )


def test_source_identity_preflight_accepts_exact_match() -> None:
    launches = (
        RankLaunch(0, "coordinator", "dgx-01", ("python",), "/srv/heretic", {}),
        RankLaunch(1, "worker", "dgx-02", ("python",), "/srv/heretic", {}),
    )
    identity = SourceIdentity("a" * 64, "b" * 64, "c" * 64, "/opt/python", "3.12.12", "2.0.0")

    accepted = verify_source_identities(launches, identity_reader=lambda _launch: identity)

    assert accepted == identity


def test_source_identity_preflight_rejects_rank_mismatch() -> None:
    launches = (
        RankLaunch(0, "coordinator", "dgx-01", ("python",), "/srv/heretic", {}),
        RankLaunch(1, "worker", "dgx-02", ("python",), "/srv/heretic", {}),
    )
    identities = {
        0: SourceIdentity("a" * 64, "b" * 64, "c" * 64, "/opt/python", "3.12.12", "2.0.0"),
        1: SourceIdentity("d" * 64, "b" * 64, "c" * 64, "/opt/python", "3.12.12", "2.0.0"),
    }

    with pytest.raises(DgxLaunchError, match="source identity mismatch"):
        verify_source_identities(launches, identity_reader=lambda launch: identities[launch.rank])


def test_reads_local_source_identity_through_configured_python() -> None:
    root = Path(__file__).resolve().parents[1]
    launch = RankLaunch(
        0,
        "coordinator",
        "dgx-01",
        (sys.executable,),
        str(root),
        {"PYTHONPATH": str(root / "src")},
    )

    identity = read_source_identity(launch)

    assert len(identity.source_sha256) == 64
    assert identity.python_version.startswith("3.12.")
    assert identity.package_version == "2.0.0.dev0"


def test_run_rank_launches_verifies_identity_before_building_commands(tmp_path: Path) -> None:
    config = ClusterConfig(
        nodes=(
            ClusterNode(host="dgx-01", rank_address="127.0.0.1"),
            ClusterNode(host="dgx-02", rank_address="127.0.0.2"),
        ),
        python=sys.executable,
        workdir=str(tmp_path),
        backend="gloo",
    )
    launches = build_rank_launches(config, ("--model", "unused"), seed=1)
    identity = SourceIdentity("a" * 64, "b" * 64, "c" * 64, sys.executable, "3.12.12", "2.0.0")
    events: list[str] = []

    def identity_reader(launch: RankLaunch) -> SourceIdentity:
        events.append(f"identity-{launch.rank}")
        return identity

    def command_builder(launch: RankLaunch) -> tuple[str, ...]:
        events.append(f"command-{launch.rank}")
        return sys.executable, "-c", "pass"

    run_rank_launches(
        launches,
        log_dir=tmp_path / "logs",
        command_builder=command_builder,
        identity_reader=identity_reader,
        poll_interval_seconds=0.01,
        local_hostname="dgx-01",
    )

    assert events[:2] == ["identity-0", "identity-1"]
    assert events[2:] == ["command-0", "command-1"]
