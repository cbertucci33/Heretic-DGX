# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from heretic.checkpoint_identity import build_checkpoint_payload_identity
from heretic.application_runner import (
    preflight_and_collect_rank_applications,
    run_rank_application_plan,
)
from heretic.dgx_runtime import (
    DgxCommand,
    DgxCoordinatorRuntime,
    run_dgx_worker,
)
from heretic.cluster import load_cluster_config
from heretic.cluster_cli import launch_cluster_application
from heretic.collective_probe import CollectiveProbeResult
from heretic.collective_probe_runner import (
    _run_probe_plan,
    preflight_and_collect_rank_collective_probes,
)
from heretic.launch_plan import build_rank_launch_plans
from heretic.launch_plan import RankLaunchPlan
from heretic.model_loading import build_model_load_kwargs
from heretic.preflight_collector import collect_rank_preflights
from heretic.rank_environment import read_rank_environment
from heretic.rank_application import run_rank_application
from heretic.rank_preflight import (
    RankPreflightIdentity,
    require_matching_rank_preflights,
)
from heretic.source_identity import build_source_identity


def _source(root: Path):
    (root / "src/heretic").mkdir(parents=True, exist_ok=True)
    (root / "src/heretic/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return build_source_identity(
        root,
        python_executable="/srv/heretic/.venv/bin/python",
        python_version="3.12.12",
        package_version="2.0.0",
    )


def _checkpoint(root: Path):
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return build_checkpoint_payload_identity(root)


class TestDgxCluster(TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_two_node_config_builds_parseable_rank_plans(self) -> None:
        cluster_file = self.root / "cluster.toml"
        cluster_file.write_text(
            """
python = "/srv/heretic/.venv/bin/python"
workdir = "/srv/heretic"
master_port = 29517
backend = "nccl"
nccl_socket_ifname = "fabric0"

[[nodes]]
host = "gx10-01"
rank_address = "10.10.10.1"

[[nodes]]
host = "gx10-02"
rank_address = "10.10.10.2"
""",
            encoding="utf-8",
        )

        plans = build_rank_launch_plans(
            load_cluster_config(cluster_file),
            ("/models/checkpoint",),
            entry_module="heretic.rank_entry",
            seed=7,
            host_environment={"HF_HOME": "/cache/hf", "HF_TOKEN": "do-not-forward"},
        )
        environments = [
            read_rank_environment(plan.environment_dict()) for plan in plans
        ]

        self.assertEqual(
            [(plan.host, plan.rank) for plan in plans],
            [("gx10-01", 0), ("gx10-02", 1)],
        )
        self.assertEqual(
            [(environment.rank, environment.role) for environment in environments],
            [(0, "coordinator"), (1, "worker")],
        )
        self.assertTrue(
            all("HF_TOKEN" not in plan.environment_dict() for plan in plans)
        )

    def test_identities_change_with_source_and_checkpoint_bytes(self) -> None:
        source_root = self.root / "source"
        checkpoint_root = self.root / "checkpoint"
        source_before = _source(source_root)
        checkpoint_before = _checkpoint(checkpoint_root)

        (source_root / "src/heretic/example.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
        (checkpoint_root / "model.safetensors").write_bytes(b"changed")

        source_after = build_source_identity(
            source_root,
            python_executable="/srv/heretic/.venv/bin/python",
            python_version="3.12.12",
            package_version="2.0.0",
        )
        self.assertNotEqual(source_before.source_sha256, source_after.source_sha256)
        self.assertNotEqual(
            checkpoint_before.digest,
            build_checkpoint_payload_identity(checkpoint_root).digest,
        )

    def test_rank_environment_rejects_unsafe_master_address(self) -> None:
        cluster_file = self.root / "cluster.toml"
        cluster_file.write_text(
            'python = "/python"\nworkdir = "/work"\n'
            '[[nodes]]\nhost = "a"\nrank_address = "10.0.0.1"\n'
            '[[nodes]]\nhost = "b"\nrank_address = "10.0.0.2"\n',
            encoding="utf-8",
        )
        plan = build_rank_launch_plans(
            load_cluster_config(cluster_file), ("model",), entry_module="entry", seed=1
        )[0]
        environment = plan.environment_dict()
        environment["MASTER_ADDR"] = "ssh://10.0.0.1"

        with self.assertRaisesRegex(ValueError, "MASTER_ADDR"):
            read_rank_environment(environment)

    def test_rank_preflight_rejects_checkpoint_mismatch(self) -> None:
        source = _source(self.root / "source")
        checkpoint = _checkpoint(self.root / "checkpoint")
        rank_zero = RankPreflightIdentity(rank=0, source=source, checkpoint=checkpoint)
        rank_one = RankPreflightIdentity(rank=1, source=source, checkpoint=checkpoint)

        self.assertIs(require_matching_rank_preflights(rank_zero, rank_one), rank_zero)

        mismatched = replace(
            rank_one,
            checkpoint=replace(rank_one.checkpoint, digest="f" * 64),
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint-payload"):
            require_matching_rank_preflights(rank_zero, mismatched)

    def test_coordinator_collects_matching_rank_preflights(self) -> None:
        cluster_file = self.root / "cluster.toml"
        cluster_file.write_text(
            'python = "/python"\nworkdir = "/work"\n'
            '[[nodes]]\nhost = "a"\nrank_address = "10.0.0.1"\n'
            '[[nodes]]\nhost = "b"\nrank_address = "10.0.0.2"\n',
            encoding="utf-8",
        )
        plans = build_rank_launch_plans(
            load_cluster_config(cluster_file), ("model",), entry_module="entry", seed=1
        )
        source = _source(self.root / "source")
        checkpoint = _checkpoint(self.root / "checkpoint")
        outputs = [
            RankPreflightIdentity(rank=rank, source=source, checkpoint=checkpoint)
            for rank in (0, 1)
        ]
        completed = [
            subprocess.CompletedProcess((), 0, identity.canonical_json(), "")
            for identity in outputs
        ]

        with patch(
            "heretic.preflight_collector.subprocess.run", side_effect=completed
        ) as run:
            result = collect_rank_preflights(
                plans, "/models/checkpoint", timeout_seconds=30
            )

        self.assertEqual(result, outputs[0])
        self.assertEqual(run.call_args_list[0].args[0][0], "env")
        self.assertEqual(run.call_args_list[1].args[0][0], "ssh")

    def test_collective_probe_runs_coordinator_locally_and_worker_over_ssh(self) -> None:
        cluster_file = self.root / "cluster.toml"
        cluster_file.write_text(
            'python = "/python"\nworkdir = "/work"\n'
            'nccl_socket_ifname = "fabric0"\n'
            '[[nodes]]\nhost = "a"\nrank_address = "10.0.0.1"\n'
            '[[nodes]]\nhost = "b"\nrank_address = "10.0.0.2"\n',
            encoding="utf-8",
        )
        plans = build_rank_launch_plans(
            load_cluster_config(cluster_file), ("model",), entry_module="entry", seed=1
        )
        completed = subprocess.CompletedProcess(
            (),
            0,
            CollectiveProbeResult(0, 2, "gloo", 3).canonical_json(),
            "",
        )

        with patch(
            "heretic.application_runner.subprocess.run", return_value=completed
        ) as run:
            _run_probe_plan(plans[0], timeout_seconds=30)

        command = run.call_args.args[0]
        self.assertEqual(command[0], "timeout")
        self.assertIn("30s", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=", command)
        self.assertIn("GLOO_SOCKET_IFNAME=fabric0", command)

        completed = subprocess.CompletedProcess(
            (),
            0,
            CollectiveProbeResult(1, 2, "gloo", 3).canonical_json(),
            "",
        )
        with patch(
            "heretic.application_runner.subprocess.run", return_value=completed
        ) as run:
            _run_probe_plan(plans[1], timeout_seconds=30)

        self.assertEqual(run.call_args.args[0][0], "ssh")

    def test_coordinator_preflights_before_collective_probe(self) -> None:
        cluster_file = self.root / "cluster.toml"
        cluster_file.write_text(
            'python = "/python"\nworkdir = "/work"\n'
            '[[nodes]]\nhost = "a"\nrank_address = "10.0.0.1"\n'
            '[[nodes]]\nhost = "b"\nrank_address = "10.0.0.2"\n',
            encoding="utf-8",
        )
        plans = build_rank_launch_plans(
            load_cluster_config(cluster_file), ("model",), entry_module="entry", seed=1
        )
        results = [CollectiveProbeResult(rank, 2, "gloo", 3) for rank in (0, 1)]
        events = []
        identity = object()

        with patch(
            "heretic.collective_probe_runner.collect_rank_preflights",
            side_effect=lambda *_, **__: events.append("preflight") or identity,
        ), patch(
            "heretic.collective_probe_runner._run_probe_plan",
            side_effect=lambda plan, **_: events.append(f"rank-{plan.rank}")
            or results[plan.rank],
        ):
            preflight, collected = preflight_and_collect_rank_collective_probes(
                plans,
                "/checkpoint",
                timeout_seconds=30,
            )

        self.assertIs(preflight, identity)
        self.assertEqual(collected, tuple(results))
        self.assertEqual(events[0], "preflight")

        with patch(
            "heretic.collective_probe_runner.collect_rank_preflights",
            side_effect=RuntimeError("identity mismatch"),
        ), patch(
            "heretic.collective_probe_runner._run_probe_plan"
        ) as run_probe:
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                preflight_and_collect_rank_collective_probes(
                    plans,
                    "/checkpoint",
                    timeout_seconds=30,
                )
        run_probe.assert_not_called()

    def test_rank_application_captures_successful_fake_rank(self) -> None:
        plan = RankLaunchPlan(
            rank=0,
            role="coordinator",
            host="local",
            argv=(sys.executable, "-c", "print('rank-zero-ok')"),
            workdir=str(self.root),
            environment=(),
        )

        result = run_rank_application_plan(plan, timeout_seconds=5)

        self.assertEqual(result.rank, 0)
        self.assertEqual(result.stdout, "rank-zero-ok\n")

    def test_rank_application_propagates_nonzero_exit(self) -> None:
        plan = RankLaunchPlan(
            rank=0,
            role="coordinator",
            host="local",
            argv=(sys.executable, "-c", "raise SystemExit(7)"),
            workdir=str(self.root),
            environment=(),
        )

        with self.assertRaisesRegex(RuntimeError, "exit code 7"):
            run_rank_application_plan(plan, timeout_seconds=5)

    def test_rank_applications_do_not_start_when_preflight_fails(self) -> None:
        plans = (
            RankLaunchPlan(0, "coordinator", "a", ("/python",), "/work", ()),
            RankLaunchPlan(1, "worker", "b", ("/python",), "/work", ()),
        )

        with patch(
            "heretic.application_runner.collect_rank_preflights",
            side_effect=RuntimeError("identity mismatch"),
        ), patch("heretic.application_runner.collect_rank_applications") as collect:
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                preflight_and_collect_rank_applications(
                    plans,
                    "/checkpoint",
                    timeout_seconds=5,
                )

        collect.assert_not_called()

    def test_rank_application_timeout_leaves_no_child_process(self) -> None:
        child_pid_file = self.root / "child.pid"
        script = (
            "import pathlib, subprocess, time; "
            "child=subprocess.Popen(['sleep', '60']); "
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
            "time.sleep(60)"
        )
        plan = RankLaunchPlan(
            rank=0,
            role="coordinator",
            host="local",
            argv=(sys.executable, "-c", script),
            workdir=str(self.root),
            environment=(),
        )

        with self.assertRaisesRegex(RuntimeError, "exit code 124"):
            run_rank_application_plan(plan, timeout_seconds=1)

        child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
            time.sleep(0.05)
        self.assertFalse(Path(f"/proc/{child_pid}").exists())

    def test_rank_entry_preserves_arguments_and_seed_without_recursion(self) -> None:
        cluster_file = self.root / "cluster.toml"
        cluster_file.write_text(
            'python = "/python"\nworkdir = "/work"\n'
            '[[nodes]]\nhost = "a"\nrank_address = "10.0.0.1"\n'
            '[[nodes]]\nhost = "b"\nrank_address = "10.0.0.2"\n',
            encoding="utf-8",
        )
        plan = build_rank_launch_plans(
            load_cluster_config(cluster_file),
            ("--model", "/checkpoint", "--max-tokens", "23"),
            entry_module="heretic.rank_application",
            seed=91,
        )[1]
        called = []

        environment, result = run_rank_application(
            plan.argv[3:],
            environment=plan.environment_dict(),
            application_main=lambda: called.append(tuple(sys.argv[1:])) or 7,
        )

        self.assertEqual(environment.rank, 1)
        self.assertEqual(result, 7)
        self.assertEqual(
            called,
            [("--model", "/checkpoint", "--max-tokens", "23", "--seed", "91")],
        )

        with self.assertRaisesRegex(ValueError, "must not contain --cluster"):
            run_rank_application(
                ("--model", "/checkpoint", "--cluster", "cluster.toml", "--seed", "91"),
                environment=plan.environment_dict(),
                application_main=lambda: 0,
            )

    def test_cluster_cli_preflights_once_and_propagates_rank_failure(self) -> None:
        cluster_file = self.root / "cluster.toml"
        cluster_file.write_text(
            'python = "/python"\nworkdir = "/work"\ntimeout_seconds = 17\n'
            '[[nodes]]\nhost = "a"\nrank_address = "10.0.0.1"\n'
            '[[nodes]]\nhost = "b"\nrank_address = "10.0.0.2"\n',
            encoding="utf-8",
        )
        settings = type("SettingsFixture", (), {})()
        settings.cluster = str(cluster_file)
        settings.seed = 44
        settings.model = "/checkpoint"

        with patch(
            "heretic.cluster_cli.preflight_and_collect_rank_applications",
            side_effect=RuntimeError("rank 1 application failed"),
        ) as launch:
            with self.assertRaisesRegex(RuntimeError, "rank 1 application failed"):
                launch_cluster_application(
                    settings,
                    (
                        "--cluster",
                        str(cluster_file),
                        "--model",
                        "/checkpoint",
                        "--seed=44",
                        "--max-tokens",
                        "23",
                    ),
                )

        plans = launch.call_args.args[0]
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.args[1], "/checkpoint")
        self.assertEqual(launch.call_args.kwargs["timeout_seconds"], 17)
        self.assertEqual(plans[0].argv, plans[1].argv)
        self.assertEqual(
            plans[0].argv,
            (
                "/python",
                "-m",
                "heretic.rank_application",
                "--model",
                "/checkpoint",
                "--max-tokens",
                "23",
                "--seed",
                "44",
            ),
        )

    def test_model_load_arguments_select_one_local_or_tp_path(self) -> None:
        common = {
            "dtype": "bfloat16",
            "quantization_config": "quantization",
            "model_commit": "revision",
            "device_map": "auto",
            "max_memory": {"0": "100GB", "cpu": "20GB"},
            "trust_remote_code": True,
        }

        local = build_model_load_kwargs(distributed=False, **common)
        distributed = build_model_load_kwargs(distributed=True, **common)

        self.assertEqual(local["device_map"], "auto")
        self.assertEqual(local["max_memory"], {0: "100GB", "cpu": "20GB"})
        self.assertNotIn("tp_plan", local)
        self.assertEqual(distributed["tp_plan"], "auto")
        self.assertNotIn("device_map", distributed)
        self.assertNotIn("max_memory", distributed)
        for name in ("dtype", "quantization_config", "revision", "trust_remote_code"):
            self.assertEqual(local[name], distributed[name])

    def test_dgx_command_rejects_unknown_operation(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported DGX runtime operation"):
            DgxCommand("unknown", (), {})  # type: ignore[arg-type]

    def test_coordinator_runtime_mirrors_operation_and_shutdown(self) -> None:
        class RuntimeFixture:
            def __init__(self) -> None:
                self.calls = []

            def get_responses(self, prompts, *, skip_special_tokens=True):
                self.calls.append(("get_responses", prompts, skip_special_tokens))
                return ["ok"]

            def shutdown(self):
                self.calls.append(("shutdown",))

        class ChannelFixture:
            def __init__(self) -> None:
                self.commands = []

            def send(self, command):
                self.commands.append(command)

            def complete(self, local_error):
                self.asserted_error = local_error
                return None, None

        local = RuntimeFixture()
        channel = ChannelFixture()
        runtime = DgxCoordinatorRuntime(local, channel)  # type: ignore[arg-type]

        self.assertEqual(runtime.get_responses([]), ["ok"])
        runtime.shutdown()

        self.assertEqual(
            [command.operation for command in channel.commands],
            ["get_responses", "shutdown"],
        )
        self.assertEqual(local.calls, [("get_responses", [], True), ("shutdown",)])

    def test_worker_executes_commands_and_reports_first_error(self) -> None:
        class RuntimeFixture:
            def __init__(self, *, fail=False) -> None:
                self.calls = []
                self.fail = fail

            def reset_model(self, model=None):
                self.calls.append(("reset_model", model))
                if self.fail:
                    raise ValueError("worker boom")

            def shutdown(self):
                self.calls.append(("shutdown",))

        class ChannelFixture:
            def __init__(self, commands) -> None:
                self.commands = list(commands)
                self.completions = []

            def receive(self):
                return self.commands.pop(0)

            def complete(self, local_error):
                self.completions.append(local_error)
                return None, local_error

        local = RuntimeFixture()
        channel = ChannelFixture(
            [DgxCommand("reset_model", (None,), {}), DgxCommand("shutdown", (), {})]
        )
        run_dgx_worker(local, channel)  # type: ignore[arg-type]
        self.assertEqual(local.calls, [("reset_model", None), ("shutdown",)])
        self.assertEqual(channel.completions, [None, None])

        failing = RuntimeFixture(fail=True)
        failing_channel = ChannelFixture([DgxCommand("reset_model", (None,), {})])
        with self.assertRaisesRegex(ValueError, "worker boom"):
            run_dgx_worker(failing, failing_channel)  # type: ignore[arg-type]
        self.assertEqual(failing_channel.completions, ["ValueError: worker boom"])
