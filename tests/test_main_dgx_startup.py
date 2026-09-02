# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace
from typing import cast

import pytest

from heretic import main
from heretic.config import QuantizationMethod, Settings
from heretic.runtime import ModelMetadata, ModelRuntime


@pytest.mark.parametrize(
    "argv",
    [
        [
            "heretic",
            "--config",
            "config.toml",
            "--cluster",
            "dgx-cluster.toml",
        ],
        [
            "heretic",
            "--config=config.toml",
            "--cluster=dgx-cluster.toml",
        ],
        [
            "heretic",
            "--model=org/model",
            "--cluster",
            "dgx-cluster.toml",
        ],
    ],
)
def test_legacy_model_rewrite_preserves_option_values(argv: list[str]) -> None:
    assert main.normalize_legacy_model_argv(argv) == argv


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["heretic", "org/model"], ["heretic", "--model", "org/model"]),
        (
            ["heretic", "--batch-size", "1", "org/model"],
            ["heretic", "--batch-size", "1", "--model", "org/model"],
        ),
        (
            ["heretic", "--startup-check", "org/model"],
            ["heretic", "--startup-check", "--model", "org/model"],
        ),
    ],
)
def test_legacy_model_rewrite_preserves_positional_compatibility(
    argv: list[str], expected: list[str]
) -> None:
    assert main.normalize_legacy_model_argv(argv) == expected


def test_coordinator_synchronizes_current_finalized_settings() -> None:
    finalized = Settings.model_construct(model="org/final", seed=789)
    observed: list[Settings | None] = []

    synchronized = main.synchronize_model_settings(
        finalized,
        lambda settings: observed.append(settings) or settings,
        worker=False,
    )

    assert synchronized is finalized
    assert observed == [finalized]


def test_worker_synchronizes_final_settings_before_model_construction(monkeypatch) -> None:
    initial = SimpleNamespace(
        collect_reproducibles=None,
        reproduce=None,
        seed=1,
        cluster=None,
    )
    finalized = SimpleNamespace(seed=2)
    events: list[tuple[str, object]] = []

    monkeypatch.setattr("sys.argv", ["heretic", "--model", "org/initial"])
    monkeypatch.setattr(main, "version", lambda package: "test")
    monkeypatch.setattr(main, "Settings", lambda: initial)
    monkeypatch.setattr(
        main.transformers,
        "set_seed",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        main,
        "Model",
        lambda settings: events.append(("model", settings)) or settings,
    )

    main.run(
        worker_runner=lambda model: events.append(("worker", model)),
        settings_synchronizer=lambda settings: (
            events.append(("synchronize", settings)) or finalized
        ),
    )

    assert events == [
        ("seed", 1),
        ("synchronize", None),
        ("seed", 2),
        ("model", finalized),
        ("worker", finalized),
    ]


def test_worker_exits_without_constructing_model_after_coordinator_abort(monkeypatch) -> None:
    initial = SimpleNamespace(
        collect_reproducibles=None,
        reproduce=None,
        seed=1,
        cluster=None,
    )
    constructed: list[object] = []

    monkeypatch.setattr("sys.argv", ["heretic", "--model", "org/initial"])
    monkeypatch.setattr(main, "version", lambda package: "test")
    monkeypatch.setattr(main, "Settings", lambda: initial)
    monkeypatch.setattr(main.transformers, "set_seed", lambda seed: None)
    monkeypatch.setattr(main, "Model", lambda settings: constructed.append(settings))

    main.run(
        worker_runner=lambda model: None,
        settings_synchronizer=lambda settings: None,
    )

    assert constructed == []


def test_worker_rejects_synchronized_settings_without_seed(monkeypatch) -> None:
    initial = SimpleNamespace(
        collect_reproducibles=None,
        reproduce=None,
        seed=1,
        cluster=None,
    )
    synchronized = SimpleNamespace(seed=None)

    monkeypatch.setattr("sys.argv", ["heretic", "--model", "org/initial"])
    monkeypatch.setattr(main, "version", lambda package: "test")
    monkeypatch.setattr(main, "Settings", lambda: initial)
    monkeypatch.setattr(main.transformers, "set_seed", lambda seed: None)

    with pytest.raises(RuntimeError, match="finalized seed"):
        main.run(
            worker_runner=lambda model: None,
            settings_synchronizer=lambda settings: synchronized,
        )


class StartupCheckRuntime:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts = []
        self.max_new_tokens = None
        self.shutdown_called = False

    def get_responses_once(self, prompts, *, max_new_tokens=None):
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        return self.responses

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            runtime_kind="dgx",
            base_model_id="org/model",
            base_model_revision="abc123",
            dtype="torch.bfloat16",
            quantization="none",
            global_layer_count=2,
            abliterable_components=("self_attn.o_proj",),
            adapter_generation=None,
        )

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_startup_check_is_an_explicit_operational_setting() -> None:
    settings = Settings.model_construct(startup_check=True)

    assert "startup_check" in Settings.model_fields
    assert settings.startup_check is True


def test_startup_check_runs_one_text_response_without_mutation() -> None:
    runtime = StartupCheckRuntime(["OK"])
    settings = Settings.model_construct(system_prompt="system")

    main.run_startup_check(cast(ModelRuntime, runtime), settings)

    assert len(runtime.prompts) == 1
    assert runtime.prompts[0].user == "Reply with the single word OK."
    assert runtime.max_new_tokens == 8
    assert runtime.shutdown_called is True


def test_startup_check_rejects_empty_response() -> None:
    runtime = StartupCheckRuntime([""])
    settings = Settings.model_construct(system_prompt="system")

    with pytest.raises(RuntimeError, match="one non-empty response"):
        main.run_startup_check(cast(ModelRuntime, runtime), settings)

    assert runtime.shutdown_called is True


def test_distributed_evaluation_model_fails_before_identity_or_runtime_mutation() -> None:
    class RecordingRuntime:
        def __init__(self) -> None:
            self.reset_models: list[str | None] = []

        def reset_model(self, model: str | None = None) -> None:
            self.reset_models.append(model)

    settings = Settings.model_construct(
        model="org/base-model",
        model_commit="a" * 40,
        evaluate_model="org/evaluation-model",
    )
    runtime = RecordingRuntime()

    with pytest.raises(RuntimeError, match="evaluation model.*not available"):
        main.prepare_evaluation_model(
            settings,
            cast(ModelRuntime, runtime),
            distributed=True,
        )

    assert settings.model == "org/base-model"
    assert settings.model_commit == "a" * 40
    assert runtime.reset_models == []


def test_distributed_evaluation_is_rejected_before_settings_synchronization() -> None:
    settings = Settings.model_construct(
        model="org/base-model",
        model_commit="a" * 40,
        evaluate_model="org/evaluation-model",
    )
    synchronized: list[Settings | None] = []

    with pytest.raises(RuntimeError, match="evaluation model"):
        main.finalize_model_settings(
            settings,
            lambda value: synchronized.append(value) or value,
        )

    assert synchronized == []


def test_export_strategy_inspection_forwards_pinned_revision(monkeypatch) -> None:
    observed: list[str | None] = []

    class MetaModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def get_memory_footprint(self) -> int:
            return 0

    monkeypatch.setattr(
        main,
        "get_model_class",
        lambda _model, revision=None: observed.append(revision) or MetaModel,
    )
    monkeypatch.setattr(main, "ask_if_unset", lambda value, _question: value)
    settings = Settings.model_construct(
        model="org/model",
        model_commit="pinned-revision",
        quantization=QuantizationMethod.BNB_4BIT,
        export_strategy=None,
    )
    model = SimpleNamespace(trusted_models=set(), revision_kwargs={})

    main.obtain_export_strategy(settings, cast(main.Model, model))

    assert observed == ["pinned-revision"]
