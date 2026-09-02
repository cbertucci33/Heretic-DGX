# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

from heretic.config import ScorerConfig, Settings
from heretic.evaluator import Evaluator, ScorerEntry
from heretic.model import AbliterationParameters, Model
from heretic.plugin import Context
from heretic.runtime import LocalModelRuntime, ModelMetadata, ModelRuntime
from heretic.scorer import Score, Scorer
from heretic.utils import Prompt


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Prompt], bool]] = []
        self.logits_calls: list[list[Prompt]] = []
        self.residual_calls: list[list[Prompt]] = []
        self.residual_mean_calls: list[list[Prompt]] = []
        self.mutation_calls: list[tuple[object, ...]] = []
        self.model = self
        self.settings = SimpleNamespace(
            model="example/model",
            model_commit="abc123",
            quantization=SimpleNamespace(value="none"),
        )
        self.dtype = torch.bfloat16
        self.config = SimpleNamespace(
            quantization_config={"quant_method": "compressed-tensors"}
        )
        self.save_calls: list[tuple[object, ...]] = []

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "base_model.model.layer.lora_A.default.weight": torch.ones(1, 2),
            "base_model.model.layer.lora_B.default.weight": torch.ones(2, 1),
        }

    def get_layers(self) -> list[object]:
        return [object(), object(), object(), object()]

    def get_abliterable_components(self) -> list[str]:
        return ["attn.o_proj", "mlp.down_proj"]

    def save_pretrained(self, save_directory: str, **kwargs: object) -> None:
        self.save_calls.append((save_directory, kwargs["max_shard_size"]))

    def reset_model(self) -> None:
        self.mutation_calls.append(("reset_model",))

    def abliterate(
        self,
        residual_directions: torch.Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None:
        self.mutation_calls.append(
            ("abliterate", residual_directions, direction_index, parameters)
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        self.calls.append((prompts, skip_special_tokens))
        return [f"response:{prompt.user}" for prompt in prompts]

    def get_logits_batched(self, prompts: list[Prompt]) -> torch.Tensor:
        self.logits_calls.append(prompts)
        return torch.tensor([[11.0]])

    def get_residuals_batched(self, prompts: list[Prompt]) -> torch.Tensor:
        self.residual_calls.append(prompts)
        return torch.tensor([[[22.0]]])

    def get_residuals_mean(self, prompts: list[Prompt]) -> torch.Tensor:
        self.residual_mean_calls.append(prompts)
        return torch.tensor([[33.0]])


class LocalModelRuntimeTests(unittest.TestCase):
    def test_reset_can_switch_model_on_the_rank_local_settings(self) -> None:
        model = RecordingModel()
        runtime = LocalModelRuntime(cast(Model, model))

        runtime.reset_model("org/evaluation-model")

        self.assertEqual(model.settings.model, "org/evaluation-model")
        self.assertEqual(model.mutation_calls, [("reset_model",)])

    def test_shutdown_is_idempotent_and_rejects_every_operation(self) -> None:
        model = RecordingModel()
        runtime = LocalModelRuntime(cast(Model, model))
        prompts = [Prompt(system="system", user="shutdown")]
        residual_directions = torch.tensor([[1.0]])

        runtime.shutdown()
        runtime.shutdown()

        self.assertIsNone(model.model)
        operations = [
            runtime.get_model_metadata,
            runtime.reset_model,
            lambda: runtime.abliterate(residual_directions, None, {}),
            lambda: runtime.save_adapter("adapter-output", max_shard_size="5GB"),
            lambda: runtime.get_responses(prompts),
            lambda: runtime.get_logits(prompts),
            lambda: runtime.get_residuals(prompts),
            lambda: runtime.get_residuals_mean(prompts),
        ]
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, "shut down"):
                    operation()

        self.assertEqual(model.mutation_calls, [])
        self.assertEqual(model.save_calls, [])
        self.assertEqual(model.calls, [])
        self.assertEqual(model.logits_calls, [])
        self.assertEqual(model.residual_calls, [])
        self.assertEqual(model.residual_mean_calls, [])

    def test_reports_declarative_local_model_metadata(self) -> None:
        runtime = LocalModelRuntime(cast(Model, RecordingModel()))

        metadata = runtime.get_model_metadata()

        self.assertEqual(
            metadata,
            ModelMetadata(
                runtime_kind="local",
                base_model_id="example/model",
                base_model_revision="abc123",
                dtype="torch.bfloat16",
                quantization="compressed-tensors",
                global_layer_count=4,
                abliterable_components=("attn.o_proj", "mlp.down_proj"),
                adapter_generation=None,
            ),
        )

    def test_reports_configured_quantization_when_model_has_no_embedded_metadata(
        self,
    ) -> None:
        model = RecordingModel()
        model.config.quantization_config = None
        model.settings.quantization.value = "bnb_4bit"
        runtime = LocalModelRuntime(cast(Model, model))

        metadata = runtime.get_model_metadata()

        self.assertEqual(metadata.quantization, "bnb_4bit")

    def test_delegates_responses_with_special_token_policy(self) -> None:
        model = RecordingModel()
        runtime = LocalModelRuntime(cast(Model, model))
        prompts = [
            Prompt(system="system", user="first"),
            Prompt(system="system", user="second"),
        ]

        responses = runtime.get_responses(prompts, skip_special_tokens=True)

        self.assertEqual(responses, ["response:first", "response:second"])
        self.assertEqual(model.calls, [(prompts, True)])

    def test_delegates_tensor_reads(self) -> None:
        model = RecordingModel()
        runtime = LocalModelRuntime(cast(Model, model))
        prompts = [Prompt(system="system", user="tensor")]

        logits = runtime.get_logits(prompts)
        residuals = runtime.get_residuals(prompts)

        self.assertEqual(logits.item(), 11.0)
        self.assertEqual(residuals.item(), 22.0)
        self.assertEqual(model.logits_calls, [prompts])
        self.assertEqual(model.residual_calls, [prompts])

    def test_delegates_residual_mean(self) -> None:
        model = RecordingModel()
        runtime = LocalModelRuntime(cast(Model, model))
        prompts = [Prompt(system="system", user="mean")]

        residual_mean = runtime.get_residuals_mean(prompts)

        self.assertEqual(residual_mean.item(), 33.0)
        self.assertEqual(model.residual_mean_calls, [prompts])

    def test_delegates_reset_then_abliteration_with_exact_arguments(self) -> None:
        model = RecordingModel()
        runtime = LocalModelRuntime(cast(Model, model))
        residual_directions = torch.tensor([[1.0], [2.0]])
        parameters = {
            "attn.o_proj": AbliterationParameters(
                max_weight=0.75,
                max_weight_position=1.5,
                min_weight=0.25,
                min_weight_distance=2.0,
            )
        }

        runtime.reset_model()
        runtime.abliterate(residual_directions, 0.5, parameters)

        self.assertEqual(model.mutation_calls[0], ("reset_model",))
        name, recorded_directions, recorded_index, recorded_parameters = (
            model.mutation_calls[1]
        )
        self.assertEqual(name, "abliterate")
        self.assertIs(recorded_directions, residual_directions)
        self.assertEqual(recorded_index, 0.5)
        self.assertIs(recorded_parameters, parameters)

    def test_propagates_mutation_exceptions_unchanged(self) -> None:
        reset_error = RuntimeError("reset failed")
        abliterate_error = ValueError("abliterate failed")

        class FailingModel(RecordingModel):
            def reset_model(self) -> None:
                raise reset_error

            def abliterate(
                self,
                residual_directions: torch.Tensor,
                direction_index: float | None,
                parameters: dict[str, AbliterationParameters],
            ) -> None:
                raise abliterate_error

        runtime = LocalModelRuntime(cast(Model, FailingModel()))
        residual_directions = torch.tensor([[1.0], [2.0]])

        with self.assertRaises(RuntimeError) as reset_context:
            runtime.reset_model()
        with self.assertRaises(ValueError) as abliterate_context:
            runtime.abliterate(residual_directions, None, {})

        self.assertIs(reset_context.exception, reset_error)
        self.assertIs(abliterate_context.exception, abliterate_error)

    def test_delegates_adapter_save_with_exact_arguments(self) -> None:
        model = RecordingModel()
        runtime = LocalModelRuntime(cast(Model, model))

        runtime.save_adapter("adapter-output", max_shard_size="5GB")

        self.assertEqual(model.save_calls, [("adapter-output", "5GB")])

    def test_adapter_save_gathers_only_bf16_lora_tensors(self) -> None:
        class CapturingAdapterModel(RecordingModel):
            def __init__(self) -> None:
                super().__init__()
                self.saved_kwargs: dict[str, object] = {}

            def state_dict(self) -> dict[str, torch.Tensor]:
                return {
                    "base_model.model.layer.base_layer.weight": torch.ones(
                        2, 2, dtype=torch.float8_e4m3fn
                    ),
                    "base_model.model.layer.lora_A.default.weight": torch.ones(
                        1, 2, dtype=torch.float32
                    ),
                    "base_model.model.layer.lora_B.default.weight": torch.ones(
                        2, 1, dtype=torch.float32
                    ),
                }

            def save_pretrained(self, directory: str, **kwargs: object) -> None:
                self.saved_kwargs = {"directory": directory, **kwargs}

        model = CapturingAdapterModel()
        runtime = LocalModelRuntime(cast(Model, model))

        runtime.save_adapter("adapter-output", max_shard_size="5GB")

        state = cast(dict[str, torch.Tensor], model.saved_kwargs["state_dict"])
        self.assertEqual(
            set(state),
            {
                "base_model.model.layer.lora_A.default.weight",
                "base_model.model.layer.lora_B.default.weight",
            },
        )
        self.assertEqual({tensor.dtype for tensor in state.values()}, {torch.bfloat16})
        self.assertEqual(model.saved_kwargs["selected_adapters"], ["default"])
        self.assertIs(model.saved_kwargs["safe_serialization"], True)
        self.assertIs(model.saved_kwargs["save_embedding_layers"], False)

    def test_distributed_worker_uses_disposable_sink_for_collective_save(self) -> None:
        class DistributedRecordingModel(RecordingModel):
            distributed = True

            def save_pretrained(
                self,
                save_directory: str,
                **kwargs: object,
            ) -> None:
                self.save_calls.append(
                    (
                        save_directory,
                        kwargs["max_shard_size"],
                        kwargs["is_main_process"],
                    )
                )

        model = DistributedRecordingModel()
        runtime = LocalModelRuntime(cast(Model, model))
        removed: list[str] = []
        with (
            patch("torch.distributed.get_rank", return_value=1),
            patch("tempfile.mkdtemp", return_value="worker-adapter-sink"),
            patch("shutil.rmtree", side_effect=lambda path: removed.append(path)),
        ):
            runtime.save_adapter("coordinator-output", max_shard_size="5GB")

        self.assertEqual(
            model.save_calls,
            [("worker-adapter-sink", "5GB", False)],
        )
        self.assertEqual(removed, ["worker-adapter-sink"])

    def test_distributed_worker_preserves_export_error_if_sink_cleanup_fails(self) -> None:
        primary_error = RuntimeError("PEFT gather failed")

        class FailingDistributedModel(RecordingModel):
            distributed = True

            def save_pretrained(self, *args, **kwargs) -> None:
                raise primary_error

        runtime = LocalModelRuntime(cast(Model, FailingDistributedModel()))
        with (
            patch("torch.distributed.get_rank", return_value=1),
            patch("tempfile.mkdtemp", return_value="worker-adapter-sink"),
            patch("shutil.rmtree", side_effect=OSError("cleanup failed")),
        ):
            with self.assertRaises(RuntimeError) as context:
                runtime.save_adapter("coordinator-output", max_shard_size="5GB")

        self.assertIs(context.exception, primary_error)


class RecordingRuntime(ModelRuntime):
    def __init__(self) -> None:
        self.calls: list[tuple[list[Prompt], bool]] = []
        self.logits_calls: list[list[Prompt]] = []
        self.residual_calls: list[list[Prompt]] = []

    def get_model_metadata(self) -> ModelMetadata:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError

    def reset_model(self, model: str | None = None) -> None:
        raise NotImplementedError

    def abliterate(
        self,
        residual_directions: torch.Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None:
        raise NotImplementedError

    def save_adapter(
        self, directory: str, *, max_shard_size: int | str
    ) -> None:
        raise NotImplementedError

    def get_responses(
        self,
        prompts: list[Prompt],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        self.calls.append((prompts, skip_special_tokens))
        return [f"runtime:{prompt.user}" for prompt in prompts]

    def get_logits(self, prompts: list[Prompt]) -> torch.Tensor:
        self.logits_calls.append(prompts)
        return torch.tensor([[float(len(self.logits_calls))]])

    def get_residuals(self, prompts: list[Prompt]) -> torch.Tensor:
        self.residual_calls.append(prompts)
        return torch.tensor([[[float(len(self.residual_calls))]]])

    def get_residuals_mean(self, prompts: list[Prompt]) -> torch.Tensor:
        return torch.tensor([[0.0]])


class ContextRuntimeTests(unittest.TestCase):
    def test_accepts_stock_model_constructor_and_preserves_model_attribute(self) -> None:
        model = RecordingModel()
        settings = cast(Settings, object())
        context = Context(settings=settings, model=cast(Model, model))
        prompts = [Prompt(system="system", user="stock-context")]

        responses = context.get_responses(prompts)

        self.assertIs(context._model, model)
        self.assertEqual(responses, ["response:stock-context"])
        self.assertEqual(model.calls, [(prompts, True)])

    def test_caches_runtime_responses_within_context(self) -> None:
        runtime = RecordingRuntime()
        context = Context(
            settings=cast(Settings, object()),
            runtime=runtime,
        )
        prompts = [Prompt(system="system", user="question")]

        first = context.get_responses(prompts)
        second = context.get_responses(prompts)

        self.assertEqual(first, ["runtime:question"])
        self.assertIs(first, second)
        self.assertEqual(runtime.calls, [(prompts, True)])

    def test_delegates_logits_without_caching(self) -> None:
        runtime = RecordingRuntime()
        context = Context(
            settings=cast(Settings, object()),
            runtime=runtime,
        )
        prompts = [Prompt(system="system", user="question")]

        first = context.get_logits(prompts)
        second = context.get_logits(prompts)

        self.assertEqual(first.item(), 1.0)
        self.assertEqual(second.item(), 2.0)
        self.assertEqual(runtime.logits_calls, [prompts, prompts])

    def test_delegates_residuals_without_caching(self) -> None:
        runtime = RecordingRuntime()
        context = Context(
            settings=cast(Settings, object()),
            runtime=runtime,
        )
        prompts = [Prompt(system="system", user="question")]

        first = context.get_residuals(prompts)
        second = context.get_residuals(prompts)

        self.assertEqual(first.item(), 1.0)
        self.assertEqual(second.item(), 2.0)
        self.assertEqual(runtime.residual_calls, [prompts, prompts])


class RecordingScorer:
    def __init__(self) -> None:
        self.contexts: list[Context] = []

    def get_score(self, context: Context) -> Score:
        self.contexts.append(context)
        responses = context.get_responses(
            [Prompt(system="system", user="evaluator")]
        )
        return Score(value=1.0, rich_display=responses[0], md_display=responses[0])


class ConstructorRecordingScorer(Scorer):
    init_context: Context
    baseline_context: Context

    def init(self, ctx: Context) -> None:
        self.init_context = ctx
        ctx.get_responses([Prompt(system="system", user="initialize")])

    def get_score(self, ctx: Context) -> Score:
        self.baseline_context = ctx
        response = ctx.get_responses(
            [Prompt(system="system", user="constructor-baseline")]
        )[0]
        return Score(value=2.0, rich_display=response, md_display=response)


class EvaluatorRuntimeTests(unittest.TestCase):
    def test_accepts_stock_positional_model_constructors(self) -> None:
        model = RecordingModel()
        settings = Settings.model_construct(
            model="unused",
            scorers=[ScorerConfig(plugin="test.Recording", optimization="none")],
        )
        context = Context(settings, cast(Model, model))

        with (
            patch(
                "heretic.evaluator.load_plugin",
                return_value=ConstructorRecordingScorer,
            ),
            patch("heretic.evaluator.print"),
        ):
            evaluator = Evaluator(settings, cast(Model, model))

        self.assertIs(context._model, model)
        self.assertIs(evaluator.model, model)

    def test_accepts_stock_model_constructor_and_preserves_model_attribute(self) -> None:
        model = RecordingModel()
        settings = Settings.model_construct(
            model="unused",
            scorers=[ScorerConfig(plugin="test.Recording", optimization="none")],
        )

        with (
            patch(
                "heretic.evaluator.load_plugin",
                return_value=ConstructorRecordingScorer,
            ),
            patch("heretic.evaluator.print"),
        ):
            evaluator = Evaluator(settings=settings, model=cast(Model, model))

        self.assertIs(evaluator.model, model)
        self.assertEqual(
            model.calls,
            [
                ([Prompt(system="system", user="initialize")], True),
                ([Prompt(system="system", user="constructor-baseline")], True),
            ],
        )

    def test_constructor_initializes_and_scores_baseline_through_runtime(self) -> None:
        runtime = RecordingRuntime()
        settings = Settings.model_construct(
            model="unused",
            scorers=[ScorerConfig(plugin="test.Recording", optimization="none")],
        )

        with (
            patch(
                "heretic.evaluator.load_plugin",
                return_value=ConstructorRecordingScorer,
            ),
            patch("heretic.evaluator.print"),
        ):
            evaluator = Evaluator(settings=settings, runtime=runtime)

        scorer = cast(ConstructorRecordingScorer, evaluator._scorer_entries[0].scorer)
        self.assertIsNot(scorer.init_context, scorer.baseline_context)
        self.assertEqual(
            runtime.calls,
            [
                ([Prompt(system="system", user="initialize")], True),
                ([Prompt(system="system", user="constructor-baseline")], True),
            ],
        )
        self.assertEqual(
            evaluator.baseline_scores,
            [
                (
                    "ConstructorRecordingScorer",
                    Score(
                        value=2.0,
                        rich_display="runtime:constructor-baseline",
                        md_display="runtime:constructor-baseline",
                    ),
                )
            ],
        )

    def test_get_scores_uses_runtime(self) -> None:
        runtime = RecordingRuntime()
        scorer = RecordingScorer()
        evaluator = object.__new__(Evaluator)
        evaluator.settings = cast(Settings, object())
        evaluator.runtime = runtime
        evaluator._scorer_entries = [
            ScorerEntry(
                scorer=cast(Scorer, scorer),
                name="recording",
                config=cast(ScorerConfig, object()),
            )
        ]

        scores = evaluator.get_scores()

        self.assertEqual(scores[0][1].rich_display, "runtime:evaluator")
        self.assertEqual(len(scorer.contexts), 1)
        self.assertEqual(len(runtime.calls), 1)


if __name__ == "__main__":
    unittest.main()


def test_runtime_types_declare_distributed_identity() -> None:
    from heretic.dgx_runtime import DgxCoordinatorRuntime
    from heretic.runtime import LocalModelRuntime

    assert LocalModelRuntime.distributed is False
    assert DgxCoordinatorRuntime.distributed is True
