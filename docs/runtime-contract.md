# Model runtime compatibility contract

## Purpose

Heretic's model execution must be replaceable without changing scorer semantics or the default single-process behavior. The implementation preserves the existing `Model` class as the local execution engine behind `LocalModelRuntime`; the two-DGX coordinator runtime must satisfy the same observable contract for every operation it qualifies.

This contract is intentionally narrower than the complete `Model` object. Runtime-native plugin code uses the operation surface rather than concrete model internals. For stock local compatibility, the legacy private `Context._model` reference remains available and points to the same concrete `Model`; distributed runtimes do not synthesize a local model.

## Observable runtime surface

The implemented surface now includes scorer reads plus the mutation operations used by local trial orchestration:

```python
class ModelRuntime(ABC):
    @abstractmethod
    def get_model_metadata(self) -> ModelMetadata: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def reset_model(self) -> None: ...

    @abstractmethod
    def abliterate(
        self,
        residual_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None: ...

    @abstractmethod
    def save_adapter(
        self, directory: str, *, max_shard_size: int | str
    ) -> None: ...

    @abstractmethod
    def get_responses(
        self, prompts: list[Prompt], *, skip_special_tokens: bool = True
    ) -> list[str]: ...

    @abstractmethod
    def get_logits(self, prompts: list[Prompt]) -> Tensor: ...

    @abstractmethod
    def get_residuals(self, prompts: list[Prompt]) -> Tensor: ...

    @abstractmethod
    def get_residuals_mean(self, prompts: list[Prompt]) -> Tensor: ...
```

`ModelMetadata` must describe at least runtime kind, base model identifier and revision, dtype/quantization, global layer count, abliterable components, and adapter generation. Metadata is declarative; callers must not use it to reach implementation objects.

## Two-rank load identity gate

The first coordinator/worker protocol slice defines an immutable `LOAD` command
from rank 0 and an acknowledgement from rank 1. The acknowledgement is accepted
only when all of the following match exactly:

- command ID, represented as an exact nonnegative integer (not a boolean or float);
- expected worker rank (`1`), also represented as an exact nonnegative integer;
- nonempty base model identifier;
- nonempty pinned base revision represented by a full 40-hex or 64-hex commit
  hash (branch names and abbreviated revisions are rejected);
- nonempty loaded dtype; and
- the canonical JSON serialization of the complete quantization configuration.

At the acceptance boundary the validator requires exact `LoadCommand`,
`LoadAcknowledgement`, and `ModelLoadIdentity` record types and reconstructs both
identities to rerun every invariant. Constructor-time validation and frozen
records are not trusted as the sole gate, so duck-typed records and forcibly
mutated records that violate any current invariant fail closed. This gate proves
current values, not provenance or the absence of post-construction mutation.

A complete quantization configuration at this protocol boundary contains a
nonempty `quant_method` plus at least one nonmetadata field whose value is
recursively nonempty. `version` is metadata and does not satisfy this requirement;
nulls and empty strings, arrays, or objects do not satisfy it either. Quantization
identity is not reduced to a method name or a selected subset of fields. The
protocol does not reinterpret backend-specific fields: the runtime loader must
supply the entire checkpoint `quantization_config` object. Arbitrary nested
mapping implementations are materialized recursively without dropping members
before canonical serialization. Both ranks must report the same full canonical
configuration before the two-node logical model may start. The same
representation is the required export target: a distributed run must not
silently dequantize, requantize to another scheme, or emit a different
quantization configuration.

The first transport tracer performs one real two-process TCP exchange on the development host.
It requires IP-literal loopback endpoints while the channel is unauthenticated.
Messages use bounded length-prefixed canonical JSON, reject unknown, duplicate,
noncanonical, truncated, empty, and oversized input, and apply one monotonic
deadline to connection acceptance, worker loading, and the exchange. On a loader
deadline the function raises; its dedicated one-shot worker-process caller must
terminate, and no retry is permitted. Rank 0 does not open the logical model gate
until the decoded acknowledgement passes the complete identity validator.

The checkpoint planner reads `model.safetensors.index.json` without modifying the
checkpoint and assigns every referenced whole safetensors shard and indexed tensor
to exactly one of two nonempty ranks. Each shard's size and SHA-256 are measured
from one verified regular-file handle; both values are bound into the deterministic
plan digest. Assignment greedily balances observed shard file bytes, and both ranks
can compare the plan digest before loading. Replacement around file opening and
descriptor metadata changes during hashing fail closed. The digest identifies the
opened-byte snapshot; it does not freeze a mutable filesystem path after hashing,
so each future rank-local loader must revalidate the planned SHA-256 against the
bytes it actually loads. Unsafe paths, missing or empty shards, duplicate index
keys, malformed maps, and plans that cannot populate both ranks fail closed. This planner and the LOAD transport are retained as historical precursor evidence;
they are not the current DGX model-execution path.

The canonical development-host tiny-model proof runs a deterministic four-layer Qwen2 model
through two real Gloo processes: rank 0 executes layers 0–1, rank 1 executes layers
2–3 plus normalization and the output head, and the boundary hidden state and final
logits cross the process-group transport. The proof requires distributed logits to
be byte-exact with monolithic logits. Both processes instantiate the complete tiny
model solely to obtain deterministic identical weights; this does not prove
memory-scaled sharded checkpoint loading, Heretic mutation, or
quantization-preserving export.

## Current two-DGX runtime status

The current DGX path launches exactly two Linux ranks, synchronizes finalized
settings before model construction, initializes one two-rank NCCL process group,
and loads compatible models through Transformers tensor parallelism with
`tp_plan="auto"`. Rank 0 coordinates ordered collective commands; rank 1 executes
the same model operations and reports completion or failure before rank 0 proceeds.
The natural invocation accepts an explicit configuration path:

```text
heretic --config config.toml --cluster dgx-cluster.toml
```

When `--config` is repeated, the last path wins consistently for both the
reported setting and the TOML contents. Missing and empty paths fail closed;
dash-prefixed relative paths use the unambiguous `--config=<path>` form.

The bounded startup check has been exercised on two physical DGX Spark nodes with
a real text forward and clean shutdown.

This qualification does not claim every checkpoint is compatible. TP topology is
accepted only when the model-level plan agrees with the target weight's actual
two-rank DTensor mesh and placement. Pipeline-only and contradictory layouts fail
closed. Full Laguna loading, Heretic mutation, temporary adapter reconstruction,
and standalone quantization-preserving export require their own real two-rank
evidence before they are claimed.

## Local behavior mapping

`LocalModelRuntime` is a behavior-preserving adapter around the existing `heretic.model.Model`:

| Runtime operation | Existing local behavior |
|---|---|
| `get_model_metadata` | Immutable declarative projection of settings, loaded dtype/quantization, layer/component discovery, and no local adapter-generation bookkeeping |
| `get_responses` | `Model.get_responses_batched`, with the caller's `skip_special_tokens` value |
| `get_logits` | `Model.get_logits_batched` |
| `get_residuals` | `Model.get_residuals_batched` |
| `get_residuals_mean` | `Model.get_residuals_mean` |
| `reset_model` | `Model.reset_model` |
| `abliterate` | `Model.abliterate`, with the exact tensor, direction index, and parameter mapping |
| `save_adapter` | Current PEFT adapter `save_pretrained` path |
| `shutdown` | Idempotent local cleanup; it must not change outputs before shutdown |

Scorer reads and trial/restore reset-and-abliterate calls use the runtime. A separate
evaluation model, chat, upload, and benchmark paths fail closed in DGX mode until
separately qualified. Distributed completion resolves only to standalone export;
adapter-only completion and the legacy full-merge path fail closed. Local behavior
continues to use the existing `Model` implementation.

## Prompt and batching semantics

1. Prompt order is preserved.
2. Empty-prompt behavior is preserved per operation. In particular, `get_residuals_mean([])` raises `ValueError`.
3. The runtime owns batching. Scorers pass a complete prompt list and do not depend on local batch boundaries.
4. Local response generation remains greedy and deterministic as currently configured.
5. The scorer-facing response path strips special tokens by default, matching the existing plugin `Context` behavior.

## Tensor contracts

### Logits

- Shape: `(prompt, vocabulary)`.
- Values are raw first-generated-token logits, not processed generation scores.
- Prompt order is preserved.
- Output placement follows runtime policy; the existing local `offload_outputs_to_cpu` behavior is preserved.

### Residuals

- Shape: `(prompt, global_layer_position, hidden_component)`.
- Position zero is the embedding residual; subsequent positions follow global transformer-layer order.
- Returned values are `float32` after existing winsorization semantics.
- `get_residuals_mean` accumulates per-batch sums in `float64` on CPU and returns `float32`, matching the current local implementation.

The distributed runtime may use rank-local accumulation internally, but its returned global order, count weighting, shape, and dtype must match this contract within an explicitly recorded tolerance.

## Adapter state contract

1. The identity state is all rank-local LoRA-B weights zeroed, matching the local reset path.
2. Reset and apply are ordered collective commands; success requires both ranks to acknowledge completion.
3. Failure or timeout makes the distributed runtime unusable and triggers shutdown rather than continuing with divergent rank state.
4. Distributed ranks may reconstruct and save one complete PEFT adapter bound to the exact base repository/revision and containing no base-model tensors, but only as a temporary handoff to standalone materialization. A real two-rank DTensor/PEFT proof remains required before this operation is runtime-qualified.
5. Adapter-only completion is forbidden. The temporary adapter is removed after successful standalone export and is never the release artifact.

## Standalone Laguna export contract

The v25 standalone path is deliberately model-specific and bounded. It must:

1. require repository `poolside/Laguna-S-2.1-FP8` at full revision `06d71e91db70a11b08ee6a09c3c4818c85a61953`;
2. verify exact `LagunaForCausalLM`/`model_type=laguna`, exactly 48 layers, and the expected FP8 dynamic-activation 128×128 block configuration;
3. require each of the 48 `model.layers.N.self_attn.o_proj.weight` targets to be a BF16 protected/ignored layer;
4. reject active non-target adapter deltas, non-finite factors, `fan_in_fan_out`, DoRA, and rsLoRA;
5. use bounded CPU FP32 LoRA delta arithmetic followed by BF16 cast/add only for the permitted target tensor;
6. copy all FP8 tensors, `.weight_scale_inv` tensors, metadata, and all bytes outside permitted target ranges unchanged;
7. reject source symlinks and unsupported source file forms and exclude Hugging Face `.cache/` bookkeeping;
8. emit and verify a complete SHA-256 manifest; and
9. contain no adapter dependency or LoRA parameter keys and load directly for deterministic two-rank inference.

This path never constructs a full BF16 Laguna base and never re-encodes FP8. Unit and fixture tests qualify the software boundary only. They do not substitute for real two-rank collective adapter reconstruction, final-directory loading, or inference.

## Errors and lifecycle

- Runtime exceptions propagate with operation identity and original cause.
- `shutdown()` is idempotent.
- No call is accepted after successful shutdown.
- A worker exit, missed deadline, stale command ID, stale adapter generation, malformed response, or module-inventory mismatch is fatal to the distributed logical run.
- The local runtime must not start network listeners or distributed workers.

## Behavior-preservation oracle

Pinned upstream behavior at `bedb94ef117a271532ac2058447fbc165d5051bd` is the acceptance oracle for the default local runtime.

1. Tests characterize intended behavior; they do not define permission to change it. When a test conflicts with intended behavior, the test is corrected or rejected.
2. `LocalModelRuntime` preserves observable local semantics: the same concrete `Model`, stock `Evaluator(settings, model)` and `Context(settings, model)` constructors, legacy local `.model`/`._model` references, call order, arguments, prompt order, cache scope, tensor values, exceptions, mutation sequence, randomness, export path, and CLI defaults.
3. Production code remains independent of test identity, CI state, caller inspection, mocks, or environment switches that exist only to satisfy verification.
4. A blocker remains visible as a failed gate or explicit limitation. Compatibility is earned by exercising the real path, not by adding a fallback that avoids it.
5. An intentional user-visible behavior change requires its own explicit scope, acceptance criteria, migration note, and independent review; it is never bundled into a runtime-boundary refactor.
6. Each runtime slice is reviewed against the complete production diff from pinned upstream and, when applicable, a real model-producing fixture with accepted output hashes or explicit numerical tolerances.

## Compatibility gates

A runtime-abstraction change is compatible only when all applicable gates pass:

1. Existing offline unit tests pass.
2. Ruff and `ty` pass.
3. The real `heretic --help` entrypoint remains functional.
4. A fake runtime proves that plugin `Context` delegates responses, logits, and residuals without requiring a Transformers model or GPU.
5. Stock positional and `model=` constructors for `Evaluator` and `Context` accept the same concrete local `Model`, preserve `Evaluator.model` and `Context._model`, and route scorer reads through a local runtime adapter.
6. Scorer initialization and constructor-time baseline scoring each receive a fresh `Context` backed by the same injected runtime; neither path may reach a concrete `Model` directly when a runtime is supplied.
7. Response caching remains scoped to one `Context` and keyed by ordered `(system, user)` prompt pairs.
8. The local adapter delegates exactly once per response cache miss and never caches logits or residuals.
9. Reset and abliteration delegate one-for-one, preserve operation order and exact argument identity/value, and propagate the original exception object unchanged.
10. The real tiny-model workflow exercises trial reset, apply, evaluation, selected-trial restoration, merge, and export, and its six-file output matches an accepted hash set with no missing or extra files.
11. The model-producing reproducibility suite remains a separately reported GPU/network gate; it is not silently represented as passed when not run.
12. Before a distributed operation is called compatible, a real two-rank model must exercise that exact operation and validate its outputs. Startup/text-forward qualification does not qualify mutation or adapter export.

## Current DGX scope and limitations

The DGX runtime is intentionally limited to exactly two Linux nodes and compatible
Transformers tensor-parallel models. It does not claim arbitrary-model support,
pipeline fallback, a separate evaluation model, interactive chat, LM Evaluation
Harness, Hub upload, adapter-only completion, or generic merged export. These
operations fail closed rather than silently running on one rank or emitting
incomplete output. Laguna FP8 loading, attention-only targeting, the smallest
Heretic mutation, real collective adapter reconstruction, standalone materialization,
and final standalone inference remain gated on their operation-specific two-rank
proofs.

## v25 software breakpoint evidence

After correction of the independent-review blockers, the exact DGX/Linux worktree
passed 209 tests plus 57 subtests, Ruff, and `compileall`. This is a software
breakpoint, not release evidence. The worktree is not a Git repository, has not been
sanitized or packaged, and has not produced or loaded a standalone Laguna model.
