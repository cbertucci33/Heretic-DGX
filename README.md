# Heretic DGX

Heretic DGX is a two-node NVIDIA DGX Spark implementation of
[`p-e-w/heretic`](https://github.com/p-e-w/heretic). It runs Heretic's
directional-ablation optimization across exactly two DGX Spark systems and
retains upstream Heretic's adapter and merged-model export workflows.

## Relationship to the original Heretic project

This repository is an independent downstream fork of Philipp Emanuel
Weidmann's original
[`p-e-w/heretic`](https://github.com/p-e-w/heretic), based on upstream commit
[`bedb94e`](https://github.com/p-e-w/heretic/commit/bedb94ef117a271532ac2058447fbc165d5051bd).
Heretic's abliteration method, scorer model, optimization approach, and core
configuration remain upstream work.

Heretic DGX adds the distributed execution layer needed to run that workflow
across two DGX Spark systems:

- coordinator-driven launch and rank supervision;
- source, checkpoint, topology, and collective preflight checks;
- mirrored prompt, residual, scoring, and optimization operations;
- coordinated cancellation, failure reporting, and teardown; and
- coordinated PEFT adapter and merged Transformers checkpoint export.

The Laguna FP8 path adds a model-specific standalone exporter. It verifies
target changes while preserving the source checkpoint's quantized and
non-target artifacts. This path does not replace the general upstream export
workflows.

For the original single-system project, documentation, and community, use the
[upstream Heretic repository](https://github.com/p-e-w/heretic). Issues specific
to the two-DGX implementation belong in this repository.

## Releases

### 0.1.1

Release 0.1.1 fixes distributed export behavior. Release 0.1.0 incorrectly
limited distributed runs to the Laguna-specific `standalone` strategy. The
fix restores both upstream export choices:

- `adapter` writes the PEFT LoRA adapter;
- `merge` writes the full merged Transformers checkpoint, tokenizer, and
  multimodal processor files when present.

Direct Hub uploads now stage the selected distributed artifact before upload,
which lets both tensor-parallel ranks participate in tensor gathering. The
Laguna `standalone` exporter remains available as a separate, model-specific
option. Regression tests cover `adapter`, `merge`, and `standalone` selection.

### 0.1.0

- One coordinator command launches one GPU-backed rank on each of two nodes.
- Both ranks load the model through Transformers tensor parallelism.
- Preflight checks verify node reachability, source identity, checkpoint
  identity, topology, and collective communication before optimization.
- Prompt ingestion, residual calculation, scoring, optimization, winner
  restoration, and model materialization are coordinated across both ranks.
- Distributed runs offer the same PEFT adapter and merged-model artifacts as
  upstream Heretic. Direct Hub uploads stage the coordinated artifact before
  upload so both ranks participate in tensor gathering.
- Failure, cancellation, timeout, and teardown behavior is bounded so a failed
  peer does not leave the other rank running indefinitely.
- The optional Laguna standalone exporter preserves non-target files and
  quantized tensors and verifies intended tensor changes before reporting
  success.

This release supports **Linux, exactly two DGX Spark nodes, NCCL, and one rank
per node**. It is not a general multi-node backend.

## Validated model

Release 0.1.0 was proven end to end with
[`poolside/Laguna-S-2.1-FP8`](https://huggingface.co/poolside/Laguna-S-2.1-FP8).
The resulting standalone checkpoint is available as
[`cbert33/Laguna-S-2.1-Heretic-FP8`](https://huggingface.co/cbert33/Laguna-S-2.1-Heretic-FP8).

The selected trial changed only the intended BF16 attention-output projection
targets in layers 29-47 while preserving the source FP8 tensors and all
non-target artifacts. It measured a KL divergence of `0.0156` from the
untouched model's first-token probability distributions across five prompts
from `mlabonne/harmless_alpaca`.

## Requirements

- Two DGX Spark systems running Linux
- CUDA, NCCL, and working node-to-node GPU collective communication
- Key-based SSH from the coordinator to both nodes
- The same clean Heretic DGX revision and model checkpoint on both nodes
- Python 3.10 or newer and [`uv`](https://docs.astral.sh/uv/)

Use a dedicated high-speed fabric for rank traffic and keep a separate
management path for SSH and recovery.

## Install

Run on both nodes at the same absolute path:

```sh
git clone https://github.com/cbertucci33/Heretic-DGX.git
cd Heretic-DGX
git checkout v0.1.1
uv sync --frozen
```

## Configure the cluster

Copy the example outside the repository and replace every placeholder:

```sh
cp cluster.example.toml ../heretic-cluster.toml
```

The two entries in `[[nodes]]` are ordered by rank: the coordinator is rank 0
and the worker is rank 1. `host` is the SSH destination; `rank_address` is the
address used for distributed traffic. `nccl_socket_ifname` must name the fabric
interface present on both systems.

Do not commit live hostnames, addresses, credentials, or private cluster
configuration.

## Run

From the coordinator:

```sh
uv run heretic \
  --cluster ../heretic-cluster.toml \
  --config ./config.default.toml \
  --model /path/to/model
```

The coordinator rejects mismatched source trees or checkpoint payloads before
launching the optimization. Output behavior and target selection are controlled
by the Heretic configuration file.

## Verification

For a source checkout:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Before using another model family or quantization, verify a full load,
optimization, selected export type, clean reload, and generation test.

## Export behavior

Heretic DGX supports the two artifact types offered by upstream Heretic:

- `adapter`: a PEFT LoRA adapter that can be merged later;
- `merge`: a full merged Transformers checkpoint, including tokenizer and
  multimodal processor files when present.

Both paths coordinate all tensor-parallel ranks before rank 0 writes the
artifact. Their checkpoint behavior follows the pinned upstream Heretic and
Transformers versions. In particular, upstream's 4-bit load mode exports a
full-precision merged model after reloading the base checkpoint on CPU.

The third strategy, `standalone`, is specific to the validated Laguna FP8
checkpoint. It preserves Laguna's native quantized tensors and patches only
the protected BF16 targets. Do not select it for another model family.

## Safety and liability

Heretic changes model refusal behavior. It does not guarantee correctness,
capability retention, safety, legality, or suitability for a particular use.
KL divergence and automated checks are limited indicators, not substitutes for
broad evaluation. Review the source model's license and usage restrictions
before creating or distributing a derivative.

**User responsibility:** this software is provided without warranty. The
creators, uploaders, and maintainers are not responsible or liable for what
others generate, publish, deploy, or otherwise do with any abliterated models
made by this. Users must operate it responsibly, apply appropriate safeguards,
comply with applicable law, and respect third-party rights. This software is
for research purposes only and is not intended for production use.

## Attribution and license

Heretic DGX retains the original project's AGPL-3.0-or-later license and
copyright notices. The two-node distributed implementation and release-specific
changes are maintained in this repository. Heretic DGX is not presented as an
official upstream release.
