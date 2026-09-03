# Heretic DGX

Heretic DGX is a two-node NVIDIA DGX Spark implementation of
[`p-e-w/heretic`](https://github.com/p-e-w/heretic). It runs Heretic's
directional-ablation optimization across exactly two DGX Spark systems and
exports a standalone checkpoint.

## Release 0.1 scope

- One coordinator command launches one GPU-backed rank on each of two nodes.
- Both ranks load the model through Transformers tensor parallelism.
- Preflight checks verify node reachability, source identity, checkpoint
  identity, topology, and collective communication before optimization.
- Prompt ingestion, residual calculation, scoring, optimization, winner
  restoration, and model materialization are coordinated across both ranks.
- Failure, cancellation, timeout, and teardown behavior is bounded so a failed
  peer does not leave the other rank running indefinitely.
- The standalone exporter preserves non-target files and quantized tensors and
  verifies intended tensor changes before reporting success.

This release is intentionally narrow: **Linux, exactly two DGX Spark nodes,
NCCL, and one rank per node**. It is not a general multi-node backend.

## Validated model

Release 0.1 was proven end to end with
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
git checkout v0.1.0
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

Model-family and quantization support must be proven independently with a full
load, optimization, standalone export, clean reload, and generation test.

## Safety and limitations

Heretic changes model refusal behavior. It does not guarantee correctness,
capability retention, safety, legality, or suitability for a particular use.
KL divergence and automated checks are limited indicators, not substitutes for
broad evaluation. Review the source model's license and usage restrictions
before creating or distributing a derivative.

## Upstream and license

Heretic DGX is based on upstream Heretic commit
[`bedb94e`](https://github.com/p-e-w/heretic/commit/bedb94ef117a271532ac2058447fbc165d5051bd)
and retains its AGPL-3.0-or-later license. The distributed implementation and
release-specific changes are maintained in this repository.
