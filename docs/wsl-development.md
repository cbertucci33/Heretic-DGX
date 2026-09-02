# WSL-first Heretic development

> **Historical only — not the accepted v25 path.** This note records an earlier WSL development environment. Distributed Heretic v25 source edits, tests, packaging, review, models, and runtime qualification are DGX/Linux-only. Windows is limited to Hermes SSH transport and note storage; WSL results do not qualify or produce the v25 candidate.

The earlier workflow built Linux, GPU, backend, and DGX-bound Heretic changes in WSL2 first. It reduced Windows/Linux packaging drift but never qualified the DGX architecture and is superseded for v25 by the exact DGX/Linux execution record.

## Storage contract

- Canonical durable repository: the native path configured as `HERETIC_REPOSITORY`
- Disposable WSL execution mirror: `$HOME/work/heretic-distributed`
- Durable reports, manifests, checksums, patches, and promoted artifacts remain under `HERETIC_REPOSITORY`.
- WSL virtual-disk contents are build state, caches, environments, or reproducible mirrors; they are not the only copy of a durable artifact.

## Verified WSL baseline

See `artifacts/baseline/wsl-cuda-environment-report.json` for the machine-readable record. The initial verified environment is:

```text
WSL2 Ubuntu 24.04.4 LTS x86_64
Python 3.12.3
uv 0.11.11
PyTorch 2.13.0+cu130
CUDA 13.0
NVIDIA GeForce RTX 5080, compute capability 12.0
Windows NVIDIA driver 610.88
```

## Bootstrap a clean execution mirror

Run from WSL:

```bash
mkdir -p "$HOME/work"
git clone --no-hardlinks \
  --branch feature/distributed-runtime \
  "$HERETIC_REPOSITORY" \
  "$HOME/work/heretic-distributed"
cd "$HOME/work/heretic-distributed"
uv sync --frozen --group dev --python 3.12
bash scripts/verify-wsl.sh
```

If the mirror already exists, first require a clean tree, then fast-forward it:

```bash
cd "$HOME/work/heretic-distributed"
test -z "$(git status --porcelain)"
git pull --ff-only
uv sync --frozen --group dev --python 3.12
bash scripts/verify-wsl.sh
```

Do not use `git reset --hard` as a routine synchronization command. A dirty mirror is evidence to inspect, not state to destroy automatically.

## Verification script

`scripts/verify-wsl.sh` fails closed unless all of the following pass:

1. The command is running under WSL.
2. The frozen environment executables exist.
3. CUDA-enabled PyTorch sees at least one GPU.
4. A real CUDA tensor operation returns the expected value.
5. All unit tests pass.
6. Ruff and `ty` pass.
7. Python compilation and installed CLI help pass.
8. The sdist and wheel build successfully.
9. The wheel contains `heretic/runtime.py`.
10. Disposable package output is removed on exit.

The script assumes `uv sync --frozen --group dev --python 3.12` has already succeeded. It deliberately does not mutate dependency resolution itself.

## What WSL proves

WSL validation covers Linux process/filesystem behavior, Linux Python dependencies, CUDA-enabled PyTorch packaging, RTX 5080 execution, CLI behavior, tests, and package construction.

It does **not** prove:

- DGX ARM64 wheel or extension availability;
- GB10 unified-memory behavior;
- Laguna FP8 loading or kernels;
- native DGX container compatibility;
- one-rank capacity;
- two-rank NCCL transport or global layer ordering; or
- production deployment readiness.

Promotion remains tiered:

```text
WSL CPU and CUDA gates
→ ARM64 artifact validation
→ one-DGX native qualification
→ two-DGX distributed qualification
→ explicitly approved production promotion
```
