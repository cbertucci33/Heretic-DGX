#!/usr/bin/env bash
set -euo pipefail

if [[ ! -r /proc/sys/kernel/osrelease ]] || ! grep -Fqi 'microsoft-standard-WSL2' /proc/sys/kernel/osrelease; then
    printf 'error: this verifier must run inside WSL2\n' >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

python_bin="$repo_root/.venv/bin/python"
ruff_bin="$repo_root/.venv/bin/ruff"
ty_bin="$repo_root/.venv/bin/ty"
heretic_bin="$repo_root/.venv/bin/heretic"

for executable in "$python_bin" "$ruff_bin" "$ty_bin" "$heretic_bin"; do
    if [[ ! -x "$executable" ]]; then
        printf 'error: missing executable: %s\n' "$executable" >&2
        printf 'run: uv sync --frozen --group dev --python 3.12\n' >&2
        exit 2
    fi
done

"$python_bin" - <<'PY'
import json
import torch

result = {
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
}
if not result["cuda_available"]:
    raise SystemExit(f"CUDA is not available: {result}")
if result["cuda_device_count"] < 1:
    raise SystemExit(f"no CUDA devices detected: {result}")
result["cuda_device"] = torch.cuda.get_device_name(0)
result["cuda_compute_capability"] = torch.cuda.get_device_capability(0)
values = torch.arange(8, dtype=torch.float32, device="cuda")
observed = (values * 2).sum()
torch.cuda.synchronize()
result["cuda_smoke_sum"] = observed.item()
if result["cuda_smoke_sum"] != 56.0:
    raise SystemExit(f"CUDA smoke result mismatch: {result}")
print(json.dumps(result, sort_keys=True))
PY

"$python_bin" -m unittest discover -s tests -p 'test_*.py' -v
"$ruff_bin" check .
"$ty_bin" check
"$python_bin" -m compileall -q src tests scripts
"$heretic_bin" --help >/dev/null
"$python_bin" scripts/verify-distributed-tiny-model.py

build_dir="$(mktemp -d)"
cleanup() {
    rm -rf "$build_dir"
}
trap cleanup EXIT

uv build --out-dir "$build_dir"
"$python_bin" - "$build_dir" <<'PY'
from pathlib import Path
from sys import argv
from zipfile import ZipFile

build_dir = Path(argv[1])
wheels = list(build_dir.glob("*.whl"))
sdists = list(build_dir.glob("*.tar.gz"))
if len(wheels) != 1:
    raise SystemExit(f"expected exactly one wheel, found: {wheels}")
if len(sdists) != 1:
    raise SystemExit(f"expected exactly one sdist, found: {sdists}")
with ZipFile(wheels[0]) as archive:
    required_modules = {
        "heretic/runtime.py",
        "heretic/distributed_protocol.py",
        "heretic/distributed_transport.py",
        "heretic/distributed_partition.py",
    }
    missing_modules = required_modules.difference(archive.namelist())
    if missing_modules:
        raise SystemExit(
            f"wheel is missing distributed runtime modules: {sorted(missing_modules)}"
        )
print("linux_package_distributed_runtime_present=true")
PY

printf 'wsl_verification_passed=true\n'
