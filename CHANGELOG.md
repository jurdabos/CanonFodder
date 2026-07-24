# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- CI unit-test collection failure `libtorch_cuda.so: undefined symbol: ncclCommResume` (run #230), root-caused for real this time: `nvidia-nccl-cu12` (xgboost's dependency) and `nvidia-nccl-cu13` (torch 2.13's dependency) both ship the same `nvidia/nccl/lib/libnccl.so.2` path, so whichever wheel installs last owns the file — and wheel inspection shows the whole cu12 line (2.27.5, 2.28.7, 2.30.7 checked) never exports `ncclCommResume`, a cu13-line API. This falsifies the assumption behind the earlier lock re-resolution (6bcd441) that both lines >= 2.28 carry the symbol; CI stayed red whenever cu12 won the install race, while local venvs where cu13 won kept masking it.
- `pyproject.toml`: torch moved from `[project.dependencies]` into mutually exclusive `gpu`/`cpu` dependency groups (uv's documented PyTorch pattern) with `default-groups = ["dev", "gpu"]`, so plain `uv sync` still installs the CUDA build from PyPI; the `cpu` group resolves `torch==2.13.0+cpu` from the new explicit `pytorch-cpu` index (`download.pytorch.org/whl/cpu`). `override-dependencies` drops `nvidia-nccl-cu12` entirely so exactly one NCCL provider exists per environment — xgboost only dlopens NCCL for distributed GPU training and imports fine without it (verified in a CPU-only env). `requirements.txt` re-exported accordingly by the `uv-export` pre-commit hook.
- `.github/workflows/test.yml`: both jobs now install with `uv sync --frozen --extra all --no-group gpu --group cpu` (`--frozen --extra all` retained from 6bcd441 to match `lint.yml`), putting CPU-only torch on the CUDA-less runners — no CUDA/NCCL wheels at all, roughly 5 GB less to install per run, and no shared-file race left to lose.

### Notes / clarifications

- Recreating a GPU venv that previously contained both NCCL packages needs `uv sync --reinstall` once: uninstalling `nvidia-nccl-cu12` deletes the shared `libnccl.so.2`/cudnn files that the surviving cu13 packages still claim.
