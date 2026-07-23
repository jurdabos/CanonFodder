# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `uv.lock`: re-resolved the torch dependency subtree (`uv lock --upgrade-package torch`) so PyPI torch 2.13.0 pulls its declared cu13 family (`nvidia-nccl-cu13==2.29.7`) instead of the stale cu12 pairing, and upgraded `nvidia-nccl-cu12` 2.27.5 -> 2.30.7 (xgboost's dependency). Both nccl wheels claim the same `nvidia/nccl/lib/libnccl.so.2` path, so the install-order race is real; with both lines >= 2.28 the `ncclCommResume` symbol exists either way. This fixes the CI unit-test collection failure `libtorch_cuda.so: undefined symbol: ncclCommResume`, which local runs masked because a side-installed `torch 2.13.0+cu130` (download.pytorch.org) had brought its own newer NCCL.
- `.github/workflows/test.yml`: both jobs now install with `uv sync --frozen --extra all`, matching `lint.yml`, so CI fails loudly on lockfile drift instead of resolving quietly.
