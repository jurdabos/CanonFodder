# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Adopted the canonical non-truncating CLI help from acidbase: the top-level command group is now built with `acidbase.cli_utils.group` (a `click.Group` subclass) instead of `click.group`. Click's default group listing truncates each command's short help at 45 characters, so longer descriptions ended in `...`; the shared group wraps the full first paragraph onto aligned continuation lines instead. `RichGroup.main` also routes output through `ensure_unicode_safe_streams()`, so non-ASCII help (em dashes, accented words) survives on Windows consoles using a legacy code page. `uv.lock` pins acidbase at the commit providing `cli_utils`.

### Fixed

- `helpers/experiment.py`: `log_model` now passes `skops_trusted_types` for `collections.OrderedDict`, `lightgbm.basic.Booster`, and `lightgbm.sklearn.LGBMClassifier` when calling `mlflow.sklearn.log_model`. MLflow's default skops serialiser audits the model object graph and rejected LightGBM pipelines mid-retrain (`c9r flow -s lb` → `retrain_model_task` / Experiment 28), failing the flow after a successful train (AUC≈0.98). The types are first-party training artefacts produced in-process, not untrusted downloads. Unit test in `tests/unit/test_experiment_extra.py` updated to assert the allow-list.
- CI unit-test collection failure `libtorch_cuda.so: undefined symbol: ncclCommResume` (run #230), root-caused for real this time: `nvidia-nccl-cu12` (xgboost's dependency) and `nvidia-nccl-cu13` (torch 2.13's dependency) both ship the same `nvidia/nccl/lib/libnccl.so.2` path, so whichever wheel installs last owns the file — and wheel inspection shows the whole cu12 line (2.27.5, 2.28.7, 2.30.7 checked) never exports `ncclCommResume`, a cu13-line API. This falsifies the assumption behind the earlier lock re-resolution (6bcd441) that both lines >= 2.28 carry the symbol; CI stayed red whenever cu12 won the install race, while local venvs where cu13 won kept masking it.
- `pyproject.toml`: torch moved from `[project.dependencies]` into mutually exclusive `gpu`/`cpu` dependency groups (uv's documented PyTorch pattern) with `default-groups = ["dev", "gpu"]`, so plain `uv sync` still installs the CUDA build from PyPI; the `cpu` group resolves `torch==2.13.0+cpu` from the new explicit `pytorch-cpu` index (`download.pytorch.org/whl/cpu`). `override-dependencies` drops `nvidia-nccl-cu12` entirely so exactly one NCCL provider exists per environment — xgboost only dlopens NCCL for distributed GPU training and imports fine without it (verified in a CPU-only env). `requirements.txt` re-exported accordingly by the `uv-export` pre-commit hook.
- `.github/workflows/test.yml`: both jobs now install with `uv sync --frozen --extra all --no-group gpu --group cpu` (`--frozen --extra all` retained from 6bcd441 to match `lint.yml`), putting CPU-only torch on the CUDA-less runners — no CUDA/NCCL wheels at all, roughly 5 GB less to install per run, and no shared-file race left to lose.

### Security

- `pyproject.toml`, `uv.lock`, `requirements.txt`: bumped `cryptography` `>=48.0.1` → `>=50.0.0` (locked `48.0.1` → `50.0.0`) to remediate CVE-2026-69247. The `acidbase patch` `uv add` failed with `UVADDFAIL` because every published mlflow (<= 3.15.1, pre-releases included) caps `cryptography < 50`; appended `"cryptography>=50.0.0"` to the existing `[tool.uv] override-dependencies` to lift the transitive cap — drop it once an mlflow release allows `cryptography >= 50`.

### Notes / clarifications

- Recreating a GPU venv that previously contained both NCCL packages needs `uv sync --reinstall` once: uninstalling `nvidia-nccl-cu12` deletes the shared `libnccl.so.2`/cudnn files that the surviving cu13 packages still claim.
- The Windows-side `acidbase patch` run (Windows uv over `\\wsl.localhost\...`) gutted the Linux `.venv` while trying to recreate it (removed `lib/`, leaving a dangling `lib64` symlink Windows could not delete); the remnant was removed and the CVE bump applied with `uv add --no-sync` from inside WSL — run `uv sync` in WSL (with `--reinstall` if shared NCCL/cudnn files complain) to rebuild the venv.
