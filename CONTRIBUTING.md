# Contributing to LightningMasterPro

Thank you for your interest in contributing! This document outlines the process
for reporting issues, proposing features, and submitting pull requests.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Branch Naming](#branch-naming)
5. [Making Changes](#making-changes)
6. [Testing](#testing)
7. [Code Style](#code-style)
8. [Pull Request Process](#pull-request-process)
9. [Reporting Issues](#reporting-issues)

---

## Code of Conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
Please read it before participating.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/<your-username>/LightningMasterPro.git
   cd LightningMasterPro
   ```

3. Set the upstream remote:

   ```bash
   git remote add upstream https://github.com/satvikpraveen/LightningMasterPro.git
   ```

---

## Development Setup

Create and activate a virtual environment, then install all dependencies:

```bash
python3 -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\activate             # Windows

pip install -e ".[dev,docs,export]"
pip install pre-commit
pre-commit install
```

Verify the setup:

```bash
pytest tests/ -q --tb=short
```

---

## Branch Naming

Use the following conventions for branch names:

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feat/<short-description>` | `feat/add-grad-monitor` |
| Bug fix | `fix/<short-description>` | `fix/kfold-stratified` |
| Documentation | `docs/<short-description>` | `docs/update-readme` |
| Refactor | `refactor/<short-description>` | `refactor/cli-cleanup` |
| Tests | `test/<short-description>` | `test/synth-timeseries` |

---

## Making Changes

1. Create a feature branch from `main`:

   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feat/your-feature
   ```

2. Make your changes. Keep commits atomic and descriptive.

3. Run pre-commit checks before committing:

   ```bash
   pre-commit run --all-files
   ```

4. Commit using a meaningful message following [Conventional Commits](https://www.conventionalcommits.org/):

   ```
   feat(callbacks): add gradient anomaly alerting to GradientMonitorCallback
   fix(loops): correct KFoldLoop stratified split indexing
   docs(api): document interpretability module public API
   ```

---

## Testing

All new features and bug fixes must include tests.

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_callbacks_ema.py -v

# Run with coverage
pytest tests/ --cov=src/lmpro --cov-report=term-missing
```

Test files live in `tests/` and mirror the source structure:

```
tests/
    test_callbacks_ema.py
    test_callbacks_swa.py
    test_data_synth_vision.py
    test_datamodules.py
    ...
```

---

## Code Style

We use the following tools (enforced via pre-commit):

| Tool | Purpose |
|------|---------|
| `black` | Code formatting |
| `isort` | Import sorting |
| `flake8` | Linting |
| `mypy` | Type checking |

Run manually:

```bash
black src/ tests/ scripts/
isort src/ tests/ scripts/
flake8 src/ tests/ scripts/
mypy src/lmpro/
```

---

## Pull Request Process

1. Ensure all tests pass locally.
2. Push your branch to your fork:

   ```bash
   git push origin feat/your-feature
   ```

3. Open a Pull Request against the `main` branch.
4. Fill in the PR template — describe *what* and *why*, not how.
5. Link any related issues with `Closes #<issue-number>`.
6. At least one maintainer review is required before merging.
7. Squash commits before merging if the PR has many WIP commits.

---

## Reporting Issues

When filing a bug report, include:

- Python, PyTorch, and PyTorch Lightning versions (`pip show torch pytorch-lightning`)
- OS and hardware (CPU/GPU)
- Minimal reproducible example
- Full stack trace

Feature requests are welcome — open an issue with the `enhancement` label and
describe the use case and proposed API.
