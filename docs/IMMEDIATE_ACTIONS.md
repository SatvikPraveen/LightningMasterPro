# Immediate Action Items - LightningMasterPro

## 🎯 Quick Start Checklist

### ✅ COMPLETED
- [x] Virtual environment created at `venv/`
- [x] Dependencies installation in progress
- [x] Comprehensive gap analysis completed

---

## 🔴 CRITICAL FIXES NEEDED (Do These First!)

### 1. Fix Broken Training Script (5 minutes)
**File**: `scripts/train.py`  
**Problem**: CLI object created but training never executes  
**Solution**: The LightningCLI with `run=True` should auto-execute, but needs verification

```python
# Current (line 12-18)
def main():
    """Main training function using LightningCLI."""
    cli = LightningMasterProCLI(
        save_config_callback=True,
        auto_configure_optimizers=False,
        parser_kwargs={"parser_mode": "omegaconf"}
    )
    # MISSING EXECUTION!

# Potential fix - verify LightningMasterProCLI class has run=True
```

**Action**: Check if LightningMasterProCLI needs explicit `run=True` or if training auto-runs

---

### 2. Complete CLI Implementation (15 minutes)
**File**: `src/lmpro/cli.py`  
**Problem**: File truncated at line 150, missing functions  
**Solution**: Complete the domain-specific CLI functions

**Missing Functions**:
- `vision_cli()`
- `nlp_cli()`
- `tabular_cli()`
- `timeseries_cli()`
- `main()`

---

### 3. Complete Utility Functions (10 minutes each)

#### a) Fix `compute_calibration_metrics()`
**File**: `src/lmpro/utils/metrics.py`  
**Status**: Function is incomplete/truncated  
**Action**: Complete the implementation

#### b) Implement `save_plot()`
**File**: `src/lmpro/utils/viz.py`  
**Status**: Imported but not implemented  
**Action**: Add the function implementation

```python
def save_plot(fig, filepath: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """Save matplotlib or plotly figure to file."""
    # Implementation needed
```

---

### 4. Create Missing Directories (2 minutes)
**Files to create**:
```bash
mkdir -p .github/workflows
mkdir -p docker
mkdir -p docs/source
mkdir -p logs
```

---

## 🟡 HIGH PRIORITY (Next Steps)

### 5. Add Critical Tests (2-4 hours)

Priority test files to create:
```bash
# Callback tests (most important)
tests/test_callbacks_ema.py
tests/test_callbacks_swa.py
tests/test_callbacks_checkpoints.py

# Loop tests
tests/test_loops_kfold.py
tests/test_loops_curriculum.py

# Utility tests
tests/test_utils_metrics.py
tests/test_utils_viz.py
```

### 6. Add CI/CD (1 hour)

**File**: `.github/workflows/ci.yml`
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e .[dev]
      - run: pytest tests/
      - run: black --check src/ tests/
      - run: isort --check src/ tests/
      - run: flake8 src/ tests/
```

### 7. Add Pre-commit Hooks (30 minutes)

**File**: `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

### 8. Add Docker Support (1 hour)

**File**: `docker/Dockerfile`
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "-m", "pytest"]
```

---

## 🟢 MEDIUM PRIORITY (This Week)

### 9. Documentation Setup
- Set up Sphinx (1-2 hours)
- Write API docs (2-3 hours)
- Add tutorials (2-4 hours)

### 10. Add CONTRIBUTING.md
- Contribution guidelines
- Development setup
- Code style guide
- PR process

### 11. Add Missing Scripts
- `scripts/evaluate.py`
- `scripts/benchmark.py`

---

## 📊 Verification Commands

After fixes, verify everything works:

```bash
# 1. Check virtual environment
source venv/bin/activate

# 2. Install package in development mode
pip install -e .

# 3. Run tests
pytest tests/ -v

# 4. Check code quality
black --check src/ tests/
isort --check src/ tests/
flake8 src/ tests/

# 5. Test training script (once fixed)
python scripts/train.py --config configs/vision/classifier.yaml --trainer.fast_dev_run=5

# 6. Test other scripts
python scripts/tune_lr.py --config configs/tuning/lr_finder.yaml
python scripts/predict.py --checkpoint path/to/checkpoint.ckpt --data path/to/data

# 7. Run notebooks (optional)
jupyter notebook notebooks/01_lightning_fundamentals/01_pl_architecture.ipynb
```

---

## 🎯 Definition of "Done" for Phase 1

- [ ] All 4 critical fixes completed
- [ ] All tests pass (`pytest tests/` returns 0 failures)
- [ ] Training script successfully runs with fast_dev_run
- [ ] Code quality checks pass (black, isort, flake8)
- [ ] All infrastructure directories created
- [ ] README updated with virtual environment setup instructions

---

## 📝 Notes

1. **Virtual Environment Location**: `venv/` (already created, in your project root)
2. **Activation Command**: `source venv/bin/activate` (macOS/Linux)
3. **Dependencies**: Currently installing from requirements.txt
4. **Python Version**: Should be Python 3.8+ (check with `python --version`)

---

## 🚀 Quick Win Strategy

**30-Minute Quick Wins**:
1. ✅ Fix train.py (check run parameter)
2. ✅ Create missing directories
3. ✅ Add .pre-commit-config.yaml
4. ✅ Create basic CI workflow

**After these 4 items**, the project will be significantly more professional and functional!

---

## 📚 Reference

See [GAP_ANALYSIS.md](GAP_ANALYSIS.md) for the complete detailed analysis of all gaps and recommendations.
