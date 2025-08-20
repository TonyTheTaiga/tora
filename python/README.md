# Tora Python SDK — Developer Guide

Private repository for the Tora Python SDK. This document describes how to set up the development environment, run tests, and contribute changes.

## Prerequisites

- Python 3.11–3.13
- A virtual environment tool (venv/pyenv)
- Optional: `pre-commit` installed locally

## Setup

```bash
# Create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate

# Install dev dependencies
pip install -e ".[dev]"

# (Optional) enable pre-commit hooks
pre-commit install
```

## Fast Iteration

- Run full test suite: `pytest -q`
- Run a single file: `pytest tests/test_wrapper.py -q`
- Lint and autofix: `ruff check . --fix`
- Clean artifacts: `make clean`

Coverage reports are written to `htmlcov/` and `coverage.xml`.

## Project Structure

- `tora/_client.py`: Core client (experiment create/load, logging, buffering)
- `tora/_wrapper.py`: Global convenience API (e.g., `setup`, `tmetric`, `tresult`)
- `tora/_http.py`: Minimal HTTP client and exceptions
- `tora/_validation.py`: Input validators
- `tora/_types.py`: Typed aliases and small data types
- `tora/__init__.py`: Public exports and `__version__`
- `tests/`: Unit tests (HTTP calls are mocked; no network required)

## Development Notes

- Public metric API:
  - Use `Tora.metric(name, value, step=None)` for training metrics.
  - Use `Tora.result(name, value)` for final results (metadata handled internally).
  - The global wrapper exposes `tmetric(...)` and `tresult(...)`. `tlog` is internal only.
- Input validation must go through validators in `tora/_validation.py`.
- Keep changes small and focused; update or add tests alongside code.
- Avoid adding new dependencies unless necessary.

## Versioning and Changelog

- Bump version in `tora/__init__.py` (`__version__`).
- Document changes in `CHANGELOG.md` using Keep a Changelog format.

Release checklist (internal):
- Update README and examples as needed.
- Ensure `pytest -q` passes locally.
- Ensure `ruff check .` is clean (or run with `--fix`).
- Build wheel: `python -m build --wheel`
- Optionally validate: `twine check dist/*`
- Publish (internal/private): `twine upload dist/*` or use `make pub-wheel`.

## Environment Variables (for manual runs)

- `TORA_API_KEY`: API key if talking to a live backend.
- `TORA_BASE_URL`: API base URL (defaults to the configured value in `_config.py`).

Tests mock the HTTP layer; neither variable is required to run tests.

## Examples

Lightweight sanity check snippet (no network; just validates shape):

```python
from tora import Tora

t = Tora("exp-local", url="https://example/exp-local")
t.metric("accuracy", 0.95, step=1)
t.result("best_accuracy", 0.95)
t.flush()
t.shutdown()
```

For a real run against an API instance, prefer using the wrapper:

```python
from tora import setup, tmetric, tresult, shutdown

setup("dev-experiment", api_key="...", workspace_id="...")
tmetric("loss", 0.5, step=1)
tresult("best_acc", 0.95)
shutdown()
```

## Makefile

- `make help` — list available targets
- `make build-wheel` — build wheel in `dist/`
- `make publish-test` — upload to TestPyPI
- `make publish` — upload to PyPI
- `make pub-wheel` — build, check, publish, clean

This is a private repo; use publish targets only for internal registries.
