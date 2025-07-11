.PHONY: help install install-dev test lint format type-check clean build publish

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest --cov=tora --cov-report=term-missing --cov-report=html

test-verbose:  ## Run tests with verbose output
	pytest -v --cov=tora --cov-report=term-missing --cov-report=html

lint:  ## Run linting
	flake8 tora tests
	black --check tora tests
	isort --check-only tora tests

format:  ## Format code
	black tora tests
	isort tora tests

type-check:  ## Run type checking
	mypy tora

pre-commit:  ## Run pre-commit hooks
	pre-commit run --all-files

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	python -m build

build-wheel:  ## Build wheel (legacy target)
	python -m build --wheel

check-build:  ## Check the built package
	twine check dist/*

publish-test:  ## Publish to Test PyPI
	twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	twine upload dist/*

pub-wheel: build check-build publish clean  ## Build and publish wheel (legacy target)

dev-setup:  ## Set up development environment
	pip install -e ".[dev]"
	pre-commit install

ci:  ## Run CI checks locally
	make lint
	make type-check
	make test
