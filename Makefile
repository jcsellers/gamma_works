.PHONY: help install install-dev test test-cov lint format pre-commit clean html

SPHINXBUILD = python -m sphinx
SPHINXSOURCEDIR = docs/sphinx
SPHINXBUILDDIR = docs/_build

help:
	@echo "Available commands:"
	@echo "  make install       - Install package"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make lint          - Run all linting checks"
	@echo "  make format        - Format code with black and isort"
	@echo "  make pre-commit    - Run pre-commit hooks on all files"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make html          - Build Sphinx HTML documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=src/xdte --cov-report=term-missing --cov-report=html

lint:
        ruff check .
        black --check .
        isort --check-only .
        mypy --strict src

format:
	black .
	isort .

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

html:
	$(SPHINXBUILD) -b html $(SPHINXSOURCEDIR) $(SPHINXBUILDDIR)/html
