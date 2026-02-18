.PHONY: install lint format test type-check check clean

install:
	uv sync --dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

type-check:
	uv run mypy .

check: lint type-check test

clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache build dist *.egg-info
	find . -name "__pycache__" -exec rm -rf {} +
