# Contributing to Your Project

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/your-project.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`

## Development Environment

### Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Standards

### Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write clear, descriptive docstrings
- Keep functions focused and modular

### Tools

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Fast linting
- **MyPy**: Static type checking (optional)
- **Bandit**: Security checks (optional)
- **pytest**: Testing framework

### Running Quality Checks

Before committing, ensure your code passes all checks:

```bash
# Format code
black .
isort .

# Check linting
ruff check .

# Run tests
pytest

# Or run all checks with pre-commit
pre-commit run --all-files
```

## Testing

- Write tests for all new features and bug fixes
- Maintain or improve code coverage
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing
```

## Pull Request Process

1. Update documentation as needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update the CHANGELOG if applicable
5. Submit a pull request with a clear description of changes

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Write clear commit messages
- Reference any related issues
- Ensure CI checks pass

## Code Review

- Be respectful and constructive
- Address all feedback
- Be open to suggestions

## Questions?

If you have questions or need help, feel free to:
- Open an issue
- Contact the maintainers

Thank you for contributing!
