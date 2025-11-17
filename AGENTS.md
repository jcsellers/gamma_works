# Agent Instructions

This document provides guidelines for AI coding agents working on this project.

## General Rules

### Code Changes
- Make minimal, surgical changes to address the issue
- Preserve existing functionality unless explicitly asked to change it
- Always validate changes don't introduce security vulnerabilities
- Update documentation when making user-facing changes

### Testing
- Run existing tests before making changes to understand baseline
- Add tests for new functionality
- Ensure all tests pass before finalizing changes
- Don't remove or modify existing tests without explicit instructions

### Code Quality
- Follow the project's style guidelines (Black, isort, Ruff)
- Run linters before committing
- Use type hints where appropriate
- Write clear, descriptive commit messages

## Development Workflow

1. **Understand the issue**: Read the problem statement carefully
2. **Explore the codebase**: Understand existing structure and patterns
3. **Plan changes**: Create a minimal-change plan
4. **Implement incrementally**: Make small, focused changes
5. **Test frequently**: Run tests after each change
6. **Validate**: Manually verify changes work as expected
7. **Report progress**: Use tools to commit and share progress

## Quality Gates

Before finalizing work, ensure:

```bash
# Format code
black .
isort .

# Lint
ruff check .

# Type check (optional)
mypy src

# Test
pytest

# Security scan (optional)
bandit -r src
```

## Project-Specific Guidelines

### File Organization
- Source code: `src/`
- Tests: `tests/unit/` and `tests/integration/`
- Documentation: `docs/`
- Configuration: Root directory

### Testing Strategy
- Unit tests for individual components
- Integration tests for component interactions
- Maintain good test coverage

### Documentation
- Keep README.md up to date
- Document complex logic with comments
- Update CONTRIBUTING.md when workflow changes

## Prohibited Actions

- Don't share sensitive data or credentials
- Don't commit secrets to source control
- Don't introduce security vulnerabilities
- Don't make breaking changes without explicit approval
- Don't modify this file without human approval

## When in Doubt

If you're unsure about:
- Whether a change is appropriate
- How to implement something correctly
- If you're following the right approach

Stop and ask the user for clarification rather than guessing.
