# Background

> Source: [Google Python Style Guide - Section 1](https://google.github.io/styleguide/pyguide.html)

## Overview

Python is the main dynamic language used at Google. This style guide provides a comprehensive list of dos and don'ts for Python programs, designed to help developers write clean, maintainable, and consistent code.

## Purpose

The purpose of this guide is to:

1. **Establish consistency** across Python codebases
2. **Improve code readability** for all team members
3. **Prevent common mistakes** and anti-patterns
4. **Facilitate code reviews** with clear, shared standards
5. **Enable better tooling** through standardized patterns

## Philosophy

Google's Python style guide follows these core principles:

- **Readability counts**: Code is read more often than it is written
- **Explicit is better than implicit**: Make your intentions clear
- **Consistency matters**: Follow existing patterns in the codebase
- **Practicality beats purity**: Rules should serve the code, not vice versa

## Automatic Formatting

To help format code correctly and avoid debates over formatting details, Google recommends using automatic formatters:

### Black

[Black](https://github.com/psf/black) is "the uncompromising Python code formatter" that automatically formats Python code to conform to PEP 8 and other standards.

**Installation:**
```bash
pip install black
```

**Usage:**
```bash
# Format a single file
black my_script.py

# Format all Python files in a directory
black src/

# Check what would be changed without modifying files
black --check src/

# Show diff of changes
black --diff src/
```

**Configuration:** Create a `pyproject.toml` in your project root:
```toml
[tool.black]
line-length = 80
target-version = ['py310']
```

### Pyink

[Pyink](https://github.com/google/pyink) is Google's fork of Black with some modifications to better align with Google's internal style.

**Installation:**
```bash
pip install pyink
```

**Usage:**
```bash
pyink my_script.py
```

**Key Differences from Black:**
- More flexible with string quote normalization
- Different handling of some edge cases
- Optimized for Google's internal practices

### Ruff (Recommended)

[Ruff](https://docs.astral.sh/ruff/) is the primary formatter and linter used in this project. It replaces Black for formatting and pylint/flake8 for linting, with significantly faster performance.

**Installation (via uv):**
```bash
uv add --dev ruff
```

**Usage:**
```bash
# Format all Python files
uv run ruff format

# Lint and auto-fix all Python files
uv run ruff check --fix
```

Run these two commands in order after any Python code change:

```bash
uv run ruff format && uv run ruff check --fix
```

**Configuration** (`pyproject.toml`):
```toml
[tool.ruff]
line-length = 80
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
```

> **Ruff is the primary formatter for this project.** Black, Pyink, and pylint are optional and not required. Use ruff for all formatting and linting tasks.

### Choosing a Formatter

Many teams at Google use Black or Pyink to:
- Eliminate debates about formatting
- Ensure consistency across the codebase
- Reduce cognitive load during code reviews
- Save time on manual formatting

**Recommendation**: Pick one formatter for your project and apply it consistently to all Python code. For this project, **ruff is the standard choice** — Black and Pyink are available as alternatives but are not required.

## Editor Setup

### Vim

Google provides a [settings file for Vim](https://google.github.io/styleguide/google_python_style.vim) to help configure your editor for Python development according to this style guide.

**Key Vim settings:**
```vim
" Set tab width to 4 spaces
set tabstop=4
set shiftwidth=4
set expandtab

" Set line length marker at 80 characters
set colorcolumn=80

" Enable Python-specific features
set autoindent
set smartindent
```

### Emacs

For Emacs users, the default settings should be fine for most Python development.

### VS Code

Recommended settings for VS Code:

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.rulers": [80],
  "python.analysis.typeCheckingMode": "basic"
}
```

### PyCharm

PyCharm settings:
1. Go to Settings → Editor → Code Style → Python
2. Set "Right margin (columns)" to 80
3. Set "Tab size" and "Indent" to 4
4. Enable "Use tab character": OFF
5. Install and configure Black or Pyink plugin

## Linting with pylint

Running `pylint` over your code is essential for catching bugs and style issues early.

### Installation

```bash
pip install pylint
```

### Configuration

Use Google's [pylintrc configuration file](https://google.github.io/styleguide/pylintrc) for consistency with this style guide.

**Download and use:**
```bash
# Download Google's pylintrc
wget https://google.github.io/styleguide/pylintrc

# Run pylint with Google's config
pylint --rcfile=pylintrc your_module.py
```

### Running pylint

```bash
# Check a single file
pylint my_script.py

# Check all Python files in a directory
pylint src/

# Generate a configuration file
pylint --generate-rcfile > .pylintrc

# Check specific error categories
pylint --disable=all --enable=errors my_script.py
```

### Understanding pylint Output

pylint provides several types of messages:

- **Error (E)**: For probable bugs in the code
- **Warning (W)**: For stylistic problems or minor programming issues
- **Refactor (R)**: For code that could be improved
- **Convention (C)**: For coding standard violations
- **Information (I)**: For informational messages

**Example output:**
```
my_script.py:15:0: C0111: Missing module docstring (missing-docstring)
my_script.py:23:4: E1101: Instance of 'dict' has no 'iteritems' member (no-member)
my_script.py:45:0: W0612: Unused variable 'result' (unused-variable)
```

### Suppressing pylint Warnings

Sometimes pylint warnings are incorrect or inappropriate for specific code. You can suppress them with comments:

```python
# Suppress specific warning for one line
def do_PUT(self):  # pylint: disable=invalid-name
    ...

# Suppress multiple warnings
result = compute()  # pylint: disable=no-member,unused-variable

# Suppress for a block
# pylint: disable=too-many-branches
def complex_function():
    if condition1:
        ...
    elif condition2:
        ...
    # ... many more branches
# pylint: enable=too-many-branches

# Suppress for entire file (use sparingly)
# pylint: disable=invalid-name
```

**Important guidelines for suppressions:**
1. Always include the symbolic name (`invalid-name`), not the code (`C0103`)
2. Add explanatory comments if the reason isn't obvious from the symbolic name
3. Keep suppressions as narrow as possible (line-level preferred over file-level)
4. Regularly review suppressions - they may become outdated

## Type Checking

In addition to pylint, use static type checking tools to catch type-related errors at build time.

### pytype

[pytype](https://github.com/google/pytype) is Google's Python type checker.

**Installation:**
```bash
pip install pytype
```

**Usage:**
```bash
pytype my_script.py
```

### mypy

[mypy](http://mypy-lang.org/) is another popular static type checker.

**Installation:**
```bash
pip install mypy
```

**Usage:**
```bash
mypy my_script.py
```

**Configuration** (`mypy.ini` or `pyproject.toml`):
```ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

## Version Compatibility

This style guide is written for Python 3. While some rules may apply to Python 2, the focus is on modern Python (3.7+).

### Minimum Python Version

Google generally targets Python 3.7 or later for new projects. Some features mentioned in this guide require specific Python versions:

- **Python 3.7+**: Required for most features
- **Python 3.9+**: For improved type hint syntax (`list[int]` instead of `List[int]`)
- **Python 3.10+**: For union type syntax (`int | str` instead of `Union[int, str]`)

### Future Imports

Use `from __future__ import annotations` to enable newer annotation syntax in older Python versions:

```python
from __future__ import annotations

def process(items: list[str]) -> dict[str, int]:  # Works in Python 3.7+
    ...
```

## Project Structure

A typical Python project following this style guide might look like:

```
myproject/
├── .pylintrc                 # pylint configuration
├── pyproject.toml           # Black/Pyink and project config
├── setup.py                 # Package setup
├── README.md
├── requirements.txt         # Dependencies
├── myproject/
│   ├── __init__.py
│   ├── main.py
│   ├── module1.py
│   └── subpackage/
│       ├── __init__.py
│       └── module2.py
└── tests/
    ├── __init__.py
    ├── test_module1.py
    └── test_module2.py
```

## Integration with Development Workflow

### Pre-commit Hooks

Use [pre-commit](https://pre-commit.com/) to automatically run formatters and linters:

**`.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/PyCQA/pylint
    rev: v3.0.0
    hooks:
      - id: pylint
        args: [--rcfile=.pylintrc]
```

### Continuous Integration

Include style checks in your CI pipeline:

```yaml
# Example GitHub Actions workflow
name: Python Style Check

on: [push, pull_request]

jobs:
  style:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install black pylint mypy
      - name: Check formatting
        run: black --check .
      - name: Run pylint
        run: pylint src/
      - name: Run type checker
        run: mypy src/
```

## Adoption Strategy

When adopting this style guide for an existing project:

1. **Start with new code**: Apply the guide to all new files and modifications
2. **Use automatic formatters**: Run Black/Pyink on the codebase
3. **Fix critical issues**: Address errors and warnings from pylint
4. **Gradual improvement**: Refactor old code incrementally during maintenance
5. **Team alignment**: Ensure all team members understand and agree on the standards
6. **Update documentation**: Keep this guide accessible to all developers

## Benefits of Following This Guide

- **Easier code reviews**: Focus on logic rather than style
- **Better collaboration**: Consistent style reduces friction
- **Fewer bugs**: Linters catch common mistakes early
- **Maintainability**: Standardized code is easier to understand and modify
- **Onboarding**: New team members can quickly learn the patterns
- **Tool support**: Consistent style enables better IDE and static analysis support

## Additional Resources

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google's pylintrc](https://google.github.io/styleguide/pylintrc)
- [Black Documentation](https://black.readthedocs.io/)
- [pylint Documentation](https://pylint.pycqa.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [pytype Documentation](https://google.github.io/pytype/)

## Next Steps

- Review [Python Language Rules](02-python-language-rules.md) for detailed guidelines on using Python features
- Check [Python Style Rules](03-python-style-rules.md) for formatting and documentation standards
- Use the [Quick Reference](00-quick-reference.md) for day-to-day coding decisions
