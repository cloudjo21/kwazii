# Python Style Checker Skill

## Skill Metadata

**Name**: Python Style Checker
**Version**: 1.0.0
**Author**: Based on Google Python Style Guide
**Purpose**: Helps AI coding agents write clean, consistent Python code following Google's style guide

## When to Use This Skill

This skill automatically activates when:
- Creating new Python files (`.py` extension)
- Modifying existing Python code
- Reviewing Python code for style issues
- Refactoring Python modules

## What This Skill Does

1. **Provides instant style guidance** during Python development
2. **Highlights common mistakes** before they're committed
3. **References detailed documentation** when needed
4. **Ensures consistency** with Google Python Style Guide

## Quick Style Checklist

Before completing any Python code task, verify:

### Imports ✓
- [ ] All imports at top of file
- [ ] `import` statements only for packages/modules — do not import individual classes, functions, or types
- [ ] Use `from x import y as z` only when y is too generic, too long, or causes a name collision
- [ ] Never use relative imports (`from . import`, `from .. import`) — always use full absolute paths
- [ ] Never use bare module imports without a path (e.g., `import jodie`) — avoid `sys.path` dependence
- [ ] Import order: `__future__` → stdlib → third-party → local
- [ ] No wildcard imports (`from x import *`)
- [ ] Exception: direct imports from `typing`, `typing_extensions`, and `collections.abc` are allowed

### Formatting ✓
- [ ] Line length ≤ 80 characters
- [ ] 4-space indentation (no tabs)
- [ ] Consistent string quotes (`'` or `"`) within file
- [ ] Proper whitespace around operators and commas
- [ ] Two blank lines between top-level functions/classes

### Documentation ✓
- [ ] Module docstring at top
- [ ] Function/method docstrings for public APIs
- [ ] Class docstrings with Attributes section
- [ ] Docstrings use `"""triple double quotes"""`
- [ ] Args, Returns, Raises sections as needed

### Type Hints ✓
- [ ] Type annotations on function signatures
- [ ] Modern syntax: `list[str]`, `str | None`
- [ ] Return type specified (or `-> None`)
- [ ] Complex types have type aliases

### Naming ✓
- [ ] `snake_case` for functions, variables, parameters
- [ ] `CapWords` for classes
- [ ] `CAPS_WITH_UNDER` for constants
- [ ] Leading `_` for internal/private names
- [ ] No single-letter names (except `i`, `j`, `k`, `e`, `f`)

### Common Pitfalls ✓
- [ ] No mutable default arguments (`def foo(x=[]):`)
- [ ] Use `is None`, not `== None`
- [ ] Use implicit false (`if not items:`)
- [ ] No bare `except:` or `except Exception:`
- [ ] Context managers for files/resources

## Critical Rules

### 🚫 NEVER Do These

```python
# ❌ Mutable default argument
def append_to(item, target=[]):
    target.append(item)
    return target

# ❌ Wildcard import
from module import *

# ❌ Relative imports
from . import sibling
from .. import parent

# ❌ Bare module import without path (sys.path dependent — unclear which module loads)
import jodie

# ❌ Direct class/function import (except typing-related)
from sound.effects.echo import EchoFilter

# ❌ Bare except
try:
    risky_operation()
except:
    pass

# ❌ Comparing to None with ==
if value == None:
    handle_none()

# ❌ Using len() for empty check
if len(items) == 0:
    print('empty')

# ❌ String concatenation in loops
result = ''
for item in items:
    result += str(item)

# ❌ f-strings in logging
logger.info(f'Processing {item}')
```

### ✅ ALWAYS Do These

```python
# ✅ None as default, create new instance
def append_to(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target

# ✅ Import modules using full absolute paths
from sound.effects import echo
from myproject.subpackage import sibling

# ✅ Direct imports from typing-related modules are allowed
from typing import Any, Optional
from collections.abc import Mapping, Sequence

# ✅ Specific exception handling
try:
    risky_operation()
except SpecificError as e:
    logger.error('Failed: %r', e)

# ✅ Use 'is None'
if value is None:
    handle_none()

# ✅ Implicit false for empty check
if not items:
    print('empty')

# ✅ Join list for string accumulation
items_str = []
for item in items:
    items_str.append(str(item))
result = ''.join(items_str)

# ✅ % formatting in logging
logger.info('Processing %s', item)
```

## Integration with Workflow

### During Code Writing

When writing Python code, the agent should:

1. **Start with proper file structure:**
   ```python
   """Module docstring goes here.

   Detailed description.
   """

   from __future__ import annotations

   import standard_library

   import third_party

   from local_package import module


   class MyClass:
       ...


   def my_function():
       ...


   if __name__ == '__main__':
       main()
   ```

2. **Apply type hints as you write:**
   ```python
   def process_data(
       items: list[dict[str, Any]],
       config: Config | None = None,
   ) -> ProcessedData:
       """Process data items with optional config."""
       ...
   ```

3. **Document while coding:**
   - Add docstrings immediately after function/class definitions
   - Include Args, Returns, Raises sections

### During Code Review

When reviewing Python code, check:

1. **Style violations** against this checklist
2. **Missing documentation** for public APIs
3. **Missing type hints** on function signatures
4. **Common anti-patterns** from the "Never Do" list

### During Refactoring

When refactoring Python code:

1. **Maintain local consistency** with surrounding code
2. **Gradually introduce** type hints if missing
3. **Extract magic numbers** to named constants
4. **Break up long functions** (>40 lines)
5. **Add docstrings** to undocumented public functions

## Quick Reference Links

When you need detailed information:

- **General overview**: `@docs/style-guides/python/README.md`
- **Quick rules**: `@docs/style-guides/python/00-quick-reference.md`
- **Setup & tools**: `@docs/style-guides/python/01-background.md`
- **Language features**: `@docs/style-guides/python/02-python-language-rules.md`
- **Formatting rules**: `@docs/style-guides/python/03-python-style-rules.md`
- **Consistency guidance**: `@docs/style-guides/python/04-parting-words.md`

## Common Patterns

### File Template

```python
"""Module for [purpose].

[Detailed description if needed.]

Typical usage example:

  manager = DataManager()
  result = manager.process(data)
"""

from __future__ import annotations

import os
import sys
from typing import Any

from third_party import library

from myproject import local_module

_PRIVATE_CONSTANT = 42
PUBLIC_CONSTANT = 'value'


class MyClass:
    """[One-line class description.]

    [Extended description if needed.]

    Attributes:
        attr_name: Description of attribute.
    """

    def __init__(self, param: str) -> None:
        """Initialize the instance.

        Args:
            param: Description of parameter.
        """
        self.attr_name = param

    def public_method(self, arg: int) -> str:
        """[One-line method description.]

        Args:
            arg: Description of argument.

        Returns:
            Description of return value.

        Raises:
            ValueError: If arg is negative.
        """
        if arg < 0:
            raise ValueError('arg must be non-negative')
        return str(arg)


def public_function(param: str, option: bool = False) -> dict[str, Any]:
    """[One-line function description.]

    Args:
        param: Description of parameter.
        option: Description of optional parameter.

    Returns:
        Description of return value.
    """
    result: dict[str, Any] = {}
    # Implementation
    return result


def _private_helper() -> None:
    """[Internal helper function description.]"""
    ...


def main() -> None:
    """Main entry point."""
    ...


if __name__ == '__main__':
    main()
```

### Common Refactoring Patterns

#### Before: Missing type hints

```python
def fetch_users(ids, include_deleted=False):
    results = []
    for id in ids:
        user = database.get_user(id)
        if user and (include_deleted or not user.deleted):
            results.append(user)
    return results
```

#### After: With type hints and docstring

```python
def fetch_users(
    ids: Sequence[int],
    include_deleted: bool = False,
) -> list[User]:
    """Fetch users by their IDs.

    Args:
        ids: Sequence of user IDs to fetch.
        include_deleted: If True, include deleted users.

    Returns:
        List of User objects found.
    """
    results: list[User] = []
    for user_id in ids:
        user = database.get_user(user_id)
        if user and (include_deleted or not user.deleted):
            results.append(user)
    return results
```

## Automated Tools

When Python code has been changed, run the following commands in order using the project's uv environment:

1. **ruff format** - Code formatting
   ```bash
   uv run ruff format
   ```

2. **ruff check** - Linting and auto-fix
   ```bash
   uv run ruff check --fix
   ```

3. **mypy** - Type checking
   ```bash
   uv run mypy
   ```

If any errors occur, fix them and re-run.

## Error Messages

When providing feedback about style violations, be specific:

### ❌ Vague Feedback
"This code doesn't follow the style guide."

### ✅ Specific Feedback
"Line 42: Use `if not items:` instead of `if len(items) == 0:` for empty checks (see section 2.14 of style guide). This is more Pythonic and handles edge cases better."

## Skill Behavior

### Auto-Activation

This skill activates when:
- File extension is `.py`
- User asks to "write Python code"
- User requests "Python style check"
- Editing files in Python projects

### Graceful Degradation

If the full style guide isn't available:
1. Fall back to PEP 8 basics
2. Focus on critical safety issues (mutable defaults, exception handling)
3. Note which guidelines couldn't be verified

### Progressive Enhancement

As code matures:
1. **Initial pass**: Get it working, follow basic style
2. **Refinement**: Add comprehensive docstrings
3. **Type safety**: Complete type hint coverage
4. **Optimization**: Improve algorithmic efficiency
5. **Polish**: Perfect formatting and documentation

## Success Criteria

Code following this skill should:
- Pass `uv run ruff format` and `uv run ruff check --fix`
- Pass `uv run mypy` type checking
- Have comprehensive docstrings for public APIs
- Follow all critical rules (no mutable defaults, proper exception handling)
- Match surrounding code style
- Be readable by developers unfamiliar with the codebase

## Notes for AI Agents

1. **Don't over-explain**: Apply the rules without lengthy justifications unless asked
2. **Be pragmatic**: Focus on high-impact issues first
3. **Respect context**: Local consistency sometimes trumps global rules
4. **Suggest improvements**: Offer to refactor when seeing violations
5. **Teach patterns**: Show correct patterns, not just what's wrong

## Version History

- **1.0.0** (2024): Initial release based on Google Python Style Guide

## Feedback

If this skill produces incorrect guidance or misses important patterns, the documentation should be updated at `docs/style-guides/python/`.
