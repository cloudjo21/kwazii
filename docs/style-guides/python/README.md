# Google Python Style Guide - Summary

> **Source**: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
>
> This is a comprehensive summary of Google's Python coding standards, adapted for AI coding agents.

## Quick Navigation

- [Quick Reference (Essential Rules)](00-quick-reference.md) - Start here for the most important rules
- [Background](01-background.md) - Introduction and setup
- [Python Language Rules](02-python-language-rules.md) - Core language features and best practices
- [Python Style Rules](03-python-style-rules.md) - Formatting and documentation standards
- [Parting Words](04-parting-words.md) - Consistency matters

## Overview

Python is the main dynamic language used at Google. This style guide provides a comprehensive set of dos and don'ts for Python programs.

**Key Tools:**
- **Formatter**: Use [Black](https://github.com/psf/black) or [Pyink](https://github.com/google/pyink) for automatic formatting
- **Linter**: Run `pylint` with [Google's pylintrc](https://google.github.io/styleguide/pylintrc)

## Critical Rules at a Glance

### Imports
- Use `import` for packages and modules only, not for individual classes or functions
- Use full package paths (e.g., `from absl import flags`, not `import flags`)
- Import order: future imports → standard library → third-party → local

### Type Annotations
- Strongly encouraged for all new code
- Use modern syntax: `str | None` instead of `Optional[str]`
- Annotate public APIs at minimum

### Formatting
- **Line length**: 80 characters maximum
- **Indentation**: 4 spaces (never tabs)
- **String quotes**: Pick `'` or `"` and be consistent within a file
- **Docstrings**: Always use `"""triple double quotes"""`

### Naming Conventions
```
module_name, package_name
ClassName, ExceptionName
method_name, function_name, function_parameter_name
GLOBAL_CONSTANT_NAME
global_var_name, instance_var_name, local_var_name
```

### Documentation
- Module docstring required at the top of every file
- Function docstring required for:
  - Non-obvious logic
  - Nontrivial size
  - Public API functions
- Use `Args:`, `Returns:`, `Raises:` sections

## Language Rules Summary

### 2.1 Lint
- Run `pylint` on all code
- Suppress warnings with `# pylint: disable=rule-name` when appropriate
- Always add explanatory comments for suppressions

### 2.2 Imports
- `import` 문은 패키지와 모듈에만 사용한다. 개별 타입, 클래스, 함수에는 사용하지 않는다
- `from x import y as z` 형식은 y가 너무 일반적이거나 길거나, 이름 충돌이 있거나, 동일 이름의 모듈이 둘 이상일 때만 허용한다
- 상대 임포트(`from . import`, `from .. import`)는 절대 사용하지 않는다 — 항상 전체 절대 경로로 임포트한다
- 예외: `typing`, `typing_extensions`, `collections.abc` 모듈의 심볼은 직접 임포트 허용

### 2.3 Packages
- 모듈을 임포트할 때는 항상 전체 패키지 경로를 사용한다
- 메인 바이너리가 있는 디렉토리가 `sys.path`에 있다고 가정하지 않는다
- `import jodie` 같이 경로 없이 임포트하면 어떤 모듈이 로드될지 불명확해지므로 금지한다

### 2.4 Exceptions
- Use built-in exception classes when appropriate
- Never use bare `except:` or catch `Exception` without re-raising
- Keep `try` blocks minimal
- Don't use `assert` for validating preconditions in production code

### 2.7 Comprehensions
```python
# Yes:
result = [mapping_expr for value in iterable if filter_expr]

# No: Multiple for clauses
result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]
```

### 2.12 Default Argument Values
```python
# Yes:
def foo(a, b=None):
    if b is None:
        b = []

# No: Mutable defaults
def foo(a, b=[]):
    ...
```

### 2.14 True/False Evaluations
```python
# Yes:
if not users:
    print('no users')

# No:
if len(users) == 0:
    print('no users')
```

### 2.21 Type Annotated Code
```python
# Yes:
def func(a: int) -> list[int]:
    ...

# Annotate variables when type is unclear:
a: SomeType = some_func()
```

## Style Rules Summary

### 3.2 Line Length
- Maximum 80 characters
- Use implicit line joining (parentheses) instead of backslashes
```python
# Yes:
foo_bar(self, width, height, color='black', design=None, x='foo',
        emphasis=None, highlight=0)

# No:
foo_bar(self, width, height, color='black', design=None, x='foo', \
        emphasis=None, highlight=0)
```

### 3.4 Indentation
- 4 spaces per indentation level
- Never use tabs
```python
# Yes: Aligned with opening delimiter
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# Yes: 4-space hanging indent
foo = long_function_name(
    var_one, var_two, var_three,
    var_four)
```

### 3.6 Whitespace
```python
# Yes:
spam(ham[1], {'eggs': 2}, [])
if x == 4:
    print(x, y)

# No:
spam( ham[ 1 ], { 'eggs': 2 }, [ ] )
if x == 4 :
    print(x , y)
```

### 3.8 Comments and Docstrings

**Module docstring:**
```python
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line. The rest of this docstring should contain an
overall description of the module or program.

Typical usage example:

  foo = ClassFoo()
  bar = foo.function_bar()
"""
```

**Function docstring:**
```python
def fetch_smalltable_rows(
    table_handle: smalltable.Table,
    keys: Sequence[bytes | str],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle. String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch. String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes. If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
```

**Class docstring:**
```python
class SampleClass:
    """Summary of class here.

    Longer class information...

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
```

### 3.10 Strings
```python
# Yes: Use f-strings, % operator, or .format()
x = f'name: {name}; score: {n}'
x = 'name: %s; score: %d' % (name, n)

# No: Don't concatenate with +
x = 'name: ' + name + '; score: ' + str(n)
```

**Logging:**
```python
# Yes:
logger.info('TensorFlow Version is: %s', tf.__version__)

# No: Don't use f-strings for logging
logger.info(f'TensorFlow Version is: {tf.__version__}')
```

### 3.13 Imports Formatting
```python
# Import order:
from __future__ import annotations  # Future imports

import sys  # Standard library
import os

import tensorflow as tf  # Third-party
from absl import app

from myproject.backend import huxley  # Local
from myproject.backend.state_machine import main_loop
```

### 3.16 Naming

| Type | Public | Internal |
|------|--------|----------|
| Packages | `lower_with_under` | |
| Modules | `lower_with_under` | `_lower_with_under` |
| Classes | `CapWords` | `_CapWords` |
| Exceptions | `CapWords` | |
| Functions | `lower_with_under()` | `_lower_with_under()` |
| Global/Class Constants | `CAPS_WITH_UNDER` | `_CAPS_WITH_UNDER` |
| Global/Class Variables | `lower_with_under` | `_lower_with_under` |
| Instance Variables | `lower_with_under` | `_lower_with_under` |
| Method Names | `lower_with_under()` | `_lower_with_under()` |
| Function/Method Parameters | `lower_with_under` | |
| Local Variables | `lower_with_under` | |

**Names to avoid:**
- Single character names (except `i`, `j`, `k` for counters, `e` for exceptions, `f` for file handles)
- Dashes (`-`) in any package/module name
- `__double_leading_and_trailing_underscore__` names (reserved by Python)
- Names that needlessly include the type (e.g., `id_to_name_dict`)

### 3.19 Type Annotations

**General rules:**
```python
# Annotate public APIs
def public_function(param: str) -> int:
    ...

# Use modern union syntax (Python 3.10+)
def func(x: str | None) -> list[int]:
    ...

# Type aliases for complex types
from typing import TypeAlias
_LossAndGradient: TypeAlias = tuple[tf.Tensor, tf.Tensor]
```

**Line breaking:**
```python
# One parameter per line with trailing comma
def my_method(
    self,
    first_var: int,
    second_var: Foo,
    third_var: Bar | None,
) -> int:
    ...
```

**Forward declarations:**
```python
from __future__ import annotations

class MyClass:
    def __init__(self, stack: Sequence[MyClass]) -> None:
        ...
```

**Imports for typing:**
```python
# Import symbols directly
from collections.abc import Mapping, Sequence
from typing import Any, Generic, cast

# Prefer abstract types
def transform(data: Sequence[int]) -> Sequence[int]:  # Yes
def transform(data: list[int]) -> list[int]:  # Less preferred
```

## Common Patterns

### File Structure
```python
#!/usr/bin/env python3
"""Module docstring.

Detailed description.
"""

from __future__ import annotations

import sys
from typing import Any

from third_party import lib

from myproject import mymodule

_PRIVATE_CONSTANT = 42
PUBLIC_CONSTANT = 'value'


class MyClass:
    """Class docstring."""

    def __init__(self) -> None:
        self.public_attr = 1
        self._protected_attr = 2

    def public_method(self) -> None:
        """Method docstring."""
        ...


def public_function(arg: str) -> int:
    """Function docstring."""
    ...


def _private_function() -> None:
    """Private function docstring."""
    ...


def main() -> None:
    """Main function."""
    ...


if __name__ == '__main__':
    main()
```

### Error Handling
```python
# Yes:
try:
    result = risky_operation()
except SpecificError as e:
    logger.error('Operation failed: %r', e)
    raise
finally:
    cleanup()

# No:
try:
    result = risky_operation()
except:  # Too broad
    pass
```

### Context Managers
```python
# Yes:
with open('file.txt') as f:
    data = f.read()

# Multiple context managers
with (
    open('input.txt') as input_file,
    open('output.txt', 'w') as output_file,
):
    process(input_file, output_file)
```

## Best Practices

1. **BE CONSISTENT** - Match the style of the surrounding code
2. **Use type hints** - Especially for public APIs and complex code
3. **Document thoroughly** - Write clear docstrings and comments
4. **Keep it simple** - Avoid power features unless necessary
5. **Test your code** - Write unit tests for all functionality
6. **Run linters** - Use pylint and address all warnings
7. **Format automatically** - Use Black or Pyink to avoid formatting debates

## When in Doubt

1. Check surrounding code for local style conventions
2. Refer to [Quick Reference](00-quick-reference.md) for common rules
3. Consult specific sections for detailed guidance
4. Remember: readability counts

## Additional Resources

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Google's pylintrc](https://google.github.io/styleguide/pylintrc)
