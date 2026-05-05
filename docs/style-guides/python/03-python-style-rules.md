# Python Style Rules

> Source: [Google Python Style Guide - Section 3](https://google.github.io/styleguide/pyguide.html)

This section covers formatting and documentation standards for Python code. These rules focus on how code should look and be documented, rather than which language features to use.

## Table of Contents

- [3.1 Semicolons](#31-semicolons)
- [3.2 Line Length](#32-line-length)
- [3.3 Parentheses](#33-parentheses)
- [3.4 Indentation](#34-indentation)
- [3.5 Blank Lines](#35-blank-lines)
- [3.6 Whitespace](#36-whitespace)
- [3.7 Shebang Line](#37-shebang-line)
- [3.8 Comments and Docstrings](#38-comments-and-docstrings)
- [3.10 Strings](#310-strings)
- [3.11 Files, Sockets, and similar Stateful Resources](#311-files-sockets-and-similar-stateful-resources)
- [3.12 TODO Comments](#312-todo-comments)
- [3.13 Imports Formatting](#313-imports-formatting)
- [3.14 Statements](#314-statements)
- [3.15 Getters and Setters](#315-getters-and-setters)
- [3.16 Naming](#316-naming)
- [3.17 Main](#317-main)
- [3.18 Function Length](#318-function-length)
- [3.19 Type Annotations](#319-type-annotations)

---

## 3.1 Semicolons

**Decision**: Do not terminate your lines with semicolons, and do not use semicolons to put two statements on the same line.

---

## 3.2 Line Length

**Decision**: Maximum line length is **80 characters**.

### Exceptions

- Long string module-level constants (URLs, pathnames)
- Pylint disable comments

### Guidelines

```python
# Yes: Use implicit line joining
foo_bar(self, width, height, color='black', design=None, x='foo',
        emphasis=None, highlight=0)

if (width == 0 and height == 0 and
    color == 'red' and emphasis == 'strong'):
    process()

# No: Don't use backslashes
if width == 0 and height == 0 and \
    color == 'red' and emphasis == 'strong':
    process()
```

**For long strings:**

```python
x = ('This will build a very long long '
     'long long long long long long string')
```

**Prefer breaking at the highest syntactic level:**

```python
# Yes:
bridgekeeper.answer(
    name="Arthur", quest=questlib.find(owner="Arthur", perilous=True))

answer = (a_long_line().of_chained_methods()
          .that_eventually_provides().an_answer())

# No:
bridgekeeper.answer(name="Arthur", quest=questlib.find(
    owner="Arthur", perilous=True))
```

---

## 3.3 Parentheses

**Decision**: Use parentheses sparingly.

```python
# Yes:
if foo:
    bar()

return foo

# Tuple with one item
onesie = (foo,)

# No:
if (x):
    bar()

return (foo)
```

---

## 3.4 Indentation

**Decision**: Indent your code blocks with **4 spaces**. Never use tabs.

### Patterns

```python
# Aligned with opening delimiter
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# 4-space hanging indent
foo = long_function_name(
    var_one, var_two, var_three,
    var_four)

# Closing bracket aligned with first character
foo = long_function_name(
    var_one, var_two, var_three,
    var_four
)
```

### Trailing Commas

Recommended when the closing bracket is on a separate line:

```python
golomb4 = [
    0,
    1,
    4,
    6,
]
```

---

## 3.5 Blank Lines

- **Two** blank lines between top-level definitions (functions and classes)
- **One** blank line between method definitions
- **No** blank line following a `def` line

---

## 3.6 Whitespace

### Rules

```python
# Yes:
spam(ham[1], {'eggs': 2}, [])
if x == 4:
    print(x, y)
x, y = y, x

# No:
spam( ham[ 1 ], { 'eggs': 2 }, [ ] )
if x == 4 :
    print(x , y)
```

### Binary Operators

Surround with single space on either side:

```python
# Yes:
x == 1
if width == 0 and height == 0:
    ...

# No:
x<1
```

### Keyword Arguments

```python
# Yes:
def complex(real, imag=0.0):
    return Magic(r=real, i=imag)

# Yes: With type annotation
def complex(real, imag: float = 0.0):
    return Magic(r=real, i=imag)

# No:
def complex(real, imag = 0.0):
    return Magic(r = real, i = imag)
```

---

## 3.7 Shebang Line

Most `.py` files don't need a `#!` line. For executable scripts:

```python
#!/usr/bin/env python3
```

Or:

```python
#!/usr/bin/python3
```

---

## 3.8 Comments and Docstrings

### 3.8.1 Docstrings

Always use `"""triple double quotes"""`. Docstring structure:

- Summary line (one physical line, ≤80 chars)
- Blank line
- Detailed description

### 3.8.2 Modules

Every file should start with a module docstring:

```python
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line. The rest of this docstring should contain an
overall description of the module or program.

Typical usage example:

  foo = ClassFoo()
  bar = foo.function_bar()
"""
```

### 3.8.3 Functions and Methods

Required for functions with:
- Non-obvious logic
- Nontrivial size
- Part of public API

```python
def fetch_data(
    table: Table,
    keys: Sequence[str],
    require_all: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Fetches rows from a table.

    Retrieves rows pertaining to the given keys.

    Args:
        table: An open Table instance.
        keys: A sequence of strings representing keys to fetch.
        require_all: If True, only returns if all keys are found.

    Returns:
        A dict mapping keys to the corresponding data.
        Each row is represented as a tuple of strings.

    Raises:
        IOError: An error occurred accessing the table.
    """
```

**Overridden methods** with `@override` decorator don't need docstrings unless behavior materially changes.

### 3.8.4 Classes

```python
class SampleClass:
    """Summary of class here.

    Longer class information...

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
```

### 3.8.5 Block and Inline Comments

```python
# We use a weighted dictionary search to find out where i is in
# the array. We extrapolate position based on the largest num
# in the array and the array size and then do binary search to
# get the exact number.

if i & (i-1) == 0:  # True if i is 0 or a power of 2.
```

Comments should start at least 2 spaces away from code.

---

## 3.10 Strings

### String Formatting

```python
# Yes: f-strings, % operator, or .format()
x = f'name: {name}; score: {n}'
x = 'name: %s; score: %d' % (name, n)
x = 'name: {}; score: {}'.format(name, n)

# No: Don't use + for formatting
x = 'name: ' + name + '; score: ' + str(n)
```

### String Accumulation

```python
# Yes: Use list and join
items = ['<table>']
for last_name, first_name in employee_list:
    items.append('<tr><td>%s, %s</td></tr>' % (last_name, first_name))
employee_table = ''.join(items)

# No: Don't accumulate with + in loops
employee_table = '<table>'
for last_name, first_name in employee_list:
    employee_table += '<tr><td>%s, %s</td></tr>' % (last_name, first_name)
```

### Quote Consistency

Pick `'` or `"` and stick with it within a file.

```python
# Yes:
Python('Why are you hiding your eyes?')
Gollum("I'm scared of lint errors.")
```

Use `"""` for multi-line strings and docstrings.

### 3.10.1 Logging

```python
# Yes: Use % placeholders (not f-strings!)
logger.info('TensorFlow Version is: %s', tf.__version__)

# No:
logger.info(f'TensorFlow Version is: {tf.__version__}')
```

### 3.10.2 Error Messages

```python
# Yes:
if not 0 <= p <= 1:
    raise ValueError(f'Not a probability: {p=}')

try:
    os.rmdir(workdir)
except OSError as error:
    logging.warning('Could not remove directory (reason: %r): %r',
                    error, workdir)
```

---

## 3.11 Files, Sockets, and similar Stateful Resources

**Decision**: Explicitly close files and sockets when done.

### Use Context Managers

```python
# Yes:
with open("hello.txt") as hello_file:
    for line in hello_file:
        print(line)

# For objects without context manager support
import contextlib

with contextlib.closing(urllib.urlopen("http://www.python.org/")) as front_page:
    for line in front_page:
        print(line)
```

---

## 3.12 TODO Comments

Format: `TODO: context - explanation`

```python
# TODO: crbug.com/192795 - Investigate cpufreq optimizations.
```

Include a very specific date or event if "at a future date":

```python
# TODO: Remove this code when all clients can handle XML responses.
```

---

## 3.13 Imports Formatting

### Separate Lines

```python
# Yes:
from collections.abc import Mapping, Sequence
import os
import sys

# No:
import os, sys
```

### Import Order

1. `from __future__ import` statements
2. Standard library imports
3. Third-party module/package imports
4. Code repository sub-package imports

```python
from __future__ import annotations

import collections
import sys

from absl import app
import tensorflow as tf

from myproject.backend import huxley
from myproject.backend.state_machine import main_loop
```

Sort lexicographically within each group, ignoring case.

---

## 3.14 Statements

**Decision**: Generally only one statement per line.

```python
# Yes:
if foo: bar(foo)

# No:
if foo: bar(foo)
else:   baz(foo)

try:               bar(foo)
except ValueError: baz(foo)
```

---

## 3.15 Getters and Setters

**Decision**: Use getter and setter functions when they provide a meaningful role.

Use when:
- Getting/setting is complex
- The cost is significant

```python
# No: Simple attribute access
class Person:
    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

# Yes: Just make it public
class Person:
    def __init__(self, name: str):
        self.name = name

# Yes: Use setter when there's validation
class Person:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not value:
            raise ValueError('Name cannot be empty')
        self._name = value
```

---

## 3.16 Naming

### Naming Convention Summary

| Type | Convention | Example |
|------|-----------|---------|
| Modules/Packages | `lower_with_under` | `my_module.py` |
| Classes | `CapWords` | `MyClass` |
| Exceptions | `CapWords` | `MyError` |
| Functions | `lower_with_under()` | `my_function()` |
| Global Constants | `CAPS_WITH_UNDER` | `MAX_SIZE` |
| Global Variables | `lower_with_under` | `global_var` |
| Instance Variables | `lower_with_under` | `instance_var` |
| Method Names | `lower_with_under()` | `method_name()` |
| Function Parameters | `lower_with_under` | `param_name` |
| Local Variables | `lower_with_under` | `local_var` |

### Internal/Private

Prepend single underscore:
- `_lower_with_under` for internal module variables/functions
- `_lower_with_under` for protected instance variables/methods
- `_CapWords` for internal classes

### 3.16.1 Names to Avoid

- Single character names (except `i`, `j`, `k` for loops, `e` for exceptions, `f` for files)
- Dashes (`-`) in module/package names
- `__double_leading_and_trailing_underscore__` names (Python reserved)
- Names including type (e.g., `id_to_name_dict`)

### 3.16.3 File Naming

- Use `.py` extension
- Never use dashes (`-`)
- For executables, use symbolic link or wrapper script

### 3.16.5 Mathematical Notation

For math-heavy code, short variable names matching established notation are acceptable:

```python
# pylint: disable=invalid-name
def compute_eigenvalues(A: np.ndarray) -> np.ndarray:
    """Computes eigenvalues using notation from Smith et al. (2020).

    See: https://example.com/paper.pdf
    """
    ...
```

---

## 3.17 Main

**Decision**: Always check `if __name__ == '__main__'` before executing main program.

```python
# With absl:
from absl import app

def main(argv: Sequence[str]):
    # process non-flag arguments
    ...

if __name__ == '__main__':
    app.run(main)

# Without absl:
def main():
    ...

if __name__ == '__main__':
    main()
```

---

## 3.18 Function Length

**Decision**: Prefer small and focused functions.

No hard limit, but if a function exceeds about 40 lines, think about whether it can be broken up.

---

## 3.19 Type Annotations

### 3.19.1 General Rules

```python
# Annotate public APIs
def greeting(name: str) -> str:
    return f'Hello {name}'

# Use Self when needed
from typing import Self

class Builder:
    def add_item(self, item: str) -> Self:
        return self

# Don't annotate __init__ return (or use -> None)
def __init__(self, name: str):
    self.name = name
```

### 3.19.2 Line Breaking

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

### 3.19.3 Forward Declarations

```python
from __future__ import annotations

class MyClass:
    def __init__(self, stack: Sequence[MyClass]) -> None:
        ...
```

### 3.19.4 Default Values

```python
# Yes: Spaces around = with type annotation
def func(a: int = 0) -> int:
    ...

# No:
def func(a:int=0) -> int:
    ...
```

### 3.19.5 NoneType

```python
# Yes: Explicit union with None
def find_user(user_id: int) -> User | None:
    ...

# No: Implicit optional
def find_user(user_id: int = None) -> User:
    ...
```

### 3.19.6 Type Aliases

```python
from typing import TypeAlias

_LossAndGradient: TypeAlias = tuple[tf.Tensor, tf.Tensor]
ComplexTFMap: TypeAlias = Mapping[str, _LossAndGradient]
```

### 3.19.7 Ignoring Types

```python
# Disable type checking for a line
result = some_function()  # type: ignore

# pytype-specific disable
# pytype: disable=attribute-error
```

### 3.19.8 Typing Variables

```python
# Annotated assignment when type is unclear
a: Foo = SomeUndecoratedFunction()
```

### 3.19.9 Tuples vs Lists

```python
a: list[int] = [1, 2, 3]  # List of ints
b: tuple[int, ...] = (1, 2, 3)  # Tuple of ints (variable length)
c: tuple[int, str, float] = (1, "2", 3.5)  # Tuple with specific types
```

### 3.19.10 Type Variables

```python
from typing import TypeVar

_T = TypeVar("_T")

def next(l: list[_T]) -> _T:
    return l.pop()

# Constrained TypeVar
AddableType = TypeVar("AddableType", int, float, str)

def add(a: AddableType, b: AddableType) -> AddableType:
    return a + b
```

### 3.19.11 String Types

Use `str` for strings, `bytes` for binary data. Don't use `typing.Text`.

### 3.19.12 Imports For Typing

```python
# Import symbols directly
from collections.abc import Mapping, Sequence
from typing import Any, Generic, cast

# Prefer abstract types
def transform(data: Sequence[int]) -> Sequence[int]:  # Good
    ...
```

### 3.19.13 Conditional Imports

```python
import typing
if typing.TYPE_CHECKING:
    import sketch

def f(x: "sketch.Sketch"):
    ...
```

### 3.19.14 Circular Dependencies

```python
from typing import Any

some_mod = Any  # some_mod.py imports this module.

def my_method(self, var: "some_mod.SomeType") -> None:
    ...
```

### 3.19.15 Generics

```python
# Yes: Specify type parameters
def get_names(employee_ids: Sequence[int]) -> Mapping[int, str]:
    ...

# No: Missing type parameters
def get_names(employee_ids: Sequence) -> Mapping:
    ...
```

---

## Summary

Key style rules:

1. **80 character** line length
2. **4 spaces** for indentation (never tabs)
3. **Module, function, and class docstrings** required
4. Use **context managers** for resources
5. **Consistent string quotes** within files
6. **Explicit type annotations** for public APIs
7. Follow **naming conventions** strictly
8. **One statement per line**
9. Keep functions **small and focused**
10. Always check `if __name__ == '__main__'`

For language-specific rules, see [Python Language Rules](02-python-language-rules.md).
