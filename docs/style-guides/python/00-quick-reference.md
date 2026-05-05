# Python Style Guide - Quick Reference

> Essential rules for immediate reference. This is a condensed version focusing on the most commonly used patterns.

## Table of Contents

- [Imports](#imports)
- [Formatting](#formatting)
- [Naming](#naming)
- [Type Annotations](#type-annotations)
- [Docstrings](#docstrings)
- [Common Patterns](#common-patterns)
- [Common Mistakes](#common-mistakes)

## Imports

### ✅ DO

```python
# Import modules, not individual classes/functions
from sound.effects import echo
import tensorflow as tf

# Use full package paths
from absl import flags
from doctor.who import jodie

# Import order: future → stdlib → third-party → local
from __future__ import annotations

import os
import sys

import tensorflow as tf
from absl import app

from myproject.backend import huxley

# Multiple imports from typing/collections.abc
from collections.abc import Mapping, Sequence
from typing import Any, Generic
```

### ❌ DON'T

```python
# Don't import individual classes (except typing/collections.abc)
from sound.effects.echo import EchoFilter

# Don't use wildcard imports
from module import *

# Don't combine imports on one line
import os, sys

# Don't use relative imports
from . import sibling
from .. import parent
```

## Formatting

### Line Length: 80 characters max

### ✅ DO

```python
# Use implicit line joining with parentheses
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# Or 4-space hanging indent
foo = long_function_name(
    var_one, var_two, var_three,
    var_four)

# Multi-line conditions
if (width == 0 and height == 0 and
    color == 'red' and emphasis == 'strong'):
    process()

# Dictionary formatting
foo = {
    'long_dictionary_key': value1 +
                          value2,
}
```

### ❌ DON'T

```python
# Don't use backslashes for line continuation
if width == 0 and height == 0 and \
    color == 'red' and emphasis == 'strong':
    process()

# Don't put stuff on first line with hanging indent
foo = long_function_name(var_one, var_two,
    var_three, var_four)
```

### Indentation: 4 spaces (never tabs)

### ✅ DO

```python
# Aligned with opening delimiter
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# 4-space hanging indent
foo = long_function_name(
    var_one, var_two, var_three,
    var_four
)
```

### ❌ DON'T

```python
# 2-space hanging indent
foo = long_function_name(
  var_one, var_two, var_three,
  var_four)

# Mixed indentation
def function():
	pass  # Tab character
```

### Whitespace

### ✅ DO

```python
spam(ham[1], {'eggs': 2}, [])
if x == 4:
    print(x, y)
x, y = y, x
x == 1

# Spaces around = for default values with type hints
def complex(real, imag: float = 0.0):
    return Magic(r=real, i=imag)
```

### ❌ DON'T

```python
spam( ham[ 1 ], { 'eggs': 2 }, [ ] )
if x == 4 :
    print(x , y)
x , y = y , x
x<1

# No spaces around = for default values without type hints
def complex(real, imag = 0.0):
    return Magic(r = real, i = imag)

# No spaces around = for keyword arguments
spam(eggs = 'two')
```

### Blank Lines

- **Two** blank lines between top-level definitions (classes, functions)
- **One** blank line between method definitions
- **No** blank line after a `def` line

## Naming

### Conventions

```python
# Modules and packages
my_module.py
my_package/

# Classes and exceptions
class MyClass:
    pass

class CustomError(Exception):
    pass

# Functions and methods
def my_function():
    pass

def _private_function():
    pass

# Constants
GLOBAL_CONSTANT = 42
_PRIVATE_CONSTANT = 42

# Variables
global_var = 1
_module_private_var = 2
instance_var = 3
local_var = 4
```

### ✅ DO

```python
class DataProcessor:
    def __init__(self):
        self.public_var = 1
        self._protected_var = 2

    def process_data(self, input_data: str) -> str:
        temp_result = self._internal_process(input_data)
        return temp_result

    def _internal_process(self, data: str) -> str:
        return data.upper()

MAX_CONNECTIONS = 100
_INTERNAL_CONSTANT = 50
```

### ❌ DON'T

```python
class data_processor:  # Should be CapWords
    pass

class DataProcessor:
    def ProcessData(self):  # Should be snake_case
        pass

maxConnections = 100  # Should be CAPS_WITH_UNDER
id_to_name_dict = {}  # Avoid type in name
l = []  # Ambiguous single letter
O = 0  # Looks like zero
```

## Type Annotations

### ✅ DO

```python
# Basic types
def greeting(name: str) -> str:
    return f'Hello {name}'

# Modern union syntax (Python 3.10+)
def find_user(user_id: int) -> User | None:
    ...

# Collections
def process_items(items: list[str]) -> dict[str, int]:
    ...

# Prefer abstract types for parameters
def sort_data(data: Sequence[int]) -> list[int]:
    ...

# Type aliases for complex types
from typing import TypeAlias

ConnectionPool: TypeAlias = dict[str, list[Connection]]

# Annotate unclear variable types
result: dict[str, Any] = complex_function()

# Function with multiple parameters
def fetch_data(
    table_handle: Table,
    keys: Sequence[str],
    timeout: float = 30.0,
) -> dict[str, Any]:
    ...
```

### ❌ DON'T

```python
# Implicit optional (deprecated)
def find_user(user_id: int = None) -> User:  # Should be int | None
    ...

# Old-style union
def find_user(user_id: int) -> Union[User, None]:  # Use User | None
    ...

# Missing return type
def process():  # Add -> None or actual return type
    ...

# Concrete types when abstract would be better
def sort_data(data: list[int]) -> list[int]:  # Use Sequence for input
    ...
```

## Docstrings

### Module Docstring

### ✅ DO

```python
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line. The rest of this docstring should contain an
overall description of the module or program.

Typical usage example:

  foo = ClassFoo()
  bar = foo.function_bar()
"""
```

### Function Docstring

### ✅ DO

```python
def fetch_data(
    table: Table,
    keys: Sequence[str],
    require_all: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Fetches data from the table.

    Retrieves rows pertaining to the given keys from the Table instance.

    Args:
        table: An open Table instance.
        keys: A sequence of strings representing the keys to fetch.
        require_all: If True, only returns if all keys are found.

    Returns:
        A dict mapping keys to the corresponding data fetched.
        Each row is represented as a tuple of strings.

    Raises:
        IOError: An error occurred accessing the table.
        ValueError: If require_all is True and not all keys found.
    """
```

### ❌ DON'T

```python
# Missing docstring for public API
def fetch_data(table, keys):
    ...

# Vague or redundant docstring
def fetch_data(table, keys):
    """Fetches data."""  # Too brief, no details
    ...

# Wrong docstring format
def fetch_data(table, keys):
    '''Single quotes for docstring'''  # Use """
    ...
```

### Class Docstring

### ✅ DO

```python
class DataCache:
    """Caches data for quick retrieval.

    This class maintains an in-memory cache with LRU eviction.

    Attributes:
        capacity: Maximum number of items to cache.
        hit_count: Number of cache hits since creation.
    """

    def __init__(self, capacity: int = 100):
        """Initializes the cache.

        Args:
            capacity: Maximum cache size.
        """
        self.capacity = capacity
        self.hit_count = 0
```

### ❌ DON'T

```python
class DataCache:
    """Class that caches data."""  # Avoid "Class that..."

    def __init__(self, capacity: int = 100):
        # Missing __init__ docstring
        self.capacity = capacity
```

## Common Patterns

### Default Argument Values

### ✅ DO

```python
def foo(a, b=None):
    if b is None:
        b = []
    b.append(a)
    return b

def foo(a, b: Sequence = ()):  # Empty tuple OK (immutable)
    ...
```

### ❌ DON'T

```python
def foo(a, b=[]):  # Mutable default value!
    b.append(a)
    return b
```

### True/False Evaluations

### ✅ DO

```python
if not users:
    print('no users')

if foo:
    bar()

if value is None:
    handle_none()

# For integers, be explicit
if count == 0:
    handle_zero()
```

### ❌ DON'T

```python
if len(users) == 0:  # Use implicit false
    print('no users')

if value == None:  # Use 'is None'
    handle_none()

if not count:  # Ambiguous: 0 or None?
    handle_zero()
```

### Comprehensions

### ✅ DO

```python
result = [mapping_expr for value in iterable if filter_expr]

result = [
    transform(value)
    for value in iterable
    if is_valid(value)
]

# For nested loops, use regular loops
result = []
for x in range(10):
    for y in range(5):
        if x * y > 10:
            result.append((x, y))
```

### ❌ DON'T

```python
# Multiple for clauses in comprehension
result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]

# Too complex
result = [
    x for x in [y for y in items if y.valid]
    if x.status == 'active' and x.count > 0
]
```

### Exception Handling

### ✅ DO

```python
try:
    risky_operation()
except SpecificError as e:
    logger.error('Failed: %r', e)
    raise
finally:
    cleanup()

# Use built-in exceptions
if value < 0:
    raise ValueError(f'Value must be positive, got {value}')
```

### ❌ DON'T

```python
# Bare except
try:
    risky_operation()
except:  # Too broad!
    pass

# Using assert for validation
assert value >= 0, 'Value must be positive'  # Use ValueError

# Too broad try block
try:
    result1 = operation1()
    result2 = operation2()  # Which one failed?
    result3 = operation3()
except Exception:
    handle_error()
```

### Context Managers

### ✅ DO

```python
with open('file.txt') as f:
    data = f.read()

# Multiple contexts
with (
    open('input.txt') as input_file,
    open('output.txt', 'w') as output_file,
):
    process(input_file, output_file)
```

### ❌ DON'T

```python
f = open('file.txt')
try:
    data = f.read()
finally:
    f.close()
```

### String Formatting

### ✅ DO

```python
# f-strings (preferred for readability)
message = f'Hello {name}, you have {count} messages'

# % formatting (good for logging)
logger.info('Processing item %d of %d', current, total)

# .format() (also acceptable)
message = 'Hello {}, you have {} messages'.format(name, count)

# String accumulation
items = []
for item in collection:
    items.append(str(item))
result = ''.join(items)
```

### ❌ DON'T

```python
# Don't concatenate with +
message = 'Hello ' + name + ', you have ' + str(count) + ' messages'

# Don't use f-strings for logging
logger.info(f'Processing item {current} of {total}')

# Don't accumulate with + in loops
result = ''
for item in collection:
    result += str(item)  # Quadratic time!
```

## Common Mistakes

### Mistake 1: Mutable Default Arguments

```python
# ❌ DON'T
def add_to_list(item, target_list=[]):
    target_list.append(item)
    return target_list

# ✅ DO
def add_to_list(item, target_list=None):
    if target_list is None:
        target_list = []
    target_list.append(item)
    return target_list
```

### Mistake 2: Comparing to None with ==

```python
# ❌ DON'T
if value == None:
    ...

# ✅ DO
if value is None:
    ...
```

### Mistake 3: Using len() for Empty Check

```python
# ❌ DON'T
if len(items) == 0:
    print('empty')

if len(items) > 0:
    process(items)

# ✅ DO
if not items:
    print('empty')

if items:
    process(items)
```

### Mistake 4: Catching Too Broad Exceptions

```python
# ❌ DON'T
try:
    data = fetch_data()
except:  # Catches everything!
    handle_error()

# ❌ DON'T
try:
    data = fetch_data()
except Exception:  # Still too broad
    handle_error()

# ✅ DO
try:
    data = fetch_data()
except (NetworkError, TimeoutError) as e:
    logger.error('Fetch failed: %r', e)
    handle_error()
```

### Mistake 5: Not Using Type Hints

```python
# ❌ DON'T (for public APIs)
def process_data(data, config):
    ...

# ✅ DO
def process_data(
    data: dict[str, Any],
    config: Config | None = None,
) -> ProcessResult:
    ...
```

### Mistake 6: Complex Comprehensions

```python
# ❌ DON'T
result = [
    transform(x, y)
    for x in collection1 if x.valid
    for y in collection2 if y.matches(x)
]

# ✅ DO
result = []
for x in collection1:
    if not x.valid:
        continue
    for y in collection2:
        if y.matches(x):
            result.append(transform(x, y))
```

### Mistake 7: Inconsistent String Quotes

```python
# ❌ DON'T (mixing quotes without reason)
name = "John"
message = 'Hello'
greeting = "Hi"

# ✅ DO (consistent within file)
name = 'John'
message = 'Hello'
greeting = 'Hi'

# ✅ DO (exception: avoid escaping)
message = "He said 'hello'"
path = 'C:\\Users\\name'  # Or use raw string r'C:\Users\name'
```

### Mistake 8: Missing Docstrings

```python
# ❌ DON'T (for public API)
def calculate_total(items, tax_rate):
    return sum(items) * (1 + tax_rate)

# ✅ DO
def calculate_total(items: list[float], tax_rate: float) -> float:
    """Calculates the total with tax applied.

    Args:
        items: List of item prices.
        tax_rate: Tax rate as a decimal (e.g., 0.08 for 8%).

    Returns:
        Total price including tax.
    """
    return sum(items) * (1 + tax_rate)
```

## Quick Checklist

Before committing Python code, verify:

- [ ] All imports are at the top, properly ordered
- [ ] Line length ≤ 80 characters
- [ ] 4-space indentation (no tabs)
- [ ] Consistent string quotes throughout file
- [ ] Module docstring at top of file
- [ ] Public functions/classes have docstrings
- [ ] Type hints on function signatures (especially public APIs)
- [ ] No mutable default arguments
- [ ] Using `is`/`is not` for None comparisons
- [ ] Using implicit false for empty checks
- [ ] Proper naming conventions (snake_case, CapWords, CAPS_WITH_UNDER)
- [ ] `pylint` runs without errors
- [ ] Code formatted with Black/Pyink (if using)

## For More Details

- [Full Summary](README.md)
- [Background](01-background.md)
- [Python Language Rules](02-python-language-rules.md)
- [Python Style Rules](03-python-style-rules.md)
