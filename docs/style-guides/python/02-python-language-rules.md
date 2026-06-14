# Python Language Rules

> Source: [Google Python Style Guide - Section 2](https://google.github.io/styleguide/pyguide.html)

This section covers rules about using Python language features. These are guidelines about what Python features to use, when to use them, and how to use them properly.

## Table of Contents

- [2.1 Lint](#21-lint)
- [2.2 Imports](#22-imports)
- [2.3 Packages](#23-packages)
- [2.4 Exceptions](#24-exceptions)
- [2.5 Mutable Global State](#25-mutable-global-state)
- [2.6 Nested/Local/Inner Classes and Functions](#26-nestedlocalinner-classes-and-functions)
- [2.7 Comprehensions & Generator Expressions](#27-comprehensions--generator-expressions)
- [2.8 Default Iterators and Operators](#28-default-iterators-and-operators)
- [2.9 Generators](#29-generators)
- [2.10 Lambda Functions](#210-lambda-functions)
- [2.11 Conditional Expressions](#211-conditional-expressions)
- [2.12 Default Argument Values](#212-default-argument-values)
- [2.13 Properties](#213-properties)
- [2.14 True/False Evaluations](#214-truefalse-evaluations)
- [2.16 Lexical Scoping](#216-lexical-scoping)
- [2.17 Function and Method Decorators](#217-function-and-method-decorators)
- [2.18 Threading](#218-threading)
- [2.19 Power Features](#219-power-features)
- [2.20 Modern Python: from __future__ imports](#220-modern-python-from-__future__-imports)
- [2.21 Type Annotated Code](#221-type-annotated-code)

---

## 2.1 Lint

**Decision**: Make sure you run `pylint` on your code.

### Definition

`pylint` is a tool for finding bugs and style problems in Python source code. It finds problems that are typically caught by a compiler for less dynamic languages like C and C++.

### Pros

- Catches easy-to-miss errors like typos
- Detects using-vars-before-assignment
- Identifies unused imports and variables
- Enforces style consistency

### Cons

`pylint` isn't perfect. Sometimes you'll need to:
- Write around it
- Suppress its warnings
- Fix it

### Usage

Suppress warnings if they are inappropriate so that other issues are not hidden:

```python
# Line-level suppression with explanation
def do_PUT(self):  # pylint: disable=invalid-name
    """WSGI requires this specific name."""
    ...

# Suppress multiple warnings
result = data.compute()  # pylint: disable=no-member,unused-variable
```

**Guidelines:**
- Use symbolic names (`empty-docstring`), not codes
- Add explanations if the reason isn't clear from the symbolic name
- Keep suppressions as narrow as possible
- Prefer `pylint: disable` to the deprecated `pylint: disable-msg`

**Handling unused arguments:**

```python
# Preferred: Delete the variable with explanation
def viking_cafe_order(spam: str, beans: str, eggs: str | None = None) -> str:
    del beans, eggs  # Unused by vikings.
    return spam + spam + spam

# Acceptable but discouraged: underscore prefix
def viking_cafe_order(spam: str, _beans: str, _eggs: str | None = None) -> str:
    return spam + spam + spam
```

---

## 2.2 Imports

**Decision**: Use `import` statements for packages and modules only, not for individual types, classes, or functions.

### Definition

Reusability mechanism for sharing code from one module to another.

### Pros

- Simple namespace management
- Source of each identifier is clear: `x.Obj` means `Obj` is defined in module `x`

### Cons

- Module names can still collide
- Some module names are inconveniently long

### Guidelines

**Use `from x import y as z` when:**
- `y` is too generic in your code context
- `y` is an inconveniently long name
- `y` conflicts with a common parameter name in public APIs
- `y` conflicts with a top-level name in the current module
- Two modules named `y` need to be imported

```python
# Yes:
from sound.effects import echo
echo.EchoFilter(input, output, delay=0.7, atten=4)

# Yes: Renaming for clarity
from storage.file_system import options as fs_options

# No: Importing individual classes
from sound.effects.echo import EchoFilter
```

**Never use relative imports:**

```python
# No:
from . import sibling
from .. import parent

# Yes:
from myproject.subpackage import sibling
from myproject import parent
```

### Exemptions

Symbols from these modules can be imported directly for static analysis and type checking:

- `typing` module
- `typing_extensions` module
- `collections.abc` module

```python
# Allowed:
from typing import Any, Optional, TypeVar
from collections.abc import Mapping, Sequence
```

---

## 2.3 Packages

**Decision**: Import each module using the full pathname location of the module.

### Pros

- Avoids conflicts in module names
- Prevents incorrect imports due to unexpected module search paths
- Makes it easier to find modules

### Cons

- Makes it harder to deploy code (not really a problem with modern deployment)

### Examples

```python
# Yes: Reference with complete name (verbose)
import absl.flags
from doctor.who import jodie

_FOO = absl.flags.DEFINE_string(...)

# Yes: Reference with module name (common)
from absl import flags
from doctor.who import jodie

_FOO = flags.DEFINE_string(...)

# No: Unclear what module will be imported
# (depends on sys.path)
import jodie
```

**Important**: The directory the main binary is located in should not be assumed to be in `sys.path`. Code should assume that `import jodie` refers to a third-party or top-level package named `jodie`, not a local `jodie.py`.

---

## 2.4 Exceptions

**Decision**: Exceptions are allowed but must be used carefully.

### Definition

Exceptions are a means of breaking out of normal control flow to handle errors or other exceptional conditions.

### Pros

- Normal operation code is not cluttered by error-handling code
- Control flow can skip multiple frames when a condition occurs
- No need to plumb error codes through multiple function calls

### Cons

- May cause confusing control flow
- Easy to miss error cases when making library calls

### Guidelines

#### Use Built-in Exception Classes

```python
# Yes:
if minimum < 1024:
    raise ValueError(f'Min. port must be at least 1024, not {minimum}.')

# Yes:
if port is None:
    raise ConnectionError(
        f'Could not connect to service on port {minimum} or higher.')
```

#### Don't Use assert for Validation

```python
# No: Don't use assert for precondition validation
def connect_to_next_port(self, minimum: int) -> int:
    assert minimum >= 1024, 'Minimum port must be at least 1024.'
    port = self._find_next_open_port(minimum)
    assert port is not None
    return port

# Yes: Use proper exceptions
def connect_to_next_port(self, minimum: int) -> int:
    """Connects to the next available port.

    Args:
        minimum: A port value greater or equal to 1024.

    Returns:
        The new minimum port.

    Raises:
        ConnectionError: If no available port is found.
    """
    if minimum < 1024:
        raise ValueError(f'Min. port must be at least 1024, not {minimum}.')
    port = self._find_next_open_port(minimum)
    if port is None:
        raise ConnectionError(
            f'Could not connect to service on port {minimum} or higher.')
    assert port >= minimum, (
        f'Unexpected port {port} when minimum was {minimum}.')
    return port
```

**Note**: `assert` is okay for:
- Test code (pytest)
- Non-critical invariant checks that don't affect program logic

#### Never Use Catch-All except

```python
# No:
try:
    risky_operation()
except:  # Catches everything!
    handle_error()

# No:
try:
    risky_operation()
except Exception:  # Still too broad
    handle_error()

# Yes: Be specific
try:
    risky_operation()
except (NetworkError, TimeoutError) as e:
    logger.error('Operation failed: %r', e)
    handle_error()
```

**Exception**: Catch-all is allowed when:
- Creating an isolation point where exceptions are recorded and suppressed (e.g., protecting a thread)
- Re-raising the exception

#### Minimize try Block Size

```python
# No: Too much in try block
try:
    data = fetch_data()
    result = process_data(data)
    save_result(result)
    log_completion()
except Exception:
    handle_error()

# Yes: Minimal try block
data = fetch_data()
try:
    result = process_data(data)
except ProcessingError:
    handle_error()
else:
    save_result(result)
finally:
    log_completion()
```

#### Custom Exceptions

```python
# Yes:
class DataValidationError(ValueError):
    """Raised when data validation fails."""

# No: Redundant naming
class FooError(Exception):
    """Foo error."""  # Should be FooException or just Error
```

**Guidelines for custom exceptions:**
- Must inherit from an existing exception class
- Names should end in `Error`
- Don't introduce repetition (not `foo.FooError`)

---

## 2.5 Mutable Global State

**Decision**: Avoid mutable global state.

### Definition

Module-level values or class attributes that can get mutated during program execution.

### Pros

Occasionally useful for caching, registries, etc.

### Cons

- Breaks encapsulation
- Makes testing difficult
- Makes concurrent execution problematic
- Can change module behavior during import

### Guidelines

```python
# No: Mutable global
cached_data = {}

def get_data(key):
    if key not in cached_data:
        cached_data[key] = fetch_data(key)
    return cached_data[key]

# Yes: Encapsulated state
class DataCache:
    def __init__(self):
        self._cache = {}

    def get_data(self, key):
        if key not in self._cache:
            self._cache[key] = fetch_data(key)
        return self._cache[key]

# Yes: Module-level constants (immutable)
_MAX_HOLY_HANDGRENADE_COUNT = 3
SIR_LANCELOTS_FAVORITE_COLOR = "blue"
```

**If you must use mutable globals:**
1. Make them internal by prepending `_`
2. Provide public functions/methods for access
3. Document why mutable global state is necessary

---

## 2.6 Nested/Local/Inner Classes and Functions

**Decision**: Nested local functions or classes are fine when used to close over a local variable. Inner classes are fine.

### Definition

A class can be defined inside a method, function, or class. A function can be defined inside a method or function. Nested functions have read-only access to variables defined in enclosing scopes.

### Pros

- Allows definition of utility classes and functions in limited scope
- Very ADT-like
- Commonly used for implementing decorators

### Cons

- Nested functions and classes cannot be directly tested
- Nesting can make the outer function longer and less readable

### Guidelines

```python
# Yes: Closing over a local value
def create_multiplier(factor: int) -> Callable[[int], int]:
    """Returns a function that multiplies by factor."""
    def multiply(x: int) -> int:
        return x * factor
    return multiply

# Yes: Helper class with limited scope
def process_data(items: list[Item]) -> ProcessedData:
    class ItemProcessor:
        def __init__(self):
            self.count = 0

        def process(self, item: Item) -> ProcessedItem:
            self.count += 1
            return item.transform()

    processor = ItemProcessor()
    results = [processor.process(item) for item in items]
    return ProcessedData(results, processor.count)

# No: Nested just to hide from users
def public_function():
    def _helper():  # Should be module-level _helper()
        ...
    return _helper()
```

**Avoid nesting except when closing over a local value.** If you just want to hide a function, prefix it with `_` at module level instead.

---

## 2.7 Comprehensions & Generator Expressions

**Decision**: Okay to use for simple cases.

### Definition

List, dict, and set comprehensions as well as generator expressions provide a concise way to create container types and iterators.

### Pros

- Simple comprehensions can be clearer than other techniques
- Generator expressions are very efficient
- Avoid creating intermediate lists

### Cons

Complicated comprehensions can be hard to read.

### Guidelines

**Allowed patterns:**

```python
# Yes: Simple comprehension
result = [mapping_expr for value in iterable if filter_expr]

# Yes: Multiline for readability
result = [
    is_valid(metric={'key': value})
    for value in interesting_iterable
    if a_longer_filter_expression(value)
]

# Yes: Descriptive name, clear logic
descriptive_name = [
    transform({'key': key, 'value': value}, color='black')
    for key, value in generate_iterable(some_input)
    if complicated_condition_is_met(key, value)
]

# Yes: Use regular loop for nested iteration
result = []
for x in range(10):
    for y in range(5):
        if x * y > 10:
            result.append((x, y))

# Yes: Dict comprehension
return {
    x: complicated_transform(x)
    for x in long_generator_function(parameter)
    if x is not None
}

# Yes: Generator expression
return (x**2 for x in range(10))

# Yes: Set comprehension
unique_names = {user.name for user in users if user is not None}
```

**Not allowed:**

```python
# No: Multiple for clauses
result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]

# No: Too complex
return (
    (x, y, z)
    for x in range(5)
    for y in range(5)
    if x != y
    for z in range(5)
    if y != z
)
```

**Rule**: No multiple `for` clauses or complex filter expressions. Optimize for readability.

---

## 2.8 Default Iterators and Operators

**Decision**: Use default iterators and operators for types that support them.

### Definition

Container types like dictionaries and lists define default iterators and membership test operators (`in` and `not in`).

### Pros

- Simple and efficient
- Express operations directly
- Generic: works with any type that supports the operation

### Cons

Can't tell object types by reading method names (unless type annotated).

### Guidelines

```python
# Yes:
for key in adict:
    ...

if obj in alist:
    ...

for line in afile:
    ...

for k, v in adict.items():
    ...

# No:
for key in adict.keys():  # .keys() is unnecessary
    ...

for line in afile.readlines():  # Creates intermediate list
    ...
```

**Note**: Don't mutate a container while iterating over it.

---

## 2.9 Generators

**Decision**: Use generators as needed.

### Definition

A generator function returns an iterator that yields a value each time it executes a `yield` statement. After yielding, the runtime state is suspended until the next value is needed.

### Pros

- Simpler code (state and control flow preserved automatically)
- Uses less memory than creating entire list at once
- Lazy evaluation

### Cons

Local variables won't be garbage collected until the generator is consumed or itself garbage collected.

### Guidelines

```python
# Yes:
def fibonacci(n: int) -> Iterator[int]:
    """Yields first n Fibonacci numbers.

    Yields:
        Fibonacci numbers.
    """
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Yes: Use context manager for cleanup
from contextlib import contextmanager

@contextmanager
def open_database(url: str) -> Iterator[Database]:
    """Context manager for database connection.

    Yields:
        Open database connection.
    """
    db = Database.connect(url)
    try:
        yield db
    finally:
        db.close()
```

**Documentation**: Use `Yields:` rather than `Returns:` in docstrings for generator functions.

**Resource management**: If the generator manages expensive resources, make sure to force cleanup, preferably using a context manager (PEP-0533).

---

## 2.10 Lambda Functions

**Decision**: Okay for one-liners. Prefer generator expressions over `map()` or `filter()` with a `lambda`.

### Definition

Lambdas define anonymous functions in an expression, as opposed to a statement.

### Pros

Convenient for simple operations.

### Cons

- Harder to read and debug than local functions
- No names means stack traces are more difficult
- Limited to expressions only

### Guidelines

```python
# Yes: Simple lambda
employees.sort(key=lambda emp: emp.last_name)

# Yes: Better alternative using operator module
import operator
employees.sort(key=operator.attrgetter('last_name'))

# Yes: Generator expression preferred
squares = (x**2 for x in range(10))

# No: Lambda too complex
result = map(lambda x: x.strip().upper() if x else '', items)

# Yes: Use regular function instead
def normalize(x: str) -> str:
    return x.strip().upper() if x else ''

result = map(normalize, items)

# Or better yet:
result = [normalize(x) for x in items]
```

**Rule**: If the lambda is longer than 60-80 characters or spans multiple concepts, define it as a regular nested function.

---

## 2.11 Conditional Expressions

**Decision**: Okay for simple cases.

### Definition

Conditional expressions (ternary operator) provide shorter syntax for if statements: `x = 1 if cond else 2`.

### Pros

Shorter and more convenient than an if statement.

### Cons

- May be harder to read than an if statement
- Condition may be difficult to locate if expression is long

### Guidelines

```python
# Yes:
one_line = 'yes' if predicate(value) else 'no'

slightly_split = ('yes' if predicate(value)
                  else 'no, nein, nyet')

the_longest_ternary_style_that_can_be_done = (
    'yes, true, affirmative, confirmed, correct'
    if predicate(value)
    else 'no, false, negative, nay')

# No: Bad line breaking
bad_line_breaking = ('yes' if predicate(value) else
                     'no')

# No: Too complex
portion_too_long = ('yes'
                    if some_long_module.some_long_predicate_function(
                        really_long_variable_name)
                    else 'no, false, negative, nay')
```

**Rule**: Each portion must fit on one line. Use complete if statement when things get complicated.

---

## 2.12 Default Argument Values

**Decision**: Okay in most cases.

### Definition

You can specify values for variables at the end of a function's parameter list: `def foo(a, b=0):`.

### Pros

- Easy way to override defaults for rare exceptions
- Provides "function overloading" behavior

### Cons

Default arguments are evaluated once at module load time. Mutable objects as defaults can cause problems.

### Guidelines

```python
# Yes:
def foo(a, b=None):
    if b is None:
        b = []
    b.append(a)
    return b

# Yes: With type hints
def foo(a, b: Sequence | None = None):
    if b is None:
        b = []
    b.append(a)
    return b

# Yes: Empty tuple OK (immutable)
def foo(a, b: Sequence = ()):
    ...

# No: Mutable default value
def foo(a, b=[]):
    b.append(a)
    return b

# No: Evaluation at module load
def foo(a, b=time.time()):
    ...

# No: Flag value evaluation at module load
from absl import flags
_FOO = flags.DEFINE_string(...)

def foo(a, b=_FOO.value):  # sys.argv not yet parsed!
    ...

# No: Mutable type hint default
def foo(a, b: Mapping = {}):
    ...
```

**Rule**: Never use mutable objects as default values in the function or method definition.

---

## 2.13 Properties

**Decision**: Properties may be used to control getting or setting attributes that require trivial computations or logic.

### Definition

A way to wrap method calls for getting and setting an attribute as standard attribute access.

### Pros

- Maintains public interface when internals evolve
- Allows calculations to be lazy
- Can make an attribute read-only
- Provides attribute access API rather than getter/setter methods

### Cons

- Can be confusing for subclasses
- Can hide side effects (like operator overloading)

### Guidelines

```python
# Yes: Simple property
class Square:
    def __init__(self, side: float):
        self._side = side

    @property
    def area(self) -> float:
        """The area of the square."""
        return self._side ** 2

    @property
    def side(self) -> float:
        """The side length."""
        return self._side

    @side.setter
    def side(self, value: float) -> None:
        if value < 0:
            raise ValueError('Side must be non-negative')
        self._side = value

# No: Unnecessary property
class Person:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:  # Just make it public!
        return self._name

# Yes: Make it public instead
class Person:
    def __init__(self, name: str):
        self.name = name
```

**Rules:**
- Properties must match expectations of regular attribute access: cheap, straightforward, unsurprising
- Use `@property` decorator (don't manually implement descriptor)
- Don't use properties for computations a subclass may want to override

---

## 2.14 True/False Evaluations

**Decision**: Use the "implicit" false if at all possible.

### Definition

Python evaluates certain values as `False` in a boolean context: `0`, `None`, `[]`, `{}`, `''`.

### Pros

- Easier to read and less error-prone
- Usually faster

### Cons

May look strange to C/C++ developers.

### Guidelines

```python
# Yes:
if not users:
    print('no users')

if foo:
    bar()

if i % 10 == 0:
    self.handle_multiple_of_ten()

def f(x=None):
    if x is None:
        x = []

# No:
if len(users) == 0:
    print('no users')

if foo != []:
    bar()

if not i % 10:
    self.handle_multiple_of_ten()

def f(x=None):
    x = x or []
```

**Special cases:**

1. **Always use `if foo is None:` or `if foo is not None:`** to check for None
2. **Never compare a boolean variable to `False` using `==`**. Use `if not x:` instead
3. **For sequences**, use the fact that empty sequences are false: `if seq:` and `if not seq:`
4. **For integers**, be explicit when checking for zero to avoid accidentally handling `None` as 0

**Note about NumPy**: NumPy arrays may raise an exception in implicit boolean context. Use `.size` attribute: `if not users.size`.

---

## 2.16 Lexical Scoping

**Decision**: Okay to use.

### Definition

A nested Python function can refer to variables defined in enclosing functions, but cannot assign to them. Variable bindings are resolved using lexical scoping.

### Pros

- Results in clearer, more elegant code
- Comforting to functional programming enthusiasts

### Cons

Can lead to confusing bugs.

### Example

```python
# Yes:
def get_adder(summand1: float) -> Callable[[float], float]:
    """Returns a function that adds numbers to a given number."""
    def adder(summand2: float) -> float:
        return summand1 + summand2

    return adder

# Confusing example (bug):
i = 4
def foo(x: Iterable[int]):
    def bar():
        print(i, end='')
    # ...lots of code...
    for i in x:  # Ah, i *is* local to foo, so this is what bar sees
        print(i, end='')
    bar()

# foo([1, 2, 3]) prints: 1 2 3 3 (not 1 2 3 4!)
```

**Decision**: Okay to use, but be aware of potential scoping issues.

---

## 2.17 Function and Method Decorators

**Decision**: Use decorators judiciously when there is a clear advantage. Avoid `staticmethod` and limit use of `classmethod`.

### Definition

Decorators allow you to wrap functions or methods to transform their behavior.

```python
class C:
    @my_decorator
    def method(self):
        # method body ...
```

### Pros

- Elegantly specifies transformations
- Eliminates repetitive code
- Enforces invariants

### Cons

- Can perform surprising implicit behavior
- Execute at import time
- Failures in decorator code are hard to recover from

### Guidelines

```python
# Yes: Clear decorator usage
@contextmanager
def database_transaction(db: Database) -> Iterator[Transaction]:
    """Context manager for database transactions."""
    transaction = db.begin()
    try:
        yield transaction
        transaction.commit()
    except Exception:
        transaction.rollback()
        raise

# Yes: Well-documented decorator
def retry(max_attempts: int = 3):
    """Decorator that retries a function on exception.

    Args:
        max_attempts: Maximum number of retry attempts.

    Returns:
        Decorated function that retries on failure.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning('Attempt %d failed: %r', attempt + 1, e)
            return wrapper
        return decorator

# No: Don't use staticmethod
class MyClass:
    @staticmethod
    def helper():  # Should be module-level function
        ...

# Yes: Use module-level function
def helper():
    ...

# OK: Use classmethod for named constructors
class MyClass:
    @classmethod
    def from_config(cls, config: Config) -> MyClass:
        """Creates instance from configuration."""
        return cls(config.value1, config.value2)
```

**Rules:**
- Follow import and naming guidelines
- Clearly document that function is a decorator
- Write unit tests for decorators
- Avoid external dependencies (files, network) in decorators
- Never use `staticmethod` (use module-level function)
- Use `classmethod` only for named constructors or class-specific state modification

---

## 2.18 Threading

**Decision**: Do not rely on the atomicity of built-in types.

### Guidelines

While Python's built-in data types like dictionaries appear to have atomic operations, there are corner cases where they aren't atomic (e.g., if `__hash__` or `__eq__` are implemented as Python methods).

```python
# Use Queue for thread communication
from queue import Queue

work_queue: Queue[WorkItem] = Queue()

# Use threading locks
import threading

lock = threading.Lock()

with lock:
    # Critical section
    shared_data.update(value)

# Prefer condition variables
condition = threading.Condition()

with condition:
    while not predicate():
        condition.wait()
    # Do work
    condition.notify_all()
```

**Don't rely on**:
- Atomic variable assignment
- Atomicity of built-in types

**Do use**:
- `queue.Queue` for inter-thread communication
- `threading.Lock`, `threading.RLock` for synchronization
- `threading.Condition` for complex coordination

---

## 2.19 Power Features

**Decision**: Avoid these features.

### Definition

Python is extremely flexible and provides many advanced features like:
- Custom metaclasses
- Access to bytecode
- On-the-fly compilation
- Dynamic inheritance
- Object reparenting
- Import hacks
- Reflection (some uses of `getattr()`)
- Modification of system internals
- `__del__` methods implementing customized cleanup

### Pros

Powerful language features that can make code more compact.

### Cons

- Very tempting to use when not necessary
- Harder to read, understand, and debug
- Tends to be more difficult than straightforward code

### Decision

**Avoid these features in your code.**

**Exception**: Standard library modules that internally use these features are okay to use (e.g., `abc.ABCMeta`, `dataclasses`, `enum`).

---

## 2.20 Modern Python: from __future__ imports

**Decision**: Use of `from __future__ import` statements is encouraged.

### Definition

Being able to turn on modern features via `from __future__ import` statements allows early use of features from expected future Python versions.

### Pros

- Makes runtime version upgrades smoother
- Enables per-file modern syntax adoption
- Modern code is more maintainable

### Cons

May not work on very old interpreter versions.

### Guidelines

```python
# Encouraged: Enable modern annotations
from __future__ import annotations

def process(items: list[str]) -> dict[str, int]:
    """Process items.

    This works even in Python 3.7+ thanks to __future__ import.
    """
    ...

# For Python 3.5 compatibility
from __future__ import generator_stop
```

**Rule**: Use `from __future__ import` to enable modern syntax. Don't remove these imports until confident code only runs in sufficiently modern environment.

---

## 2.21 Type Annotated Code

**Decision**: Annotate Python code with type hints. Type-check the code with tools like [pytype](https://github.com/google/pytype).

### Definition

Type annotations (or "type hints") provide type information for function/method arguments and return values:

```python
def func(a: int) -> list[int]:
    ...

# Variable annotations
a: SomeType = some_func()
```

### Pros

- Improves readability and maintainability
- Type checker converts many runtime errors to build-time errors
- Reduces ability to use problematic power features

### Cons

- Must keep type declarations up to date
- May see type errors for seemingly valid code
- Type checkers may limit use of power features

### Guidelines

```python
# Yes: Annotate public APIs
def fetch_data(
    table: Table,
    keys: Sequence[str],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetches data from table."""
    ...

# Yes: Annotate when type is unclear
result: dict[str, list[int]] = complex_function()

# Yes: Modern union syntax
def find_user(user_id: int) -> User | None:
    ...

# Don't annotate self or cls
class MyClass:
    def method(self) -> int:  # No type for self
        ...

    @classmethod
    def create(cls) -> MyClass:  # No type for cls
        ...

# Use Self when needed
from typing import Self

class Builder:
    def add_item(self, item: str) -> Self:
        """Returns self for chaining."""
        self.items.append(item)
        return self

# Don't annotate __init__ return
def __init__(self) -> None:  # Unnecessary, but acceptable
    ...
```

**When to annotate:**
- All public APIs
- Code that is hard to understand
- Code prone to type-related errors
- Mature, stable code
- At least the most important functions

**Type checking tools:**
- [pytype](https://github.com/google/pytype) - Google's type checker
- [mypy](http://mypy-lang.org/) - Popular alternative

---

## Summary

This section covered essential Python language rules:

1. **Run pylint** on all code
2. **Import modules**, not individual classes
3. **Use full package paths** for imports
4. **Handle exceptions** carefully and specifically
5. **Avoid mutable global state**
6. **Use comprehensions** for simple cases only
7. **Use default iterators** and operators
8. **Leverage generators** for memory efficiency
9. **Keep lambdas simple** or use named functions
10. **Don't use mutable default arguments**
11. **Use properties** judiciously
12. **Leverage implicit false** for cleaner conditionals
13. **Use decorators** when there's clear advantage
14. **Avoid power features** unless absolutely necessary
15. **Annotate code with type hints**, especially public APIs

For formatting and documentation standards, see [Python Style Rules](03-python-style-rules.md).
