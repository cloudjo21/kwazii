# Parting Words

> Source: [Google Python Style Guide - Section 4](https://google.github.io/styleguide/pyguide.html)

## BE CONSISTENT

If you're editing code, take a few minutes to look at the code around you and determine its style.

**If they use:**
- `_idx` suffixes in index variable names → **you should too**
- Little boxes of hash marks around comments → **make your comments have little boxes too**
- Specific string quote character (`'` or `"`) → **use the same**
- Particular import ordering → **follow the same pattern**

## The Purpose of Style Guidelines

The point of having style guidelines is to have a **common vocabulary of coding** so people can concentrate on **what you're saying** rather than on **how you're saying it**.

We present global style rules here so people know the vocabulary, but **local style is also important**.

If code you add to a file looks **drastically different** from the existing code around it, it throws readers out of their rhythm when they go to read it. Avoid this.

## Limits to Consistency

However, there are limits to consistency:

1. **Local consistency matters more** for choices unspecified by the global style
2. **Consistency should not be used as a justification** to do things in an old style without considering:
   - The benefits of the new style
   - The tendency of the codebase to converge on newer styles over time
3. **Don't blindly maintain consistency** if it means using deprecated or problematic patterns

## When to Deviate from Consistency

### Valid Reasons

- **New style is objectively better**: More readable, safer, or more maintainable
- **Project-wide decision**: Team has agreed to migrate to new pattern
- **Fixing bugs or security issues**: Old pattern was problematic
- **Modern Python features**: Using new language features appropriately

### Invalid Reasons

- **Personal preference**: "I like it this way"
- **Code golf**: Making code shorter without improving clarity
- **Showing off**: Using advanced features unnecessarily
- **Laziness**: Not wanting to follow existing patterns

## Practical Application

### Example 1: Joining a New Project

```python
# Existing codebase uses this style:
class UserManager:
    def get_user_by_id(self, user_id):
        return self._database.fetch_user(user_id)

# Your code should match:
class OrderManager:
    def get_order_by_id(self, order_id):  # Match the naming pattern
        return self._database.fetch_order(order_id)

# Don't do this:
class OrderManager:
    def getOrderById(self, orderId):  # Different naming convention
        return self._database.fetchOrder(orderId)
```

### Example 2: Modernizing Gradually

```python
# Old style in codebase (Python 2 era):
def process_items(items):
    # type: (List[str]) -> Dict[str, int]
    ...

# When adding new code, use modern type hints:
def process_orders(orders: list[Order]) -> dict[str, int]:
    ...

# This is okay because:
# 1. It's objectively better (native Python 3 syntax)
# 2. It's moving the codebase forward
# 3. Both styles can coexist during migration
```

### Example 3: Comment Style

```python
# If the file has this style of comments:
################################################################################
# Section: User Authentication
################################################################################

def login(username: str, password: str) -> User:
    ...

# Match it in your new code:
################################################################################
# Section: User Authorization
################################################################################

def check_permission(user: User, resource: str) -> bool:
    ...

# Don't introduce a different style:
# ===================================================================
# Section: User Authorization
# ===================================================================
```

## Balance Between Global and Local Style

1. **Global rules (this guide)** take precedence for:
   - Safety issues (e.g., mutable default arguments)
   - Readability fundamentals (e.g., line length)
   - Compatibility concerns (e.g., import conventions)

2. **Local consistency** takes precedence for:
   - Variable naming patterns within a module
   - Comment formatting styles
   - Internal helper function organization
   - Choices not specified in the global guide

## Improving the Codebase

It's okay to gradually improve code style, but:

1. **Make consistency fixes separate** from functional changes
2. **Get team buy-in** before large-scale style changes
3. **Use automatic tools** (Black, Pyink) for mechanical changes
4. **Document exceptions** when you must deviate from style guide
5. **Educate team members** about new patterns

## Key Takeaways

### DO:
- ✅ Match surrounding code style
- ✅ Follow this guide for global standards
- ✅ Gradually modernize using better patterns
- ✅ Use automatic formatters for consistency
- ✅ Consider reader's experience
- ✅ Discuss style changes with the team

### DON'T:
- ❌ Introduce drastically different styles
- ❌ Mix multiple string quote styles in one file
- ❌ Use personal preference to override local patterns
- ❌ Make large style changes without discussion
- ❌ Prioritize cleverness over clarity
- ❌ Blindly copy old patterns without understanding them

## Final Thoughts

**Code is read much more often than it is written.**

Every line of code you write will be read by:
- Future you (in 6 months)
- Your teammates
- Code reviewers
- New team members
- AI coding assistants

Make it easy for them.

**Be consistent. Be readable. Be thoughtful.**

---

## Related Resources

- [Quick Reference](00-quick-reference.md) - Essential rules for daily use
- [Python Language Rules](02-python-language-rules.md) - What Python features to use
- [Python Style Rules](03-python-style-rules.md) - How to format your code
- [Full Guide Summary](README.md) - Complete overview

---

> "Consistency is important, but it serves the code, not the other way around."
>
> — Google Python Style Guide
