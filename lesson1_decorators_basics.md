# Lesson 1: Introduction to Python Decorators

## What is a Decorator?

A **decorator** is a design pattern in Python that allows you to modify or extend the behavior of functions or classes without permanently modifying their code. Think of it as "wrapping" a function with additional functionality.

## Basic Decorator Syntax

### Method 1: Using @ Symbol (Recommended)
```python
@decorator_function
def my_function():
    pass
```

### Method 2: Manual Application
```python
def my_function():
    pass

my_function = decorator_function(my_function)
```

## Simple Decorator Example

```python
def my_decorator(func):
    def wrapper():
        print("Something before the function")
        func()
        print("Something after the function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# When called:
say_hello()
# Output:
# Something before the function
# Hello!
# Something after the function
```

## How Decorators Work

1. **Decorator receives a function** as an argument
2. **Wrapper function** is defined inside the decorator
3. **Wrapper calls the original function** and adds extra behavior
4. **Decorator returns the wrapper** function

## Real-World Analogy

Think of a decorator like gift wrapping:
- The **gift** is your original function
- The **wrapping paper** is the decorator
- The **wrapped gift** is the enhanced function you get back

## Key Points to Remember

- ✅ Decorators are functions that take functions as arguments
- ✅ They return a new function (usually a wrapper)
- ✅ The original function is preserved but enhanced
- ✅ Multiple decorators can be applied to one function
- ✅ Use `@functools.wraps(func)` to preserve function metadata

## Next Steps

In the next lesson, we'll learn how to create decorators that accept arguments and return values!

---

## Practice Exercise

Try to create a decorator that prints "Starting..." before a function runs and "Finished!" after it completes.

```python
# Your code here
def timing_decorator(func):
    # Fill in the implementation
    pass

@timing_decorator
def test_function():
    print("This is the main function")

test_function()
# Expected output:
# Starting...
# This is the main function  
# Finished!
```
