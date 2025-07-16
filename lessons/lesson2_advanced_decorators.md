# Lesson 2: Advanced Decorators with Arguments and Return Values

## Handling Function Arguments

Real functions often take arguments. Our decorators need to handle this:

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"Function returned: {result}")
        return result
    return wrapper

@my_decorator
def add_numbers(a, b):
    """Adds two numbers together"""
    return a + b

result = add_numbers(3, 5)
print(f"Final result: {result}")
```

## The `*args` and `**kwargs` Pattern

- **`*args`**: Captures positional arguments as a tuple
- **`**kwargs`**: Captures keyword arguments as a dictionary
- This allows the decorator to work with any function signature

## Why Use `@functools.wraps`?

Without `@wraps`, decorated functions lose their metadata:

```python
# WITHOUT @wraps
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@bad_decorator
def my_function():
    """This is my function"""
    pass

print(my_function.__name__)  # Output: "wrapper" ❌
print(my_function.__doc__)   # Output: None ❌

# WITH @wraps
from functools import wraps

def good_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@good_decorator  
def my_function():
    """This is my function"""
    pass

print(my_function.__name__)  # Output: "my_function" ✅
print(my_function.__doc__)   # Output: "This is my function" ✅
```

## Building a Timer Decorator

Let's create our time measurement decorator step by step:

```python
import time
from functools import wraps

def measure_time(func):
    """Decorator that measures function execution time"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Record start time
        start_time = time.time()
        
        # Execute the original function
        result = func(*args, **kwargs)
        
        # Record end time
        end_time = time.time()
        
        # Calculate execution time
        execution_time = end_time - start_time
        
        # Return both result and timing
        return result, execution_time
    
    return wrapper
```

## Using Our Timer Decorator

```python
@measure_time
def slow_function(n):
    """A function that takes some time to complete"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

# Call the decorated function
result, time_taken = slow_function(100000)
print(f"Result: {result}")
print(f"Time taken: {time_taken:.6f} seconds")
```

## Multiple Return Values

Our decorator returns a tuple `(result, execution_time)`. This pattern is useful when you want to:
- Keep the original function's return value
- Add additional information (like timing)

## Decorator Templates

### Basic Template
```python
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Before function execution
        result = func(*args, **kwargs)
        # After function execution  
        return result
    return wrapper
```

### Template with Logging
```python
def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper
```

### Template with Error Handling
```python
def safe_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return None
    return wrapper
```

## Practice Exercises

### Exercise 1: Debug Decorator
Create a decorator that prints the function name and its arguments before calling it:

```python
@debug
def multiply(x, y):
    return x * y

multiply(3, 4)
# Should print: "Calling multiply with args: (3, 4), kwargs: {}"
# Should return: 12
```

### Exercise 2: Retry Decorator
Create a decorator that retries a function up to 3 times if it raises an exception:

```python
@retry(max_attempts=3)
def unreliable_function():
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise Exception("Random failure!")
    return "Success!"
```

---

## Key Takeaways

- ✅ Use `*args, **kwargs` to handle any function signature
- ✅ Always use `@functools.wraps(func)` to preserve metadata
- ✅ Decorators can modify arguments, return values, or add side effects
- ✅ Return tuples when you need to provide additional information
- ✅ Test your decorators with functions that have different signatures
