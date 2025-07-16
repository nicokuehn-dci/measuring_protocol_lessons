import time
from functools import wraps


def measure_time(func):
    """
    A decorator that measures the execution time of a function.
    
    Usage:
        @measure_time
        def my_function():
            # function code here
            pass
            
        result, execution_time = my_function()
    
    Returns:
        tuple: (function_result, execution_time_in_seconds)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


# Example usage of the decorator
@measure_time
def example_function(n):
    """Simple example function that sleeps for a bit"""
    time.sleep(0.1)  # Simulate some work
    return n * 2


# Demonstrate the decorator
if __name__ == "__main__":
    print("Decorator Example")
    print("=" * 30)
    
    # Call the decorated function
    result, execution_time = example_function(5)
    
    print(f"Function result: {result}")
    print(f"Execution time: {execution_time:.4f} seconds")
