import time
from functools import wraps


def timer_wrapper(func):
    
    @wraps(func)
    def timer(*args, **kwargs):
        start_time = time.time() 
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"--- Выполнено за {end_time - start_time:.4f} секунд ---")
        return result
    return timer
