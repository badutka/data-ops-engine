import time
import statistics
from functools import wraps


def measure_execution_time(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            execution_times = []
            for _ in range(n):
                start_time = time.perf_counter()
                func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                execution_times.append(execution_time)

            mean = statistics.mean(execution_times)
            standard_deviation = statistics.stdev(execution_times)

            print(f"Average execution time: {strftime(mean)} ± {strftime(standard_deviation)}")
            print(f"Average execution time: {mean:.6f}s ± {standard_deviation:.6f}s")

            return execution_times

        return wrapper

    return decorator


def timefunc(cls=None):
    def timefunc_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            return_value = func(*args, **kwargs)
            print(f"Elapsed: {strftime(time.perf_counter() - start_time)} "
                  f"(function/method: {func.__module__}.{cls + '.' if cls is not None else ''}{func.__name__})")

            return return_value

        return wrapper

    return timefunc_decorator


def strftime(seconds):
    """
    start_time = time.perf_counter()
    $END$
    print(f"Elapsed: {strftime(time.perf_counter() - start_time)}")
    :param seconds:
    :return:
    """
    hours, rem = divmod(seconds, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, milliseconds = divmod(rem, 1)
    milliseconds = milliseconds * 1000
    milliseconds, microseconds = divmod(milliseconds, 1)
    microseconds = microseconds * 1000

    return f"{int(seconds):01d}s {int(milliseconds):01d}ms {int(microseconds):01d}µs."


def strfdelta(tdelta):
    """
    start = timeit.default_timer()
    $END$
    print(f"Elapsed: {strfdelta(timedelta(seconds=(timeit.default_timer() - start)))}")
    :param tdelta:
    :return:
    """
    hours, rem = divmod(tdelta.total_seconds(), 3600)
    minutes, rem = divmod(rem, 60)
    seconds, milliseconds = divmod(rem, 1)
    milliseconds = milliseconds * 1000
    milliseconds, microseconds = divmod(milliseconds, 1)
    microseconds = microseconds * 1000

    return f"{int(hours):01d} H, {int(minutes):01d} m, {int(seconds):01d} s, {int(milliseconds):01d} ms, {int(microseconds):01d} µs."
