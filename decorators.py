from typing import List, Callable

included_method_names: List[Callable] | None = None


def include_plot(func):
    global included_method_names
    if included_method_names:
        included_method_names.append(func.__name__)
    else:
        included_method_names = [func.__name__]
    return func
