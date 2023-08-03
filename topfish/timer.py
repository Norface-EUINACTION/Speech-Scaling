""" Function to show time neeed for the evaluation of a function.
"""

import functools
import time


def timer(function):
    """ Function to show time neeed for the evaluation of a function.
        Args:
            function (function): Function that is executed.
        Returns:
            result (object): Result of the executed function.
    """

    @functools.wraps(function)
    def time_measurement(*args, **kwargs):
        """ Function to measure time elapsed by executing a function.
        """
        time_start = time.perf_counter()
        result = function(*args, **kwargs)
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        print("Finished function: {} in {} seconds.".format(repr(function.__name__), round(time_elapsed, 2)))
        return result

    return time_measurement